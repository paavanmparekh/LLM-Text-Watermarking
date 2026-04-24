import hashlib
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

from ...binarizer import build_binary_vocab, compute_bit_probs
from ...config import Config, config as default_config
from .prc import LDPCPRC0, LDPCPRC0Params


def _seed_to_uint64(seed: bytes) -> int:
    digest = hashlib.sha256(seed).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _mask_from_seed(seed: bytes, *, block_idx_1based: int, n_bits: int) -> int:
    """
    Deterministically derive a uniform-looking n-bit mask a_block from a seed.
    """
    if block_idx_1based <= 0:
        raise ValueError("block_idx_1based must be >= 1")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")

    out = 0
    produced = 0
    counter = 0
    while produced < n_bits:
        h = hashlib.sha256()
        h.update(seed)
        h.update(b"PRC_MASK")
        h.update(block_idx_1based.to_bytes(4, "big", signed=False))
        h.update(counter.to_bytes(4, "big", signed=False))
        chunk = h.digest()
        for byte in chunk:
            for bit in range(8):
                if produced >= n_bits:
                    break
                if (byte >> bit) & 1:
                    out |= 1 << produced
                produced += 1
        counter += 1
    return out


def _bernoulli_param(p1: float, x_bit: int) -> float:
    """
    Figure 3 / Algorithm 2 line 7:
      t <- Ber(p - (-1)^x * min(p, 1-p))
    where p is the model's conditional probability of bit=1.
    """
    m = p1 if p1 <= 0.5 else (1.0 - p1)
    if x_bit == 0:
        q = p1 - m
    else:
        q = p1 + m
    if q < 0.0:
        return 0.0
    if q > 1.0:
        return 1.0
    return q


class PRCWatermark:
    """
    PRC-based watermarking (Section 7 / Figure 3), adapted to this repo's
    binary-vocabulary sampling interface.

    We treat the model as emitting a stream of binary decisions (bits) via
    `binarizer.compute_bit_probs`, and embed PRC codeword bits into that stream
    using the Ber(.) rule from Figure 3.
    """

    NAME = "prc"

    def __init__(
        self,
        cfg: Config = None,
        key: bytes = None,
        lambda_entropy: float = 0.0,
        prc_params: Optional[LDPCPRC0Params] = None,
    ) -> None:
        self.cfg = cfg or default_config
        self.lambda_entropy = lambda_entropy

        self.key = key if key is not None else os.urandom(32)

        if prc_params is None:
            # Practical defaults with *low false positives*.
            #
            # The PRC detector checks whether a length-n block "looks like" an
            # LDPC-PRC0 encoding by testing whether its syndrome has unusually
            # low weight. If r is small, this event is not rare for uniform
            # random strings, and scanning multiple blocks can lead to many
            # false positives on baseline text.
            #
            # Using a larger r (≈ n) and a constant zeta keeps the decode
            # acceptance probability for random strings exponentially small,
            # which is the intended regime in Section 5/7 of the paper.
            prc_params = LDPCPRC0Params(
                n=256,
                g=64,
                t=6,
                r=253,     # ≈ 0.99n
                eta=0.02,
                zeta=0.25, # threshold = (1/2 - zeta) r = 0.25 r
            )

        self.prc_params = prc_params
        self._prc = LDPCPRC0(self.prc_params, seed=_seed_to_uint64(self.key))
        self._sk, self._pk = self._prc.keygen()

    def _mask(self, block_idx_1based: int) -> int:
        return _mask_from_seed(self.key, block_idx_1based=block_idx_1based, n_bits=self.prc_params.n)

    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature = self.cfg.temperature
        top_p = self.cfg.top_p

        warpers = LogitsProcessorList()
        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p))

        bit_length, _, _ = build_binary_vocab(tokenizer)
        vocab_size = len(tokenizer)
        special_token_ids = sorted({
            tid for tid in getattr(tokenizer, "all_special_ids", [])
            if tid is not None and 0 <= tid < vocab_size
        })

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attn_mask = torch.ones_like(input_ids)
        past: Any = None

        generated_ids: List[int] = []
        bitstring: str = ""
        bit_surprisals: List[float] = []

        n = int(self.prc_params.n)
        block_idx = 1
        j_in_block = 0
        x_block = self._prc.encode(self._pk) ^ self._mask(block_idx)

        eos_id = getattr(tokenizer, "eos_token_id", None)

        t0 = time.time()

        for step in range(max_new_tokens):
            with torch.no_grad():
                if past is not None:
                    output = model(
                        input_ids[:, -1:],
                        past_key_values=past,
                        attention_mask=attn_mask,
                    )
                else:
                    output = model(input_ids, attention_mask=attn_mask)

            logits = output.logits[:, -1, :vocab_size]
            if len(warpers) > 0:
                logits = warpers(input_ids, logits)
            if special_token_ids:
                logits[:, special_token_ids] = float("-inf")

            probs = torch.nn.functional.softmax(logits, dim=-1).cpu()[0]
            past = output.past_key_values

            token_id = 0
            for bit_idx in range(bit_length):
                p0, p1 = compute_bit_probs(probs, bit_idx, bit_length, token_id)
                total_mass = (p0 + p1).item()
                if total_mass == 0.0:
                    break

                prob_1 = p1.item() / total_mass

                if j_in_block >= n:
                    block_idx += 1
                    j_in_block = 0
                    x_block = self._prc.encode(self._pk) ^ self._mask(block_idx)

                xj = (x_block >> j_in_block) & 1
                q = _bernoulli_param(prob_1, int(xj))
                chosen = 1 if torch.rand(1).item() <= q else 0

                chosen_prob = prob_1 if chosen == 1 else (1.0 - prob_1)
                chosen_prob = max(chosen_prob, 1e-12)
                bit_surprisals.append(-math.log2(chosen_prob))

                token_id = (token_id << 1) | chosen
                bitstring += str(chosen)
                j_in_block += 1

            token_id = min(max(token_id, 0), vocab_size - 1)
            generated_ids.append(token_id)

            if eos_id is not None and token_id == int(eos_id):
                break

            next_token = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=-1)

        generation_time = time.time() - t0
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "num_tokens": len(generated_ids),
            "generation_time": round(generation_time, 2),
            "generated_ids": generated_ids,
            "watermark_bitstring": bitstring,
            "bit_length": bit_length,
            "bit_surprisals": bit_surprisals,
            "total_empirical_entropy": sum(bit_surprisals),
            "mode": self.NAME,
            "key_hex": self.key.hex(),  # seed for deterministic Setup()
            "prc_params": {
                "n": self.prc_params.n,
                "g": self.prc_params.g,
                "t": self.prc_params.t,
                "r": self.prc_params.r,
                "eta": self.prc_params.eta,
                "zeta": self.prc_params.zeta,
            },
        }
