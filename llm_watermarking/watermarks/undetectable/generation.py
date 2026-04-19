
import hashlib
import hmac
import math
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

from ...binarizer import build_binary_vocab, compute_bit_probs
from ...config import Config, config as default_config
from .wm_logger import logger

def _prf_binary(key: bytes, r_bits: str, global_bit_pos: int) -> float:
    msg = bytes(r_bits, "utf-8") + bytes(bin(global_bit_pos), "utf-8")
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    return int.from_bytes(digest, "big") / ((1 << 256) - 1)

class UndetectableWatermark:
    
    NAME = "undetectable"

    def __init__(
        self,
        cfg: Config = None,
        key: bytes = None,
        lambda_entropy: float = 5.0,
    ) -> None:
        self.cfg = cfg or default_config
        default_key = bytes.fromhex("7ba6c066a1f7784bf688f01556d92f7f45d2b9ec1039b4dfdfc4af07a07974f8")
        self.key = key if key is not None else default_key
        self.lambda_entropy = lambda_entropy

    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature    = self.cfg.temperature
        top_p          = self.cfg.top_p

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

        H: float = 0.0
        r: Optional[str] = None
        bitstring: str = ""
        anchor_bit_index: Optional[int] = None
        anchor_entropy_bits: Optional[float] = None

        bit_surprisals: List[float] = []
        generated_ids: List[int] = []

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

                global_bit_pos = step * bit_length + bit_idx
                prob_1 = p1.item() / total_mass
                prob_0 = 1.0 - prob_1

                token_id <<= 1

                if H < self.lambda_entropy:
                    chosen = 1 if torch.rand(1).item() <= prob_1 else 0
                    chosen_prob = prob_1 if chosen == 1 else prob_0
                    bit_surprisal = -math.log2(chosen_prob)
                    bit_surprisals.append(bit_surprisal)
                    H += bit_surprisal
                    if H >= self.lambda_entropy and r is None:
                        r = bitstring + str(chosen)
                        anchor_bit_index = global_bit_pos
                        anchor_entropy_bits = H
                        logger.info(f"  ANCHOR FOUND: r='{r}' at pos={global_bit_pos}")
                else:
                    prf_val = _prf_binary(self.key, r, global_bit_pos)
                    chosen  = 1 if prf_val <= prob_1 else 0
                    chosen_prob = prob_1 if chosen == 1 else prob_0
                    bit_surprisal = -math.log2(chosen_prob)
                    bit_surprisals.append(bit_surprisal)
                
                token_id += chosen
                bitstring += str(chosen)

            token_id = min(max(token_id, 0), vocab_size - 1)
            generated_ids.append(token_id)

            next_token = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=-1)

        generation_time = time.time() - t0
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        detector_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
        text_roundtrip_match = generated_ids == detector_token_ids
        if not text_roundtrip_match:
            logger.warning(
                "  TEXT ROUNDTRIP MISMATCH: detector tokenization does not exactly match "
                "the internally sampled token sequence."
            )

        return {
            "prompt":                       prompt,
            "generated_text":               generated_text,
            "num_tokens":                   len(generated_ids),
            "generation_time":              round(generation_time, 2),
            "bit_surprisals":               bit_surprisals,
            "total_empirical_entropy":      sum(bit_surprisals),
            "generated_ids":                generated_ids,
            "watermark_bitstring":          bitstring,
            "bit_length":                   bit_length,
            "anchor_bit_index":             anchor_bit_index,
            "anchor_length_bits":           len(r) if r is not None else 0,
            "anchor_entropy_bits":          anchor_entropy_bits,
            "text_roundtrip_match":         text_roundtrip_match,
            "detector_token_count":         len(detector_token_ids),
            "mode":                         self.NAME,
            "key_hex":                      self.key.hex(),
            "lambda_entropy":               self.lambda_entropy,
        }
