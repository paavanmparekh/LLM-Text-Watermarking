"""
watermarks/undetectable.py — Dummy Undetectable Watermarking scheme (Phase 1).

Implements binary-token generation as described in Miranda et al. (2023),
Section 4.3 Algorithm 3, but with random.random() in place of a real PRF.
This "dummy" version preserves the binary decomposition structure to validate
that the binarized sampler produces coherent text with correct entropy metrics.

Detection is NOT implemented in this phase.

Usage
-----
    from llm_watermarking.watermarks.undetectable import UndetectableWatermark

    scheme = UndetectableWatermark(cfg)
    result = scheme.generate(model, tokenizer, prompt)
"""

import math
import random
import time
from typing import Any, Dict, List, Optional

import torch

from ..binarizer import build_binary_vocab, compute_bit_probs
from ..config import Config, config as default_config


class UndetectableWatermark:
    """
    Dummy Undetectable watermarking via binary token decomposition.

    Generates text by decomposing each token choice into `bit_length` binary
    decisions and sampling each bit with random.random() (placeholder for a
    real PRF keyed with a secret key in later phases).

    Entropy metrics produced
    ------------------------
    shannon_entropies : per-token Shannon entropy (bits), computed as the sum
        of per-bit binary entropies H(p0/(p0+p1)) over the `bit_length` bits.
    token_surprisals  : per-token empirical surprisal (bits), computed as
        -log2(product of chosen-bit probabilities) = -sum(log2(p_chosen_bit)).
    """

    NAME = "undetectable-dummy"

    def __init__(self, cfg: Config = None) -> None:
        self.cfg = cfg or default_config

    # ------------------------------------------------------------------ #

    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response using the binarized sampling loop.

        Returns a dict with the same keys as LLMGenerator.generate_text(),
        plus `mode` and `bit_length` for identification.
        """
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens

        # ---- binarisation setup ---------------------------------------- #
        bit_length, _, _ = build_binary_vocab(tokenizer)
        vocab_size = len(tokenizer)
        print(f"  [Binarized] bit_length={bit_length}, vocab_size={vocab_size}")

        # ---- tokenise prompt ------------------------------------------- #
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]                        # (1, prompt_len)
        attn_mask = torch.ones_like(input_ids)

        past: Any = None

        # ---- metric accumulators --------------------------------------- #
        shannon_entropies: List[float] = []   # per token: sum of per-bit H
        token_surprisals: List[float] = []    # per token: -log2 p(chosen path)
        cumulative_empirical: List[float] = []
        total_empirical: float = 0.0
        generated_ids: List[int] = []

        t0 = time.time()

        for step in range(max_new_tokens):
            # ---- one forward pass -------------------------------------- #
            with torch.no_grad():
                if past is not None:
                    output = model(
                        input_ids[:, -1:],
                        past_key_values=past,
                        attention_mask=attn_mask,
                    )
                else:
                    output = model(input_ids, attention_mask=attn_mask)

            # Softmax over true vocab slice; keep on CPU for bit-loop
            probs = torch.nn.functional.softmax(
                output.logits[:, -1, :vocab_size], dim=-1
            ).cpu()[0]  # shape: [vocab_size]

            past = output.past_key_values

            # ---- binary decomposition loop ----------------------------- #
            token_id = 0
            token_log_prob = 0.0    # sum of log2(p_chosen_bit) for this token
            token_shannon = 0.0     # sum of H(bit) for this token

            for bit_idx in range(bit_length):
                p0, p1 = compute_bit_probs(probs, bit_idx, bit_length, token_id)
                total = (p0 + p1).item()

                if total < 1e-12:
                    # No valid tokens under this prefix — stop early
                    break

                prob_1 = p1.item() / total
                prob_0 = 1.0 - prob_1

                # Shannon entropy of this binary split (bits)
                h_bit = 0.0
                if prob_0 > 1e-12:
                    h_bit -= prob_0 * math.log2(prob_0)
                if prob_1 > 1e-12:
                    h_bit -= prob_1 * math.log2(prob_1)
                token_shannon += h_bit

                # Sample bit with random.random() (PRF placeholder)
                token_id <<= 1
                if random.random() < prob_1:
                    token_id += 1
                    token_log_prob += math.log2(prob_1) if prob_1 > 1e-12 else -1000.0
                else:
                    token_log_prob += math.log2(prob_0) if prob_0 > 1e-12 else -1000.0

            # ---- clamp to valid range ---------------------------------- #
            token_id = min(max(token_id, 0), vocab_size - 1)

            # ---- record metrics ---------------------------------------- #
            surprisal = -token_log_prob          # empirical surprisal in bits
            total_empirical += surprisal
            token_surprisals.append(surprisal)
            cumulative_empirical.append(total_empirical)
            shannon_entropies.append(token_shannon)
            generated_ids.append(token_id)

            # ---- feed token back to model ------------------------------ #
            next_token = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=-1)

            # ---- EOS check --------------------------------------------- #
            if token_id == tokenizer.eos_token_id:
                break

        generation_time = time.time() - t0
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        n = len(token_surprisals)

        return {
            "prompt":                       prompt,
            "generated_text":               generated_text,
            "num_tokens":                   n,
            "generation_time":              round(generation_time, 2),
            "shannon_entropies":            shannon_entropies,
            "token_surprisals":             token_surprisals,
            "cumulative_empirical_entropy": cumulative_empirical,
            "total_empirical_entropy":      total_empirical,
            "total_shannon_entropy":        sum(shannon_entropies),
            "top_k_distributions":          [],   # not tracked in binary mode
            "mode":                         self.NAME,
            "bit_length":                   bit_length,
        }
