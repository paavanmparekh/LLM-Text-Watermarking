"""
watermarks/undetectable.py — Undetectable Watermarking scheme (Christ et al. 2023).

Implements Algorithm 3 (Wat_sk) from Section 4.3:
  - Phase 1 (H < λ): sample tokens normally from the model, accumulating empirical
    entropy H += -log2 p(x_i). This "burn-in" collects λ bits of true randomness.
  - Once H ≥ λ: freeze r = (x_1,...,x_i) as the PRF seed.
  - Phase 2 (H ≥ λ): x_i = 1 iff F_sk(r, global_bit_pos) ≤ p_i(1).

PRF: HMAC-SHA256 keyed on `sk`, input = JSON([r_list, global_bit_pos]).
     Output mapped to [0,1] by treating the full 256-bit hash as an int
     and dividing by 2^256 - 1.

λ choice: OrZamir's repo skips the entropy phase entirely (uses position index
as PRF seed). We implement the paper faithfully with λ=20 bits (default) — this
means roughly the first ~3-5 tokens (depending on model entropy) are sampled
normally before watermarking begins. We default to λ=10 bits (for robust ML
dataset metrics under 300 tokens), though the paper originally uses 20 bits for
strict cryptographic guarantees.

Usage
-----
    scheme = UndetectableWatermark(cfg, key=b"...", lambda_entropy=10.0)
    result = scheme.generate(model, tokenizer, prompt)
    # result["bit_trace"]  — list of per-bit records for detection
    # result["key_hex"]    — hex-encoded secret key
"""

import hashlib
import hmac
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from ...binarizer import build_binary_vocab, compute_bit_probs
from ...config import Config, config as default_config


# ------------------------------------------------------------------ #
#  PRF                                                                #
# ------------------------------------------------------------------ #

def _prf(key: bytes, r: Tuple, global_bit_pos: int) -> float:
    """
    Keyed PRF: (key, r, bit_pos) → float in [0, 1].

    Uses HMAC-SHA256. The context r is the tuple of token IDs generated in
    Phase 1 (frozen once H ≥ λ). global_bit_pos indexes every binary decision
    across all Phase-2 tokens (not reset per token).

    Why HMAC over random.seed: HMAC is a cryptographically secure PRF —
    without the key, its outputs are computationally indistinguishable from
    random, which is the foundation of the undetectability proof.
    """
    msg = json.dumps([list(r), global_bit_pos], separators=(',', ':')).encode()
    digest = hmac.new(key, msg, hashlib.sha256).digest()
    # Map full 256 bits to [0, 1]
    return int.from_bytes(digest, "big") / ((1 << 256) - 1)


# ------------------------------------------------------------------ #
#  Watermark scheme                                                   #
# ------------------------------------------------------------------ #

class UndetectableWatermark:
    """
    Undetectable watermarking via binary token decomposition (Christ et al. 2023).

    Parameters
    ----------
    cfg : Config
    key : bytes, optional
        Secret key. If None, a fresh random 32-byte key is generated.
    lambda_entropy : float
        Security parameter λ (bits). Phase-1 runs until H ≥ λ.
        Default 10 bits — balanced for robust ML testing on short sequences.
        OrZamir's reference implementation omits Phase 1 entirely (λ=0),
        trading the formal entropy-seed argument for simplicity.
    """

    NAME = "undetectable"

    def __init__(
        self,
        cfg: Config = None,
        key: bytes = None,
        lambda_entropy: float = 10.0,
    ) -> None:
        self.cfg = cfg or default_config
        self.key = key if key is not None else os.urandom(32)
        self.lambda_entropy = lambda_entropy

    # ------------------------------------------------------------------ #

    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate watermarked text using Algorithm 3 (Wat_sk).

        Returns
        -------
        dict with all standard generation keys plus:
            key_hex        : hex-encoded secret key
            lambda_entropy : λ value used
            phase1_tokens  : number of tokens in burn-in phase
            all_tokens     : generated token strings (for detector anchor search)
            generated_ids  : generated token IDs (for detector bit extraction)
        """
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens

        bit_length, _, _ = build_binary_vocab(tokenizer)
        vocab_size = len(tokenizer)
        print(f"  [Undetectable] bit_length={bit_length}, vocab_size={vocab_size}, λ={self.lambda_entropy}")

        # ---- tokenise prompt ------------------------------------------- #
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]
        attn_mask = torch.ones_like(input_ids)
        past: Any = None

        # ---- state ----------------------------------------------------- #
        H: float = 0.0
        r: Optional[Tuple] = None

        # ---- metric accumulators --------------------------------------- #
        shannon_entropies: List[float] = []
        token_surprisals: List[float] = []
        cumulative_empirical: List[float] = []
        total_empirical: float = 0.0
        generated_ids: List[int] = []
        phase1_count: int = 0

        t0 = time.time()

        for step in range(max_new_tokens):
            # ---- forward pass ------------------------------------------ #
            with torch.no_grad():
                if past is not None:
                    output = model(
                        input_ids[:, -1:],
                        past_key_values=past,
                        attention_mask=attn_mask,
                    )
                else:
                    output = model(input_ids, attention_mask=attn_mask)

            probs = torch.nn.functional.softmax(
                output.logits[:, -1, :vocab_size], dim=-1
            ).cpu()[0]
            past = output.past_key_values

            # ---- binary decomposition ---------------------------------- #
            token_id = 0
            token_log_prob = 0.0
            token_shannon = 0.0

            in_phase1 = (H < self.lambda_entropy)

            for bit_idx in range(bit_length):
                p0, p1 = compute_bit_probs(probs, bit_idx, bit_length, token_id)
                total_mass = (p0 + p1).item()

                if total_mass == 0.0:
                    break

                prob_1 = p1.item() / total_mass
                prob_0 = 1.0 - prob_1

                # Shannon entropy of this bit split
                h_bit = 0.0
                if prob_0 > 0.0:
                    h_bit -= prob_0 * math.log2(prob_0)
                if prob_1 > 0.0:
                    h_bit -= prob_1 * math.log2(prob_1)
                token_shannon += h_bit

                token_id <<= 1

                if in_phase1:
                    chosen = 1 if torch.rand(1).item() < prob_1 else 0
                    token_id += chosen
                    chosen_prob = prob_1 if chosen == 1 else prob_0
                    token_log_prob += math.log2(chosen_prob) if chosen_prob > 0.0 else -1000.0
                else:
                    prf_val = _prf(self.key, r, step * bit_length + bit_idx)
                    chosen = 1 if prf_val <= prob_1 else 0
                    token_id += chosen
                    chosen_prob = prob_1 if chosen == 1 else prob_0
                    token_log_prob += math.log2(chosen_prob) if chosen_prob > 0.0 else -1000.0

            # ---- clamp token_id ---------------------------------------- #
            token_id = min(max(token_id, 0), vocab_size - 1)

            # ---- metrics ----------------------------------------------- #
            surprisal = -token_log_prob
            total_empirical += surprisal
            token_surprisals.append(surprisal)
            cumulative_empirical.append(total_empirical)
            shannon_entropies.append(token_shannon)
            generated_ids.append(token_id)

            # ---- Phase 1 → Phase 2 transition check -------------------- #
            if in_phase1:
                phase1_count += 1
                H += surprisal
                if H >= self.lambda_entropy and r is None:
                    r = tuple(tokenizer.convert_ids_to_tokens(generated_ids))
                    print(f"  [Undetectable] Phase 1→2 at token {step+1}, H={H:.2f} bits")

            # ---- feed token back --------------------------------------- #
            next_token = torch.tensor([[token_id]], device=model.device)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=-1)

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
            "top_k_distributions":          [],
            "mode":                         self.NAME,
            "bit_length":                   bit_length,
            # ── watermark-specific ──
            "key_hex":                      self.key.hex(),
            "lambda_entropy":               self.lambda_entropy,
            "phase1_tokens":                phase1_count,
            # all_tokens and generated_ids let the detector reconstruct
            # bit decisions without any stored trace or model re-run.
            "all_tokens":                   list(tokenizer.convert_ids_to_tokens(generated_ids)),
            "generated_ids":                generated_ids,
        }
