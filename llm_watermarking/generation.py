"""
generation.py — Text generation with per-step entropy tracking.

Classes
-------
BaselineLogitTracker
    A LogitsProcessor that records Shannon entropy of each token's
    distribution and the empirical surprisal once the token is sampled.

LLMGenerator
    Wraps a HuggingFace causal LM and provides generate_text().
    Accepts an optional custom LogitsProcessor hook so future
    watermarking schemes can be plugged in with zero code changes.

Usage
-----
    from llm_watermarking.generation import LLMGenerator

    generator = LLMGenerator(model, tokenizer)
    result = generator.generate_text(prompt)
    print(result["generated_text"])
    print(result["total_empirical_entropy"])
"""

import math
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper, TopPLogitsWarper

from .config import Config, config as default_config


# ============================================================ #
#  BaselineLogitTracker                                         #
# ============================================================ #

class BaselineLogitTracker(LogitsProcessor):
    """
    Intercepts the raw logits at every decoding step to record:

    * shannon_entropies        — H(D_i) in bits for each step i
    * token_surprisals         — -log₂ p(x_i | x_{<i}) after sampling
    * total_empirical          — final empirical entropy H_e(x)

    The tracker is inserted into HuggingFace's LogitsProcessorList,
    which is called *before* sampling, so the surprisal for step i is
    computed at step i+1 (when input_ids already contains x_i).
    """

    def __init__(self, temperature: float = 1.0, top_p: float = 1.0) -> None:
        self.warpers = LogitsProcessorList()
        if temperature is not None and temperature != 1.0:
            self.warpers.append(TemperatureLogitsWarper(temperature))
        if top_p is not None and top_p < 1.0:
            self.warpers.append(TopPLogitsWarper(top_p))
        self.reset()

    def reset(self) -> None:
        self.shannon_entropies: List[float] = []
        self.token_surprisals: List[float] = []
        self.total_empirical: float = 0.0
        self._last_log_probs: Optional[torch.Tensor] = None

    # LogitsProcessor protocol ---------------------------------------- #

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # --- 1. Empirical contribution from the PREVIOUS step ---------- #
        if self._last_log_probs is not None:
            sampled_token = input_ids[0, -1]
            surprisal = -self._last_log_probs[0, sampled_token].item()

            self.token_surprisals.append(surprisal)
            self.total_empirical += surprisal

        # Apply temperature and top_p to a CLONE of the scores
        warped_scores = self.warpers(input_ids, scores.clone())

        # --- 2. Shannon entropy of the CURRENT distribution ------------ #
        probs = torch.nn.functional.softmax(warped_scores, dim=-1)
        # Use log2 and clamp to handle zeros from top_p
        log_probs_b2 = torch.log2(torch.clamp(probs, min=1e-10))
        shannon_h = -(probs * log_probs_b2).sum(dim=-1)
        self.shannon_entropies.append(shannon_h.item())

        # --- 3. Save for next step (using warped log probs) -------------- #
        self._last_log_probs = log_probs_b2.detach()

        return scores  # pass through — we never modify logits here


# ============================================================ #
#  LLMGenerator                                                 #
# ============================================================ #

class LLMGenerator:
    """
    Wrapper around a HuggingFace causal LM for watermarking research.

    Tracks per-token entropy / surprisal via BaselineLogitTracker and
    allows callers to inject additional LogitsProcessor objects (e.g.
    a watermarking processor) through the *custom_processor* argument.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    cfg : Config, optional
    """

    def __init__(self, model, tokenizer, cfg: Config = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or default_config

    # ------------------------------------------------------------------ #

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        custom_processor: Optional[LogitsProcessor] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response for *prompt* and return rich metrics.

        Parameters
        ----------
        prompt : str
            Input text (including [INST] tags for Mistral).
        max_new_tokens : int, optional
            Override config default.
        temperature : float, optional
            Override config default.
        top_p : float, optional
            Override config default.
        custom_processor : LogitsProcessor, optional
            An additional processor injected *after* the tracker.
            Use this to plug in watermarking schemes.

        Returns
        -------
        dict with keys:
            prompt, generated_text, num_tokens,
            shannon_entropies, token_surprisals,
            total_empirical_entropy, total_shannon_entropy
        """
        max_new_tokens = max_new_tokens or self.cfg.max_new_tokens
        temperature    = temperature    or self.cfg.temperature
        top_p          = top_p          or self.cfg.top_p

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        tracker = BaselineLogitTracker(
            temperature=temperature if temperature is not None else 1.0,
            top_p=top_p if top_p is not None else 1.0
        )
        processors = LogitsProcessorList([tracker])
        if custom_processor:
            processors.append(custom_processor)

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=temperature,
                top_p=top_p,
                logits_processor=processors,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generation_time = time.time() - t0

        input_length  = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Align: surprisals are 1 step behind shannon_entropies
        n = len(tracker.token_surprisals)

        return {
            "prompt":                       prompt,
            "generated_text":               generated_text,
            "num_tokens":                   n,
            "generation_time":              round(generation_time, 2),
            "shannon_entropies":            tracker.shannon_entropies[:n],
            "token_surprisals":             tracker.token_surprisals,
            "total_empirical_entropy":      tracker.total_empirical,
            "total_shannon_entropy":        sum(tracker.shannon_entropies[:n]),
        }
