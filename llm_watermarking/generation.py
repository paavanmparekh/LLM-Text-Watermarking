"""
generation.py — Text generation

Classes
-------
LLMGenerator
    Wraps a HuggingFace causal LM and provides generate_text().

Usage
-----
    from llm_watermarking.generation import LLMGenerator

    generator = LLMGenerator(model, tokenizer)
    result = generator.generate_text(prompt)
    print(result["generated_text"])
"""

import time
from typing import Any, Dict, Optional, List

import torch
from transformers import LogitsProcessorList, LogitsProcessor, TemperatureLogitsWarper, TopPLogitsWarper

from .config import Config, config as default_config


# ============================================================ #
#  BaselineLogitTracker                                         #
# ============================================================ #

class BaselineLogitTracker(LogitsProcessor):
    def __init__(self, temperature: float = 1.0, top_p: float = 1.0) -> None:
        self.warpers = LogitsProcessorList()
        if temperature is not None and temperature != 1.0:
            self.warpers.append(TemperatureLogitsWarper(temperature))
        if top_p is not None and top_p < 1.0:
            self.warpers.append(TopPLogitsWarper(top_p))
        self.reset()

    def reset(self) -> None:
        self.total_empirical: float = 0.0
        self._last_log_probs: Optional[torch.Tensor] = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self._last_log_probs is not None:
            sampled_token = input_ids[0, -1]
            surprisal = -self._last_log_probs[0, sampled_token].item()
            self.total_empirical += surprisal

        warped_scores = self.warpers(input_ids, scores.clone())
        probs = torch.nn.functional.softmax(warped_scores, dim=-1)
        self._last_log_probs = torch.log(torch.clamp(probs, min=1e-10)).detach()
        return scores


# ============================================================ #
#  LLMGenerator                                                 #
# ============================================================ #

class LLMGenerator:
    """
    Wrapper around a HuggingFace causal LM for standard text generation.

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

        Returns
        -------
        dict with keys:
            prompt, generated_text, num_tokens, generation_time, total_empirical_entropy
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
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return {
            "prompt":                       prompt,
            "generated_text":               generated_text,
            "num_tokens":                   len(generated_ids),
            "generation_time":              round(generation_time, 2),
            "total_empirical_entropy":      tracker.total_empirical,
        }
