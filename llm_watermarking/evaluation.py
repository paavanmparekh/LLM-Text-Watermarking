"""
evaluation.py — Evaluation metrics for generated text.

Metrics
-------
* Perplexity       — model-based, self-evaluated
* Distinct-N       — lexical diversity (n-gram overlap); stored but not in
                     the pipeline summary CSV
* Log Diversity    — Herdan's C: log(unique types) / log(total tokens),
                     the displayed diversity metric in the summary
* Shannon entropy  — expectation over distribution (bits)
* Empirical entropy — average surprisal of sampled tokens (bits)

Usage
-----
    from llm_watermarking.evaluation import Evaluator

    evaluator = Evaluator(model, tokenizer)
    result = evaluator.evaluate(generation_data)
    # result["eval"]["perplexity"], result["eval"]["log_diversity"], ...
"""

import math
from typing import Any, Dict, List

import torch


class Evaluator:
    """
    Computes text quality and information-theoretic metrics.

    Parameters
    ----------
    model : PreTrainedModel
        The language model used for perplexity computation.
    tokenizer : PreTrainedTokenizer
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Lexical diversity
    # ------------------------------------------------------------------

    def distinct_n(self, text: str, n: int) -> float:
        """
        Distinct-N: fraction of unique n-grams in *text*.

        Returns 0.0 for empty or too-short texts.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < n:
            return 0.0

        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0

        return len(set(ngrams)) / len(ngrams)

    def log_diversity(self, text: str) -> float:
        """
        Log Diversity (Herdan's C): log(unique types) / log(total tokens).

        This is the standard log-type-token ratio, bounded in [0, 1] and
        more robust to text length than the raw log(unique)/N variant.
        Returns 0.0 for texts shorter than 2 tokens.
        """
        tokens = self.tokenizer.tokenize(text)
        n = len(tokens)
        if n < 2:
            return 0.0
        unique = len(set(tokens))
        if unique < 2:
            return 0.0

        return math.log(unique) / math.log(n)

    # ------------------------------------------------------------------
    # Perplexity
    # ------------------------------------------------------------------

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity of *text* under the reference model.

        Returns float('inf') if the text exceeds the model's context window.
        """
        encodings = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        seq_len = encodings.input_ids.shape[1]

        if seq_len > self.model.config.max_position_embeddings:
            return float("inf")

        with torch.no_grad():
            outputs = self.model(encodings.input_ids, labels=encodings.input_ids)
            ppl = torch.exp(outputs.loss)

        return ppl.item()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(self, generation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augment *generation_data* with an ``"eval"`` sub-dict.

        Parameters
        ----------
        generation_data : dict
            Output from ``LLMGenerator.generate_text()``.

        Returns
        -------
        The same dict with an added ``"eval"`` key containing:
            log_diversity (Herdan's C),
            perplexity,
        """
        text = generation_data["generated_text"]

        generation_data["eval"] = {
            "log_diversity":          self.log_diversity(text),
            "perplexity":             self.compute_perplexity(text),
        }
        return generation_data
