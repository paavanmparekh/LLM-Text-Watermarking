"""
evaluation.py — Evaluation metrics for generated text.

Metrics
-------
* Perplexity       — model-based, self-evaluated
* Distinct-N       — lexical diversity (n-gram overlap)
* Shannon entropy  — expectation over distribution (bits)
* Empirical entropy — average surprisal of sampled tokens (bits)

Usage
-----
    from llm_watermarking.evaluation import Evaluator

    evaluator = Evaluator(model, tokenizer)
    result = evaluator.evaluate(generation_data)
    # result["eval"]["perplexity"], result["eval"]["distinct_2"], ...
"""

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
            distinct_1, distinct_2, distinct_3,
            perplexity,
            avg_shannon_entropy, avg_empirical_entropy,
            total_shannon_entropy, total_empirical_entropy
        """
        text = generation_data["generated_text"]
        shannon  = generation_data.get("shannon_entropies", [])
        surprisals = generation_data.get("token_surprisals", [])

        def _safe_mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        generation_data["eval"] = {
            "distinct_1":             self.distinct_n(text, 1),
            "distinct_2":             self.distinct_n(text, 2),
            "distinct_3":             self.distinct_n(text, 3),
            "perplexity":             self.compute_perplexity(text),
            "avg_shannon_entropy":    _safe_mean(shannon),
            "avg_empirical_entropy":  _safe_mean(surprisals),
            "total_shannon_entropy":  generation_data.get("total_shannon_entropy", 0.0),
            "total_empirical_entropy": generation_data.get("total_empirical_entropy", 0.0),
        }
        return generation_data
