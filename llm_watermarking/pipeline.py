"""
pipeline.py — End-to-end orchestrator for the LLM Watermarking baseline.

run_pipeline()
    Iterates over prompts, calls LLMGenerator, evaluates each result,
    persists to disk, and returns both a raw-results list and a summary
    pandas DataFrame.

Usage
-----
    from llm_watermarking.pipeline import run_pipeline
    from llm_watermarking.model_loader import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer()
    results, df = run_pipeline(model, tokenizer)
"""

import json
import os
import time
from typing import List, Optional, Tuple

import pandas as pd

from .config import Config, config as default_config
from .evaluation import Evaluator
from .generation import LLMGenerator
from .prompts import PromptLoader


def run_pipeline(
    model,
    tokenizer,
    cfg: Config = None,
    prompt_loader: Optional[PromptLoader] = None,
    custom_processor=None,
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Run the full baseline generation + evaluation pipeline.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    cfg : Config, optional
    prompt_loader : PromptLoader, optional
        Custom prompt set. Falls back to default prompts.
    custom_processor : LogitsProcessor, optional
        Hook for watermarking logits processors.

    Returns
    -------
    results : list of dict
        One entry per prompt, each containing generation data + eval.
    df : pd.DataFrame
        Human-readable summary table.
    """
    if cfg is None:
        cfg = default_config
    if prompt_loader is None:
        prompt_loader = PromptLoader()

    generator = LLMGenerator(model, tokenizer, cfg)
    evaluator = Evaluator(model, tokenizer)

    results: List[dict] = []
    prompts = prompt_loader.get_prompts()

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating for prompt: {prompt[:60]}...")
        gen_data  = generator.generate_text(prompt, custom_processor=custom_processor)
        eval_data = evaluator.evaluate(gen_data)
        results.append(eval_data)
        print(f"  Done in {gen_data['generation_time']}s | {gen_data['num_tokens']} tokens")

    # ------------------------------------------------------------------ #
    #  Persist to disk                                                     #
    # ------------------------------------------------------------------ #
    _save_results(results, cfg.results_path)
    print(f"\n--- PIPELINE COMPLETE ---")
    print(f"Results saved → {cfg.results_path}")

    df = _build_summary_df(results)
    return results, df


# ---------------------------------------------------------------------- #
#  Internal helpers                                                        #
# ---------------------------------------------------------------------- #

def _save_results(results: List[dict], path: str) -> None:
    """Write results as newline-delimited JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Tensors are not JSON-serialisable; convert lists before dumping.
    def _make_serialisable(obj):
        if isinstance(obj, list):
            return [_make_serialisable(x) for x in obj]
        if hasattr(obj, "item"):          # numpy / torch scalar
            return obj.item()
        return obj

    with open(path, "w", encoding="utf-8") as fh:
        for res in results:
            safe = {k: _make_serialisable(v) for k, v in res.items()}
            fh.write(json.dumps(safe) + "\n")


def _build_summary_df(results: List[dict]) -> pd.DataFrame:
    """Build a concise summary DataFrame."""
    rows = []
    for i, r in enumerate(results):
        ev = r.get("eval", {})
        rows.append({
            "Prompt #":              i + 1,
            "Prompt Preview":       r["prompt"].replace("[INST]", "").replace("[/INST]", "").strip()[:45] + "…",
            "Tokens":               r["num_tokens"],
            "Gen Time (s)":         r.get("generation_time", float("nan")),
            "Perplexity":           round(ev.get("perplexity", float("nan")), 2),
            "Dist-1":               round(ev.get("distinct_1", 0.0), 3),
            "Dist-2":               round(ev.get("distinct_2", 0.0), 3),
            "Avg Shannon (bits)":   round(ev.get("avg_shannon_entropy", 0.0), 3),
            "Avg Empirical (bits)": round(ev.get("avg_empirical_entropy", 0.0), 3),
        })
    return pd.DataFrame(rows)


def load_results(path: str) -> List[dict]:
    """
    Load previously saved JSONL results from disk.

    Parameters
    ----------
    path : str
        Path to the ``.jsonl`` file written by run_pipeline.

    Returns
    -------
    list of dict
    """
    results = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            results.append(json.loads(line))
    return results
