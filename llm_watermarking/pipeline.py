import json
import os
import time
from typing import List, Optional, Tuple

import pandas as pd

from .config import Config, config as default_config
from .evaluation import Evaluator
from .generation import LLMGenerator
from .prompts import PromptLoader
from .watermarks.undetectable import WatermarkDetector


def run_pipeline(
    model,
    tokenizer,
    cfg: Config = None,
    prompt_loader: Optional[PromptLoader] = None,
    custom_processor=None,
    watermark_scheme=None,
) -> Tuple[List[dict], pd.DataFrame]:
    """
    Run the full generation + evaluation + detection pipeline.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    cfg : Config, optional
    prompt_loader : PromptLoader, optional
    custom_processor : LogitsProcessor, optional
    watermark_scheme : Any, optional

    Returns
    -------
    results : list of dict
    df : pd.DataFrame
    """
    if cfg is None:
        cfg = default_config
    if prompt_loader is None:
        prompt_loader = PromptLoader()

    generator = LLMGenerator(model, tokenizer, cfg)
    evaluator = Evaluator(model, tokenizer)
    
    detector = None
    if watermark_scheme is not None and hasattr(watermark_scheme, "key"):
        detector = WatermarkDetector(
            key=watermark_scheme.key,
            lambda_entropy=watermark_scheme.lambda_entropy,
            tokenizer=tokenizer
        )

    results: List[dict] = []
    prompts = prompt_loader.get_prompts()

    use_binary = watermark_scheme is not None
    mode_label = watermark_scheme.NAME if use_binary else "baseline"

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] [{mode_label}] Generating for prompt: {prompt[:60]}...")
        if use_binary:
            gen_data = watermark_scheme.generate(
                model, tokenizer, prompt,
                max_new_tokens=cfg.max_new_tokens if cfg else None,
            )
        else:
            gen_data = generator.generate_text(prompt, custom_processor=custom_processor)
            
        gen_data["mode"] = mode_label
        eval_data = evaluator.evaluate(gen_data)
        
        if detector and use_binary:
            det = detector.detect(eval_data)
            print(f"  [Detect] stat={det['stat']:.2f} | threshold={det['threshold']:.2f} | detected={det['detected']}")
            
        results.append(eval_data)
        print(f"  Done in {gen_data['generation_time']}s | {gen_data['num_tokens']} tokens")

    # ------------------------------------------------------------------ #
    #  Persist to disk                                                     #
    # ------------------------------------------------------------------ #
    _save_results(results, cfg.results_path)
    print(f"\n--- PIPELINE COMPLETE ---")
    print(f"Results saved → {cfg.results_path}")

    df = _build_summary_df(results)
    
    csv_path = cfg.results_path.replace(".jsonl", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary saved → {csv_path}")
    
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
        if isinstance(obj, tuple):
            return list(obj)
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
        det = r.get("detection", {})
        
        row = {
            "Prompt #":              i + 1,
            "Prompt Preview":        r["prompt"].replace("[INST]", "").replace("[/INST]", "").strip()[:45] + "…",
            "Tokens":                r["num_tokens"],
            "Gen Time (s)":          r.get("generation_time", float("nan")),
            "Perplexity":            round(ev.get("perplexity", float("nan")), 2),
            "Dist-1":                round(ev.get("distinct_1", 0.0), 3),
            "Dist-2":                round(ev.get("distinct_2", 0.0), 3),
            "Avg Shannon (bits)":    round(ev.get("avg_shannon_entropy", 0.0), 3),
            "Avg Empirical (bits)":  round(ev.get("avg_empirical_entropy", 0.0), 3),
        }
        
        if det:
            row["Phase-1"] = r.get("phase1_tokens", 0)
            row["Phase-2"] = det.get("num_bits", 0)
            row["Detection Score"] = round(det.get("detection_score", 0.0), 2)
            row["Watermarked"] = "Yes" if det.get("detected") else "No"
            
        rows.append(row)
        
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
            res = json.loads(line)
            # Reconstruct tuples since JSON turns them into lists
            results.append(res)
    return results
