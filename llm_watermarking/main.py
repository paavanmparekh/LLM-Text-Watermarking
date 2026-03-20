"""
main.py — Entry point for the LLM Watermarking baseline pipeline.

Run from the project root (Project/) with:

    python -m llm_watermarking.main
    python -m llm_watermarking.main --prompts my_prompts.txt
    python -m llm_watermarking.main --no-plots
    python -m llm_watermarking.main --load-results outputs/baseline_results.jsonl

Environment variables
---------------------
HF_TOKEN : str, optional
    HuggingFace access token for authenticated downloads.
"""

import argparse

from .config import Config
from .model_loader import load_model_and_tokenizer
from .pipeline import load_results, run_pipeline
from .prompts import PromptLoader
from .visualization import plot_evaluation_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Watermarking — Baseline Generation Pipeline"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to a plain-text file containing prompts (one per line). "
             "Uses default prompts if omitted.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum new tokens per generation (default: 150).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for JSONL results and plot images (default: outputs/).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots.",
    )
    parser.add_argument(
        "--load-results",
        type=str,
        default=None,
        metavar="JSONL_PATH",
        help="Skip generation; load saved JSONL results and re-plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
    )

    # ------------------------------------------------------------------ #
    #  (Re-)plot from an existing results file                             #
    # ------------------------------------------------------------------ #
    if args.load_results:
        print(f"Loading results from {args.load_results} …")
        results = load_results(args.load_results)
        if not args.no_plots:
            plot_evaluation_metrics(results, output_dir=args.output_dir)
        return

    # ------------------------------------------------------------------ #
    #  Full run                                                            #
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model_and_tokenizer(cfg)

    prompt_loader = (
        PromptLoader.from_file(args.prompts) if args.prompts else PromptLoader()
    )
    print(f"Using {len(prompt_loader)} prompt(s).")

    results, df = run_pipeline(model, tokenizer, cfg=cfg, prompt_loader=prompt_loader)

    print("\n=== Summary ===")
    print(df.to_string(index=False))

    if not args.no_plots:
        plot_evaluation_metrics(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
