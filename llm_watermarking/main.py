"""
main.py — Entry point for the LLM Watermarking baseline pipeline.

Run from the project root (Project/) with:

    python -m llm_watermarking.main
    python -m llm_watermarking.main --watermark Undetectable
    python -m llm_watermarking.main --watermark Undetectable --lambda 20.0
    python -m llm_watermarking.main --prompts my_prompts.txt
    python -m llm_watermarking.main --load-results outputs/undetectable_results.jsonl --detect-only

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
from .visualization import plot_evaluation_metrics, plot_detection_metrics
from .watermarks import WATERMARK_REGISTRY
from .watermarks.undetectable import WatermarkDetector


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
        help="Skip generation; load saved JSONL results.",
    )
    parser.add_argument(
        "--watermark",
        type=str,
        default=None,
        choices=list(WATERMARK_REGISTRY.keys()),
        help="Watermarking scheme to use (e.g. 'Undetectable'). "
             "Omit for standard baseline generation.",
    )
    parser.add_argument(
        "--watermark-key",
        type=str,
        default=None,
        help="Hex string of the secret key (for generation or detection).",
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_entropy",
        type=float,
        default=10.0,
        help="Security parameter λ for Undetectable watermarking (default: 10.0).",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="If --load-results is used, run watermark detection on the loaded sequence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = Config(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        watermark=args.watermark,
        watermark_key=args.watermark_key,
        lambda_entropy=args.lambda_entropy,
    )

    # ------------------------------------------------------------------ #
    #  (Re)-load and (Re)-detect                                          #
    # ------------------------------------------------------------------ #
    if args.load_results:
        print(f"Loading results from {args.load_results} …")
        results = load_results(args.load_results)
        
        if args.detect_only:
            # Figure out key and lambda from results or args
            key_hex = cfg.watermark_key
            lam = cfg.lambda_entropy
            
            if not key_hex and results and "key_hex" in results[0]:
                key_hex = results[0]["key_hex"]
                print(f"Using key from results file: {key_hex}")
            if results and "lambda_entropy" in results[0]:
                lam = results[0]["lambda_entropy"]
                print(f"Using λ from results file: {lam}")
                
            if not key_hex:
                print("Error: Detection requires a watermark key.")
                return
                
            detector = WatermarkDetector(bytes.fromhex(key_hex), lam)
            
            # Since we're just loading one file, we don't have true labels for
            # mixed metrics, but we can set them all to 1 if it was a watermarked run
            is_wm = any("undetectable" in r.get("mode", "") for r in results)
            true_labels = [1 if is_wm else 0] * len(results)
            
            print(f"Running detection on {len(results)} sequences...")
            detector.detect_batch(results, true_labels=true_labels)
            
            if is_wm:
                metrics = detector.compute_metrics(results)
                print(f"\nDetection Metrics: TPR={metrics['tpr']:.2f}, F1={metrics['f1']:.2f}")
                if not args.no_plots:
                    plot_detection_metrics(results, output_dir=args.output_dir, detector_metrics=metrics)
            else:
                if not args.no_plots:
                    plot_detection_metrics(results, output_dir=args.output_dir)
                    
            return

        # Regular reload (just plot everything)
        if not args.no_plots:
            plot_evaluation_metrics(results, output_dir=args.output_dir)
        return

    # ------------------------------------------------------------------ #
    #  Resolve watermarking scheme (if any)                               #
    # ------------------------------------------------------------------ #
    watermark_scheme = None
    if args.watermark:
        scheme_cls = WATERMARK_REGISTRY[args.watermark]
        key_bytes = bytes.fromhex(cfg.watermark_key) if cfg.watermark_key else None
        watermark_scheme = scheme_cls(
            cfg, 
            key=key_bytes, 
            lambda_entropy=cfg.lambda_entropy
        )
        print(f"Watermarking scheme: {args.watermark} → {watermark_scheme.NAME} (λ={watermark_scheme.lambda_entropy})")
        print(f"Using Secret Key: {watermark_scheme.key.hex()}")
    else:
        print("Mode: baseline (no watermark)")

    # ------------------------------------------------------------------ #
    #  Full run                                                            #
    # ------------------------------------------------------------------ #
    model, tokenizer = load_model_and_tokenizer(cfg)

    prompt_loader = (
        PromptLoader.from_file(args.prompts) if args.prompts else PromptLoader()
    )
    print(f"Using {len(prompt_loader)} prompt(s).")

    results, df = run_pipeline(
        model, tokenizer,
        cfg=cfg,
        prompt_loader=prompt_loader,
        watermark_scheme=watermark_scheme,
    )

    print("\n=== Summary ===")
    print(df.to_string(index=False))

    if not args.no_plots:
        # We don't have true negatives here to compute real F1 metrics in one go,
        # but plot_evaluation_metrics will plot the scores/scatter/dist.
        plot_evaluation_metrics(results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
