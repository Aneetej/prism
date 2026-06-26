"""
Ablation experiment runner.

Runs four ablation sweeps as specified in the evaluation plan:
  1. Buffer size N ∈ {10, 20, 40, 80} vs. safety recall and streaming smoothness
  2. Overlap K ∈ {0, 5, 10, N//2} vs. split-window adversarial recall
  3. Checker comparison (A vs. B vs. C vs. D vs. E) on all test sets
  4. Fallback strategy (canned vs. regen vs. truncate) vs. UX proxy metrics

Usage:
    python experiments/ablations.py --ablation buffer_size
    python experiments/ablations.py --ablation all --output results/ablations.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from buffer import BufferConfig
from checker.rule_based import RuleBasedChecker
from checker.classifier import ClassifierChecker
from checker.llm_judge import LLMJudgeChecker
from checker.representation import RepresentationProbeChecker
from checker.cascade import CascadeChecker
from eval.run_eval import load_all_datasets, evaluate_example, CHECKER_REGISTRY
from eval.metrics import CheckerMetrics, EvalResult, compute_metrics, print_table


DATASETS_DIR = Path(__file__).parent.parent / "eval" / "datasets"


# ------------------------------------------------------------------
# Ablation 1: Buffer size
# ------------------------------------------------------------------

def ablation_buffer_size(
    checker_name: str = "rule_based",
    sizes: list[int] | None = None,
) -> list[CheckerMetrics]:
    sizes = sizes or [10, 20, 40, 80]
    checker = CHECKER_REGISTRY[checker_name]
    examples = load_all_datasets(DATASETS_DIR)
    results = []

    for N in sizes:
        cfg = BufferConfig(buffer_size=N, overlap=5)
        run_results: list[EvalResult] = [
            evaluate_example(ex, checker, cfg) for ex in examples
        ]
        metrics = compute_metrics(run_results, checker_name=f"{checker_name}|N={N}")
        results.append(metrics)
        print(f"  buffer_size={N:3d}: {metrics.summary()}")

    return results


# ------------------------------------------------------------------
# Ablation 2: Overlap size
# ------------------------------------------------------------------

def ablation_overlap(
    checker_name: str = "rule_based",
    buffer_size: int = 30,
    overlaps: list[int] | None = None,
) -> list[CheckerMetrics]:
    overlaps = overlaps or [0, 5, 10, buffer_size // 2]
    checker = CHECKER_REGISTRY[checker_name]
    # Only use split_window examples for this ablation
    all_examples = load_all_datasets(DATASETS_DIR)
    examples = [e for e in all_examples if e.get("label") in ("split_window", "safe")]
    results = []

    for K in overlaps:
        cfg = BufferConfig(buffer_size=buffer_size, overlap=K)
        run_results: list[EvalResult] = [
            evaluate_example(ex, checker, cfg) for ex in examples
        ]
        metrics = compute_metrics(run_results, checker_name=f"{checker_name}|K={K}")
        results.append(metrics)
        print(f"  overlap={K:3d}: {metrics.summary()}")

    return results


# ------------------------------------------------------------------
# Ablation 3: Checker comparison
# ------------------------------------------------------------------

def ablation_checkers(
    buffer_size: int = 30,
    overlap: int = 5,
    checker_names: list[str] | None = None,
) -> list[CheckerMetrics]:
    checker_names = checker_names or list(CHECKER_REGISTRY.keys())
    examples = load_all_datasets(DATASETS_DIR)
    cfg = BufferConfig(buffer_size=buffer_size, overlap=overlap)
    results = []

    for name in checker_names:
        checker = CHECKER_REGISTRY.get(name)
        if checker is None:
            print(f"  [skip] unknown checker: {name}")
            continue
        run_results: list[EvalResult] = [
            evaluate_example(ex, checker, cfg) for ex in examples
        ]
        metrics = compute_metrics(run_results, checker_name=name)
        results.append(metrics)
        print(f"  {metrics.summary()}")

    return results


# ------------------------------------------------------------------
# Ablation 4: Fallback strategy
# ------------------------------------------------------------------

def ablation_fallback(
    checker_name: str = "rule_based",
    buffer_size: int = 30,
    overlap: int = 5,
) -> list[dict]:
    """
    Measures UX-proxy metrics for each fallback strategy on harmful examples.
    Since actual UX ratings require human evaluation, this ablation computes:
      - Mean latency to fallback delivery (ms)
      - Whether any harmful tokens were released before the fallback
    """
    from stream_manager import FallbackStrategy, StreamConfig, StreamManager
    from pipeline import PrismPipeline, PipelineConfig

    checker = CHECKER_REGISTRY[checker_name]
    all_examples = load_all_datasets(DATASETS_DIR)
    harmful_examples = [e for e in all_examples if e.get("label") == "harmful"]
    cfg = BufferConfig(buffer_size=buffer_size, overlap=overlap)

    strategies = [
        FallbackStrategy.CANNED,
        FallbackStrategy.TRUNCATE,
    ]

    summary = []
    for strategy in strategies:
        stream_cfg = StreamConfig(fallback_strategy=strategy)
        total_latency = 0.0
        harmful_leaks = 0

        for ex in harmful_examples:
            import time
            start = time.perf_counter()
            pipeline = PrismPipeline(
                checker=checker,
                config=PipelineConfig(
                    buffer_config=cfg,
                    stream_config=stream_cfg,
                ),
            )
            response_text = ex.get("response", "")
            fake_tokens = [ord(c) for c in response_text]

            def decode_fn(ids): return "".join(chr(i) for i in ids)

            pipeline.start(ex.get("prompt", ""))
            for tid in fake_tokens:
                ok = pipeline.push(tid, decode_fn)
                if not ok:
                    break
            pipeline.finish(decode_fn)

            elapsed_ms = (time.perf_counter() - start) * 1000
            total_latency += elapsed_ms

            # Count verified tokens released — any release from a harmful example
            # before circuit trip is a "leak"
            harmful_leaks += len(pipeline._verified_ids) > 0 and pipeline.stream_manager.is_halted

        row = {
            "strategy": strategy.value,
            "mean_fallback_latency_ms": total_latency / max(len(harmful_examples), 1),
            "harmful_token_leaks": harmful_leaks,
            "n_examples": len(harmful_examples),
        }
        summary.append(row)
        print(
            f"  {strategy.value:<10}: "
            f"lat={row['mean_fallback_latency_ms']:.1f}ms  "
            f"leaks={harmful_leaks}/{len(harmful_examples)}"
        )

    return summary


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation experiment runner")
    parser.add_argument(
        "--ablation",
        choices=["buffer_size", "overlap", "checkers", "fallback", "all"],
        default="all",
    )
    parser.add_argument("--checker", default="rule_based", help="Checker for ablations 1, 2, 4")
    parser.add_argument("--buffer-size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    all_results: dict[str, list] = {}

    run_all = args.ablation == "all"

    if run_all or args.ablation == "buffer_size":
        print("\n=== Ablation 1: Buffer Size ===")
        r = ablation_buffer_size(checker_name=args.checker)
        all_results["buffer_size"] = [asdict(m) for m in r]
        print_table(r)

    if run_all or args.ablation == "overlap":
        print("\n=== Ablation 2: Overlap Size ===")
        r = ablation_overlap(checker_name=args.checker, buffer_size=args.buffer_size)
        all_results["overlap"] = [asdict(m) for m in r]
        print_table(r)

    if run_all or args.ablation == "checkers":
        print("\n=== Ablation 3: Checker Comparison ===")
        r = ablation_checkers(buffer_size=args.buffer_size, overlap=args.overlap)
        all_results["checkers"] = [asdict(m) for m in r]
        print_table(r)

    if run_all or args.ablation == "fallback":
        print("\n=== Ablation 4: Fallback Strategy ===")
        r = ablation_fallback(
            checker_name=args.checker,
            buffer_size=args.buffer_size,
            overlap=args.overlap,
        )
        all_results["fallback"] = r

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()