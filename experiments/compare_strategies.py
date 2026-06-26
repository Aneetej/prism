"""
Primary experiment: Strategy A (sliding_window) vs. Strategy B (full_output).

For each prompt in the test set:
  - Run pipeline in full_output mode → record latency, safety decision
  - Run pipeline in sliding_window mode → record TTFT, total latency, safety decision
  - Compare against ground-truth label

Outputs:
  results/latency_comparison.csv
  results/accuracy_comparison.csv
  results/summary.json

Usage:
    python experiments/compare_strategies.py
    python experiments/compare_strategies.py --config config.yaml --checker rule_based
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from buffer import BufferConfig
from checker.rule_based import RuleBasedChecker
from checker.base import SafetyChecker
from llm.base import GenerationConfig
from pre_check import PreCheck
from pipeline import PrismPipeline, PipelineResult


RESULTS_DIR = Path(__file__).parent.parent / "results"
TEST_SET_PATH = Path(__file__).parent.parent / "eval" / "datasets" / "test_set.json"


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class StrategyResult:
    example_id: str
    label: str                  # "harmful" | "safe"
    strategy: str               # "full_output" | "sliding_window"
    output: str
    passed: bool
    blocked_at: str | None
    total_latency_ms: float
    checker_latency_ms: float
    pre_check_latency_ms: float
    llm_latency_ms: float
    ttft_ms: float              # sliding_window only; 0 for full_output
    correct: bool               # True if decision matches ground truth


def _mock_llm(response: str):
    """Minimal stand-in for LLMAdapter that replays a fixed response."""
    from types import SimpleNamespace
    from llm.base import GenerationResult
    return SimpleNamespace(
        model_id="mock",
        generate=lambda prompt, config: GenerationResult(
            text=response, tokens_used=len(response.split()), model_id="mock", latency_ms=0.0
        ),
        stream=lambda prompt, config: (w + " " for w in response.split()),
    )


# ------------------------------------------------------------------
# Per-example runner
# ------------------------------------------------------------------

def run_example(
    example: dict,
    checker: SafetyChecker,
    pre_check: PreCheck | None,
    buffer_config: BufferConfig,
    error_message: str,
) -> tuple[StrategyResult, StrategyResult]:
    prompt = example["prompt"]
    response = example["response"]
    label = example["label"]
    eid = example["id"]

    mock_llm = _mock_llm(response)

    # --- Strategy B: full_output ---
    pipeline_b = PrismPipeline(
        llm=mock_llm,
        checker=checker,
        pre_check=pre_check,
        buffer_config=buffer_config,
        error_message=error_message,
        mode="full_output",
    )
    result_b = pipeline_b.run(prompt)
    # A correct decision: harmful → blocked (passed=False), safe → passed
    correct_b = (label == "harmful") == (not result_b.passed)

    strat_b = StrategyResult(
        example_id=eid,
        label=label,
        strategy="full_output",
        output=result_b.output,
        passed=result_b.passed,
        blocked_at=result_b.blocked_at,
        total_latency_ms=result_b.latency_ms,
        checker_latency_ms=result_b.checker_latency_ms,
        pre_check_latency_ms=result_b.pre_check_latency_ms,
        llm_latency_ms=result_b.llm_latency_ms,
        ttft_ms=0.0,
        correct=correct_b,
    )

    # --- Strategy A: sliding_window ---
    pipeline_a = PrismPipeline(
        llm=_mock_llm(response),
        checker=checker,
        pre_check=pre_check,
        buffer_config=buffer_config,
        error_message=error_message,
        mode="sliding_window",
    )
    ttft_start = time.perf_counter()
    chunks: list[str] = []
    ttft_ms = 0.0
    wall_start = time.perf_counter()
    for chunk in pipeline_a.stream(prompt):
        if not chunks:
            ttft_ms = (time.perf_counter() - ttft_start) * 1000
        chunks.append(chunk)
    total_ms_a = (time.perf_counter() - wall_start) * 1000

    output_a = "".join(chunks)
    passed_a = output_a != error_message and error_message not in output_a
    correct_a = (label == "harmful") == (not passed_a)

    strat_a = StrategyResult(
        example_id=eid,
        label=label,
        strategy="sliding_window",
        output=output_a,
        passed=passed_a,
        blocked_at=None,
        total_latency_ms=total_ms_a,
        checker_latency_ms=0.0,
        pre_check_latency_ms=0.0,
        llm_latency_ms=0.0,
        ttft_ms=ttft_ms,
        correct=correct_a,
    )

    return strat_b, strat_a


# ------------------------------------------------------------------
# Aggregated metrics
# ------------------------------------------------------------------

def _summarise(results: list[StrategyResult], strategy: str) -> dict:
    sr = [r for r in results if r.strategy == strategy]
    harmful = [r for r in sr if r.label == "harmful"]
    safe = [r for r in sr if r.label == "safe"]

    tp = sum(1 for r in harmful if not r.passed)
    fn = sum(1 for r in harmful if r.passed)
    fp = sum(1 for r in safe if not r.passed)
    tn = sum(1 for r in safe if r.passed)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    lats = [r.total_latency_ms for r in sr]
    mean_lat = sum(lats) / len(lats) if lats else 0.0
    mean_checker = sum(r.checker_latency_ms for r in sr) / len(sr) if sr else 0.0
    mean_ttft = sum(r.ttft_ms for r in sr) / len(sr) if sr else 0.0

    return {
        "strategy": strategy,
        "n_examples": len(sr),
        "safety_recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1, 4),
        "mean_total_latency_ms": round(mean_lat, 2),
        "mean_checker_latency_ms": round(mean_checker, 2),
        "mean_ttft_ms": round(mean_ttft, 2),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_experiment(
    test_set_path: Path = TEST_SET_PATH,
    checker_name: str = "rule_based",
    buffer_size: int = 30,
    overlap: int = 5,
    results_dir: Path = RESULTS_DIR,
    error_message: str = "I'm not able to respond to that request.",
) -> dict:
    with open(test_set_path) as f:
        examples = json.load(f)

    checker_map = {
        "rule_based": RuleBasedChecker,
    }
    checker_cls = checker_map.get(checker_name, RuleBasedChecker)
    checker = checker_cls()
    pre_check = PreCheck()
    buffer_config = BufferConfig(buffer_size=buffer_size, overlap=overlap)

    all_results: list[StrategyResult] = []
    for ex in examples:
        b, a = run_example(ex, checker, pre_check, buffer_config, error_message)
        all_results.extend([b, a])
        status_b = "PASS" if b.passed else "BLOCK"
        status_a = "PASS" if a.passed else "BLOCK"
        print(f"  [{ex['label']:7s}] {ex['id']:20s}  full={status_b}  sliding={status_a}")

    results_dir.mkdir(parents=True, exist_ok=True)

    # latency CSV
    lat_path = results_dir / "latency_comparison.csv"
    with open(lat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "example_id", "label", "strategy",
            "total_latency_ms", "checker_latency_ms", "ttft_ms",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "example_id": r.example_id,
                "label": r.label,
                "strategy": r.strategy,
                "total_latency_ms": round(r.total_latency_ms, 3),
                "checker_latency_ms": round(r.checker_latency_ms, 3),
                "ttft_ms": round(r.ttft_ms, 3),
            })

    # accuracy CSV
    acc_path = results_dir / "accuracy_comparison.csv"
    with open(acc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "example_id", "label", "strategy", "passed", "blocked_at", "correct",
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "example_id": r.example_id,
                "label": r.label,
                "strategy": r.strategy,
                "passed": r.passed,
                "blocked_at": r.blocked_at or "",
                "correct": r.correct,
            })

    # summary JSON
    summary = {
        "checker": checker_name,
        "buffer_size": buffer_size,
        "overlap": overlap,
        "full_output": _summarise(all_results, "full_output"),
        "sliding_window": _summarise(all_results, "sliding_window"),
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nResults written to {results_dir}/")
    _print_summary(summary)
    return summary


def _print_summary(summary: dict) -> None:
    print("\n=== Strategy Comparison ===")
    header = f"{'Metric':<30} {'full_output':>14} {'sliding_window':>16}"
    print(header)
    print("-" * len(header))
    keys = ["safety_recall", "false_positive_rate", "f1_score",
            "mean_total_latency_ms", "mean_checker_latency_ms", "mean_ttft_ms"]
    for k in keys:
        fo = summary["full_output"].get(k, 0)
        sw = summary["sliding_window"].get(k, 0)
        print(f"{k:<30} {fo:>14.4f} {sw:>16.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checker",
        default="rule_based",
        choices=["rule_based", "classifier", "llm_judge", "probe"],
    )
    parser.add_argument("--buffer-size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--test-set", type=Path, default=TEST_SET_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    run_experiment(
        test_set_path=args.test_set,
        checker_name=args.checker,
        buffer_size=args.buffer_size,
        overlap=args.overlap,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()