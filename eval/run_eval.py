"""
PRISM evaluation harness.

Runs each checker against all test sets and reports metrics.

Usage:
    python eval/run_eval.py --checkers rule_based classifier cascade
    python eval/run_eval.py --checkers all --buffer-size 30 --overlap 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from buffer import BufferConfig, TokenBuffer
from checker.base import SafetyChecker
from checker.rule_based import RuleBasedChecker
from checker.classifier import ClassifierChecker
from checker.llm_judge import LLMJudgeChecker
from checker.llama_guard import LlamaGuardChecker
from checker.cascade import CascadeChecker
from checker.probe import RepresentationProbeChecker

from eval.metrics import CheckerMetrics, EvalResult, compute_metrics, print_table


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_all_datasets(datasets_dir: Path) -> list[dict]:
    all_examples = []
    # benchmark_test is the primary eval set when available (run eval/fetch_benchmark.py to download)
    # benchmark_cbrn is a CBRN-specific supplement from HarmBench (prompt-only, tests Stage 1)
    optional = {"xstest_fp.jsonl", "boundary.jsonl", "split_window.jsonl",
                "benchmark_test.jsonl", "benchmark_cbrn.jsonl"}
    for fname in ["benchmark_test.jsonl", "benchmark_cbrn.jsonl", "harmful.jsonl", "safe.jsonl",
                  "boundary.jsonl", "split_window.jsonl", "xstest_fp.jsonl"]:
        fpath = datasets_dir / fname
        if fpath.exists():
            all_examples.extend(load_dataset(fpath))
        elif fname not in optional:
            print(f"  [warn] dataset not found: {fpath}", file=sys.stderr)
    return all_examples


# ------------------------------------------------------------------
# Per-example evaluation
# ------------------------------------------------------------------

def evaluate_example_full_output(
    example: dict,
    checker: SafetyChecker,
) -> EvalResult:
    """Single-shot check on the complete response — matches production full_output mode."""
    prompt = example.get("prompt", "")
    response = example.get("response", prompt)

    start = time.perf_counter()
    result = checker.check(prompt, response)
    latency_ms = (time.perf_counter() - start) * 1000

    return EvalResult(
        example_id=example.get("id", "unknown"),
        label=example.get("label", "unknown"),
        predicted_safe=result.passed,
        latency_ms=latency_ms,
        ttft_ms=latency_ms,   # full_output: user waits the full check time before any output
        category=result.category,
    )


def evaluate_example_sliding_window(
    example: dict,
    checker: SafetyChecker,
    buffer_config: BufferConfig,
) -> EvalResult:
    """Sliding-window check on token buffer chunks — matches production sliding_window mode."""
    response_text: str = example.get("response", example.get("prompt", ""))
    fake_tokens: list[int] = [ord(c) for c in response_text]

    buf = TokenBuffer(buffer_config)
    context = example.get("prompt", "")
    latencies: list[float] = []

    check_passed = True
    category = None

    ttft_start = time.perf_counter()
    first_token_delivered = False
    ttft_ms = 0.0
    inter_token_times: list[float] = []
    last_token_time = time.perf_counter()

    for tid in fake_tokens:
        ready = buf.push(tid)
        if ready:
            window = "".join(chr(t) for t in buf.window())
            result = checker.check(context, window)
            latencies.append(result.latency_ms)

            if result.passed:
                released = buf.release()
                if not first_token_delivered and released:
                    ttft_ms = (time.perf_counter() - ttft_start) * 1000
                    first_token_delivered = True

                now = time.perf_counter()
                inter_token_times.append((now - last_token_time) * 1000)
                last_token_time = now
            else:
                check_passed = False
                category = result.category
                break

    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return EvalResult(
        example_id=example.get("id", "unknown"),
        label=example.get("label", "unknown"),
        predicted_safe=check_passed,
        latency_ms=mean_latency,
        ttft_ms=ttft_ms,
        category=category,
        inter_token_times_ms=inter_token_times,
    )


# ------------------------------------------------------------------
# Checker registry (lazy factories — models load only when selected)
# ------------------------------------------------------------------

def _load_probe() -> SafetyChecker:
    probe_path = Path("models/probe.npz")
    if not probe_path.exists():
        print(
            "  [probe] models/probe.npz not found — run experiments/train_probe.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return RepresentationProbeChecker.from_file(probe_path)


CHECKER_FACTORIES: dict[str, Callable[[], SafetyChecker]] = {
    "rule_based":  lambda: RuleBasedChecker(),
    "classifier":  lambda: ClassifierChecker(),
    "llm_judge":   lambda: LLMJudgeChecker(backend="ollama", ollama_url="http://localhost:11434"),
    "llama_guard": lambda: LlamaGuardChecker(device="mps"),
    "cascade":     lambda: CascadeChecker(
        fast=ClassifierChecker(),
        slow=LlamaGuardChecker(device="mps"),
        skip_below=0.4,
    ),
    # Probe needs hidden states per example; run via experiments/train_probe.py --mode eval
    # for activation-based evaluation. Listed here for completeness (falls back to probe_unavailable).
    "probe":       _load_probe,
}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_eval(
    checker_names: list[str],
    datasets_dir: Path,
    mode: str = "full_output",
    buffer_size: int = 30,
    overlap: int = 5,
) -> list[CheckerMetrics]:
    examples = load_all_datasets(datasets_dir)
    if not examples:
        print("No examples found. Generate datasets first.", file=sys.stderr)
        return []

    buffer_config = BufferConfig(buffer_size=buffer_size, overlap=overlap)
    all_metrics: list[CheckerMetrics] = []

    for name in checker_names:
        if name not in CHECKER_FACTORIES:
            print(f"Unknown checker: {name}", file=sys.stderr)
            continue
        checker = CHECKER_FACTORIES[name]()

        print(f"\nEvaluating {name} [{mode}] on {len(examples)} examples...")
        results: list[EvalResult] = []
        for ex in examples:
            if mode == "full_output":
                r = evaluate_example_full_output(ex, checker)
            else:
                r = evaluate_example_sliding_window(ex, checker, buffer_config)
            results.append(r)

        metrics = compute_metrics(results, checker_name=f"{name} [{mode}]")
        all_metrics.append(metrics)
        print(f"  {metrics.summary()}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="PRISM evaluation harness")
    parser.add_argument(
        "--checkers",
        nargs="+",
        default=["rule_based"],
        choices=list(CHECKER_FACTORIES.keys()) + ["all"],
        help="Which checkers to evaluate",
    )
    parser.add_argument("--mode", choices=["full_output", "sliding_window"], default="full_output")
    parser.add_argument("--buffer-size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path(__file__).parent / "datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write metrics JSON to this file",
    )
    args = parser.parse_args()

    checker_names = (
        list(CHECKER_FACTORIES.keys()) if "all" in args.checkers else args.checkers
    )

    metrics_list = run_eval(
        checker_names=checker_names,
        datasets_dir=args.datasets_dir,
        mode=args.mode,
        buffer_size=args.buffer_size,
        overlap=args.overlap,
    )

    if metrics_list:
        print("\n=== Summary ===")
        print_table(metrics_list)

    if args.output and metrics_list:
        import dataclasses
        args.output.write_text(
            json.dumps([dataclasses.asdict(m) for m in metrics_list], indent=2)
        )
        print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()