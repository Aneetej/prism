"""
Definitive checker comparison: rule_based vs classifier vs llama_guard vs cascade vs probe,
all evaluated on the same held-out benchmark_test.jsonl.

Text-based checkers (rule_based, classifier, cascade) run without a loaded LLM.
The probe requires a HuggingFace model to extract hidden states — use --no-models to skip it.

Usage:
    python experiments/compare_checkers.py
    python experiments/compare_checkers.py --no-models   # skip probe + llama_guard
    python experiments/compare_checkers.py --output results/checker_comparison.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import CheckerMetrics, EvalResult, compute_metrics, print_table

DATASETS_DIR = Path(__file__).parent.parent / "eval" / "datasets"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_benchmark_test(datasets_dir: Path) -> list[dict]:
    path = datasets_dir / "benchmark_test.jsonl"
    if not path.exists():
        print(
            f"benchmark_test.jsonl not found at {path}\n"
            "Run: python eval/fetch_benchmark.py",
            file=sys.stderr,
        )
        sys.exit(1)
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def eval_text_checker(checker, examples: list[dict], name: str) -> CheckerMetrics:
    results = []
    for ex in examples:
        start = time.perf_counter()
        r = checker.check(ex.get("prompt", ""), ex.get("response", ""))
        latency_ms = (time.perf_counter() - start) * 1000
        results.append(EvalResult(
            example_id=ex.get("id", ""),
            label=ex.get("label", "unknown"),
            predicted_safe=r.passed,
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            category=r.category,
        ))
    return compute_metrics(results, checker_name=name)


def eval_probe(probe, examples: list[dict], model_id: str, device: str,
               hf_token: str | None) -> CheckerMetrics:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_id} for hidden-state extraction...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto" if device == "cuda" else "cpu",
        torch_dtype=torch.bfloat16,
    )
    if device == "mps":
        model = model.to("mps")
    model.eval()

    layer = model.config.num_hidden_layers // 2
    results = []

    for ex in examples:
        text = ex.get("prompt", "") + "\n\n" + ex.get("response", "")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[layer][0].mean(dim=0).cpu().float().numpy()
        r = probe.check(ex.get("prompt", ""), ex.get("response", ""), hidden_states=hidden)
        latency_ms = (time.perf_counter() - t0) * 1000

        results.append(EvalResult(
            example_id=ex.get("id", ""),
            label=ex.get("label", "unknown"),
            predicted_safe=r.passed,
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            category=r.category,
        ))

    return compute_metrics(results, checker_name="probe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all checkers on benchmark_test")
    parser.add_argument("--no-models", action="store_true",
                        help="Skip checkers that require a loaded model (probe, llama_guard)")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct",
                        help="HuggingFace model for probe hidden-state extraction")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--probe-path", type=Path, default=Path("models/probe.npz"))
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    examples = load_benchmark_test(args.datasets_dir)
    harmful = sum(1 for e in examples if e.get("label") == "harmful")
    safe = sum(1 for e in examples if e.get("label") == "safe")
    print(f"Loaded {len(examples)} examples ({harmful} harmful, {safe} safe)")

    hf_token = os.environ.get("HF_TOKEN")
    all_metrics: list[CheckerMetrics] = []

    from checker.rule_based import RuleBasedChecker
    print("\nEvaluating rule_based...")
    all_metrics.append(eval_text_checker(RuleBasedChecker(), examples, "rule_based"))

    from checker.classifier import ClassifierChecker
    print("Evaluating classifier (toxic-bert)...")
    all_metrics.append(eval_text_checker(ClassifierChecker(), examples, "classifier"))

    if not args.no_models:
        from checker.llama_guard import LlamaGuardChecker
        print("Evaluating llama_guard...")
        all_metrics.append(eval_text_checker(
            LlamaGuardChecker(device=args.device, hf_token=hf_token),
            examples, "llama_guard",
        ))

        from checker.cascade import CascadeChecker
        print("Evaluating cascade...")
        all_metrics.append(eval_text_checker(
            CascadeChecker(
                fast=ClassifierChecker(),
                slow=LlamaGuardChecker(device=args.device, hf_token=hf_token),
            ),
            examples, "cascade",
        ))

        if args.probe_path.exists():
            from checker.probe import RepresentationProbeChecker
            print("Evaluating probe (with LLM forward pass)...")
            probe = RepresentationProbeChecker.from_file(args.probe_path)
            all_metrics.append(eval_probe(
                probe, examples, args.model_id, args.device, hf_token
            ))
        else:
            print(f"  [skip] probe weights not found at {args.probe_path}")
            print(f"         Run: python experiments/train_probe.py --mode train")
    else:
        from checker.cascade import CascadeChecker
        print("Evaluating cascade (classifier skip gate only — no llama_guard in --no-models mode)...")
        all_metrics.append(eval_text_checker(
            CascadeChecker(fast=ClassifierChecker(), slow=ClassifierChecker()),
            examples, "cascade (no llama_guard)",
        ))

    print("\n=== Checker Comparison ===")
    print_table(all_metrics)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([asdict(m) for m in all_metrics], indent=2))
        print(f"\nResults written to {args.output}")
    else:
        out = RESULTS_DIR / "checker_comparison.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(m) for m in all_metrics], indent=2))
        print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
