"""
Download benchmark data for PRISM evaluation.

Sources:
  WildGuardMix (allenai/wildguardmix) — 13 harm categories, proper train/test split.
    Used for: probe training (train split) and all-checker evaluation (test split).

  HarmBench CBRN behaviors (centerforaisafety/HarmBench, GitHub CSV) — Chemical,
    Biological, Radiological, Nuclear, and Explosives queries. Prompt-only (no
    responses), so these test Stage 1 pre-check recall on CBRN content specifically.
    Written to benchmark_cbrn.jsonl.

Usage:
    python eval/fetch_benchmark.py
    python eval/fetch_benchmark.py --train-size 1000 --test-size 600
    python eval/fetch_benchmark.py --skip-cbrn
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / "datasets"
HF_DATASET = "allenai/wildguardmix"
SEED = 42


def _label(row: dict) -> str | None:
    """Map WildGuardMix fields to 'harmful' | 'safe'. Returns None to skip the row."""
    prompt_harm = row.get("prompt_harm_label", "")
    response_harm = row.get("response_harm_label", "")
    if not isinstance(prompt_harm, str) or not isinstance(response_harm, str):
        return None
    if "harmful" in prompt_harm.lower() or "harmful" in response_harm.lower():
        return "harmful"
    if "unharmful" in prompt_harm.lower() or "benign" in prompt_harm.lower():
        return "safe"
    return None


def _to_example(row: dict, idx: int, split: str) -> dict | None:
    label = _label(row)
    if label is None:
        return None
    prompt = row.get("prompt", "")
    response = row.get("response", "") or row.get("assistant_response", "") or ""
    if not prompt:
        return None
    return {
        "id": f"wildguard_{split}_{idx:05d}",
        "prompt": prompt,
        "response": response,
        "label": label,
        "source": HF_DATASET,
    }


def fetch_split(
    dataset,
    split: str,
    n_harmful: int,
    n_safe: int,
) -> list[dict]:
    rng = random.Random(SEED)
    harmful, safe = [], []

    for i, row in enumerate(dataset[split]):
        ex = _to_example(row, i, split)
        if ex is None:
            continue
        if ex["label"] == "harmful":
            harmful.append(ex)
        else:
            safe.append(ex)

    rng.shuffle(harmful)
    rng.shuffle(safe)

    selected = harmful[:n_harmful] + safe[:n_safe]
    rng.shuffle(selected)

    print(f"  {split}: {len(harmful)} harmful, {len(safe)} safe available → "
          f"sampling {min(n_harmful, len(harmful))} + {min(n_safe, len(safe))}")
    return selected


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  wrote {len(examples)} examples → {path}")


HARMBENCH_CSV = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_val.csv"
)
CBRN_CATEGORY = "Chemical/Biological/Radiological/Nuclear/Explosives"


def fetch_cbrn(output_dir: Path) -> None:
    """Download HarmBench behaviors and write CBRN entries to benchmark_cbrn.jsonl."""
    import csv
    import urllib.request

    print(f"Downloading HarmBench behaviors from GitHub...")
    try:
        with urllib.request.urlopen(HARMBENCH_CSV, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"  [warn] Could not fetch HarmBench CSV: {exc}", file=sys.stderr)
        return

    reader = csv.DictReader(content.splitlines())
    examples = []
    for i, row in enumerate(reader):
        category = row.get("SemanticCategory", "").strip()
        behavior = row.get("Behavior", "").strip()
        if not behavior or CBRN_CATEGORY not in category:
            continue
        examples.append({
            "id": f"harmbench_cbrn_{i:04d}",
            "prompt": behavior,
            "response": "",  # prompt-only; no LLM response provided
            "label": "harmful",
            "category": "cbrn",
            "source": "centerforaisafety/HarmBench",
        })

    write_jsonl(examples, output_dir / "benchmark_cbrn.jsonl")
    print(f"  CBRN examples: {len(examples)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets for PRISM")
    parser.add_argument("--train-size", type=int, default=1000,
                        help="Total train examples (split equally harmful/safe)")
    parser.add_argument("--test-size", type=int, default=600,
                        help="Total test examples (split equally harmful/safe)")
    parser.add_argument("--skip-cbrn", action="store_true",
                        help="Skip HarmBench CBRN download")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install the datasets library: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET)

    n_train_each = args.train_size // 2
    n_test_each = args.test_size // 2

    print("Sampling train split...")
    train_examples = fetch_split(ds, "train", n_train_each, n_train_each)

    print("Sampling test split...")
    test_examples = fetch_split(ds, "test", n_test_each, n_test_each)

    train_ids = {ex["id"] for ex in train_examples}
    test_ids = {ex["id"] for ex in test_examples}
    assert not train_ids & test_ids, "Train/test overlap detected — check split logic"

    write_jsonl(train_examples, args.output_dir / "benchmark_train.jsonl")
    write_jsonl(test_examples, args.output_dir / "benchmark_test.jsonl")

    if not args.skip_cbrn:
        print("\nFetching CBRN supplement (HarmBench)...")
        fetch_cbrn(args.output_dir)

    print(f"\nDone. Train: {len(train_examples)}  Test: {len(test_examples)}")
    print("Next: python experiments/train_probe.py --mode train")


if __name__ == "__main__":
    main()
