"""
Download the XSTest false-positive benchmark and save it as a local JSONL file.

Usage:
    python eval/fetch_xstest.py
    python eval/fetch_xstest.py --out eval/datasets/xstest_fp.jsonl --model gpt4

Source: https://huggingface.co/datasets/natolambert/xstest-v2-copy
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


DATASET = "natolambert/xstest-v2-copy"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

# Types that begin with "contrast_" are the *actually* harmful prompts included for
# calibration. We exclude them — we want only the safe prompts.
_SAFE_TYPES = {
    "homonyms",
    "figurative_language",
    "safe_targets",
    "safe_contexts",
    "definitions",
    "historical_events",
    "nons_group_real_discr",
    "real_group_nons_discr",
    "privacy_public",
    "privacy_fictional",
}


def _fetch_all_rows(split: str) -> list[dict]:
    import requests

    rows: list[dict] = []
    offset = 0
    total = None

    while total is None or offset < total:
        resp = requests.get(
            HF_ROWS_API,
            params={
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": PAGE_SIZE,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        total = data["num_rows_total"]
        batch = [r["row"] for r in data["rows"]]
        rows.extend(batch)
        offset += len(batch)
        print(f"  fetched {offset}/{total}", end="\r", flush=True)
        if batch:
            time.sleep(0.2)

    print()
    return rows


def fetch_xstest(split: str = "gpt4", out_path: Path | None = None) -> Path:
    if out_path is None:
        out_path = Path(__file__).parent / "datasets" / "xstest_fp.jsonl"

    print(f"Fetching {DATASET} (split={split})...")
    rows = _fetch_all_rows(split)
    print(f"  total rows: {len(rows)}")

    # Keep safe-typed prompts where the model answered normally (no refusal).
    # final_label values: 1_full_compliance | 2_full_refusal | 3_partial_refusal
    kept = [
        r for r in rows
        if r.get("type") in _SAFE_TYPES
        and r.get("final_label") == "1_full_compliance"
    ]
    print(f"  after filtering: {len(kept)} safe+compliant examples")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i, row in enumerate(kept):
            record = {
                "id": f"xstest_{i:04d}",
                "prompt": row["prompt"],
                "response": row["completion"],
                "label": "safe",
                "xstest_type": row.get("type", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"  saved → {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download XSTest FP benchmark")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: eval/datasets/xstest_fp.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="gpt4",
        choices=["gpt4"],
        help="Which model's completions to use",
    )
    args = parser.parse_args()
    fetch_xstest(split=args.model, out_path=args.out)


if __name__ == "__main__":
    main()
