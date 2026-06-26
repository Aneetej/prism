"""
Train and evaluate the RepresentationProbeChecker for a given HuggingFace LLM.

Usage:
    python experiments/train_probe.py --mode train --model-id meta-llama/Llama-3.2-1B-Instruct
    python experiments/train_probe.py --mode eval --probe models/probe.npz
    python experiments/train_probe.py --mode both --model-id meta-llama/Llama-3.2-1B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from typing import NamedTuple

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from checker.probe import RepresentationProbeChecker, train_probe


# ------------------------------------------------------------------
# Activation extraction
# ------------------------------------------------------------------

class ActivationBatch(NamedTuple):
    safe: np.ndarray       # (N_safe,   hidden_dim)
    unsafe: np.ndarray     # (N_unsafe, hidden_dim)


def extract_activations(
    examples: list[dict],
    model_id: str,
    device: str = "cpu",
    layer_index: int | None = None,
    hf_token: str | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Run a single forward pass per example and return mean-pooled hidden states.

    Returns (activations, labels) where labels[i] = 1 means unsafe, 0 means safe.
    Uses a direct model() forward pass — not generate() — so we get clean activations
    for the full prompt+response sequence without the complexity of per-step generation states.
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = hf_token or os.environ.get("HF_TOKEN")

    print(f"  Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, device_map="cpu", torch_dtype=torch.bfloat16
        )
        if device == "mps":
            model = model.to("mps")

    model.eval()
    num_layers = model.config.num_hidden_layers
    layer = layer_index if layer_index is not None else num_layers // 2
    print(f"  Extracting layer {layer}/{num_layers} activations for {len(examples)} examples...")

    activations_list: list[np.ndarray] = []
    labels: list[int] = []

    for i, ex in enumerate(examples):
        text = ex.get("prompt", "") + "\n\n" + ex.get("response", ex.get("prompt", ""))
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (num_layers+1,) tensors, each (batch, seq, hidden)
        hidden = out.hidden_states[layer]          # (1, seq_len, hidden_dim)
        vec = hidden[0].mean(dim=0).cpu().float().numpy()  # (hidden_dim,)
        activations_list.append(vec)

        label_str = ex.get("label", "safe")
        labels.append(1 if label_str == "harmful" else 0)

        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(examples)}", end="\r", flush=True)

    print()
    return np.stack(activations_list).astype(np.float32), labels


def load_training_examples(datasets_dir: Path) -> list[dict]:
    examples = []
    # xstest_fp provides diverse safe examples beyond the 10 in safe.jsonl
    for fname in ["harmful.jsonl", "safe.jsonl", "xstest_fp.jsonl"]:
        fpath = datasets_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
        elif fname != "xstest_fp.jsonl":
            print(f"  [warn] not found: {fpath}", file=sys.stderr)
    return examples


def load_eval_examples(datasets_dir: Path) -> list[dict]:
    examples = []
    for fname in ["harmful.jsonl", "safe.jsonl", "boundary.jsonl", "xstest_fp.jsonl"]:
        fpath = datasets_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        examples.append(json.loads(line))
    return examples


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def evaluate_probe(probe: RepresentationProbeChecker, examples: list[dict]) -> None:
    harmful = [e for e in examples if e.get("label") == "harmful"]
    safe    = [e for e in examples if e.get("label") == "safe"]

    def run(ex_list: list[dict]) -> tuple[int, int, float]:
        tp = fn = 0
        total_ms = 0.0
        for ex in ex_list:
            r = probe.check(ex.get("prompt", ""), ex.get("response", ex.get("prompt", "")))
            total_ms += r.latency_ms
            if not r.passed:
                tp += 1
            else:
                fn += 1
        return tp, fn, total_ms / len(ex_list) if ex_list else 0.0

    if harmful:
        tp, fn, lat = run(harmful)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  harmful  n={len(harmful):3d}  recall={recall:.1%}  lat={lat:.1f}ms")

    if safe:
        fp, tn, lat = run(safe)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        print(f"  safe     n={len(safe):3d}  FPR   ={fpr:.1%}  lat={lat:.1f}ms")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> RepresentationProbeChecker:
    datasets_dir = args.datasets_dir
    examples = load_training_examples(datasets_dir)
    if not examples:
        print("No training examples found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(examples)} training examples.")

    activations, labels = extract_activations(
        examples,
        model_id=args.model_id,
        device=args.device,
        layer_index=args.layer,
        hf_token=args.hf_token,
    )

    safe_act   = activations[np.array(labels) == 0]
    unsafe_act = activations[np.array(labels) == 1]
    print(f"  safe: {len(safe_act)}  unsafe: {len(unsafe_act)}")

    if len(safe_act) == 0 or len(unsafe_act) == 0:
        print("Need both safe and unsafe examples to train.", file=sys.stderr)
        sys.exit(1)

    print("Training logistic regression probe...")
    probe = train_probe(safe_act, unsafe_act)

    probe_path = args.probe
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    probe.save(probe_path)
    print(f"Probe saved → {probe_path}")
    return probe


def cmd_eval(args: argparse.Namespace, probe: RepresentationProbeChecker | None = None) -> None:
    if probe is None:
        if not args.probe.exists():
            print(f"Probe file not found: {args.probe}", file=sys.stderr)
            sys.exit(1)
        probe = RepresentationProbeChecker.from_file(args.probe)
        print(f"Loaded probe from {args.probe}")

    examples = load_eval_examples(args.datasets_dir)
    if not examples:
        print("No eval examples found.", file=sys.stderr)
        return

    # The probe's check() needs hidden_states. For eval, we extract them here.
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = args.hf_token or os.environ.get("HF_TOKEN")
    print(f"Loading {args.model_id} for eval...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, token=token, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, token=token, device_map="cpu", torch_dtype=torch.bfloat16
        )
        if args.device == "mps":
            model = model.to("mps")
    model.eval()

    num_layers = model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else num_layers // 2

    harmful = [e for e in examples if e.get("label") == "harmful"]
    safe    = [e for e in examples if e.get("label") == "safe"]

    def run_group(ex_list: list[dict], expect_unsafe: bool) -> tuple[int, int, float]:
        correct = wrong = 0
        total_ms = 0.0
        for ex in ex_list:
            text = ex.get("prompt", "") + "\n\n" + ex.get("response", ex.get("prompt", ""))
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[layer][0].mean(dim=0).cpu().float().numpy()
            result = probe.check(ex.get("prompt", ""), ex.get("response", ""), hidden_states=hidden)
            total_ms += (time.perf_counter() - t0) * 1000

            blocked = not result.passed
            if blocked == expect_unsafe:
                correct += 1
            else:
                wrong += 1
        mean_lat = total_ms / len(ex_list) if ex_list else 0.0
        return correct, wrong, mean_lat

    print(f"\nEvaluating probe (layer {layer}) on {len(examples)} examples...")
    if harmful:
        tp, fn, lat = run_group(harmful, expect_unsafe=True)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"  harmful  n={len(harmful):3d}  recall={recall:.1%}  lat={lat:.1f}ms")
    if safe:
        tn, fp, lat = run_group(safe, expect_unsafe=False)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        print(f"  safe     n={len(safe):3d}  FPR   ={fpr:.1%}  lat={lat:.1f}ms")

        xstest = [e for e in safe if e.get("id", "").startswith("xstest_")]
        if xstest:
            tn2, fp2, lat2 = run_group(xstest, expect_unsafe=False)
            fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) else 0.0
            print(f"  xstest   n={len(xstest):3d}  FPR   ={fpr2:.1%}  lat={lat2:.1f}ms  (hard FP set)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train / evaluate the representation probe")
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument(
        "--model-id",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model ID to extract activations from",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--layer", type=int, default=None, help="Layer index (default: middle layer)")
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path("models/probe.npz"),
        help="Path to save/load probe weights",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("eval/datasets"),
    )
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    probe = None
    if args.mode in ("train", "both"):
        probe = cmd_train(args)
    if args.mode in ("eval", "both"):
        cmd_eval(args, probe=probe)


if __name__ == "__main__":
    main()
