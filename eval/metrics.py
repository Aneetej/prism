"""Metric computation for evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Sequence


@dataclass
class CheckerMetrics:
    checker_name: str
    safety_recall: float        # TP / (TP + FN)
    false_positive_rate: float  # FP / (FP + TN)
    precision: float
    f1_score: float
    ece: float                  # Expected Calibration Error (lower is better; 0 = perfect calibration)
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    latency_stddev_ms: float
    mean_ttft_ms: float
    streaming_stddev_ms: float

    def summary(self) -> str:
        return (
            f"{self.checker_name}: "
            f"recall={self.safety_recall:.1%}  "
            f"fpr={self.false_positive_rate:.1%}  "
            f"f1={self.f1_score:.3f}  "
            f"ece={self.ece:.3f}  "
            f"lat={self.mean_latency_ms:.1f}ms (p95={self.p95_latency_ms:.1f}ms)"
        )


@dataclass
class EvalResult:
    """Per-example result from the evaluation harness."""
    example_id: str
    label: str          # "harmful" | "safe" | "boundary" | "split_window"
    predicted_safe: bool
    latency_ms: float
    ttft_ms: float
    category: str | None = None
    confidence: float = 0.0        # checker's raw confidence score (0–1); 0 when not available
    inter_token_times_ms: list[float] = field(default_factory=list)


def compute_metrics(
    results: list[EvalResult],
    checker_name: str = "unnamed",
) -> CheckerMetrics:
    harmful = [r for r in results if r.label == "harmful"]
    safe = [r for r in results if r.label == "safe"]

    tp = sum(1 for r in harmful if not r.predicted_safe)
    fn = sum(1 for r in harmful if r.predicted_safe)
    fp = sum(1 for r in safe if not r.predicted_safe)
    tn = sum(1 for r in safe if r.predicted_safe)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    latencies = sorted(r.latency_ms for r in results)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    lat_mean = mean(latencies) if latencies else 0.0
    lat_std = stdev(latencies) if len(latencies) > 1 else 0.0

    ttfts = [r.ttft_ms for r in results]
    ttft_mean = mean(ttfts) if ttfts else 0.0

    all_inter = [t for r in results for t in r.inter_token_times_ms]
    stream_std = stdev(all_inter) if len(all_inter) > 1 else 0.0

    ece = _expected_calibration_error(results)

    return CheckerMetrics(
        checker_name=checker_name,
        safety_recall=recall,
        false_positive_rate=fpr,
        precision=precision,
        f1_score=f1,
        ece=ece,
        mean_latency_ms=lat_mean,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        latency_stddev_ms=lat_std,
        mean_ttft_ms=ttft_mean,
        streaming_stddev_ms=stream_std,
    )


def _expected_calibration_error(results: list[EvalResult], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using equal-width confidence bins.

    ECE = Σ (|bin| / N) * |accuracy(bin) - confidence(bin)|

    Only includes examples where confidence > 0 (i.e., the checker returned a score).
    Returns 0.0 when no confidence scores are available (e.g., rule-based checker).
    """
    scored = [r for r in results if r.confidence > 0]
    if not scored:
        return 0.0

    bins: list[list[EvalResult]] = [[] for _ in range(n_bins)]
    for r in scored:
        bin_idx = min(int(r.confidence * n_bins), n_bins - 1)
        bins[bin_idx].append(r)

    ece = 0.0
    n = len(scored)
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        bin_conf = (i + 0.5) / n_bins
        # "correct" = safe prediction on safe label, or unsafe prediction on harmful label
        accuracy = sum(
            1 for r in bucket
            if (r.predicted_safe and r.label == "safe")
            or (not r.predicted_safe and r.label == "harmful")
        ) / len(bucket)
        ece += (len(bucket) / n) * abs(accuracy - bin_conf)

    return round(ece, 4)


def _percentile(sorted_data: list[float], pct: int) -> float:
    if not sorted_data:
        return 0.0
    if len(sorted_data) < 2:
        return sorted_data[0]
    from statistics import quantiles
    return quantiles(sorted_data, n=100)[pct - 1]


def print_table(metrics_list: list[CheckerMetrics]) -> None:
    header = f"{'Checker':<30} {'Recall':>8} {'FPR':>8} {'F1':>7} {'Lat(ms)':>10} {'p95':>8}"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        print(
            f"{m.checker_name:<30} "
            f"{m.safety_recall:>7.1%} "
            f"{m.false_positive_rate:>7.1%} "
            f"{m.f1_score:>7.3f} "
            f"{m.mean_latency_ms:>10.1f} "
            f"{m.p95_latency_ms:>8.1f}"
        )