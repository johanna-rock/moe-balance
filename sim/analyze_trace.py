#!/usr/bin/env python3
import argparse
import json
from collections import Counter, defaultdict
from typing import Dict, List, Set

from replication import replicate_freqs


def _percentiles(values: List[float], ps: List[float]) -> Dict[float, float]:
    if not values:
        return {p: 0.0 for p in ps}
    s = sorted(values)
    n = len(s)
    out = {}
    for p in ps:
        idx = min(n - 1, int(round(p * (n - 1))))
        out[p] = s[idx]
    return out


def _parse_ranges(spec: str) -> Set[int]:
    values: Set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            try:
                start_s, end_s = part.split("-", 1)
                start = int(start_s.strip())
                end = int(end_s.strip())
            except Exception:
                raise SystemExit("Invalid --layers. Use comma-separated ints or ranges, e.g. 3-60 or 0,2,5-7")
            if end < start:
                start, end = end, start
            for v in range(start, end + 1):
                values.add(v)
        else:
            try:
                values.add(int(part))
            except Exception:
                raise SystemExit("Invalid --layers. Use comma-separated ints or ranges, e.g. 3-60 or 0,2,5-7")
    return values


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a trace JSONL file")
    ap.add_argument("--trace", required=True, help="Path to trace JSONL")
    ap.add_argument("--experts", type=int, default=256, help="Number of experts")
    ap.add_argument("--layers", default="3-60", help="Comma-separated layer indices or ranges (default: 3-60)")
    ap.add_argument("--bins", default="", help="Frequency bins in percent for per-layer histograms")
    ap.add_argument("--replication-slots", type=int, default=0, help="Number of replication slots to add")
    ap.add_argument("--hist-out", default="", help="Output PNG for per-layer frequency histogram")
    args = ap.parse_args()

    layer_filter = _parse_ranges(args.layers) if args.layers else None

    bins = None

    # counts[layer][expert] = activations
    counts: Dict[int, List[int]] = {}
    totals: Dict[int, int] = defaultdict(int)
    seq_lengths: List[int] = []
    request_ids = set()
    records = 0

    with open(args.trace, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            layer = rec.get("layer", 0)
            if layer_filter is not None and layer not in layer_filter:
                continue

            topk = rec.get("topk_experts", [])
            seq_lengths.append(len(topk))
            rid = rec.get("request_id", rec.get("batch_id", None))
            if rid is not None:
                request_ids.add(rid)
            records += 1

            if layer not in counts:
                counts[layer] = [0 for _ in range(args.experts)]

            for experts_list in topk:
                for e in experts_list:
                    if 0 <= e < args.experts:
                        counts[layer][e] += 1
                        totals[layer] += 1

    print("trace summary")
    print(f"  records: {records}")
    print(f"  samples (unique request_id): {len(request_ids)}")

    if seq_lengths:
        seq_stats = _percentiles(seq_lengths, [0.0, 0.5, 0.9, 0.99, 1.0])
        print("sequence length distribution")
        print(f"  min: {seq_stats[0.0]}  p50: {seq_stats[0.5]}  p90: {seq_stats[0.9]}  p99: {seq_stats[0.99]}  max: {seq_stats[1.0]}")

    # Max frequency per layer
    max_freqs = []
    per_layer_freqs: Dict[int, List[float]] = {}
    for layer in sorted(counts.keys()):
        total = totals.get(layer, 0)
        if total <= 0:
            freqs = [0.0 for _ in range(args.experts)]
        else:
            freqs = [c / total * 100.0 for c in counts[layer]]
        if args.replication_slots > 0:
            freqs, _ = replicate_freqs(freqs, args.replication_slots)
        per_layer_freqs[layer] = freqs
        max_freqs.append(max(freqs) if freqs else 0.0)

    if max_freqs:
        mf_stats = _percentiles(max_freqs, [0.0, 0.5, 0.9, 0.99, 1.0])
        print("max expert frequency per layer (%)")
        print(f"  min: {mf_stats[0.0]:.4f}  p50: {mf_stats[0.5]:.4f}  p90: {mf_stats[0.9]:.4f}  p99: {mf_stats[0.99]:.4f}  max: {mf_stats[1.0]:.4f}")

    # Build bins if not provided (10 equal-width bins between global min/max)
    if args.bins:
        try:
            bins = [float(x.strip()) for x in args.bins.split(",") if x.strip()]
        except Exception:
            raise SystemExit("Invalid --bins. Use comma-separated numbers, e.g. 0,0.5,1,2,4,8")
        if len(bins) < 2:
            raise SystemExit("--bins must have at least two values")
    else:
        all_freqs = []
        for layer in sorted(per_layer_freqs.keys()):
            all_freqs.extend(per_layer_freqs[layer])
        if not all_freqs:
            bins = [0.0, 1.0]
        else:
            minv = min(all_freqs)
            maxv = max(all_freqs)
            if maxv == minv:
                maxv = minv + 1e-6
            step = (maxv - minv) / 10.0
            bins = [minv + i * step for i in range(11)]

    # Frequency distribution per layer (histogram of expert %)
    print("per-layer expert frequency histogram (% of activations)")
    def _cf(pct: float) -> float:
        return pct * 256.0 / 100.0

    bin_labels = [
        f"[{bins[i]:.2f}, {bins[i+1]:.2f})\nCF [{_cf(bins[i]):.2f}, {_cf(bins[i+1]):.2f})"
        for i in range(len(bins) - 1)
    ]
    per_layer_hist = {}
    for layer in sorted(per_layer_freqs.keys()):
        freqs = per_layer_freqs[layer]
        if not freqs:
            print(f"  layer {layer}: no data")
            continue
        hist = [0 for _ in range(len(bins) - 1)]
        for v in freqs:
            for i in range(len(bins) - 1):
                is_last = i == (len(bins) - 2)
                if bins[i] <= v < bins[i + 1] or (is_last and v <= bins[i + 1]):
                    hist[i] += 1
                    break
        parts = [f"{bin_labels[i]}={hist[i]}" for i in range(len(hist))]
        print(f"  layer {layer}: " + " ".join(parts))
        per_layer_hist[layer] = hist

    if args.hist_out and per_layer_hist:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e
        layers_sorted = sorted(per_layer_hist.keys())
        data = [per_layer_hist[l] for l in layers_sorted]
        plt.figure(figsize=(10, max(6, 0.2 * len(layers_sorted))))
        # Log-scale color normalization to emphasize low counts.
        try:
            from matplotlib.colors import LogNorm
            norm = LogNorm(vmin=1, vmax=max(max(row) for row in data if row))
        except Exception:
            norm = None
        plt.imshow(data, aspect="auto", cmap="magma", interpolation="nearest", norm=norm)
        plt.title("Per-layer Expert Frequency Histogram (counts)")
        plt.xlabel("Frequency bin (%)")
        plt.ylabel("Layer")
        plt.xticks(range(len(bin_labels)), bin_labels, rotation=45, ha="right")
        plt.yticks(range(len(layers_sorted)), [str(x) for x in layers_sorted], fontsize=8)
        plt.colorbar(label="Experts per bin")
        plt.tight_layout()
        plt.savefig(args.hist_out, dpi=200)


if __name__ == "__main__":
    main()
