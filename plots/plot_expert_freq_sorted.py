#!/usr/bin/env python3
import argparse
import os
import sys
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sim.replication import replicate_freqs


def _print_progress(label: str, done: int, total: int) -> None:
    if total <= 0:
        return
    width = 40
    frac = min(1.0, max(0.0, done / total))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(frac * 100)
    print(f"\r{label} [{bar}] {pct:3d}%", end="", file=sys.stderr, flush=True)


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
                raise SystemExit("Invalid range spec. Use comma-separated ints or ranges, e.g. 0,1,4-8")
            if end < start:
                start, end = end, start
            for v in range(start, end + 1):
                values.add(v)
        else:
            try:
                values.add(int(part))
            except Exception:
                raise SystemExit("Invalid range spec. Use comma-separated ints or ranges, e.g. 0,1,4-8")
    return values


def _parse_position_ids(spec: str) -> Set[int]:
    return _parse_ranges(spec)


def _load_counts(trace_path: str, experts: int, layer_filter: Optional[List[int]],
                 position_id: Optional[int], position_ids: Optional[Set[int]],
                 show_progress: bool, dedupe_origin_rows: bool, min_layer: int,
                 request_id: Optional[str]) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    counts: Dict[int, List[int]] = {}
    totals: Dict[int, int] = {}
    total_bytes = os.path.getsize(trace_path)
    try:
        import numpy as np
    except Exception:
        np = None
    bytes_read = 0
    with open(trace_path, "r", encoding="utf-8") as f:
        line_idx = 0
        for line in f:
            bytes_read += len(line.encode("utf-8"))
            rec = json.loads(line)
            if dedupe_origin_rows and rec.get("origin_row") not in (None, 0):
                continue
            if request_id is not None:
                rid = rec.get("request_id", rec.get("batch_id", None))
                if rid is None or str(rid) != request_id:
                    continue
            layer = rec.get("layer", 0)
            if layer < min_layer:
                continue
            if layer_filter is not None and layer not in layer_filter:
                continue
            if layer not in counts:
                counts[layer] = [0 for _ in range(experts)]
                totals[layer] = 0
            topk = rec.get("topk_experts", [])
            if position_id is None and position_ids is None:
                if np is not None:
                    flat = np.asarray(topk, dtype=int).reshape(-1)
                    binc = np.bincount(flat, minlength=experts)
                    counts[layer] = (np.asarray(counts[layer]) + binc).tolist()
                    totals[layer] += int(len(flat))
                else:
                    for experts_list in topk:
                        for e in experts_list:
                            counts[layer][e] += 1
                            totals[layer] += 1
            else:
                positions = []
                if position_ids is not None:
                    positions = [p for p in position_ids if 0 <= p < len(topk)]
                elif position_id is not None and 0 <= position_id < len(topk):
                    positions = [position_id]
                if positions:
                    if np is not None:
                        rows = np.asarray([topk[p] for p in positions], dtype=int).reshape(-1)
                        binc = np.bincount(rows, minlength=experts)
                        counts[layer] = (np.asarray(counts[layer]) + binc).tolist()
                        totals[layer] += int(len(rows))
                    else:
                        for p in positions:
                            for e in topk[p]:
                                counts[layer][e] += 1
                                totals[layer] += 1
            line_idx += 1
            if show_progress and (line_idx % 200 == 0):
                _print_progress("counting", bytes_read, total_bytes)
    if show_progress:
        print("", file=sys.stderr)
    return counts, totals


def _sorted_freq_vector(counts: Dict[int, List[int]], totals: Dict[int, int], experts: int, replication_slots: int) -> List[float]:
    layers_sorted = sorted(counts.keys())
    values: List[float] = []
    for layer in layers_sorted:
        total = totals.get(layer, 0)
        if total <= 0:
            values.extend([0.0 for _ in range(experts)])
        else:
            freqs = [(c / total) * 100.0 for c in counts[layer]]
            if replication_slots > 0:
                freqs, _ = replicate_freqs(freqs, replication_slots)
            values.extend(freqs)
    values.sort()
    return values


def _sorted_freq_per_layer(counts: Dict[int, List[int]], totals: Dict[int, int], experts: int,
                           replication_slots: int) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for layer in sorted(counts.keys()):
        total = totals.get(layer, 0)
        if total <= 0:
            values = [0.0 for _ in range(experts)]
        else:
            values = [(c / total) * 100.0 for c in counts[layer]]
        if replication_slots > 0:
            values, _ = replicate_freqs(values, replication_slots)
        values.sort()
        out[layer] = values
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot sorted expert frequencies across layers")
    ap.add_argument("--trace", action="append", required=True, help="Trace JSONL (repeatable)")
    ap.add_argument("--label", action="append", default=[], help="Label for each trace (repeatable)")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--experts", type=int, default=256, help="Number of experts")
    ap.add_argument("--layers", default="", help="Comma-separated layer indices to include (empty = all)")
    ap.add_argument("--position-id", type=int, default=-1, help="Token position to use (default: all)")
    ap.add_argument("--position-ids", default="", help="Comma-separated positions or ranges, e.g. 0,1,4-8")
    ap.add_argument("--title", default="Sorted Expert Frequencies (All Layers)")
    ap.add_argument("--progress", action="store_true", help="Show approximate progress while reading trace")
    ap.add_argument("--dedupe-origin-rows", action="store_true", help="If origin_row exists, only use origin_row=0")
    ap.add_argument("--request-id", default="", help="Filter to a specific request_id")
    ap.add_argument("--series-by-layer", action="store_true", help="Plot one curve per dataset+layer instead of collapsed")
    ap.add_argument("--replication-slots", type=int, default=0, help="Number of replication slots to add")
    args = ap.parse_args()

    if args.layers:
        layer_filter = sorted(_parse_ranges(args.layers))
        min_layer = 0
    else:
        layer_filter = None
        min_layer = 3

    position_id = None if args.position_id < 0 else args.position_id
    position_ids = _parse_position_ids(args.position_ids) if args.position_ids else None
    if position_id is not None and position_ids is not None:
        raise SystemExit("Use either --position-id or --position-ids, not both.")
    request_id = args.request_id.strip() if args.request_id else None

    labels = args.label if args.label else []
    if labels and len(labels) != len(args.trace):
        raise SystemExit("Number of --label entries must match number of --trace entries")

    series = []
    for i, trace_path in enumerate(args.trace):
        counts, totals = _load_counts(
            trace_path,
            args.experts,
            layer_filter,
            position_id,
            position_ids,
            show_progress=args.progress,
            dedupe_origin_rows=args.dedupe_origin_rows,
            min_layer=min_layer,
            request_id=request_id,
        )
        if not counts:
            raise SystemExit(f"No data found for trace: {trace_path}")
        label_base = labels[i] if labels else os.path.basename(trace_path)
        # Summary: combined frequency of top-32 experts per layer
        print(f"summary: top-32 combined frequency per layer for {label_base}")
        for layer in sorted(counts.keys()):
            total = totals.get(layer, 0)
            if total <= 0:
                print(f"  layer {layer}: 0.0000%")
                continue
            freqs = [(c / total) * 100.0 for c in counts[layer]]
            if args.replication_slots > 0:
                freqs, _ = replicate_freqs(freqs, args.replication_slots)
            freqs = sorted(freqs)
            top32 = sum(freqs[-32:]) if len(freqs) >= 32 else sum(freqs)
            print(f"  layer {layer}: {top32:.4f}%")
        if args.series_by_layer:
            per_layer = _sorted_freq_per_layer(counts, totals, args.experts, args.replication_slots)
            for layer, values in per_layer.items():
                series.append((label_base, layer, values))
        else:
            values = _sorted_freq_vector(counts, totals, args.experts, args.replication_slots)
            series.append((label_base, None, values))

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e

    plt.figure(figsize=(10, 6))
    # Color scheme: dataset 0 -> cool, dataset 1 -> warm (extend if more datasets).
    try:
        import matplotlib.cm as cm
    except Exception:
        cm = None
    dataset_labels = list({s[0] for s in series})
    dataset_labels.sort()
    dataset_cmaps = {}
    if cm is not None:
        for idx, ds in enumerate(dataset_labels):
            dataset_cmaps[ds] = cm.Blues if idx % 2 == 0 else cm.Oranges

    # Precompute layer ranges per dataset for gradients.
    layer_ranges = {}
    if args.series_by_layer:
        for ds in dataset_labels:
            layers = [s[1] for s in series if s[0] == ds and s[1] is not None]
            if layers:
                layer_ranges[ds] = (min(layers), max(layers))

    for ds, layer, values in series:
        if args.series_by_layer and layer is not None and cm is not None and ds in dataset_cmaps:
            lo, hi = layer_ranges.get(ds, (layer, layer))
            denom = max(1, hi - lo)
            t = (layer - lo) / denom
            color = dataset_cmaps[ds](0.3 + 0.7 * t)
            label = f"{ds}:L{layer}"
        else:
            # Fallback colors for collapsed series
            if cm is not None and ds in dataset_cmaps:
                color = dataset_cmaps[ds](0.6)
            else:
                color = None
            label = ds
        plt.plot(range(len(values)), values, label=label, linewidth=1.2, color=color, alpha=0.5)
    plt.title(args.title)
    plt.xlabel("Sorted expert activations (flattened across layers)")
    plt.ylabel("Frequency (%)")
    # Vertical reference line at position 224
    plt.axvline(224, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    if len(series) > 1:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        plt.tight_layout()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
