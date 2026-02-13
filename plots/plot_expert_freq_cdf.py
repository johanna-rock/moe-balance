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


def _cdf_per_layer(counts: Dict[int, List[int]], totals: Dict[int, int], experts: int,
                   replication_slots: int) -> Dict[int, List[float]]:
    out: Dict[int, List[float]] = {}
    for layer in sorted(counts.keys()):
        total = totals.get(layer, 0)
        if total <= 0:
            out[layer] = [0.0 for _ in range(experts)]
            continue
        freqs = [(c / total) * 100.0 for c in counts[layer]]
        if replication_slots > 0:
            freqs, _ = replicate_freqs(freqs, replication_slots)
        freqs = sorted(freqs, reverse=True)
        cdf = []
        running = 0.0
        for v in freqs:
            running += v
            cdf.append(running)
        out[layer] = cdf
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot cumulative frequency (CDF) of ranked experts per layer")
    ap.add_argument("--trace", required=True, help="Trace JSONL")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--experts", type=int, default=256, help="Number of experts")
    ap.add_argument("--layers", default="", help="Comma-separated layer indices to include (empty = all)")
    ap.add_argument("--position-id", type=int, default=-1, help="Token position to use (default: all)")
    ap.add_argument("--position-ids", default="", help="Comma-separated positions or ranges, e.g. 0,1,4-8")
    ap.add_argument("--title", default="Cumulative Frequency of Ranked Experts")
    ap.add_argument("--dataset-name", default="", help="Dataset name to append to title")
    ap.add_argument("--progress", action="store_true", help="Show approximate progress while reading trace")
    ap.add_argument("--dedupe-origin-rows", action="store_true", help="If origin_row exists, only use origin_row=0")
    ap.add_argument("--request-id", default="", help="Filter to a specific request_id")
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

    counts, totals = _load_counts(
        args.trace,
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
        raise SystemExit("No data found for the requested layers/position.")

    cdf = _cdf_per_layer(counts, totals, args.experts, args.replication_slots)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e

    plt.figure(figsize=(10, 6))
    # Compute average cumulative frequency at x=32 and x=64 across layers.
    idxs = [32, 64]
    avg_at = {}
    for idx in idxs:
        vals = [values[idx - 1] for values in cdf.values() if len(values) >= idx]
        avg_at[idx] = sum(vals) / len(vals) if vals else 0.0

    for layer, values in cdf.items():
        plt.plot(range(1, len(values) + 1), values, label=f"L{layer}", linewidth=1.0, alpha=0.6)
    title = args.title
    if args.dataset_name:
        title = f"{title} ({args.dataset_name})"
    plt.title(title)
    plt.xlabel("Expert rank (1 = most active)")
    plt.ylabel("Cumulative frequency (%)")
    plt.axvline(32, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.axvline(64, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.ylim(0, 100)
    # Annotations for average cumulative frequency at x=32 and x=64
    plt.text(32, avg_at.get(32, 0.0), f"E=32\n{avg_at.get(32, 0.0):.2f}%", ha="left", va="bottom", fontsize=8)
    plt.text(64, avg_at.get(64, 0.0), f"E=64\n{avg_at.get(64, 0.0):.2f}%", ha="left", va="bottom", fontsize=8)

    print(f"avg cumulative frequency at E=32: {avg_at.get(32, 0.0):.4f}%")
    print(f"avg cumulative frequency at E=64: {avg_at.get(64, 0.0):.4f}%")
    if len(cdf) > 1:
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
    else:
        plt.tight_layout()
    plt.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
