#!/usr/bin/env python3
import argparse
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple


def load_placement(path: str) -> Dict[int, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # keys may be strings
    return {int(k): v for k, v in data.items()}


def compute_freqs(trace_path: str, experts: int, include_shared: bool, shared_id: int = 256) -> List[float]:
    counts = [0 for _ in range(experts)]
    total = 0
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            topk = rec.get("topk_experts", [])
            for experts_list in topk:
                if include_shared and shared_id < experts:
                    experts_list = list(experts_list) + [shared_id]
                for e in experts_list:
                    if 0 <= e < experts:
                        counts[e] += 1
                        total += 1
    if total == 0:
        return [0.0 for _ in range(experts)]
    return [c / total for c in counts]


def build_instance_mapping(placement: Dict[int, List[int]]) -> Tuple[List[int], List[int]]:
    # instance_to_device, instance_to_original
    instance_to_device = []
    instance_to_original = []
    for expert_id in sorted(placement.keys()):
        for did in placement[expert_id]:
            instance_to_device.append(did)
            instance_to_original.append(expert_id)
    return instance_to_device, instance_to_original


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot placement grid with expert bars")
    ap.add_argument("--placement", required=True, help="placement.json from search run")
    ap.add_argument("--trace", required=True, help="trace JSONL")
    ap.add_argument("--out", required=True, help="Output PNG")
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--include-shared", action="store_true", default=True)
    ap.add_argument("--shared-id", type=int, default=256)
    ap.add_argument("--only-original", action="store_true", help="Plot only one bar per original expert")
    ap.add_argument("--baseline-original", action="store_true", help="Ignore placement; use row-wise expert ids + shared per device")
    ap.add_argument("--text", action="store_true", help="Render text values instead of bar heights")
    ap.add_argument("--freq-folder", default="", help="Folder containing expert_frequency.csv")
    ap.add_argument("--layer", type=int, default=0, help="Layer id to select from expert_frequency.csv")
    args = ap.parse_args()

    if args.freq_folder:
        freq_csv = os.path.join(args.freq_folder, "expert_frequency.csv")
        try:
            with open(freq_csv, "r", encoding="utf-8") as f:
                rows = f.readlines()
            freqs = None
            for line in rows[1:]:
                if line.startswith(f"Layer {args.layer},"):
                    parts = line.strip().split(",")[1:]
                    freqs = [float(x) for x in parts]
                    break
            if freqs is None:
                freqs = compute_freqs(args.trace, args.experts + (1 if args.include_shared else 0), args.include_shared, args.shared_id)
        except Exception:
            freqs = compute_freqs(args.trace, args.experts + (1 if args.include_shared else 0), args.include_shared, args.shared_id)
    else:
        freqs = compute_freqs(args.trace, args.experts + (1 if args.include_shared else 0), args.include_shared, args.shared_id)
    if freqs:
        if args.include_shared and args.experts > 0:
            max_freq = max(freqs[:args.experts])
        else:
            max_freq = max(freqs)
    else:
        max_freq = 1.0

    if args.baseline_original:
        # Baseline: gated experts placed in row-wise order, one shared expert per device.
        instance_to_device = []
        instance_to_original = []
        num_devices = args.rows * args.cols
        experts_per_device = args.experts // num_devices
        for did in range(num_devices):
            start = did * experts_per_device
            for oid in range(start, min(start + experts_per_device, args.experts)):
                instance_to_device.append(did)
                instance_to_original.append(oid)
            if args.include_shared:
                instance_to_device.append(did)
                instance_to_original.append(args.shared_id)
        # Adjust shared expert frequency to be evenly split per device
        if args.include_shared and args.shared_id < len(freqs):
            shared_freq_total = freqs[args.shared_id]
            per_device = shared_freq_total / max(1, num_devices)
            freqs = list(freqs)
            freqs[args.shared_id] = per_device
    else:
        placement = load_placement(args.placement)
        instance_to_device, instance_to_original = build_instance_mapping(placement)

    # group instances per device
    device_instances = defaultdict(list)
    for inst_id, did in enumerate(instance_to_device):
        device_instances[did].append(inst_id)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except Exception as e:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e

    # color per original expert
    cmap = plt.colormaps.get_cmap("tab20").resampled(args.experts + (1 if args.include_shared else 0))

    fig, ax = plt.subplots(figsize=(args.cols * 1.2, args.rows * 1.2))
    ax.set_xlim(0, args.cols)
    ax.set_ylim(0, args.rows)
    ax.set_xticks([c + 0.5 for c in range(args.cols)])
    ax.set_yticks([r + 0.5 for r in range(args.rows)])
    ax.set_xticklabels([str(c) for c in range(args.cols)])
    ax.set_yticklabels([str(r) for r in range(args.rows)])
    ax.invert_yaxis()
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title("Expert Placement Grid")

    # draw device grid (thicker)
    for r in range(args.rows + 1):
        ax.axhline(r, color="gray", linewidth=1.0)
    for c in range(args.cols + 1):
        ax.axvline(c, color="gray", linewidth=1.0)

    for did, insts in device_instances.items():
        r = did // args.cols
        c = did % args.cols
        insts_sorted = sorted(insts, key=lambda i: instance_to_original[i])
        if args.only_original:
            # one bar per original expert (unique)
            seen = set()
            filtered = []
            for inst in insts_sorted:
                oid = instance_to_original[inst]
                if oid in seen:
                    continue
                seen.add(oid)
                filtered.append(inst)
            insts_sorted = filtered

        if not insts_sorted:
            continue

        bar_width = 1.0 / max(1, len(insts_sorted))
        for idx, inst in enumerate(insts_sorted):
            oid = instance_to_original[inst]
            freq = freqs[oid] if oid < len(freqs) else 0.0
            height = freq / max_freq if max_freq > 0 else 0.0
            x0 = c + idx * bar_width
            color = cmap(oid)
            # expert sub-cell grid
            ax.add_patch(plt.Rectangle((x0, r), bar_width, 1.0, fill=False, edgecolor="lightgray", linewidth=0.3))
            if args.text:
                ax.text(
                    x0 + bar_width / 2,
                    r + 0.5,
                    f"E:{oid}\n{freq:.4f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
            else:
                y0 = r + (1.0 - height)
                rect = plt.Rectangle((x0, y0), bar_width, height, color=color)
                ax.add_patch(rect)
                # Add expert id at top of bar and frequency at bottom
                # Expert id at top of cell, frequency at bottom of cell
                ax.text(
                    x0 + bar_width / 2,
                    r + 0.10,
                    f"{oid}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    color="black",
                )
                ax.text(
                    x0 + bar_width / 2,
                    r + 0.90,
                    f"{freq:.4f}",
                    ha="center",
                    va="top",
                    fontsize=5,
                    color="black",
                )

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()
