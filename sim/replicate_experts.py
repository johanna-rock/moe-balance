#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

from replication import replicate_freqs


def _parse_ranges(spec: str) -> Set[int]:
    values: Set[int] = set()
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                start, end = end, start
            for v in range(start, end + 1):
                values.add(v)
        else:
            values.add(int(part))
    return values


def _load_counts(trace_path: str, experts: int, layer_filter: Optional[List[int]]) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    counts: Dict[int, List[int]] = {}
    totals: Dict[int, int] = {}
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            layer = rec.get("layer", 0)
            if layer_filter is not None and layer not in layer_filter:
                continue
            if layer not in counts:
                counts[layer] = [0 for _ in range(experts)]
                totals[layer] = 0
            for experts_list in rec.get("topk_experts", []):
                for e in experts_list:
                    if 0 <= e < experts:
                        counts[layer][e] += 1
                        totals[layer] += 1
    return counts, totals


def main() -> None:
    ap = argparse.ArgumentParser(description="Replicate experts by splitting highest-frequency experts")
    ap.add_argument("--trace", required=True, help="Trace JSONL")
    ap.add_argument("--replication-slots", type=int, required=True, help="Number of replication slots")
    ap.add_argument("--experts", type=int, default=256, help="Number of base experts")
    ap.add_argument("--layers", default="", help="Comma-separated layer indices or ranges")
    ap.add_argument("--out", default="", help="Output JSON file (defaults to stdout)")
    args = ap.parse_args()

    layer_filter = sorted(_parse_ranges(args.layers)) if args.layers else None

    counts, totals = _load_counts(args.trace, args.experts, layer_filter)
    result = {"replication_slots": args.replication_slots, "experts": args.experts, "layers": {}}

    for layer in sorted(counts.keys()):
        total = totals.get(layer, 0)
        if total <= 0:
            freqs = [0.0 for _ in range(args.experts)]
        else:
            freqs = [(c / total) * 100.0 for c in counts[layer]]
        replicated, mapping = replicate_freqs(freqs, args.replication_slots)
        result["layers"][str(layer)] = {
            "replica_to_original": mapping,
            "replica_frequencies": replicated,
        }

    out_text = json.dumps(result, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_text)
    else:
        print(out_text)


if __name__ == "__main__":
    main()
