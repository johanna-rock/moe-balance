#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from typing import Any, List


def _as_request_id(value: Any, fallback: int) -> Any:
    # Normalize request_id to int/str for JSON
    try:
        if hasattr(value, "shape") and value.shape == ():
            value = value.item()
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return fallback


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert NPZ routing data to trace JSONL")
    ap.add_argument("--data-dir", required=True, help="Directory containing .npz files")
    ap.add_argument("--file", default="", help="Convert only this .npz file (basename or path)")
    ap.add_argument("--out", required=True, help="Output JSONL file")
    ap.add_argument("--rows", type=int, default=16, help="Number of DP rows")
    ap.add_argument("--skip-origin-rows", action="store_true", help="Do not populate origin_rows or duplicate per row")
    ap.add_argument("--token-slice", choices=["all", "prompt", "output"], default="all")
    ap.add_argument("--layers", default="", help="Comma-separated layer indices to include (empty = all)")
    # origin rows are assigned in round-robin blocks of 32 tokens
    ap.add_argument("--max-requests", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    import numpy as np  # local import so script can exist without numpy installed

    if args.file:
        if os.path.isabs(args.file) or os.path.dirname(args.file):
            files = [os.path.basename(args.file)]
            data_dir = os.path.dirname(args.file) or args.data_dir
        else:
            files = [args.file]
            data_dir = args.data_dir
    else:
        data_dir = args.data_dir
        files = [f for f in os.listdir(args.data_dir) if f.endswith(".npz")]
    files.sort()
    if args.max_requests and args.max_requests > 0:
        files = files[: args.max_requests]

    skipped = 0
    processed = 0
    if args.layers:
        try:
            layer_filter = {int(x.strip()) for x in args.layers.split(",") if x.strip() != ""}
        except Exception:
            raise SystemExit("Invalid --layers. Use comma-separated integers, e.g. 0,1,2")
    else:
        layer_filter = None

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out_f:
        for idx, name in enumerate(files):
            path = os.path.join(data_dir, name)
            npz = np.load(path, allow_pickle=True)

            if "routed_experts" not in npz:
                print(f"warn: skipping {name}: missing routed_experts", file=sys.stderr)
                skipped += 1
                continue

            routed = npz["routed_experts"]
            # Some files store routed_experts as a 0-d object containing the real array.
            if routed.ndim == 0 and routed.dtype == object:
                try:
                    item = routed.item()
                    if item is None:
                        print(f"warn: skipping {name}: routed_experts is None", file=sys.stderr)
                        skipped += 1
                        continue
                    routed = np.array(item)
                except Exception:
                    pass
            if routed is None or (hasattr(routed, "size") and routed.size == 0):
                # Skip malformed entries with no routing data
                print(f"warn: skipping {name}: routed_experts empty", file=sys.stderr)
                skipped += 1
                continue
            if routed.ndim != 3:
                print(
                    f"warn: skipping {name}: expected routed_experts 3D [seq, layers, k], got {routed.shape}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            prompt_len = len(npz.get("prompt_token_ids", []))
            output_len = len(npz.get("output_token_ids", []))
            total_len = routed.shape[0]

            if prompt_len + output_len == 0:
                # fallback to routed_experts length
                prompt_len = total_len
                output_len = 0

            if args.token_slice == "prompt":
                start, end = 0, min(prompt_len, total_len)
            elif args.token_slice == "output":
                start = min(prompt_len, total_len)
                end = min(prompt_len + output_len, total_len)
            else:
                start, end = 0, total_len

            token_indices = list(range(start, end))

            request_id = _as_request_id(npz.get("request_id", idx), idx)
            num_layers = routed.shape[1]

            for layer in range(num_layers):
                if layer_filter is not None and layer not in layer_filter:
                    continue
                topk_experts: List[List[int]] = []
                for t_i, t in enumerate(token_indices):
                    experts = routed[t, layer].tolist()
                    topk_experts.append([int(x) for x in experts])

                if args.skip_origin_rows:
                    record = {
                        "layer": layer,
                        "batch_id": request_id,
                        "origin_rows": [],
                        "topk_experts": topk_experts,
                        "request_id": request_id,
                        "source_file": name,
                    }
                    out_f.write(json.dumps(record) + "\n")
                else:
                    # Duplicate each request for every row: all tokens share the same row.
                    for req_row in range(args.rows):
                        origin_rows: List[int] = [req_row for _ in token_indices]
                        record = {
                            "layer": layer,
                            "batch_id": request_id,
                            "origin_rows": origin_rows,
                            "topk_experts": topk_experts,
                            "request_id": request_id,
                            "source_file": name,
                            "origin_row": req_row,
                        }
                        out_f.write(json.dumps(record) + "\n")
            processed += 1

    print(f"summary: processed={processed} skipped={skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
