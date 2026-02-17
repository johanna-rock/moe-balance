#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict


def _strip_jsonc(text: str) -> str:
    out = []
    in_string = False
    escape = False
    line_comment = False
    block_comment = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if ch == "\n":
                line_comment = False
                out.append(ch)
            i += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            i += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        if ch == "\"":
            in_string = True
            out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _load_jsonc(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return json.loads(_strip_jsonc(raw))


def load_trace(path: str, layer: int, max_records: int = 0) -> List[Dict]:
    batches = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["layer"] == layer:
                if "tokens" in rec:
                    origin_rows = [t["origin_row"] for t in rec["tokens"]]
                    topk_experts = [t["topk_experts"] for t in rec["tokens"]]
                    rec = {
                        "layer": rec["layer"],
                        "batch_id": rec.get("batch_id", 0),
                        "origin_rows": origin_rows,
                        "topk_experts": topk_experts,
                    }
                batches.append(rec)
                if max_records and len(batches) >= max_records:
                    break
    return batches


def build_coact_matrix(trace: List[Dict], experts: int, include_shared: bool, shared_ids: List[int]) -> List[List[int]]:
    coact = [[0 for _ in range(experts)] for _ in range(experts)]
    for rec in trace:
        for experts_list in rec["topk_experts"]:
            if include_shared and shared_ids:
                extras = [sid for sid in shared_ids if sid < experts]
                if extras:
                    experts_list = list(experts_list) + extras
            for i in range(len(experts_list)):
                ei = experts_list[i]
                for j in range(i + 1, len(experts_list)):
                    ej = experts_list[j]
                    coact[ei][ej] += 1
                    coact[ej][ei] += 1
    return coact


def write_coact_csv(coact: List[List[int]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in coact:
            f.write(",".join(str(x) for x in row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export co-activation CSV from trace")
    ap.add_argument("--trace", default="")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--num-shared-experts", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--system-config", default="", help="JSONC file with system/dataset parameters")
    ap.add_argument("--out", required=True, help="CSV output path")
    args = ap.parse_args()

    if args.system_config:
        sys_cfg = _load_jsonc(args.system_config)
        for k in ["trace", "layer", "experts", "num_shared_experts", "max_records"]:
            if k in sys_cfg:
                setattr(args, k, sys_cfg[k])
    if not args.trace:
        raise SystemExit("--trace or --system-config with trace is required")

    include_shared = args.num_shared_experts > 0
    total_experts = args.experts + max(0, args.num_shared_experts)
    shared_ids = list(range(args.experts, args.experts + max(0, args.num_shared_experts)))

    trace = load_trace(args.trace, args.layer, max_records=args.max_records)
    coact = build_coact_matrix(trace, total_experts, include_shared, shared_ids)
    write_coact_csv(coact, args.out)
    print(f"wrote co-activation CSV to {args.out}")


if __name__ == "__main__":
    main()
