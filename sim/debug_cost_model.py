#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List, Tuple

from cost_model import Mesh, Placement, CostParams, calibrate_dispatch_coeffs, _dispatch_time
from expert_selection import balanced_expert_selection_replicas


def load_trace_record(trace_path: str, layer: int) -> Dict:
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("layer", 0) == layer:
                return rec
    raise SystemExit(f"No record found for layer {layer}")


def _parse_placement_json(raw: dict) -> Placement:
    expert_replicas = {}
    for k, v in raw.items():
        eid = int(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            expert_replicas[eid] = [int(item["device"]) for item in v]
        else:
            expert_replicas[eid] = [int(d) for d in v]
    return Placement(expert_replicas=expert_replicas)

def build_instance_mapping(placement: Placement) -> Tuple[List[List[int]], List[int], List[int]]:
    max_expert_id = max(placement.expert_replicas.keys()) if placement.expert_replicas else -1
    expert_id_mapping: List[List[int]] = [[] for _ in range(max_expert_id + 1)]
    instance_to_device: List[int] = []
    instance_to_original: List[int] = []
    for expert_id in sorted(placement.expert_replicas.keys()):
        for did in placement.expert_replicas[expert_id]:
            instance_id = len(instance_to_device)
            instance_to_device.append(did)
            instance_to_original.append(expert_id)
            expert_id_mapping[expert_id].append(instance_id)
    return expert_id_mapping, instance_to_device, instance_to_original


def ring_hop(src_row: int, dst_row: int, rows: int) -> int:
    linear = abs(dst_row - src_row)
    return min(linear, rows - linear)


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug cost model on two tokens")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--placement", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--capacity-factor", type=float, default=2.0)
    ap.add_argument("--token-indices", default="0,1", help="Comma-separated token indices")
    ap.add_argument("--origin-rows", default="", help="Comma-separated origin rows for the selected tokens")
    ap.add_argument("--include-shared", action="store_true", default=True)
    ap.add_argument("--shared-id", type=int, default=256)
    args = ap.parse_args()

    with open(args.placement, "r", encoding="utf-8") as f:
        raw = json.load(f)
    placement = _parse_placement_json(raw)

    rec = load_trace_record(args.trace, args.layer)
    topk = rec.get("topk_experts", [])
    if args.include_shared:
        topk = [list(t) + [args.shared_id] for t in topk]

    token_indices = [int(x) for x in args.token_indices.split(",") if x.strip() != ""]
    if len(token_indices) != 2:
        raise SystemExit("Provide exactly two token indices, e.g. --token-indices 0,1")

    selected_topk = [topk[i] for i in token_indices]

    if args.origin_rows:
        origin_rows = [int(x) for x in args.origin_rows.split(",") if x.strip() != ""]
        if len(origin_rows) != 2:
            raise SystemExit("Provide exactly two origin rows, e.g. --origin-rows 0,15")
    else:
        origin_rows = [0, 0]

    mesh = Mesh(args.rows, args.cols)
    coeffs = calibrate_dispatch_coeffs(mesh.rows)
    params = CostParams(
        device_capacity=1000.0,
        row_bandwidth=333.3333333333,
        dispatch_coeffs=coeffs,
        combine_coeffs=coeffs,
    )

    expert_id_mapping, instance_to_device, instance_to_original = build_instance_mapping(placement)
    num_expert_instances = len(instance_to_device)
    active_experts, _ = balanced_expert_selection_replicas(
        selected_topk,
        expert_id_mapping,
        args.capacity_factor,
        num_expert_instances,
    )

    print("== Debug Cost Model (2 tokens) ==")
    print(f"layer: {args.layer}")
    print(f"token indices: {token_indices}")
    print(f"origin rows: {origin_rows}")
    print(f"routing: balanced-replicas, capacity_factor={args.capacity_factor}")
    print(f"dispatch coeffs (A,B,C,D): {coeffs}")
    print("")

    device_expert_counts: Dict[int, Dict[int, int]] = {}
    expert_token_counts: Dict[int, int] = {}
    replica_token_counts: Dict[int, int] = {}
    dispatch_times = []
    combine_times = []

    for t_idx, (token_id, experts, origins) in enumerate(zip(token_indices, active_experts, origin_rows)):
        print(f"Token {token_id}:")
        dests: List[Tuple[int, int]] = []
        for inst in experts:
            if inst < 0 or inst >= len(instance_to_device):
                continue
            did = instance_to_device[inst]
            oid = instance_to_original[inst] if inst < len(instance_to_original) else -1
            dests.append((oid, did))
            device_expert_counts.setdefault(did, {})
            device_expert_counts[did][oid] = device_expert_counts[did].get(oid, 0) + 1
            expert_token_counts[oid] = expert_token_counts.get(oid, 0) + 1
            replica_token_counts[inst] = replica_token_counts.get(inst, 0) + 1

        hops = []
        col_to_experts: Dict[int, set] = {}
        devices = set()
        for oid, did in dests:
            row = did // mesh.cols
            col = did % mesh.cols
            hop = ring_hop(origins, row, mesh.rows)
            hops.append(hop)
            col_to_experts.setdefault(col, set()).add(oid)
            devices.add(did)

        e_col = max((len(v) for v in col_to_experts.values()), default=0)
        avg_hop = sum(hops) / len(hops) if hops else 0.0
        max_hop = max(hops) if hops else 0.0
        u = len(devices)

        dispatch_t = _dispatch_time(e_col, avg_hop, max_hop, params.dispatch_coeffs)
        combine_t = _dispatch_time(u, avg_hop, max_hop, params.combine_coeffs)
        dispatch_times.append(dispatch_t)
        combine_times.append(combine_t)

        print(f"  topk original experts: {selected_topk[t_idx]}")
        print(f"  active instances: {experts}")
        print(f"  active originals: {[instance_to_original[i] if 0 <= i < len(instance_to_original) else -1 for i in experts]}")
        print(f"  dests (orig_expert, device_id): {dests}")
        print(f"  E_col: {e_col}, U: {u}")
        print(f"  hops: {hops}")
        print(f"  avg_hop: {avg_hop:.2f}, max_hop: {max_hop:.2f}")
        print(f"  dispatch_us: {dispatch_t:.3f}")
        print(f"  combine_us: {combine_t:.3f}")
        print("")

    # compute cost per device
    compute_times = []
    device_expert_counts_summary: Dict[int, int] = {}
    for did, ec in device_expert_counts.items():
        per_dev = 0.0
        for t_e in ec.values():
            if t_e <= 0:
                continue
            blocks = (t_e + params.compute_block - 1) // params.compute_block
            per_dev += params.compute_base_us * blocks
        compute_times.append(per_dev)
        device_expert_counts_summary[did] = len(ec)

    max_dispatch = max(dispatch_times) if dispatch_times else 0.0
    max_combine = max(combine_times) if combine_times else 0.0
    max_compute = max(compute_times) if compute_times else 0.0

    print("== Aggregate over the 2 tokens ==")
    print(f"max_dispatch_us: {max_dispatch:.3f}")
    print(f"max_compute_us: {max_compute:.3f}")
    print(f"max_combine_us: {max_combine:.3f}")
    print(f"total_us: {max_dispatch + max_compute + max_combine:.3f}")
    print("")
    print("experts per device (over selected tokens):")
    for did in sorted(device_expert_counts_summary.keys()):
        print(f"  device {did}: {device_expert_counts_summary[did]}")
    print("")
    print("tokens per original expert (over selected tokens):")
    for oid in sorted(expert_token_counts.keys()):
        print(f"  expert {oid}: {expert_token_counts[oid]}")
    print("")
    print("tokens per expert replica (over selected tokens):")
    for inst in sorted(replica_token_counts.keys()):
        oid = instance_to_original[inst] if 0 <= inst < len(instance_to_original) else -1
        did = instance_to_device[inst] if 0 <= inst < len(instance_to_device) else -1
        print(f"  inst {inst} (orig {oid}, device {did}): {replica_token_counts[inst]}")


if __name__ == "__main__":
    main()
