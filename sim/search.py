#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Tuple, Optional

from cost_model import Mesh, Placement, CostParams, simulate_batch, simulate_batch_instances, aggregate
from expert_selection import balanced_expert_selection_replicas


def _print_progress(label: str, done: int, total: int) -> None:
    if total <= 0:
        return
    width = 40
    frac = min(1.0, max(0.0, done / total))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(frac * 100)
    print(f"\r{label} [{bar}] {pct:3d}%", end="", file=sys.stderr, flush=True)


def load_trace(path: str, layer: int, max_records: int = 0, show_progress: bool = False) -> List[Dict]:
    batches = []
    total_bytes = os.path.getsize(path)
    bytes_read = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            bytes_read += len(line.encode("utf-8"))
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
            if show_progress:
                _print_progress("load_trace", bytes_read, total_bytes)
    if show_progress:
        print("", file=sys.stderr)
    return batches


def initial_replication(expert_counts: List[int], total_slots: int,
                        shared_id: Optional[int] = None, shared_replicas: int = 0) -> Dict[int, int]:
    # Allocate at least 1 slot per expert, then assign extra by frequency.
    # shared_replicas is the TOTAL replicas for the shared expert (e.g., 16 = 1 base + 15 extra).
    n = len(expert_counts)
    slots = [1 for _ in range(n)]
    if shared_id is not None and 0 <= shared_id < n and shared_replicas > 0:
        slots[shared_id] = shared_replicas
    remaining = total_slots - sum(slots)
    if remaining <= 0:
        return {i: slots[i] for i in range(n)}
    # greedy by counts (ignore shared expert for extra replicas)
    order = sorted(
        [i for i in range(n) if i != shared_id],
        key=lambda i: expert_counts[i],
        reverse=True,
    )
    idx = 0
    while remaining > 0:
        slots[order[idx % len(order)]] += 1
        remaining -= 1
        idx += 1
    return {i: slots[i] for i in range(n)}


def random_placement(replication: Dict[int, int], mesh: Mesh, max_per_device: int) -> Placement:
    num_devices = mesh.rows * mesh.cols
    device_load = [0 for _ in range(num_devices)]
    expert_replicas: Dict[int, List[int]] = {}
    for expert, reps in replication.items():
        # sample devices with available capacity
        choices = [i for i in range(num_devices) if device_load[i] < max_per_device]
        if len(choices) < reps:
            raise SystemExit("Not enough device capacity for random placement")
        selected = random.sample(choices, reps)
        expert_replicas[expert] = selected
        for did in selected:
            device_load[did] += 1
    return Placement(expert_replicas)


def build_coact_matrix(trace: List[Dict], experts: int, include_shared: bool = False, shared_id: int = 256) -> List[List[int]]:
    coact = [[0 for _ in range(experts)] for _ in range(experts)]
    for rec in trace:
        for experts_list in rec["topk_experts"]:
            if include_shared and shared_id < experts:
                experts_list = list(experts_list) + [shared_id]
            for i in range(len(experts_list)):
                ei = experts_list[i]
                for j in range(i + 1, len(experts_list)):
                    ej = experts_list[j]
                    coact[ei][ej] += 1
                    coact[ej][ei] += 1
    return coact


def write_coact_csv(coact: List[List[int]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in coact:
            f.write(",".join(str(x) for x in row) + "\n")


def row_partition(replication: Dict[int, int], expert_counts: List[int],
                  coact: List[List[int]], mesh: Mesh, max_per_device: int) -> Dict[int, List[int]]:
    # Assign base experts to rows using co-activation affinity + load balance
    rows = mesh.rows
    experts = len(expert_counts)
    row_load = [0 for _ in range(rows)]
    row_slots = [0 for _ in range(rows)]
    row_capacity = max_per_device * mesh.cols
    row_experts: List[List[int]] = [[] for _ in range(rows)]

    order = sorted(range(experts), key=lambda e: expert_counts[e], reverse=True)
    for e in order:
        best_row = 0
        best_score = None
        for r in range(rows):
            affinity = sum(coact[e][x] for x in row_experts[r])
            score = affinity - 0.1 * row_load[r]
            if best_score is None or score > best_score:
                best_score = score
                best_row = r
        if row_slots[best_row] >= row_capacity:
            candidates = [r for r in range(rows) if row_slots[r] < row_capacity]
            if not candidates:
                raise SystemExit("All rows saturated during row_partition")
            best_row = min(candidates, key=lambda r: (row_load[r], r))
        row_experts[best_row].append(e)
        row_load[best_row] += expert_counts[e]
        row_slots[best_row] += 1

    # Place replicas: spread to balance row load, prefer rows without that expert
    replica_rows: Dict[int, List[int]] = {e: [r for r in range(rows) if e in row_experts[r]] for e in range(experts)}
    for e in range(experts):
        extra = replication[e] - 1
        for _ in range(extra):
            candidates = [r for r in range(rows) if row_slots[r] < row_capacity]
            if not candidates:
                raise SystemExit("All rows saturated during replica placement")
            candidates.sort(key=lambda r: (e in row_experts[r], row_load[r], r))
            r = candidates[0]
            row_experts[r].append(e)
            replica_rows[e].append(r)
            row_load[r] += expert_counts[e]
            row_slots[r] += 1

    return replica_rows


def hot_tier_partition(replication: Dict[int, int], expert_counts: List[int],
                       coact: List[List[int]], mesh: Mesh, max_per_device: int, top_n: int = 32) -> Dict[int, List[int]]:
    # Spread hot experts across rows, then use row_partition for the rest.
    rows = mesh.rows
    experts = len(expert_counts)
    hot = sorted(range(experts), key=lambda e: expert_counts[e], reverse=True)[:top_n]
    cold = [e for e in range(experts) if e not in hot]

    row_experts: List[List[int]] = [[] for _ in range(rows)]
    row_load = [0 for _ in range(rows)]
    row_slots = [0 for _ in range(rows)]
    row_capacity = max_per_device * mesh.cols

    # Place hot experts round-robin by load
    for e in hot:
        candidates = [r for r in range(rows) if row_slots[r] < row_capacity]
        if not candidates:
            raise SystemExit("All rows saturated during hot-tier placement")
        r = min(candidates, key=lambda x: (row_load[x], x))
        row_experts[r].append(e)
        row_load[r] += expert_counts[e]
        row_slots[r] += 1

    # Place cold experts by co-activation affinity
    order = sorted(cold, key=lambda e: expert_counts[e], reverse=True)
    for e in order:
        best_row = 0
        best_score = None
        for r in range(rows):
            affinity = sum(coact[e][x] for x in row_experts[r])
            score = affinity - 0.1 * row_load[r]
            if best_score is None or score > best_score:
                best_score = score
                best_row = r
        if row_slots[best_row] >= row_capacity:
            candidates = [r for r in range(rows) if row_slots[r] < row_capacity]
            if not candidates:
                raise SystemExit("All rows saturated during hot-tier placement")
            best_row = min(candidates, key=lambda r: (row_load[r], r))
        row_experts[best_row].append(e)
        row_load[best_row] += expert_counts[e]
        row_slots[best_row] += 1

    # Place replicas: spread to balance row load, prefer rows without that expert
    replica_rows: Dict[int, List[int]] = {e: [r for r in range(rows) if e in row_experts[r]] for e in range(experts)}
    for e in range(experts):
        extra = replication[e] - 1
        for _ in range(extra):
            candidates = [r for r in range(rows) if row_slots[r] < row_capacity]
            if not candidates:
                raise SystemExit("All rows saturated during hot-tier replica placement")
            candidates.sort(key=lambda r: (e in row_experts[r], row_load[r], r))
            r = candidates[0]
            row_experts[r].append(e)
            replica_rows[e].append(r)
            row_load[r] += expert_counts[e]
            row_slots[r] += 1

    return replica_rows


def row_first_balance(replication: Dict[int, int], expert_counts: List[int],
                      mesh: Mesh, max_per_device: int) -> Dict[int, List[int]]:
    # Assign experts to rows to balance row load, ignoring co-activation.
    rows = mesh.rows
    experts = len(expert_counts)
    row_load = [0 for _ in range(rows)]
    row_slots = [0 for _ in range(rows)]
    row_capacity = max_per_device * mesh.cols
    row_experts: List[List[int]] = [[] for _ in range(rows)]

    order = sorted(range(experts), key=lambda e: expert_counts[e], reverse=True)
    for e in order:
        candidates = [r for r in range(rows) if row_slots[r] < row_capacity]
        if not candidates:
            raise SystemExit("All rows saturated during row-balance placement")
        r = min(candidates, key=lambda x: (row_load[x], x))
        row_experts[r].append(e)
        row_load[r] += expert_counts[e]
        row_slots[r] += 1

    replica_rows: Dict[int, List[int]] = {e: [r for r in range(rows) if e in row_experts[r]] for e in range(experts)}
    for e in range(experts):
        extra = replication[e] - 1
        for _ in range(extra):
            candidates = [r for r in range(rows) if row_slots[r] < row_capacity]
            if not candidates:
                raise SystemExit("All rows saturated during row-balance replica placement")
            r = min(candidates, key=lambda x: (row_load[x], x))
            row_experts[r].append(e)
            replica_rows[e].append(r)
            row_load[r] += expert_counts[e]
            row_slots[r] += 1

    return replica_rows


def place_within_rows(replica_rows: Dict[int, List[int]], mesh: Mesh, max_per_device: int) -> Placement:
    # Assign each expert replica to a device within its row (least-loaded)
    num_devices = mesh.rows * mesh.cols
    device_load = [0 for _ in range(num_devices)]
    expert_replicas: Dict[int, List[int]] = {e: [] for e in replica_rows.keys()}
    for e, rows in replica_rows.items():
        for r in rows:
            # devices in row r: r*cols .. r*cols+cols-1
            start = r * mesh.cols
            row_devs = list(range(start, start + mesh.cols))
            # pick least-loaded device that is under capacity
            candidates = [did for did in row_devs if device_load[did] < max_per_device]
            if not candidates:
                raise SystemExit(f"Row {r} is saturated: cannot place more than {max_per_device} replicas per device")
            best = min(candidates, key=lambda did: (device_load[did], did))
            expert_replicas[e].append(best)
            device_load[best] += 1
    return Placement(expert_replicas)


def shared_expert_row_replication(mesh: Mesh, replicas: int) -> List[int]:
    # Place shared expert replicas: one per row first, then round-robin across columns.
    devices = []
    rows, cols = mesh.rows, mesh.cols
    for r in range(rows):
        if len(devices) >= replicas:
            break
        devices.append(r * cols)  # first column in each row
    remaining = replicas - len(devices)
    if remaining > 0:
        # fill remaining across rows/cols round-robin, skipping already used
        for r in range(rows):
            for c in range(1, cols):
                if remaining <= 0:
                    break
                did = r * cols + c
                devices.append(did)
                remaining -= 1
            if remaining <= 0:
                break
    return devices

def placement_to_device_map(placement: Placement) -> Dict[int, List[int]]:
    # device_id -> list of experts (one entry per replica)
    device_map: Dict[int, List[int]] = {}
    for e, devs in placement.expert_replicas.items():
        for did in devs:
            device_map.setdefault(did, []).append(e)
    return device_map


def device_map_to_placement(device_map: Dict[int, List[int]]) -> Placement:
    expert_replicas: Dict[int, List[int]] = {}
    for did, experts in device_map.items():
        for e in experts:
            expert_replicas.setdefault(e, []).append(did)
    return Placement(expert_replicas)


def local_search(trace: List[Dict], placement: Placement, mesh: Mesh,
                 params: CostParams, routing_strategy: str, capacity_factor: float,
                 include_shared: bool, objective: str, iters: int = 200) -> Placement:
    best = placement
    best_score = evaluate(trace, best, mesh, params, routing_strategy, capacity_factor, include_shared)[objective]
    device_map = placement_to_device_map(best)
    device_ids = list(range(mesh.rows * mesh.cols))

    for _ in range(iters):
        d1, d2 = random.sample(device_ids, 2)
        if not device_map.get(d1) or not device_map.get(d2):
            continue
        e1 = random.choice(device_map[d1])
        e2 = random.choice(device_map[d2])
        if e1 == e2:
            continue
        # swap one replica
        device_map[d1].remove(e1)
        device_map[d2].remove(e2)
        device_map[d1].append(e2)
        device_map[d2].append(e1)

        candidate = device_map_to_placement(device_map)
        score = evaluate(trace, candidate, mesh, params, routing_strategy, capacity_factor, include_shared)[objective]
        if score < best_score:
            best_score = score
            best = candidate
        else:
            # revert swap
            device_map[d1].remove(e2)
            device_map[d2].remove(e1)
            device_map[d1].append(e1)
            device_map[d2].append(e2)

    return best


def simulated_annealing(trace: List[Dict], placement: Placement, mesh: Mesh,
                        params: CostParams, routing_strategy: str, capacity_factor: float,
                        include_shared: bool, objective: str,
                        iters: int = 500, t0: float = 1.0, t1: float = 0.01) -> Placement:
    current = placement
    current_score = evaluate(trace, current, mesh, params, routing_strategy, capacity_factor, include_shared)[objective]
    best = current
    best_score = current_score

    device_map = placement_to_device_map(current)
    device_ids = list(range(mesh.rows * mesh.cols))

    for i in range(iters):
        t = t0 * ((t1 / t0) ** (i / max(1, iters - 1)))
        d1, d2 = random.sample(device_ids, 2)
        if not device_map.get(d1) or not device_map.get(d2):
            continue
        e1 = random.choice(device_map[d1])
        e2 = random.choice(device_map[d2])
        if e1 == e2:
            continue

        # swap one replica
        device_map[d1].remove(e1)
        device_map[d2].remove(e2)
        device_map[d1].append(e2)
        device_map[d2].append(e1)

        candidate = device_map_to_placement(device_map)
        score = evaluate(trace, candidate, mesh, params, routing_strategy, capacity_factor, include_shared)[objective]
        delta = score - current_score
        if delta <= 0 or random.random() < math.exp(-delta / max(1e-9, t)):
            current = candidate
            current_score = score
            if score < best_score:
                best_score = score
                best = candidate
        else:
            # revert swap
            device_map[d1].remove(e2)
            device_map[d2].remove(e1)
            device_map[d1].append(e1)
            device_map[d2].append(e2)

    return best


def build_instance_mapping(placement: Placement) -> Dict[str, object]:
    max_expert_id = max(placement.expert_replicas.keys()) if placement.expert_replicas else -1
    expert_id_mapping: List[List[int]] = [[] for _ in range(max_expert_id + 1)]
    instance_to_device: List[int] = []
    for expert_id in sorted(placement.expert_replicas.keys()):
        replicas = placement.expert_replicas[expert_id]
        for did in replicas:
            instance_id = len(instance_to_device)
            instance_to_device.append(did)
            expert_id_mapping[expert_id].append(instance_id)
    return {
        "expert_id_mapping": expert_id_mapping,
        "instance_to_device": instance_to_device,
        "num_expert_instances": len(instance_to_device),
    }


def _add_shared_expert(topk_experts: List[List[int]], shared_id: int) -> List[List[int]]:
    out = []
    for experts in topk_experts:
        if experts and experts[-1] == shared_id:
            out.append(experts)
        else:
            out.append(list(experts) + [shared_id])
    return out


def _build_origin_rows(num_tokens: int, rows: int, mode: str, rng: random.Random) -> List[int]:
    if mode == "round-robin":
        return [i % rows for i in range(num_tokens)]
    if mode == "random":
        return [rng.randrange(rows) for _ in range(num_tokens)]
    # average-rows handled separately
    return [0 for _ in range(num_tokens)]


def _avg_results(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {"max_dispatch": 0.0, "max_compute": 0.0, "max_combine": 0.0, "latency": 0.0}
    keys = results[0].keys()
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


def evaluate(trace: List[Dict], placement: Placement, mesh: Mesh, params: CostParams,
             routing_strategy: str, capacity_factor: float, include_shared: bool,
             show_progress: bool = False, label: str = "eval",
             origin_mode: str = "average-rows", seed: int = 0) -> Dict[str, float]:
    results = []
    mapping = build_instance_mapping(placement)
    expert_id_mapping = mapping["expert_id_mapping"]
    instance_to_device = mapping["instance_to_device"]
    num_expert_instances = mapping["num_expert_instances"]
    total = len(trace)
    rng = random.Random(seed)
    for idx, rec in enumerate(trace):
        topk = rec["topk_experts"]
        if include_shared:
            topk = _add_shared_expert(topk, shared_id=256)
        origin_rows = rec.get("origin_rows", [])
        num_tokens = len(topk)
        if not origin_rows:
            if origin_mode == "average-rows":
                per_row = []
                for rrow in range(mesh.rows):
                    origin_rows = [rrow for _ in range(num_tokens)]
                    if routing_strategy == "balanced-replicas":
                        active_experts, _ = balanced_expert_selection_replicas(
                            topk,
                            expert_id_mapping,
                            capacity_factor,
                            num_expert_instances,
                        )
                        per_row.append(simulate_batch_instances(origin_rows, active_experts, instance_to_device, mesh, params))
                    else:
                        per_row.append(simulate_batch(origin_rows, topk, placement, mesh, params))
                r = _avg_results(per_row)
            else:
                origin_rows = _build_origin_rows(num_tokens, mesh.rows, origin_mode, rng)
                if routing_strategy == "balanced-replicas":
                    active_experts, _ = balanced_expert_selection_replicas(
                        topk,
                        expert_id_mapping,
                        capacity_factor,
                        num_expert_instances,
                    )
                    r = simulate_batch_instances(origin_rows, active_experts, instance_to_device, mesh, params)
                else:
                    r = simulate_batch(origin_rows, topk, placement, mesh, params)
        else:
            if routing_strategy == "balanced-replicas":
                active_experts, _ = balanced_expert_selection_replicas(
                    topk,
                    expert_id_mapping,
                    capacity_factor,
                    num_expert_instances,
                )
                r = simulate_batch_instances(origin_rows, active_experts, instance_to_device, mesh, params)
            else:
                r = simulate_batch(origin_rows, topk, placement, mesh, params)
        results.append(r)
        if show_progress and total > 0 and (idx % 10 == 0 or idx == total - 1):
            pct = int((idx + 1) / total * 100)
            print(f"\r{label}: {pct:3d}%", end="", flush=True)
    if show_progress:
        print("")
    agg = aggregate(results)
    # Also aggregate dispatch/compute/combine components.
    if results:
        agg["max_dispatch_mean"] = sum(r["max_dispatch"] for r in results) / len(results)
        agg["max_compute_mean"] = sum(r["max_compute"] for r in results) / len(results)
        agg["max_combine_mean"] = sum(r["max_combine"] for r in results) / len(results)
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Search for a better MoE placement")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--slots", type=int, default=384)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--search", choices=["random", "row-aware", "anneal", "row-balance", "hot-tier"], default="row-balance")
    ap.add_argument("--local-search-iters", type=int, default=200)
    ap.add_argument("--anneal-iters", type=int, default=500)
    ap.add_argument("--anneal-t0", type=float, default=1.0)
    ap.add_argument("--anneal-t1", type=float, default=0.01)
    ap.add_argument("--coact-csv", default="", help="Write co-activation matrix to CSV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=0, help="Limit number of trace records (0 = no limit)")
    ap.add_argument("--progress", action="store_true", default=True, help="Show approximate progress while loading/evaluating trace")
    ap.add_argument("--coact-only", action="store_true", help="Only write co-activation CSV and exit")
    ap.add_argument("--routing-strategy", choices=["least-loaded", "balanced-replicas"], default="least-loaded")
    ap.add_argument("--capacity-factor", type=float, default=2.0)
    ap.add_argument("--no-shared-expert", action="store_true", help="Disable shared expert (id=256)")
    ap.add_argument("--no-shared-expert-row-replication", action="store_true",
                    help="Disable shared expert row-first replication placement")
    ap.add_argument("--objective", choices=["mean", "p95", "p99"], default="mean")
    ap.add_argument("--origin-mode", choices=["average-rows", "round-robin", "random"], default="average-rows")
    ap.add_argument("--save-placement", default="", help="Folder to store placement artifacts")
    args = ap.parse_args()

    random.seed(args.seed)
    mesh = Mesh(args.rows, args.cols)
    trace = load_trace(args.trace, args.layer, max_records=args.max_records, show_progress=args.progress)

    # Count expert usage for replication
    include_shared = not args.no_shared_expert
    total_experts = args.experts + (1 if include_shared else 0)
    counts = [0 for _ in range(total_experts)]
    for rec in trace:
        topk = rec["topk_experts"]
        if include_shared:
            topk = _add_shared_expert(topk, shared_id=256)
        for experts in topk:
            for e in experts:
                counts[e] += 1

    replication = initial_replication(counts, args.slots, shared_id=256 if include_shared else None, shared_replicas=16)
    # Compute:comm ratio = 1:3 (comm is 3x slower than compute for same token count).
    params = CostParams(device_capacity=1000.0, row_bandwidth=333.3333333333)

    best = None
    best_place = None

    coact = build_coact_matrix(trace, total_experts, include_shared=include_shared, shared_id=256)
    if args.coact_csv:
        write_coact_csv(coact, args.coact_csv)
        if args.coact_only:
            print(f"wrote co-activation CSV to {args.coact_csv}")
            return

    shared_devices = None
    if include_shared and not args.no_shared_expert_row_replication:
        shared_reps = replication.get(256, 0)
        shared_devices = shared_expert_row_replication(mesh, shared_reps)
        if shared_devices:
            shared_row_list = [mesh.devices()[did].row for did in shared_devices]
    else:
        shared_row_list = []

    max_per_device = int(math.ceil(args.slots / (args.rows * args.cols)))

    if args.search == "row-aware":
        replica_rows = row_partition(replication, counts, coact, mesh, max_per_device)
        if shared_row_list:
            replica_rows[256] = shared_row_list
        placement = place_within_rows(replica_rows, mesh, max_per_device)
        placement = local_search(
            trace, placement, mesh, params,
            routing_strategy=args.routing_strategy,
            capacity_factor=args.capacity_factor,
            include_shared=include_shared,
            objective=args.objective,
            iters=args.local_search_iters,
        )
        best_place = placement
        best = evaluate(trace, placement, mesh, params, args.routing_strategy, args.capacity_factor, include_shared, show_progress=args.progress, label="eval", origin_mode=args.origin_mode, seed=args.seed)
        print(
            f"row-aware[{args.objective}]: mean={best['mean']:.4f} p95={best['p95']:.4f} p99={best['p99']:.4f} "
            f"dispatch={best.get('max_dispatch_mean', 0.0):.4f} "
            f"compute={best.get('max_compute_mean', 0.0):.4f} "
            f"combine={best.get('max_combine_mean', 0.0):.4f}"
        )
    elif args.search == "anneal":
        replica_rows = row_partition(replication, counts, coact, mesh, max_per_device)
        if shared_row_list:
            replica_rows[256] = shared_row_list
        placement = place_within_rows(replica_rows, mesh, max_per_device)
        placement = simulated_annealing(
            trace, placement, mesh, params,
            routing_strategy=args.routing_strategy,
            capacity_factor=args.capacity_factor,
            include_shared=include_shared,
            objective=args.objective,
            iters=args.anneal_iters, t0=args.anneal_t0, t1=args.anneal_t1
        )
        best_place = placement
        best = evaluate(trace, placement, mesh, params, args.routing_strategy, args.capacity_factor, include_shared, show_progress=args.progress, label="eval", origin_mode=args.origin_mode, seed=args.seed)
        print(
            f"anneal[{args.objective}]: mean={best['mean']:.4f} p95={best['p95']:.4f} p99={best['p99']:.4f} "
            f"dispatch={best.get('max_dispatch_mean', 0.0):.4f} "
            f"compute={best.get('max_compute_mean', 0.0):.4f} "
            f"combine={best.get('max_combine_mean', 0.0):.4f}"
        )
    elif args.search == "row-balance":
        replica_rows = row_first_balance(replication, counts, mesh, max_per_device)
        if shared_row_list:
            replica_rows[256] = shared_row_list
        placement = place_within_rows(replica_rows, mesh, max_per_device)
        best_place = placement
        best = evaluate(trace, placement, mesh, params, args.routing_strategy, args.capacity_factor, include_shared, show_progress=args.progress, label="eval", origin_mode=args.origin_mode, seed=args.seed)
        print(
            f"row-balance[{args.objective}]: mean={best['mean']:.4f} p95={best['p95']:.4f} p99={best['p99']:.4f} "
            f"dispatch={best.get('max_dispatch_mean', 0.0):.4f} "
            f"compute={best.get('max_compute_mean', 0.0):.4f} "
            f"combine={best.get('max_combine_mean', 0.0):.4f}"
        )
    elif args.search == "hot-tier":
        replica_rows = hot_tier_partition(replication, counts, coact, mesh, max_per_device, top_n=32)
        if shared_row_list:
            replica_rows[256] = shared_row_list
        placement = place_within_rows(replica_rows, mesh, max_per_device)
        best_place = placement
        best = evaluate(trace, placement, mesh, params, args.routing_strategy, args.capacity_factor, include_shared, show_progress=args.progress, label="eval", origin_mode=args.origin_mode, seed=args.seed)
        print(
            f"hot-tier[{args.objective}]: mean={best['mean']:.4f} p95={best['p95']:.4f} p99={best['p99']:.4f} "
            f"dispatch={best.get('max_dispatch_mean', 0.0):.4f} "
            f"compute={best.get('max_compute_mean', 0.0):.4f} "
            f"combine={best.get('max_combine_mean', 0.0):.4f}"
        )
    else:
        for i in range(args.iters):
            placement = random_placement(replication, mesh, max_per_device)
            score = evaluate(trace, placement, mesh, params, args.routing_strategy, args.capacity_factor, include_shared, show_progress=args.progress, label=f"eval {i}", origin_mode=args.origin_mode, seed=args.seed)
            if best is None or score[args.objective] < best[args.objective]:
                best = score
                best_place = placement
            print(
                f"iter {i}[{args.objective}]: mean={score['mean']:.4f} p95={score['p95']:.4f} p99={score['p99']:.4f} "
                f"dispatch={score.get('max_dispatch_mean', 0.0):.4f} "
                f"compute={score.get('max_compute_mean', 0.0):.4f} "
                f"combine={score.get('max_combine_mean', 0.0):.4f}"
            )

    print("best:", best)
    if best_place:
        print("replicas per expert (non-1):", {k: v for k, v in replication.items() if v > 1})
        # Determine output folder
        out_dir = args.save_placement
        if not out_dir:
            from datetime import datetime
            base = os.path.basename(os.path.dirname(args.trace)) or "dataset"
            if base == "processed":
                base = os.path.basename(os.path.dirname(os.path.dirname(args.trace))) or "dataset"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join("analysis_artifacts", f"{base}_layer{args.layer}_{ts}")

        import json, os, subprocess
        os.makedirs(out_dir, exist_ok=True)

        # Save placement
        placement_path = os.path.join(out_dir, "placement.json")
        with open(placement_path, "w", encoding="utf-8") as f:
            json.dump(best_place.expert_replicas, f, indent=2)

        # Save run config
        commit = "unknown"
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd(), text=True).strip()
        except Exception:
            pass
        config = {
            "command": " ".join(sys.argv),
            "trace": args.trace,
            "layer": args.layer,
            "rows": args.rows,
            "cols": args.cols,
            "experts": args.experts,
            "slots": args.slots,
            "search": args.search,
            "routing_strategy": args.routing_strategy,
            "capacity_factor": args.capacity_factor,
            "objective": args.objective,
            "origin_mode": args.origin_mode,
            "include_shared": not args.no_shared_expert,
            "shared_row_replication": not args.no_shared_expert_row_replication,
            "commit": commit,
        }
        with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Render plots
        trace_path = args.trace
        full_plot = os.path.join(out_dir, "placement_instances.png")
        baseline_plot = os.path.join(out_dir, "placement_baseline_originals.png")
        plot_cmd = [
            sys.executable, "plots/plot_placement_grid.py",
            "--placement", placement_path,
            "--trace", trace_path,
            "--out", full_plot,
            "--rows", str(args.rows),
            "--cols", str(args.cols),
            "--experts", str(args.experts),
        ]
        plot_cmd_base = plot_cmd + ["--baseline-original", "--only-original", "--out", baseline_plot]
        subprocess.run(plot_cmd, check=False)
        subprocess.run(plot_cmd_base, check=False)


if __name__ == "__main__":
    main()
