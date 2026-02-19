#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
import time
import subprocess
from typing import Dict, List, Tuple, Optional

from cost_model import Mesh, Placement, CostParams, simulate_batch, simulate_batch_instances, aggregate, calibrate_dispatch_coeffs
from expert_selection import balanced_expert_selection_replicas

def _format_eta(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return f"{seconds:0.1f}s"
    m, s = divmod(seconds, 60)
    m = int(m)
    s = int(s)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"

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

def _print_progress(label: str, done: int, total: int, start_time: float) -> None:
    if total <= 0:
        return
    width = 40
    frac = min(1.0, max(0.0, done / total))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = int(frac * 100)
    elapsed = time.perf_counter() - start_time
    eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
    print(
        f"\r{label} [{bar}] {pct:3d}%  elapsed { _format_eta(elapsed) }  eta { _format_eta(eta) }",
        end="",
        file=sys.stderr,
        flush=True,
    )

def load_trace(path: str, layer: int, max_records: int = 0, show_progress: bool = False,
               fast_test_pct: float = 0.0, seed: int = 0,
               max_seq_len: int = 0) -> List[Dict]:
    batches = []
    total_bytes = os.path.getsize(path)
    bytes_read = 0
    start_time = time.perf_counter()
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
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
                if max_seq_len and max_seq_len > 0:
                    rec = dict(rec)
                    rec["topk_experts"] = rec.get("topk_experts", [])[:max_seq_len]
                    if "origin_rows" in rec and rec["origin_rows"]:
                        rec["origin_rows"] = rec["origin_rows"][:max_seq_len]
                if fast_test_pct and 0.0 < fast_test_pct < 1.0:
                    topk = rec.get("topk_experts", [])
                    if topk:
                        k = max(1, int(len(topk) * fast_test_pct))
                        rng = random.Random(seed + idx)
                        pick = sorted(rng.sample(range(len(topk)), k))
                        rec = dict(rec)
                        rec["topk_experts"] = [topk[i] for i in pick]
                        if "origin_rows" in rec and rec["origin_rows"]:
                            rec["origin_rows"] = [rec["origin_rows"][i] for i in pick]
                batches.append(rec)
                if max_records and len(batches) >= max_records:
                    break
            if show_progress:
                _print_progress("load_trace", bytes_read, total_bytes, start_time)
    if show_progress:
        print("", file=sys.stderr)
    return batches

def initial_replication(expert_counts: List[int], total_slots: int,
                        shared_ids: Optional[List[int]] = None, shared_replicas: int = 0) -> Dict[int, int]:
    # Allocate at least 1 slot per expert, then assign extra by frequency.
    # shared_replicas is the TOTAL replicas for the shared expert (e.g., 16 = 1 base + 15 extra).
    n = len(expert_counts)
    slots = [1 for _ in range(n)]
    if shared_ids and shared_replicas > 0:
        for shared_id in shared_ids:
            if 0 <= shared_id < n:
                slots[shared_id] = shared_replicas
    remaining = total_slots - sum(slots)
    if remaining <= 0:
        return {i: slots[i] for i in range(n)}
    # greedy by counts (ignore shared expert for extra replicas)
    order = sorted(
        [i for i in range(n) if not shared_ids or i not in shared_ids],
        key=lambda i: expert_counts[i],
        reverse=True,
    )
    idx = 0
    while remaining > 0:
        slots[order[idx % len(order)]] += 1
        remaining -= 1
        idx += 1
    return {i: slots[i] for i in range(n)}

def random_placement(replication: Dict[int, int], mesh: Mesh, max_per_device: int, placement: Optional[Placement] = None) -> Placement:
    num_devices = mesh.rows * mesh.cols
    total_capacity = num_devices * max_per_device
    total_replicas = sum(replication.values())
    used = 0
    expert_replicas: Dict[int, List[int]] = {}
    device_load = [0 for _ in range(num_devices)]
    if placement is not None:
        for e, devs in placement.expert_replicas.items():
            expert_replicas[e] = list(devs)
            for did in devs:
                device_load[did] += 1
                used += 1
    if total_replicas + used > total_capacity:
        raise SystemExit("Not enough device capacity for random placement")

    # Build a list of all replica slots to place (one entry per replica).
    replicas = []
    for expert, reps in replication.items():
        replicas.extend([expert] * reps)
    random.shuffle(replicas)

    # Build a list of available device slots (each device repeated by remaining capacity).
    device_slots = []
    for did in range(num_devices):
        remaining = max_per_device - device_load[did]
        if remaining > 0:
            device_slots.extend([did] * remaining)
    random.shuffle(device_slots)

    if len(device_slots) < len(replicas):
        raise SystemExit("Not enough device capacity for random placement")

    for expert, did in zip(replicas, device_slots):
        expert_replicas.setdefault(expert, []).append(did)

    return Placement(expert_replicas)

def build_coact_matrix(trace: List[Dict], experts: int, include_shared: bool = False,
                       shared_ids: Optional[List[int]] = None) -> List[List[int]]:
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

def row_capacity_from_shared(mesh: Mesh, max_per_device: int, placement: Optional[Placement]) -> List[int]:
    rows, cols = mesh.rows, mesh.cols
    row_capacity = [max_per_device * cols for _ in range(rows)]
    if placement is None:
        return row_capacity
    for devs in placement.expert_replicas.values():
        for did in devs:
            r = did // cols
            row_capacity[r] -= 1
    for r, cap in enumerate(row_capacity):
        if cap < 0:
            raise SystemExit(f"Row {r} over capacity after shared placement")
    return row_capacity

def parse_search_sequence(value, default_t0=1.0, default_t1=0.01):
    if not value:
        return []
    if isinstance(value, list):
        steps = []
        for item in value:
            if isinstance(item, str):
                steps.extend(parse_search_sequence(item, default_t0, default_t1))
            elif isinstance(item, dict):
                stype = item.get("type")
                if stype not in {"local", "anneal"}:
                    raise SystemExit(f"Unknown search step type: {stype}")
                iters = int(item.get("iters", 0))
                if iters <= 0:
                    raise SystemExit(f"Search step {stype} requires positive iters")
                step = {"type": stype, "iters": iters}
                if stype == "anneal":
                    step["t0"] = float(item.get("t0", default_t0))
                    step["t1"] = float(item.get("t1", default_t1))
                steps.append(step)
            else:
                raise SystemExit("search_sequence entries must be strings or objects")
        return steps
    if isinstance(value, str):
        raw = [v.strip() for v in value.split(",") if v.strip()]
        steps = []
        for part in raw:
            pieces = part.split(":")
            stype = pieces[0]
            if stype not in {"local", "anneal"}:
                raise SystemExit(f"Unknown search step type: {stype}")
            if len(pieces) < 2:
                raise SystemExit(f"Search step {stype} requires iters")
            iters = int(pieces[1])
            if iters <= 0:
                raise SystemExit(f"Search step {stype} requires positive iters")
            step = {"type": stype, "iters": iters}
            if stype == "anneal":
                t0 = float(pieces[2]) if len(pieces) > 2 else default_t0
                t1 = float(pieces[3]) if len(pieces) > 3 else default_t1
                step["t0"] = t0
                step["t1"] = t1
            steps.append(step)
        return steps
    raise SystemExit("search_sequence must be list or string")
def write_coact_csv(coact: List[List[int]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in coact:
            f.write(",".join(str(x) for x in row) + "\n")

def place_shared_replicas(
    placement: Placement,
    mesh: Mesh,
    max_per_device: int,
    shared_ids: List[int],
    replication: Dict[int, int],
) -> Placement:
    # Round-robin across rows; within a row, pick least-loaded device (tie: lowest col).
    rows, cols = mesh.rows, mesh.cols
    num_devices = rows * cols
    device_load = [0 for _ in range(num_devices)]
    for devs in placement.expert_replicas.values():
        for did in devs:
            device_load[did] += 1

    for sid in shared_ids:
        reps = replication.get(sid, 0)
        placement.expert_replicas.setdefault(sid, [])
        for i in range(reps):
            r = i % rows
            preferred_col = i % cols
            chosen = None
            # Try preferred column first, then round-robin within the same row.
            for off in range(cols):
                c = (preferred_col + off) % cols
                did = r * cols + c
                if device_load[did] < max_per_device:
                    chosen = did
                    break
            if chosen is None:
                # Fallback: find any row/col with remaining capacity.
                for rr in range(rows):
                    row_devs = [rr * cols + c for c in range(cols)]
                    candidates = [did for did in row_devs if device_load[did] < max_per_device]
                    if not candidates:
                        continue
                    chosen = min(candidates, key=lambda did: (device_load[did], did % cols))
                    break
            if chosen is None:
                raise SystemExit(f"All devices saturated: cannot place shared replicas (max_per_device={max_per_device})")
            placement.expert_replicas[sid].append(chosen)
            device_load[chosen] += 1
    return placement

# Axis-generic placement helpers (row/col share same code path)

def _axis_params(mesh: Mesh, axis: str) -> tuple:
    if axis == "row":
        axis_len = mesh.rows
        other_len = mesh.cols
        axis_index = lambda did: did // mesh.cols
        axis_devices = lambda a: [a * mesh.cols + c for c in range(mesh.cols)]
    elif axis == "col":
        axis_len = mesh.cols
        other_len = mesh.rows
        axis_index = lambda did: did % mesh.cols
        axis_devices = lambda a: [r * mesh.cols + a for r in range(mesh.rows)]
    else:
        raise SystemExit(f"Unknown axis: {axis}")
    return axis_len, other_len, axis_index, axis_devices

def axis_partition(replication: Dict[int, int], expert_counts: List[int],
                   coact: List[List[int]], mesh: Mesh, max_per_device: int,
                   axis: str, axis_capacity: Optional[List[int]] = None, top_n: int = 0) -> Dict[int, List[int]]:
    axis_len, other_len, _, _ = _axis_params(mesh, axis)
    experts = len(expert_counts)
    axis_load = [0 for _ in range(axis_len)]
    axis_slots = [0 for _ in range(axis_len)]
    if axis_capacity is None:
        axis_capacity = [max_per_device * other_len for _ in range(axis_len)]
    axis_experts: List[List[int]] = [[] for _ in range(axis_len)]

    hot = []
    if top_n and top_n > 0:
        hot = sorted(range(experts), key=lambda e: expert_counts[e], reverse=True)[:top_n]
        for e in hot:
            candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
            if not candidates:
                raise SystemExit("All axis saturated during hot-tier placement")
            a = min(candidates, key=lambda x: (axis_load[x], x))
            axis_experts[a].append(e)
            axis_load[a] += expert_counts[e]
            axis_slots[a] += 1
    order = [e for e in sorted(range(experts), key=lambda e: expert_counts[e], reverse=True) if e not in hot]
    for e in order:
        best_axis = 0
        best_score = None
        for a in range(axis_len):
            affinity = sum(coact[e][x] for x in axis_experts[a])
            score = affinity - 0.1 * axis_load[a]
            if best_score is None or score > best_score:
                best_score = score
                best_axis = a
        if axis_slots[best_axis] >= axis_capacity[best_axis]:
            candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
            if not candidates:
                raise SystemExit("All axis saturated during axis_partition")
            best_axis = min(candidates, key=lambda a: (axis_load[a], a))
        axis_experts[best_axis].append(e)
        axis_load[best_axis] += expert_counts[e]
        axis_slots[best_axis] += 1

    replica_axis: Dict[int, List[int]] = {e: [a for a in range(axis_len) if e in axis_experts[a]] for e in range(experts)}
    for e in range(experts):
        extra = replication[e] - 1
        for _ in range(extra):
            candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
            if not candidates:
                raise SystemExit("All axis saturated during replica placement")
            candidates.sort(key=lambda a: (e in axis_experts[a], axis_load[a], a))
            a = candidates[0]
            axis_experts[a].append(e)
            replica_axis[e].append(a)
            axis_load[a] += expert_counts[e]
            axis_slots[a] += 1

    return replica_axis

def axis_first_balance(replication: Dict[int, int], expert_counts: List[int],
                       mesh: Mesh, max_per_device: int, axis: str,
                       axis_capacity: Optional[List[int]] = None, top_n: int = 0) -> Dict[int, List[int]]:
    axis_len, other_len, _, _ = _axis_params(mesh, axis)
    experts = len(expert_counts)
    axis_load = [0 for _ in range(axis_len)]
    axis_slots = [0 for _ in range(axis_len)]
    if axis_capacity is None:
        axis_capacity = [max_per_device * other_len for _ in range(axis_len)]
    axis_experts: List[List[int]] = [[] for _ in range(axis_len)]

    hot = []
    if top_n and top_n > 0:
        hot = sorted(range(experts), key=lambda e: expert_counts[e], reverse=True)[:top_n]
        for e in hot:
            candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
            if not candidates:
                raise SystemExit("All axis saturated during hot-tier placement")
            a = min(candidates, key=lambda x: (axis_load[x], x))
            axis_experts[a].append(e)
            axis_load[a] += expert_counts[e]
            axis_slots[a] += 1
    order = [e for e in sorted(range(experts), key=lambda e: expert_counts[e], reverse=True) if e not in hot]
    for e in order:
        candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
        if not candidates:
            raise SystemExit("All axis saturated during balance placement")
        a = min(candidates, key=lambda x: (axis_load[x], x))
        axis_experts[a].append(e)
        axis_load[a] += expert_counts[e]
        axis_slots[a] += 1

    replica_axis: Dict[int, List[int]] = {e: [a for a in range(axis_len) if e in axis_experts[a]] for e in range(experts)}
    for e in range(experts):
        extra = replication[e] - 1
        for _ in range(extra):
            candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
            if not candidates:
                raise SystemExit("All axis saturated during replica placement")
            a = min(candidates, key=lambda x: (axis_load[x], x))
            axis_experts[a].append(e)
            replica_axis[e].append(a)
            axis_load[a] += expert_counts[e]
            axis_slots[a] += 1

    return replica_axis

def axis_hot_tier(replication: Dict[int, int], expert_counts: List[int],
                  coact: List[List[int]], mesh: Mesh, max_per_device: int,
                  axis: str, top_n: int = 32, axis_capacity: Optional[List[int]] = None) -> Dict[int, List[int]]:
    axis_len, other_len, _, _ = _axis_params(mesh, axis)
    experts = len(expert_counts)
    hot = sorted(range(experts), key=lambda e: expert_counts[e], reverse=True)[:top_n]
    cold = [e for e in range(experts) if e not in hot]

    axis_experts: List[List[int]] = [[] for _ in range(axis_len)]
    axis_load = [0 for _ in range(axis_len)]
    axis_slots = [0 for _ in range(axis_len)]
    if axis_capacity is None:
        axis_capacity = [max_per_device * other_len for _ in range(axis_len)]

    for e in hot:
        candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
        if not candidates:
            raise SystemExit("All axis saturated during hot-tier placement")
        a = min(candidates, key=lambda x: (axis_load[x], x))
        axis_experts[a].append(e)
        axis_load[a] += expert_counts[e]
        axis_slots[a] += 1

    order = sorted(cold, key=lambda e: expert_counts[e], reverse=True)
    for e in order:
        best_axis = 0
        best_score = None
        for a in range(axis_len):
            affinity = sum(coact[e][x] for x in axis_experts[a])
            score = affinity - 0.1 * axis_load[a]
            if best_score is None or score > best_score:
                best_score = score
                best_axis = a
        if axis_slots[best_axis] >= axis_capacity[best_axis]:
            candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
            if not candidates:
                raise SystemExit("All axis saturated during hot-tier placement")
            best_axis = min(candidates, key=lambda a: (axis_load[a], a))
        axis_experts[best_axis].append(e)
        axis_load[best_axis] += expert_counts[e]
        axis_slots[best_axis] += 1

    replica_axis: Dict[int, List[int]] = {e: [a for a in range(axis_len) if e in axis_experts[a]] for e in range(experts)}
    for e in range(experts):
        extra = replication[e] - 1
        for _ in range(extra):
            candidates = [a for a in range(axis_len) if axis_slots[a] < axis_capacity[a]]
            if not candidates:
                raise SystemExit("All axis saturated during hot-tier replica placement")
            candidates.sort(key=lambda a: (e in axis_experts[a], axis_load[a], a))
            a = candidates[0]
            axis_experts[a].append(e)
            replica_axis[e].append(a)
            axis_load[a] += expert_counts[e]
            axis_slots[a] += 1

    return replica_axis

def place_within_axis(
    replica_axis: Dict[int, List[int]],
    mesh: Mesh,
    max_per_device: int,
    expert_counts: List[int],
    axis: str,
    placement: Optional[Placement] = None,
) -> Placement:
    axis_len, _, axis_index, axis_devices = _axis_params(mesh, axis)
    num_devices = mesh.rows * mesh.cols
    device_load = [0.0 for _ in range(num_devices)]
    device_slots = [0 for _ in range(num_devices)]
    expert_replicas: Dict[int, List[int]] = {e: [] for e in replica_axis.keys()}
    if placement is not None:
        for e, devs in placement.expert_replicas.items():
            expert_replicas[e] = list(devs)
            for did in devs:
                device_slots[did] += 1
                if e < len(expert_counts):
                    device_load[did] += float(expert_counts[e])
    for e, axes in replica_axis.items():
        for a in axes:
            axis_devs = axis_devices(a)
            candidates = [did for did in axis_devs if device_slots[did] < max_per_device]
            if not candidates:
                axis_candidates = []
                for aa in range(axis_len):
                    axis_devs_aa = axis_devices(aa)
                    if any(device_slots[d] < max_per_device for d in axis_devs_aa):
                        a_load = sum(device_load[d] for d in axis_devs_aa)
                        axis_candidates.append((a_load, aa))
                if not axis_candidates:
                    raise SystemExit(f"All devices saturated: cannot place more than {max_per_device} replicas per device")
                _, best_axis = min(axis_candidates, key=lambda x: (x[0], x[1]))
                axis_devs = axis_devices(best_axis)
                candidates = [did for did in axis_devs if device_slots[did] < max_per_device]
            best = min(candidates, key=lambda did: (device_load[did], device_slots[did], did))
            expert_replicas[e].append(best)
            device_slots[best] += 1
            if e < len(expert_counts):
                device_load[best] += float(expert_counts[e])
    return Placement(expert_replicas)

def build_initial_placement(
    strategy: str,
    base_replication: Dict[int, int],
    base_counts: List[int],
    base_coact: List[List[int]],
    mesh: Mesh,
    max_per_device: int,
    shared_first_placement: Optional[Placement],
    row_capacity_per_row: List[int],
    col_capacity_per_col: List[int],
    hot_n: int = 32,
    placement_axis: str = "row",
    hot_tier_axis: str = "col",
) -> Placement:
    if strategy == "aware":
        replica_axis = axis_partition(
            base_replication,
            base_counts,
            base_coact,
            mesh,
            max_per_device,
            placement_axis,
            row_capacity_per_row if placement_axis == "row" else col_capacity_per_col,
            top_n=hot_n,
        )
        return place_within_axis(
            replica_axis,
            mesh,
            max_per_device,
            base_counts,
            placement_axis,
            placement=shared_first_placement,
        )
    elif strategy == "balance":
        replica_axis = axis_first_balance(
            base_replication,
            base_counts,
            mesh,
            max_per_device,
            placement_axis,
            row_capacity_per_row if placement_axis == "row" else col_capacity_per_col,
            top_n=hot_n,
        )
        return place_within_axis(
            replica_axis,
            mesh,
            max_per_device,
            base_counts,
            placement_axis,
            placement=shared_first_placement,
        )
    elif strategy == "hot-tier":
        axis = hot_tier_axis
        replica_axis = axis_hot_tier(
            base_replication,
            base_counts,
            base_coact,
            mesh,
            max_per_device,
            axis,
            top_n=hot_n,
            axis_capacity=row_capacity_per_row if axis == "row" else col_capacity_per_col,
        )
        return place_within_axis(
            replica_axis,
            mesh,
            max_per_device,
            base_counts,
            axis,
            placement=shared_first_placement,
        )
    else:
        raise SystemExit(f"Unknown placement strategy: {strategy}")
def _strip_shared_replica_rows(replica_rows: Dict[int, List[int]], shared_ids: List[int]) -> Dict[int, List[int]]:
    for sid in shared_ids:
        if sid in replica_rows:
            replica_rows.pop(sid)
    return replica_rows

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
                 include_shared: bool, objective: str, iters: int = 200,
                 show_progress: bool = False, label: str = "local_search",
                 profile_eval: bool = False,
                 shared_ids: Optional[List[int]] = None) -> Placement:
    best = placement
    eval_show = profile_eval
    best_score = evaluate(
        trace,
        best,
        mesh,
        params,
        routing_strategy,
        capacity_factor,
        include_shared,
        show_progress=eval_show,
        profile_eval=profile_eval,
        shared_ids=shared_ids,
    )[objective]
    device_map = placement_to_device_map(best)
    device_ids = list(range(mesh.rows * mesh.cols))

    start_time = time.perf_counter()
    for i in range(iters):
        if show_progress:
            _print_progress(label, i + 1, iters, start_time)
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
        score = evaluate(
            trace,
            candidate,
            mesh,
            params,
            routing_strategy,
            capacity_factor,
            include_shared,
            show_progress=eval_show,
            profile_eval=profile_eval,
            shared_ids=shared_ids,
        )[objective]
        if score < best_score:
            best_score = score
            best = candidate
        else:
            # revert swap
            device_map[d1].remove(e2)
            device_map[d2].remove(e1)
            device_map[d1].append(e1)
            device_map[d2].append(e2)

    if show_progress:
        print("")
    return best

def simulated_annealing(trace: List[Dict], placement: Placement, mesh: Mesh,
                        params: CostParams, routing_strategy: str, capacity_factor: float,
                        include_shared: bool, objective: str,
                        iters: int = 500, t0: float = 1.0, t1: float = 0.01,
                        show_progress: bool = False, label: str = "anneal",
                        profile_eval: bool = False,
                        shared_ids: Optional[List[int]] = None) -> Placement:
    current = placement
    eval_show = profile_eval
    current_score = evaluate(
        trace,
        current,
        mesh,
        params,
        routing_strategy,
        capacity_factor,
        include_shared,
        show_progress=eval_show,
        profile_eval=profile_eval,
        shared_ids=shared_ids,
    )[objective]
    best = current
    best_score = current_score

    device_map = placement_to_device_map(current)
    device_ids = list(range(mesh.rows * mesh.cols))

    start_time = time.perf_counter()
    for i in range(iters):
        if show_progress:
            _print_progress(label, i + 1, iters, start_time)
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
        score = evaluate(
            trace,
            candidate,
            mesh,
            params,
            routing_strategy,
            capacity_factor,
            include_shared,
            show_progress=eval_show,
            profile_eval=profile_eval,
            shared_ids=shared_ids,
        )[objective]
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

    if show_progress:
        print("")
    return best

def build_instance_mapping(placement: Placement) -> Dict[str, object]:
    max_expert_id = max(placement.expert_replicas.keys()) if placement.expert_replicas else -1
    expert_id_mapping: List[List[int]] = [[] for _ in range(max_expert_id + 1)]
    instance_to_device: List[int] = []
    instance_to_original: List[int] = []
    for expert_id in sorted(placement.expert_replicas.keys()):
        replicas = placement.expert_replicas[expert_id]
        for did in replicas:
            instance_id = len(instance_to_device)
            instance_to_device.append(did)
            instance_to_original.append(expert_id)
            expert_id_mapping[expert_id].append(instance_id)
    return {
        "expert_id_mapping": expert_id_mapping,
        "instance_to_device": instance_to_device,
        "instance_to_original": instance_to_original,
        "num_expert_instances": len(instance_to_device),
    }

def _add_shared_experts(topk_experts: List[List[int]], shared_ids: List[int]) -> List[List[int]]:
    if not shared_ids:
        return topk_experts
    out = []
    shared_set = set(shared_ids)
    for experts in topk_experts:
        cur = list(experts)
        for sid in shared_ids:
            if sid not in shared_set:
                continue
            if sid not in cur:
                cur.append(sid)
        out.append(cur)
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

def write_expert_frequency_csv(trace: List[Dict], placement: Placement, total_experts: int,
                               include_shared: bool, out_path: str,
                               shared_ids: Optional[List[int]] = None) -> None:
    mapping = build_instance_mapping(placement)
    instance_to_original = mapping.get("instance_to_original")
    if instance_to_original is None:
        instance_to_original = []
        for expert_id in sorted(placement.expert_replicas.keys()):
            for _ in placement.expert_replicas[expert_id]:
                instance_to_original.append(expert_id)
    layer_counts: Dict[int, List[int]] = {}
    for rec in trace:
        layer = rec.get("layer", 0)
        topk = rec.get("topk_experts", [])
        if include_shared and shared_ids:
            topk = _add_shared_experts(topk, shared_ids)
        layer_counts.setdefault(layer, [0 for _ in range(total_experts)])
        for experts in topk:
            for e in experts:
                if 0 <= e < total_experts:
                    layer_counts[layer][e] += 1
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Layer," + ",".join(f"Inst {i}" for i in range(len(instance_to_original))) + "\n")
        for layer in sorted(layer_counts.keys()):
            total = sum(layer_counts[layer]) or 1
            per_orig = [c / total for c in layer_counts[layer]]
            per_inst = [per_orig[oid] if oid < len(per_orig) else 0.0 for oid in instance_to_original]
            f.write(f"Layer {layer}," + ",".join(f"{v:.6f}" for v in per_inst) + "\n")

def evaluate(trace: List[Dict], placement: Placement, mesh: Mesh, params: CostParams,
             routing_strategy: str, capacity_factor: float, include_shared: bool,
             show_progress: bool = False, label: str = "eval",
             origin_mode: str = "average-rows", seed: int = 0,
             profile_eval: bool = False,
             shared_ids: Optional[List[int]] = None) -> Dict[str, float]:
    if routing_strategy != "balanced-replicas":
        raise AssertionError("routing_strategy must be balanced-replicas (instance ids required by cost model)")
    results = []
    mapping = build_instance_mapping(placement)
    expert_id_mapping = mapping["expert_id_mapping"]
    instance_to_device = mapping["instance_to_device"]
    num_expert_instances = mapping["num_expert_instances"]
    total = len(trace)
    rng = random.Random(seed)
    start_time = time.perf_counter()
    for idx, rec in enumerate(trace):
        topk = rec.get("topk_with_shared") if include_shared and shared_ids and "topk_with_shared" in rec else rec["topk_experts"]
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
                        per_row.append(simulate_batch_instances(origin_rows, active_experts, instance_to_device, mesh, params, mapping["instance_to_original"]))
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
                    r = simulate_batch_instances(origin_rows, active_experts, instance_to_device, mesh, params, mapping["instance_to_original"])
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
                r = simulate_batch_instances(origin_rows, active_experts, instance_to_device, mesh, params, mapping["instance_to_original"])
            else:
                r = simulate_batch(origin_rows, topk, placement, mesh, params)
        results.append(r)
        if show_progress and total > 0 and (profile_eval or idx % 10 == 0 or idx == total - 1):
            pct = int((idx + 1) / total * 100)
            elapsed = time.perf_counter() - start_time
            frac = (idx + 1) / total
            eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
            print(
                f"\r{label}: {pct:3d}%  elapsed {_format_eta(elapsed)}  eta {_format_eta(eta)}",
                end="",
                flush=True,
            )
    if show_progress:
        print("")
    agg = aggregate(results)
    # Also aggregate dispatch/compute/combine components.
    if results:
        agg["max_dispatch_mean"] = sum(r["max_dispatch"] for r in results) / len(results)
        agg["max_compute_mean"] = sum(r["max_compute"] for r in results) / len(results)
        agg["max_combine_mean"] = sum(r["max_combine"] for r in results) / len(results)
    if profile_eval:
        elapsed = time.perf_counter() - start_time
        print(
            f"\n{label} done in {_format_eta(elapsed)} for {len(trace)} records",
            file=sys.stderr,
            flush=True,
        )
    return agg

def main() -> None:
    ap = argparse.ArgumentParser(description="Search for a better MoE placement")
    ap.add_argument("--trace", default="")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--slots", type=int, default=384)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--placement-strategy", choices=["balance", "aware", "hot-tier"], default="balance")
    ap.add_argument("--search", choices=["random", "none", "local", "anneal", "hybrid"], default="none")
    ap.add_argument("--search-config", default="", help="JSON file with search-specific parameters")
    ap.add_argument("--search-sequence", default="", help="Comma-separated search steps, e.g. local:200,anneal:500:1.0:0.01,local:10")
    ap.add_argument("--system-config", default="", help="JSON file with system/dataset parameters")
    ap.add_argument("--initial-placement-config", default="", help="JSON file with initial placement parameters")
    ap.add_argument("--local-search-iters", type=int, default=200)
    ap.add_argument("--local-search-final-iters", type=int, default=10, help="Final local search iters")
    ap.add_argument("--anneal-iters", type=int, default=500)
    ap.add_argument("--anneal-t0", type=float, default=1.0)
    ap.add_argument("--anneal-t1", type=float, default=0.01)
    ap.add_argument("--restarts", type=int, default=1, help="Number of randomized restarts per search")
    ap.add_argument("--fast-test-pct", type=float, default=0.0, help="Subsample tokens per record (0.01 = 1%)")
    ap.add_argument("--max-seq-len", type=int, default=0, help="Cap tokens per record (0 = no cap)")
    ap.add_argument("--profile-eval", action="store_true", help="Print timing for each evaluate call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-records", type=int, default=0, help="Limit number of trace records (0 = no limit)")
    ap.add_argument("--routing-strategy", choices=["least-loaded", "balanced-replicas"], default="least-loaded")
    ap.add_argument("--capacity-factor", type=float, default=2.0)
    ap.add_argument("--num-shared-experts", type=int, default=1, help="Number of shared experts to append")
    ap.add_argument(
        "--shared-expert-round-robin-in-rows",
        action="store_true",
        help="Place shared expert replicas round-robin across rows (initial placement)",
    )
    ap.add_argument("--objective", choices=["mean", "p95", "p99"], default="mean")
    ap.add_argument("--origin-mode", choices=["average-rows", "round-robin", "random"], default="average-rows")
    ap.add_argument("--save-placement", default="", help="Folder to store placement artifacts")
    ap.add_argument("--plot-only", action="store_true", help="Only render plots from existing placement.json")
    args = ap.parse_args()

    # Apply system config overrides.
    sys_cfg = {}
    if args.system_config:
        sys_cfg = _load_jsonc(args.system_config)
        allowed = {
            "trace",
            "layer",
            "rows",
            "cols",
            "experts",
            "slots",
            "num_shared_experts",
            "routing_strategy",
            "capacity_factor",
            "objective",
            "origin_mode",
            "max_records",
            "fast_test_pct",
            "max_seq_len",
        }
        unknown = [k for k in sys_cfg.keys() if k not in allowed]
        if unknown:
            raise SystemExit(f"--system-config has unknown keys: {', '.join(unknown)}")
        for k, v in sys_cfg.items():
            setattr(args, k, v)
    init_cfg = {}
    if args.initial_placement_config:
        init_cfg = _load_jsonc(args.initial_placement_config)
        allowed = {"shared_expert_round_robin_in_rows", "placement_strategy", "placement_axis", "top_n", "hot_tier_axis"}
        unknown = [k for k in init_cfg.keys() if k not in allowed]
        if unknown:
            raise SystemExit(f"--initial-placement-config has unknown keys: {', '.join(unknown)}")
        for k, v in init_cfg.items():
            setattr(args, k, v)
    if not args.trace:
        raise SystemExit("--trace or --system-config with trace is required")
    search_config = {}
    required_by_search = {
        "random": {"search", "iters", "restarts"},
        "none": {"search", "restarts"},
        "local": {"search", "local_search_iters", "restarts"},
        "anneal": {"search", "anneal_iters", "anneal_t0", "anneal_t1", "restarts"},
        "sequence": {"search", "search_sequence", "restarts"},
    }
    if args.search_config:
        search_config = _load_jsonc(args.search_config)
        if "search" not in search_config:
            raise SystemExit("--search-config must include a 'search' field")
        search_name = search_config["search"]
        if "search_sequence" in search_config and search_name != "sequence":
            raise SystemExit("--search-config with search_sequence must set search=sequence")
        if search_name not in required_by_search:
            raise SystemExit(f"--search-config has unknown search type: {search_name}")
        required = required_by_search[search_name]
        missing = [k for k in required if k not in search_config]
        if missing:
            raise SystemExit(f"--search-config missing required keys for {search_name}: {', '.join(missing)}")
        # Apply config values
        for k, v in search_config.items():
            setattr(args, k, v)

    if args.search == "random" and args.seed == 0:
        args.seed = random.SystemRandom().randint(1, 2**31 - 1)
    random.seed(args.seed)

    if args.search_sequence:
        args.search = "sequence"

    search_sequence_steps = parse_search_sequence(args.search_sequence) if args.search_sequence else []
    if args.search == "sequence":
        if not search_sequence_steps:
            # maybe provided in config as list
            if isinstance(getattr(args, "search_sequence", None), list):
                search_sequence_steps = parse_search_sequence(args.search_sequence)
        if not search_sequence_steps:
            raise SystemExit("search=sequence requires search_sequence")

    # Print grouped configuration (including defaults)
    print("search config:", file=sys.stderr)
    sys_keys = [
        "trace", "layer", "rows", "cols", "experts", "num_shared_experts", "slots",
        "routing_strategy", "capacity_factor", "objective", "origin_mode",
        "max_records", "fast_test_pct", "max_seq_len",
    ]
    init_keys = ["shared_expert_round_robin_in_rows", "placement_strategy", "placement_axis", "top_n", "hot_tier_axis"]
    search_keys = [
        "search", "search_sequence", "iters", "local_search_iters", "local_search_final_iters",
        "anneal_iters", "anneal_t0", "anneal_t1", "restarts",
    ]
    misc_keys = [
        "seed", "plot_only", "profile_eval", "save_placement",
        "system_config", "initial_placement_config", "search_config",
    ]
    def _print_group(title: str, keys: List[str]) -> None:
        print(f"\n{title}:", file=sys.stderr)
        for k in keys:
            if hasattr(args, k):
                print(f"  {k}: {getattr(args, k)}", file=sys.stderr)
    _print_group("system", sys_keys)
    _print_group("initial_placement", init_keys)
    # Only show search parameters relevant to the selected search type.
    active_search_keys = list(required_by_search.get(args.search, {"search"}))
    # Keep a stable, readable order.
    ordered = [k for k in search_keys if k in active_search_keys]
    _print_group("search", ordered)
    _print_group("misc", misc_keys)

    # Plot-only mode: validate config and render plots without running search.
    if args.plot_only:
        if not args.save_placement:
            raise SystemExit("--plot-only requires --save-placement pointing to the run folder")
        run_cfg_path = os.path.join(args.save_placement, "run_config.json")
        placement_path = os.path.join(args.save_placement, "placement.json")
        if not os.path.exists(run_cfg_path) or not os.path.exists(placement_path):
            raise SystemExit("run_config.json or placement.json missing in save-placement folder")
        with open(run_cfg_path, "r", encoding="utf-8") as f:
            run_cfg = json.load(f)
        with open(placement_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        placement_data = Placement(expert_replicas={int(k): v for k, v in raw.items()})
        # Compare essential config fields
        expected = {
            "trace": args.trace,
            "layer": args.layer,
            "rows": args.rows,
            "cols": args.cols,
            "experts": args.experts,
            "slots": args.slots,
            "search": args.search,
            "placement_strategy": args.placement_strategy,
            "search_sequence": args.search_sequence,
            "routing_strategy": args.routing_strategy,
            "capacity_factor": args.capacity_factor,
            "objective": args.objective,
            "origin_mode": args.origin_mode,
            "num_shared_experts": args.num_shared_experts,
            "shared_expert_round_robin_in_rows": args.shared_expert_round_robin_in_rows,
        }
        for k, v in expected.items():
            if k == "shared_expert_round_robin_in_rows" and k not in run_cfg and "shared_row_replication" in run_cfg:
                legacy = not bool(run_cfg.get("shared_row_replication"))
                if str(legacy) != str(v):
                    raise SystemExit(
                        f"run_config mismatch for {k}: expected {v}, found legacy shared_row_replication={run_cfg.get('shared_row_replication')}"
                    )
                continue
            if str(run_cfg.get(k)) != str(v):
                raise SystemExit(f"run_config mismatch for {k}: expected {v}, found {run_cfg.get(k)}")

        trace_path = args.trace
        full_plot = os.path.join(args.save_placement, "placement_instances.png")
        baseline_plot = os.path.join(args.save_placement, "placement_baseline_originals.png")
        plot_cmd = [
            sys.executable, "plots/plot_placement_grid.py",
            "--placement", placement_path,
            "--trace", trace_path,
            "--out", full_plot,
            "--rows", str(args.rows),
            "--cols", str(args.cols),
            "--experts", str(args.experts),
            "--freq-folder", args.save_placement,
            "--layer", str(args.layer),
        ]
        plot_cmd_base = plot_cmd + ["--baseline-original", "--only-original", "--out", baseline_plot]
        # Ensure frequency CSV exists for plot-only runs.
        freq_csv = os.path.join(args.save_placement, "expert_frequency.csv")
        if not os.path.exists(freq_csv):
            trace = load_trace(
                args.trace,
                args.layer,
                max_records=args.max_records,
                show_progress=True,
                fast_test_pct=args.fast_test_pct,
                seed=args.seed,
                max_seq_len=args.max_seq_len,
            )
            include_shared = args.num_shared_experts > 0
            total_experts = args.experts + max(0, args.num_shared_experts)
            shared_ids = list(range(args.experts, args.experts + max(0, args.num_shared_experts)))
            write_expert_frequency_csv(trace, placement_data, total_experts, include_shared, freq_csv, shared_ids=shared_ids)
        subprocess.run(plot_cmd, check=False)
        subprocess.run(plot_cmd_base, check=False)
        return

    mesh = Mesh(args.rows, args.cols)
    trace = load_trace(
        args.trace,
        args.layer,
        max_records=args.max_records,
        show_progress=True,
        fast_test_pct=args.fast_test_pct,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
    )

    # Count expert usage for replication
    include_shared = args.num_shared_experts > 0
    total_experts = args.experts + max(0, args.num_shared_experts)
    shared_ids = list(range(args.experts, args.experts + max(0, args.num_shared_experts)))
    if include_shared and shared_ids:
        for rec in trace:
            rec["topk_with_shared"] = _add_shared_experts(rec["topk_experts"], shared_ids)
    counts = [0 for _ in range(total_experts)]
    for rec in trace:
        topk = rec.get("topk_with_shared") if include_shared and shared_ids and "topk_with_shared" in rec else rec["topk_experts"]
        for experts in topk:
            for e in experts:
                counts[e] += 1

    replication = initial_replication(counts, args.slots, shared_ids=shared_ids if include_shared else None, shared_replicas=16)
    # Compute:comm ratio = 1:3 (comm is 3x slower than compute for same token count).
    coeffs = calibrate_dispatch_coeffs(mesh.rows)
    params = CostParams(
        device_capacity=1000.0,
        row_bandwidth=333.3333333333,
        dispatch_coeffs=coeffs,
        combine_coeffs=coeffs,
    )

    best = None
    best_place = None

    coact = build_coact_matrix(trace, total_experts, include_shared=include_shared, shared_ids=shared_ids)

    shared_row_replication_enabled = include_shared and args.shared_expert_round_robin_in_rows
    base_counts = counts
    base_coact = coact
    base_replication = replication
    if shared_row_replication_enabled and shared_ids:
        base_counts = counts[:args.experts]
        base_coact = [row[:args.experts] for row in coact[:args.experts]]
        base_replication = {k: v for k, v in replication.items() if k < args.experts}

    max_per_device = int(math.ceil(args.slots / (args.rows * args.cols)))

    shared_first_placement: Optional[Placement] = None
    if shared_row_replication_enabled and shared_ids:
        shared_first_placement = Placement(expert_replicas={})
        shared_first_placement = place_shared_replicas(
            shared_first_placement,
            mesh,
            max_per_device,
            shared_ids,
            replication,
        )
    row_capacity_per_row = row_capacity_from_shared(mesh, max_per_device, shared_first_placement)
    col_capacity_per_col = col_capacity_from_shared(mesh, max_per_device, shared_first_placement)

    def _evaluate_and_print(label: str, placement: Placement, seed: int) -> Tuple[Dict[str, float], Placement]:
        score = evaluate(
            trace,
            placement,
            mesh,
            params,
            args.routing_strategy,
            args.capacity_factor,
            include_shared,
            show_progress=False,
            label="eval",
            origin_mode=args.origin_mode,
            seed=seed,
            profile_eval=args.profile_eval,
            shared_ids=shared_ids,
        )
        print(
            f"{label}[{args.objective}]: mean={score['mean']:.4f} p95={score['p95']:.4f} p99={score['p99']:.4f} "
            f"dispatch={score.get('max_dispatch_mean', 0.0):.4f} "
            f"compute={score.get('max_compute_mean', 0.0):.4f} "
            f"combine={score.get('max_combine_mean', 0.0):.4f}"
        )
        return score, placement

    for r_idx in range(max(1, args.restarts)):
        restart_start = time.perf_counter()
        if args.restarts > 1:
            print(f"restart {r_idx + 1}/{args.restarts}")
        random.seed(args.seed + r_idx)

        if args.search != "random":
            placement = build_initial_placement(
                args.placement_strategy,
                base_replication,
                base_counts,
                base_coact,
                mesh,
                max_per_device,
                shared_first_placement,
                row_capacity_per_row,
                col_capacity_per_col,
                hot_n=getattr(args, "top_n", 32),
                placement_axis=getattr(args, "placement_axis", "row"),
                hot_tier_axis=getattr(args, "hot_tier_axis", "col"),
            )

        if args.search == "none":
            score, placement = _evaluate_and_print("placement-only", placement, args.seed + r_idx)
        elif args.search == "local":
            placement = local_search(
                trace, placement, mesh, params,
                routing_strategy=args.routing_strategy,
                capacity_factor=args.capacity_factor,
                include_shared=include_shared,
                objective=args.objective,
                iters=args.local_search_iters,
                show_progress=True,
                label="local_search",
                profile_eval=args.profile_eval,
                shared_ids=shared_ids,
            )
            score, placement = _evaluate_and_print("local", placement, args.seed + r_idx)
        elif args.search == "anneal":
            placement = simulated_annealing(
                trace, placement, mesh, params,
                routing_strategy=args.routing_strategy,
                capacity_factor=args.capacity_factor,
                include_shared=include_shared,
                objective=args.objective,
                iters=args.anneal_iters, t0=args.anneal_t0, t1=args.anneal_t1,
                show_progress=True,
                label="anneal",
                profile_eval=args.profile_eval,
                shared_ids=shared_ids,
            )
            score, placement = _evaluate_and_print("anneal", placement, args.seed + r_idx)
        # hybrid removed; use search=sequence
        
        elif args.search == "sequence":
            for step in search_sequence_steps:
                if step["type"] == "local":
                    placement = local_search(
                        trace, placement, mesh, params,
                        routing_strategy=args.routing_strategy,
                        capacity_factor=args.capacity_factor,
                        include_shared=include_shared,
                        objective=args.objective,
                        iters=step["iters"],
                        show_progress=True,
                        label="local_search",
                        profile_eval=args.profile_eval,
                        shared_ids=shared_ids,
                    )
                elif step["type"] == "anneal":
                    placement = simulated_annealing(
                        trace, placement, mesh, params,
                        routing_strategy=args.routing_strategy,
                        capacity_factor=args.capacity_factor,
                        include_shared=include_shared,
                        objective=args.objective,
                        iters=step["iters"], t0=step["t0"], t1=step["t1"],
                        show_progress=True,
                        label="anneal",
                        profile_eval=args.profile_eval,
                        shared_ids=shared_ids,
                    )
            score, placement = _evaluate_and_print("sequence", placement, args.seed + r_idx)
        else:
            score = None
            placement = None
            start_time = time.time()
            for i in range(args.iters):
                if shared_row_replication_enabled:
                    base_rep = base_replication
                    cand = random_placement(base_rep, mesh, max_per_device, placement=shared_first_placement)
                else:
                    cand = random_placement(replication, mesh, max_per_device)
                cand_score = evaluate(
                    trace,
                    cand,
                    mesh,
                    params,
                    args.routing_strategy,
                    args.capacity_factor,
                    include_shared,
                    show_progress=True,
                    label=f"eval {i}",
                    origin_mode=args.origin_mode,
                    seed=args.seed + r_idx,
                    profile_eval=args.profile_eval,
                    shared_ids=shared_ids,
                )
                if score is None or cand_score[args.objective] < score[args.objective]:
                    score = cand_score
                    placement = cand
                if args.iters > 0:
                    frac = (i + 1) / args.iters
                    elapsed = time.perf_counter() - start_time
                    eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
                    print(
                        f"\rrandom_search: {int(frac*100):3d}%  elapsed {_format_eta(elapsed)}  eta {_format_eta(eta)}",
                        end="",
                        flush=True,
                    )
                print(
                    f"iter {i}[{args.objective}]: mean={cand_score['mean']:.4f} p95={cand_score['p95']:.4f} p99={cand_score['p99']:.4f} "
                    f"dispatch={cand_score.get('max_dispatch_mean', 0.0):.4f} "
                    f"compute={cand_score.get('max_compute_mean', 0.0):.4f} "
                    f"combine={cand_score.get('max_combine_mean', 0.0):.4f}"
                )
            print("")
            # re-print best for random
            if score is not None and placement is not None:
                print(
                    f"random-best[{args.objective}]: mean={score['mean']:.4f} p95={score['p95']:.4f} p99={score['p99']:.4f} "
                    f"dispatch={score.get('max_dispatch_mean', 0.0):.4f} "
                    f"compute={score.get('max_compute_mean', 0.0):.4f} "
                    f"combine={score.get('max_combine_mean', 0.0):.4f}"
                )

        if score is not None and placement is not None:
            if best is None or score[args.objective] < best[args.objective]:
                best = score
                best_place = placement
        restart_elapsed = time.perf_counter() - restart_start
        print(f"restart {r_idx + 1} done in {_format_eta(restart_elapsed)}")

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
            out_dir = os.path.join("results", base, f"layer{args.layer}", args.search, ts)

        os.makedirs(out_dir, exist_ok=True)

        # Save placement
        placement_path = os.path.join(out_dir, "placement.json")
        with open(placement_path, "w", encoding="utf-8") as f:
            json.dump(best_place.expert_replicas, f, indent=2)

        # Save eval summary
        if best is not None:
            eval_path = os.path.join(out_dir, "eval.md")
            with open(eval_path, "w", encoding="utf-8") as f:
                f.write("# Search Evaluation Summary\n\n")
                f.write(f"- objective: {args.objective}\n")
                f.write(f"- mean: {best.get('mean', 0.0):.4f}\n")
                f.write(f"- p95: {best.get('p95', 0.0):.4f}\n")
                f.write(f"- p99: {best.get('p99', 0.0):.4f}\n")
                f.write(f"- dispatch_mean: {best.get('max_dispatch_mean', 0.0):.4f}\n")
                f.write(f"- compute_mean: {best.get('max_compute_mean', 0.0):.4f}\n")
                f.write(f"- combine_mean: {best.get('max_combine_mean', 0.0):.4f}\n")

        # Save expert frequency table (layer x expert instance)
        freq_csv = os.path.join(out_dir, "expert_frequency.csv")
        write_expert_frequency_csv(trace, best_place, total_experts, include_shared, freq_csv, shared_ids=shared_ids)

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
            "placement_strategy": args.placement_strategy,
            "search_sequence": args.search_sequence,
            "routing_strategy": args.routing_strategy,
            "capacity_factor": args.capacity_factor,
            "objective": args.objective,
            "origin_mode": args.origin_mode,
            "num_shared_experts": args.num_shared_experts,
            "shared_expert_round_robin_in_rows": args.shared_expert_round_robin_in_rows,
            "search_config": args.search_config,
            "search_config_content": search_config,
            "system_config": args.system_config,
            "system_config_content": sys_cfg,
            "initial_placement_config": args.initial_placement_config,
            "initial_placement_config_content": init_cfg,
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
            "--freq-folder", args.save_placement,
            "--layer", str(args.layer),
        ]
        plot_cmd_base = plot_cmd + ["--baseline-original", "--only-original", "--out", baseline_plot]
        subprocess.run(plot_cmd, check=False)
        subprocess.run(plot_cmd_base, check=False)

if __name__ == "__main__":
    main()
