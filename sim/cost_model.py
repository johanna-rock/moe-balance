#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math


@dataclass
class Device:
    device_id: int
    row: int
    col: int


@dataclass
class Placement:
    # expert_id -> list of device_ids (replicas)
    expert_replicas: Dict[int, List[int]]


@dataclass
class Mesh:
    rows: int
    cols: int

    def devices(self) -> List[Device]:
        devs = []
        did = 0
        for r in range(self.rows):
            for c in range(self.cols):
                devs.append(Device(did, r, c))
                did += 1
        return devs


@dataclass
class CostParams:
    # Focused model: only dispatch/compute/combine are modeled.
    # device_capacity / row_bandwidth are not used in the calibrated model but kept for compatibility.
    device_capacity: float  # tokens per unit time (legacy)
    row_bandwidth: float    # tokens per unit time (legacy)
    dispatch_coeffs: Tuple[float, float, float, float]  # A,B,C,D
    combine_coeffs: Tuple[float, float, float, float]   # A,B,C,D
    compute_base_us: float = 118.0
    compute_step_us: float = 118.0
    compute_block: int = 32


def _dispatch_time(e_col: int, avg_hop: float, max_hop: float, coeffs: Tuple[float, float, float, float]) -> float:
    a, b, c, d = coeffs
    return a + b * e_col + c * avg_hop + d * max_hop


def _compute_time_for_device(expert_token_counts: Dict[int, int],
                             compute_base_us: float,
                             compute_step_us: float,
                             compute_block: int) -> float:
    # Sum over experts: 118us per 1-32 token block (ceil)
    total = 0.0
    for t_e in expert_token_counts.values():
        if t_e <= 0:
            continue
        blocks = math.ceil(t_e / float(compute_block))
        total += compute_base_us * blocks
    return total


def _collect_metrics(origin_rows: List[int], routed: List[List[Tuple[int, int]]], mesh: Mesh,
                     coeffs_dispatch: Tuple[float, float, float, float],
                     coeffs_combine: Tuple[float, float, float, float],
                     compute_base_us: float,
                     compute_step_us: float,
                     compute_block: int) -> Dict[str, float]:
    # routed: per token list of (orig_expert_id, device_id)
    tokens = len(routed)
    if tokens == 0:
        return {"max_dispatch": 0.0, "max_combine": 0.0, "max_compute": 0.0, "latency": 0.0}

    # per-device per-expert counts
    device_expert_counts: Dict[int, Dict[int, int]] = {}
    dispatch_times = []
    combine_times = []

    for t_idx, dests in enumerate(routed):
        if not dests:
            dispatch_times.append(0.0)
            combine_times.append(0.0)
            continue
        origin_row = origin_rows[t_idx]
        # hops and column stats
        hops = []
        col_to_experts: Dict[int, set] = {}
        devices = set()
        for item in dests:
            if len(item) != 3:
                raise AssertionError("routed entries must include (orig_id, device_id, inst_id)")
            expert_id, did, inst_id = item
            if inst_id is None:
                raise AssertionError("instance id required for cost model")
            row = did // mesh.cols
            col = did % mesh.cols
            linear = abs(row - origin_row)
            hop = min(linear, mesh.rows - linear)
            hops.append(hop)
            col_to_experts.setdefault(col, set()).add(inst_id)
            devices.add(did)
            device_expert_counts.setdefault(did, {})
            key = inst_id if inst_id is not None else expert_id
            device_expert_counts[did][key] = device_expert_counts[did].get(key, 0) + 1

        e_col = max((len(v) for v in col_to_experts.values()), default=0)
        avg_hop = sum(hops) / len(hops) if hops else 0.0
        max_hop = max(hops) if hops else 0.0
        u = len(devices)

        dispatch_times.append(_dispatch_time(e_col, avg_hop, max_hop, coeffs_dispatch))
        combine_times.append(_dispatch_time(u, avg_hop, max_hop, coeffs_combine))

    # compute per-device cost
    compute_times = []
    for did, ec in device_expert_counts.items():
        compute_times.append(_compute_time_for_device(ec, compute_base_us, compute_step_us, compute_block))

    max_dispatch = max(dispatch_times) if dispatch_times else 0.0
    max_combine = max(combine_times) if combine_times else 0.0
    max_compute = max(compute_times) if compute_times else 0.0

    return {
        "max_dispatch": max_dispatch,
        "max_combine": max_combine,
        "max_compute": max_compute,
        "latency": max_dispatch + max_compute + max_combine,
    }


def simulate_batch(origin_rows: List[int], topk_experts: List[List[int]],
                   placement: Placement, mesh: Mesh, params: CostParams) -> Dict[str, float]:
    raise AssertionError("simulate_batch() without instance ids is not supported in this cost model")
    devices = mesh.devices()
    dev_by_id = {d.device_id: d for d in devices}

    device_load = [0 for _ in devices]

    routed: List[List[Tuple[int, int]]] = []
    # Routing policy: send to least-loaded replica (ties by lower device id)
    for token_idx, experts in enumerate(topk_experts):
        token_dests: List[Tuple[int, int]] = []
        for e in experts:
            replicas = placement.expert_replicas.get(e, [])
            if not replicas:
                continue
            best = min(replicas, key=lambda did: (device_load[did], did))
            device_load[best] += 1
            token_dests.append((e, best))
        routed.append(token_dests)

    return _collect_metrics(
        origin_rows,
        routed,
        mesh,
        params.dispatch_coeffs,
        params.combine_coeffs,
        params.compute_base_us,
        params.compute_step_us,
        params.compute_block,
    )


def simulate_batch_instances(origin_rows: List[int], active_experts: List[List[int]],
                             instance_to_device: List[int], mesh: Mesh,
                             params: CostParams, instance_to_original: List[int]) -> Dict[str, float]:
    routed: List[List[Tuple[int, int, int]]] = []
    for token_idx, instances in enumerate(active_experts):
        token_dests: List[Tuple[int, int, int]] = []
        for inst in instances:
            if inst < 0 or inst >= len(instance_to_device):
                continue
            did = instance_to_device[inst]
            oid = instance_to_original[inst] if inst < len(instance_to_original) else -1
            token_dests.append((oid, did, inst))
        routed.append(token_dests)

    return _collect_metrics(
        origin_rows,
        routed,
        mesh,
        params.dispatch_coeffs,
        params.combine_coeffs,
        params.compute_base_us,
        params.compute_step_us,
        params.compute_block,
    )


def calibrate_dispatch_coeffs(
    rows: int,
    best_hop: float = 0.0,
    avg_hop: Optional[float] = None,
    worst_hop: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    # Fit coefficients to the provided table using assumed hop stats.
    # Best/avg/worst hop assumptions:
    #   best: avg=best_hop, max=best_hop
    #   avg:  avg=avg_hop, max=avg_hop
    #   worst: avg=worst_hop, max=worst_hop
    table = {
        "best":  [26.868, 96.183, 33.005, 36.073, 39.141, 42.209, 45.277, 48.345],
        "avg":   [55.258, 66.713, 78.169, 89.624, 101.079, 112.535, 123.990, 135.445],
        "worst": [88.238, 96.183, 104.129, 112.074, 120.020, 127.965, 135.911, 143.856],
    }
    if avg_hop is None:
        avg_hop = (rows - 1) / 2.0 if rows > 1 else 0.0
    if worst_hop is None:
        worst_hop = rows / 2.0 if rows > 1 else 0.0
    data = []
    for i, e in enumerate(range(1, 9)):
        data.append((e, best_hop, best_hop, table["best"][i]))
        data.append((e, avg_hop, avg_hop, table["avg"][i]))
        data.append((e, worst_hop, worst_hop, table["worst"][i]))

    # Least squares for A + B*E + C*avg + D*max
    import numpy as np
    X = []
    y = []
    for e, avg_h, max_h, t in data:
        X.append([1.0, e, avg_h, max_h])
        y.append(t)
    X = np.asarray(X)
    y = np.asarray(y)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3]))


def aggregate(results: List[Dict[str, float]]) -> Dict[str, float]:
    if not results:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0}
    lat = sorted(r["latency"] for r in results)
    n = len(lat)
    def pct(p: float) -> float:
        idx = min(n - 1, int(math.ceil(p * n)) - 1)
        return lat[idx]
    return {
        "mean": sum(lat) / n,
        "p95": pct(0.95),
        "p99": pct(0.99),
    }
