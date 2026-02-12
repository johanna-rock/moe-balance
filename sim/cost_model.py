#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, List, Tuple
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
    device_capacity: float  # tokens per unit time
    row_bandwidth: float    # tokens per unit time
    # Focused model: only dispatch/compute/combine are modeled.


def simulate_batch(origin_rows: List[int], topk_experts: List[List[int]],
                   placement: Placement, mesh: Mesh, params: CostParams) -> Dict[str, float]:
    devices = mesh.devices()
    dev_by_id = {d.device_id: d for d in devices}

    device_load = [0 for _ in devices]
    row_outgoing = [0 for _ in range(mesh.rows)]

    # Routing policy: send to least-loaded replica (ties by lower device id)
    for token_idx, experts in enumerate(topk_experts):
        origin_row = origin_rows[token_idx]
        for e in experts:
            replicas = placement.expert_replicas.get(e, [])
            if not replicas:
                # unplaced expert: skip (counts as miss)
                continue
            # pick least loaded replica
            best = min(replicas, key=lambda did: (device_load[did], did))
            device_load[best] += 1
            if dev_by_id[best].row != origin_row:
                row_outgoing[origin_row] += abs(dev_by_id[best].row - origin_row)

    # Focused model: dispatch + expert compute + combine.
    # Dispatch/combine traffic both scale with row-crossing volume.
    compute_times = [load / params.device_capacity for load in device_load]
    dispatch_times = [traffic / params.row_bandwidth for traffic in row_outgoing]
    combine_times = [traffic / params.row_bandwidth for traffic in row_outgoing]

    max_compute = max(compute_times) if compute_times else 0.0
    max_dispatch = max(dispatch_times) if dispatch_times else 0.0
    max_combine = max(combine_times) if combine_times else 0.0

    return {
        "max_compute": max_compute,
        "max_dispatch": max_dispatch,
        "max_combine": max_combine,
        "latency": max_dispatch + max_compute + max_combine,
    }


def simulate_batch_instances(origin_rows: List[int], active_experts: List[List[int]],
                             instance_to_device: List[int], mesh: Mesh,
                             params: CostParams) -> Dict[str, float]:
    devices = mesh.devices()
    dev_by_id = {d.device_id: d for d in devices}

    device_load = [0 for _ in devices]
    row_outgoing = [0 for _ in range(mesh.rows)]

    for token_idx, instances in enumerate(active_experts):
        origin_row = origin_rows[token_idx]
        for inst in instances:
            if inst < 0:
                continue
            if inst >= len(instance_to_device):
                continue
            did = instance_to_device[inst]
            device_load[did] += 1
            if dev_by_id[did].row != origin_row:
                row_outgoing[origin_row] += abs(dev_by_id[did].row - origin_row)

    compute_times = [load / params.device_capacity for load in device_load]
    dispatch_times = [traffic / params.row_bandwidth for traffic in row_outgoing]
    combine_times = [traffic / params.row_bandwidth for traffic in row_outgoing]

    max_compute = max(compute_times) if compute_times else 0.0
    max_dispatch = max(dispatch_times) if dispatch_times else 0.0
    max_combine = max(combine_times) if combine_times else 0.0

    return {
        "max_compute": max_compute,
        "max_dispatch": max_dispatch,
        "max_combine": max_combine,
        "latency": max_dispatch + max_compute + max_combine,
    }


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
