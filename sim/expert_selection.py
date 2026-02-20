#!/usr/bin/env python3
from typing import List, Tuple


def balanced_expert_selection_with_rerouting(
    scores_with_bias: List[List[float]],
    expert_id_mapping: List[List[int]],
    capacity_factor: float,
    k: int,
    num_expert_instances: int,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Balanced expert selection with capacity and replica-balance_load_coactivation rerouting.

    scores_with_bias: [BATCH][NUM_GATED_EXPERTS]
    expert_id_mapping: [NUM_ORIGINAL_EXPERTS][MAX_NUM_REPLICAS] -> expert instance ids
    capacity_factor: scalar
    k: number of experts per token
    num_expert_instances: total expert instances (e.g., 384)

    Returns:
      active_experts: [BATCH][K] expert instance ids
      active_weights: [BATCH][K] routing weights aligned to active_experts
    """
    batch = len(scores_with_bias)
    num_gated = len(scores_with_bias[0]) if batch > 0 else 0

    # Pre-sort experts by score per token (descending).
    sorted_indices = []
    sorted_scores = []
    for t in range(batch):
        pairs = list(enumerate(scores_with_bias[t]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_indices.append([p[0] for p in pairs])
        sorted_scores.append([p[1] for p in pairs])

    capacity = int(capacity_factor * (batch * k / max(1, num_expert_instances)))
    if capacity < 1:
        capacity = 1

    token_count_per_instance = [0 for _ in range(num_expert_instances)]
    active_experts = [[-1 for _ in range(k)] for _ in range(batch)]
    active_weights = [[0.0 for _ in range(k)] for _ in range(batch)]
    current_rank_idx = [0 for _ in range(batch)]

    for token_rank in range(k):
        for token_id in range(batch):
            found = False
            for expert_rank in range(current_rank_idx[token_id], num_gated):
                next_expert = sorted_indices[token_id][expert_rank]
                # Try replicas for this original expert
                for rep_id in expert_id_mapping[next_expert]:
                    if rep_id < 0:
                        continue
                    if token_count_per_instance[rep_id] < capacity:
                        token_count_per_instance[rep_id] += 1
                        active_experts[token_id][token_rank] = rep_id
                        active_weights[token_id][token_rank] = sorted_scores[token_id][expert_rank]
                        current_rank_idx[token_id] = expert_rank
                        found = True
                        break
                if found:
                    break
            if not found:
                # Leave as -1 / 0.0 if no capacity available
                continue

    return active_experts, active_weights


def balanced_expert_selection_replicas(
    topk_experts: List[List[int]],
    expert_id_mapping: List[List[int]],
    capacity_factor: float,
    num_expert_instances: int,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Balanced expert selection with capacity and replica-balance_load_coactivation routing.
    Uses top-k experts (no rerouting across original experts). If all replicas
    of an expert are full, it round-robins across replicas (no dropping).

    topk_experts: [BATCH][K] original expert ids
    expert_id_mapping: [NUM_ORIGINAL_EXPERTS][MAX_NUM_REPLICAS] -> expert instance ids
    capacity_factor: scalar
    num_expert_instances: total expert instances (e.g., 384)

    Returns:
      active_experts: [BATCH][K] expert instance ids
      active_weights: [BATCH][K] routing weights (uniform = 1.0 for selected)
    """
    batch = len(topk_experts)
    k = len(topk_experts[0]) if batch > 0 else 0

    capacity = int(capacity_factor * (batch * k / max(1, num_expert_instances)))
    if capacity < 1:
        capacity = 1

    # Fast path: if per-instance capacity can hold the entire batch, routing always
    # chooses the first replica (capacity never fills).
    if capacity >= batch:
        active_experts = [[-1 for _ in range(k)] for _ in range(batch)]
        active_weights = [[0.0 for _ in range(k)] for _ in range(batch)]
        for token_id in range(batch):
            for token_rank in range(k):
                expert_id = topk_experts[token_id][token_rank]
                if expert_id < 0 or expert_id >= len(expert_id_mapping):
                    raise AssertionError(f"expert_id out of range: {expert_id} (mapping size {len(expert_id_mapping)})")
                replicas = expert_id_mapping[expert_id]
                chosen = replicas[0] if replicas else -1
                if chosen >= 0:
                    active_experts[token_id][token_rank] = chosen
                    active_weights[token_id][token_rank] = 1.0
        return active_experts, active_weights

    token_count_per_instance = [0 for _ in range(num_expert_instances)]
    rr_index = [0 for _ in range(len(expert_id_mapping))]
    active_experts = [[-1 for _ in range(k)] for _ in range(batch)]
    active_weights = [[0.0 for _ in range(k)] for _ in range(batch)]

    for token_id in range(batch):
        for token_rank in range(k):
            expert_id = topk_experts[token_id][token_rank]
            if expert_id < 0 or expert_id >= len(expert_id_mapping):
                raise AssertionError(f"expert_id out of range: {expert_id} (mapping size {len(expert_id_mapping)})")
            replicas = expert_id_mapping[expert_id]
            chosen = -1
            for rep_id in replicas:
                if rep_id < 0:
                    continue
                if token_count_per_instance[rep_id] < capacity:
                    token_count_per_instance[rep_id] += 1
                    chosen = rep_id
                    break
            if chosen < 0 and replicas:
                # All replicas full: round-robin (still accept token)
                idx = rr_index[expert_id] % len(replicas)
                chosen = replicas[idx]
                rr_index[expert_id] += 1
                if chosen >= 0:
                    token_count_per_instance[chosen] += 1
            if chosen >= 0:
                active_experts[token_id][token_rank] = chosen
                active_weights[token_id][token_rank] = 1.0

    return active_experts, active_weights


def topk_expert_selection(
    scores_with_bias: List[List[float]],
    expert_id_mapping: List[List[int]],
    k: int,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Simple top-k selection (no capacity) choosing first replica for each expert.
    """
    batch = len(scores_with_bias)
    num_gated = len(scores_with_bias[0]) if batch > 0 else 0
    active_experts = [[-1 for _ in range(k)] for _ in range(batch)]
    active_weights = [[0.0 for _ in range(k)] for _ in range(batch)]

    for t in range(batch):
        pairs = list(enumerate(scores_with_bias[t]))
        pairs.sort(key=lambda x: x[1], reverse=True)
        for i in range(min(k, num_gated)):
            expert_id = pairs[i][0]
            replicas = expert_id_mapping[expert_id]
            active_experts[t][i] = replicas[0] if replicas else -1
            active_weights[t][i] = pairs[i][1]

    return active_experts, active_weights
