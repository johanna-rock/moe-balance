#!/usr/bin/env python3
from typing import List, Tuple, Optional


def replicate_freqs(freqs: List[float], slots: int, original_ids: List[int] = None,
                    shared_id: Optional[int] = None, shared_replicas: int = 16) -> Tuple[List[float], List[int]]:
    """
    Replicate experts by repeatedly splitting the highest-frequency expert.

    freqs: list of frequencies (percent), length = num_experts
    slots: number of replication slots to add
    original_ids: mapping from current replica index -> original expert id
    """
    if slots <= 0:
        return list(freqs), list(original_ids) if original_ids is not None else list(range(len(freqs)))

    out_freqs = list(freqs)
    if original_ids is None:
        replica_to_original = list(range(len(freqs)))
    else:
        replica_to_original = list(original_ids)

    # Reserve fixed replicas for shared expert (if present).
    # shared_replicas is TOTAL replicas for shared expert (e.g., 16 = 1 base + 15 extra).
    if shared_id is not None and 0 <= shared_id < len(out_freqs) and shared_replicas > 1:
        shared_freq = out_freqs[shared_id]
        per_rep = shared_freq / shared_replicas
        out_freqs[shared_id] = per_rep
        for _ in range(shared_replicas - 1):
            out_freqs.append(per_rep)
            replica_to_original.append(shared_id)
        slots -= (shared_replicas - 1)

    for _ in range(slots):
        # pick highest frequency (tie -> lowest index)
        # Ignore shared expert for extra replication.
        candidate_idxs = [i for i in range(len(out_freqs)) if replica_to_original[i] != shared_id]
        if not candidate_idxs:
            break
        max_i = max(candidate_idxs, key=lambda i: (out_freqs[i], -i))
        v = out_freqs[max_i]
        half = v / 2.0
        out_freqs[max_i] = half
        out_freqs.append(half)
        replica_to_original.append(replica_to_original[max_i])

    return out_freqs, replica_to_original
