# Search Loop Documentation

This document describes how the **MoE placement/replication search** is structured, and how the **routing strategy** and **cost model** are wired together.

---

## Inputs

### 1) Trace Data
- `trace.jsonl` with **top‑k experts per token**:
  - `topk_experts`: `[num_tokens][K]`
  - `origin_rows`: `[num_tokens]`
- No `scores_with_bias` is required for the current routing strategy.

### 2) Search Parameters
- `experts = 256` (original experts)
- `replication_slots` (e.g., 128 → 384 expert instances)
- `rows, cols` (device mesh)
- `routing_strategy`:
- **`balanced_expert_selection_replicas`** (current)
  - **`balanced_expert_selection_with_rerouting`** (later; requires scores)

### 3) Routing Inputs
- `expert_id_mapping`: `[NUM_ORIGINAL_EXPERTS][MAX_NUM_REPLICAS]`
  - Maps each original expert to a list of **expert instance IDs** (replicas).
  - Built from the replication plan.

### 4) Cost Model Inputs
- `active_experts`: `[BATCH][K]` (expert instance IDs)
- `origin_rows`: `[BATCH]`
- `instance → device` mapping (from placement)

---

## Routing Strategy (Current)

**`balanced_expert_selection_replicas`**

For each token and each of its top‑k original experts:
1. Try replicas in order (using `expert_id_mapping`).
2. If a replica has available capacity, select it.
3. If all replicas are full, **round‑robin** across replicas (no dropping).
