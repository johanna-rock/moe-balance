# Balanced Expert Selection (Proposed Op)

## Context
This note summarizes the proposed `balanced_expert_selection` operation that will replace the grouped top‑k selection in the optimized MoE flow. It is meant to improve load balance by enforcing per‑expert (or per‑expert‑instance) capacity and rerouting when capacity is exhausted.

## High‑Level Ideas
- Replace grouped top‑k with a fixed, deterministic selection algorithm that enforces expert capacity.
- Make expert placement and replication fully data‑driven via input tensors (runtime‑configurable).
- Support expert rerouting:
  - No replication: overflow goes to the next best expert.
  - With replication: overflow uses another replica instance of the same expert; if all replicas are full, reroute to the next best expert.
- Allow per‑device preferred replica ordering (device‑specific instance priority mapping).

## Inputs and Outputs (From Pseudocode)
- `BATCH` = `512`
- `NUM_GATED_EXPERTS` = `256`
- `NUM_ORIGINAL_EXPERTS` = `NUM_GATED_EXPERTS+1`
- `K` = `8`
- `MAX_NUM_REPLICAS` = `16`
- `scores_with_bias`: `[1,1,BATCH,NUM_GATED_EXPERTS]` per device (replicated rows/cols)
- `expert_id_mapping`: `[1,1,NUM_ORIGINAL_EXPERTS,MAX_NUM_REPLICAS]` per device (maps original expert IDs to expert instances/replica IDs)
- `capacity_factor`: `[1,1,1,1]`
- `NUM_EXPERTS` = `384` (3 per device in example)
- Output `active_experts`: `[1,1,BATCH,K]` (expert instance IDs)
- Output `active_weights`: `[1,1,BATCH,K]` (routing weights aligned to `active_experts`)

Also: `expert_device_mapping`: total `[NUM_DEVICES,1,NUM_EXPERTS,NUM_DEVICES]`, per device `[1,1,NUM_EXPERTS,NUM_DEVICES]` is expert to device mapping, it now maps expert instances (384) to devices, not original expert IDs. Should not impact A2A dispatch.

## Pseudocode Flow (Condensed)
1. Build indices `[0..NUM_GATED_EXPERTS]` for expert scores per token.
2. Sort scores and indices along expert dimension.
3. Initialize:
   - `token_count_per_expert_instance` to zero.
   - `capacity = CF * (B*K/NUM_EXPERTS)`.
   - `active_experts = -1`.
   - `current_expert_rank_index` per token to 0.
4. For each `token_rank` in `0..K-1`, each `token_id` in `0..B-1`:
   - Scan ranked experts from `current_expert_rank_index[token_id]` upward.
   - For each candidate expert, try replica instances in preferred order.
   - If instance capacity available: select it, update counts, advance rank index, break.
   - If all instances full: continue to next best expert.
5. Return `active_experts` and `active_weights`.

## Pseudocode (Original, With `active_weights`)

```
input1: affiliate scores with bias, [1,1,BATCH,NUM_GATED_EXPERTS] total shape, replicated in rows and columns -> [1,1,BATCH,NUM_GATED_EXPERTS] per device
input2: expert_id_mapping [1,1,NUM_EXPERTS,MAX_NUM_REPLICAS] (rep,rep) [1,1,NUM_EXPERTS,MAX_NUM_REPLICAS] per device (e.g. [1,1,384,16])
input3: capacity_factor [1,1,1,1]
NUM_EXPERTS = 384 (3 per device)
output: active_experts [1,1,BATCH,K]
output: active_weights [1,1,BATCH,K]

1. Create an index tensor mapping the scores to experts [1,1,BATCH,NUM_GATED_EXPERTS] = where [1,1,BATCH,:] = 0,1,...,NUM_GATED_EXPERTS
2. Sort the scores/indices based on the score tensor in dim=3; sorted_scores, sorted_indices
3. Init counters:
    token_count_per_expert [1,1,1,NUM_EXPERTS] = 0
    capacity = floor(CF * (B_global*k/NUM_EXPERTS)) = 2 * (BATCH*8/NUM_EXPERTS)
    active_experts = [1,1,BATCH,K] = -1
    active_weights = [1,1,BATCH,K] = 0
    current_expert_rank_index = [1,1,1,BATCH] = 0
4. Loop over tokens to build active_experts
   For each token_rank in 0,...,K-1
        for each token_id in 0,...BATCH-1
            for each expert_id in current_expert_rank_index[token_id],...,NUM_GATED_EXPERTS
                next_active_expert = sorted_indices[1,1,token_id,expert_id]
                found_next_expert = false
                for each expert_replica_id 0,...,MAX_NUM_REPLICAS-1
                    expert_instance_id = expert_id_mapping[0,0,next_active_expert,expert_replica_id]
                    if token_count_per_expert[expert_instance_id] < capacity
                       active_experts[1,1,token_id,token_rank] = expert_instance_id
                       active_weights[1,1,token_id,token_rank] = sorted_scores[expert_id]
                       current_expert_rank_index[token_id] = expert_id
                       found_next_expert = true
                       break
                if found_next_expert: # continue with next token_id
                    break
5. return active_experts, active_weights
```


**Potential Improvements / Changes**
- **Complexity**:
  - Vectorized mask‑based selection, or pre‑computing a candidate list with top‑m where `m` is small but > K.
  - Two‑phase selection: pick top‑k experts, then only reroute among a capped candidate pool.
