# DeepSeek V3 MoE Module - Optimized Dense Expert Computation

## Overview

This document describes the **optimized MoE implementation** that replaces sparse computation with dense computation through intelligent token reordering during the all-to-all dispatch phase.

### Key Optimization Principles

1. **Dense Token Packing**: Instead of sparse tensors with garbage values, tokens are reordered and packed densely
2. **Dynamic Expert Height**: Each expert receives a dynamically-sized dense input (variable number of tokens)
3. **Lazy Expert Loading**: Experts with zero assigned tokens are not loaded
4. **Standard Dense MatMul**: Uses efficient dense matrix multiplication instead of sparse_matmul

### Device Mesh Configuration

Same as functional implementation:
- **TG/T3K**: `(4, 8)` = 32 devices
- **DUAL**: `(8, 8)` = 64 devices  
- **QUAD**: `(16, 8)` = 128 devices

The MoE module uses **256 routed experts** with `num_experts_per_tok=8` (8 experts selected per token).

---

## 1. Activation Replication Before All-to-All Dispatch

### Input State
Identical to the functional implementation - MoE receives input as **Data Parallel (DP) + Tensor Parallel (TP)** sharded tensor:
- **Shape per device**: `(1, 1, batch_size_per_device, hidden_size / TP_size)`
- **DP dimension** = `mesh.shape[0]` (rows)
- **TP dimension** = `mesh.shape[1]` (columns) = 8

### Step 1: All-Gather to Replicate Across TP Dimension

Same as functional:

```python
x = ttnn.experimental.all_gather_async(x, **cfg["ccl"].populate_all_gather_runtime_args(cfg["revert_tp"]))
```

**Data Movement Pattern:**
```
Before All-Gather (per row of 8 devices):
  Device[row,0]: [B_local, H/8]  |  Device[row,1]: [B_local, H/8]  | ... | Device[row,7]: [B_local, H/8]
                    ↓ all-gather along dim=-1 (width/TP axis) ↓
After All-Gather (replicated across columns):
  Device[row,0]: [B_local, H]    |  Device[row,1]: [B_local, H]    | ... | Device[row,7]: [B_local, H]
```

**Result**: Full hidden dimension `H=7168` is now available on all devices. Each device row has tokens replicated across columns.

---

## 2. MoE Gate: Expert Selection

Identical to functional implementation:

1. **Gate Projection**: `scores = sigmoid(x @ gate_weights)` → `[batch, n_routed_experts]` = `[batch, 256]`
2. **Score Bias Addition**: Add `e_score_correction_bias` to scores
3. **Expert Group Selection**: 
   - Reshape to `[batch, n_group=16, experts_per_group=16]`
   - TopK within groups (k=2) to find top 2 experts per group
   - TopK on group scores (k=topk_group=4) to find top 4 expert groups
4. **Final TopK**: Select top 8 experts from the active groups

**Output**: 
- `topk_experts_weights`: `[1, 1, batch, 8]` - routing weights
- `topk_experts_indices`: `[1, 1, batch, 8]` - expert indices (0-255)

---

## 3. All-to-All Dispatch with Dense Token Reordering

### Key Difference from Functional Implementation

In the **functional** implementation, all-to-all dispatch produces a sparse output where only relevant token positions are filled, and the rest contain garbage values. 

In the **optimized** implementation, all-to-all dispatch performs **token reordering** to pack tokens densely:

### Dispatch Operation with Reordering

```python
all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors = ttnn.all_to_all_dispatch_dense(
    x_rm,                        # [B_local, 1, seq_len, H]
    topk_experts_indices_rm,     # [B_local, 1, seq_len, K=8]
    expert_mapping_tensors,      # [1, 1, E=256, D=32]
    cluster_axis=0,              # Dispatch along row dimension (DP axis)
    reorder_dense=True,          # Enable dense packing with reordering
)
```

### Detailed Data Movement with Reordering

**Cluster Axis = 0** means dispatch operates along the DP (row) dimension.

**Example for TG (4 rows, 8 cols):**

```
Input: Each device has B_local=32 tokens (total 128 tokens across DP)
       Each token routes to K=8 experts

FUNCTIONAL (Sparse) Output:
  Shape: [1, B_global=128, S, H] - Fixed size, sparse with garbage values
  Token positions match original positions

OPTIMIZED (Dense) Output:
  Shape: [1, T_local, 1, H] - Dynamic height T_local = count of tokens for local experts
  Tokens are reordered and densely packed per expert
```

**Dense Packing Visual:**
```
ALL-TO-ALL DISPATCH WITH DENSE REORDERING (cluster_axis=0):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Consider Device[row=0, col=0] which owns Experts 0-7:                           │
│                                                                                 │
│ From Row 0: Token 3 → Expert 2, Token 7 → Expert 0, Token 15 → Expert 5         │
│ From Row 1: Token 40 → Expert 0, Token 52 → Expert 2                            │
│ From Row 2: Token 70 → Expert 7, Token 88 → Expert 0                            │
│ From Row 3: Token 100 → Expert 2, Token 110 → Expert 5                          │
│                                                                                 │
│ FUNCTIONAL writes to sparse positions (most positions = garbage):               │
│   Position 3:   Token 3 data  (for Expert 2)                                    │
│   Position 7:   Token 7 data  (for Expert 0)                                    │
│   Position 15:  Token 15 data (for Expert 5)                                    │
│   Position 40:  Token 40 data (for Expert 0)                                    │
│   ... etc                                                                       │
│                                                                                 │
│ OPTIMIZED reorders and packs densely by expert:                                 │
│   Expert 0 section (3 tokens): [Token 7, Token 40, Token 88]                    │
│   Expert 2 section (3 tokens): [Token 3, Token 52, Token 100]                   │
│   Expert 5 section (2 tokens): [Token 15, Token 110]                            │
│   Expert 7 section (1 token):  [Token 70]                                       │
│   (Experts 1, 3, 4, 6 have 0 tokens - no allocation needed)                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Output Format

**Dense Dispatch Output:**
- **Shape**: `[1, 1, T_total_local, H]` where `T_total_local = sum(tokens_per_expert)` for local experts
- **Layout**: Tokens contiguously packed, grouped by expert
- **No garbage values**: Every position contains valid token data

**Metadata Tensors:**
- **Expert Offsets**: `[E_local]` - Starting position of each expert's token block
- **Token Counts**: `[E_local]` - Number of tokens assigned to each expert
- **Reverse Mapping**: `[T_total_local]` - Original (device, position) for each token (for combine phase)
- **Expert Weights per Token**: `[T_total_local]` - Routing weight for each token-expert pair

---

## 4. Expert Sharding Strategy

### Weight Distribution

Same as functional implementation:
- Total experts: `n_routed_experts = 256`
- TG (32 devices): `256 / 32 = 8` experts per device
- DUAL (64 devices): `256 / 64 = 4` experts per device
- QUAD (128 devices): `256 / 128 = 2` experts per device

### Weight Shapes

Each expert has 3 weight matrices:
- `w1_experts` (gate_proj): `[1, 1, H, intermediate_size]` per expert
- `w2_experts` (down_proj): `[1, 1, intermediate_size, H]` per expert
- `w3_experts` (up_proj): `[1, 1, H, intermediate_size]` per expert

### Lazy Expert Loading

**Key Optimization**: Experts with zero assigned tokens are **not loaded** from memory.

```python
# Only load experts that have tokens assigned
active_experts = [e for e in range(E_local) if token_counts[e] > 0]
for expert_id in active_experts:
    load_expert_weights(expert_id)
```

---

## 5. Dense Expert Computation

### Key Difference from Functional

- **Functional**: Uses `sparse_matmul` with sparsity masks on fixed-size sparse tensors
- **Optimized**: Uses standard `dense_matmul` on variable-height dense tensors

### Expert Forward Pass

```python
def _forward_dense(cls, x: ttnn.Tensor, expert_offsets: ttnn.Tensor, 
                   token_counts: ttnn.Tensor, cfg: RunDecodeConfig):
    outputs = []
    
    for expert_id in range(E_local):
        num_tokens = token_counts[expert_id]
        if num_tokens == 0:
            continue  # Skip experts with no tokens
        
        # Extract dense token block for this expert
        start = expert_offsets[expert_id]
        end = start + num_tokens
        x_expert = x[:, :, start:end, :]  # [1, 1, num_tokens, H]
        
        # Load expert weights (lazy loading)
        w1 = load_weight(expert_id, "w1")  # [1, 1, H, intermediate]
        w3 = load_weight(expert_id, "w3")  # [1, 1, H, intermediate]
        w2 = load_weight(expert_id, "w2")  # [1, 1, intermediate, H]
        
        # FF1: Gate projection (dense matmul)
        w1_out = ttnn.matmul(x_expert, w1)  # [1, 1, num_tokens, intermediate]
        
        # FF3: Up projection (dense matmul)
        w3_out = ttnn.matmul(x_expert, w3)  # [1, 1, num_tokens, intermediate]
        
        # SiLU activation and element-wise multiply
        activated = ttnn.mul(ttnn.silu(w1_out), w3_out)  # [1, 1, num_tokens, intermediate]
        
        # FF2: Down projection (dense matmul)
        output = ttnn.matmul(activated, w2)  # [1, 1, num_tokens, H]
        
        outputs.append((expert_id, output))
    
    return outputs
```

### Local Reduction for Same-Token Results

Since a token may be routed to multiple experts **on the same device**, results must be reduced locally:

```python
def local_reduce_same_token(expert_outputs, token_info, expert_weights_per_token):
    """
    Reduce outputs for tokens that went to multiple local experts.
    
    Example: If Token 42 was routed to Expert 0 and Expert 2 (both on this device),
    their outputs are weighted-summed here before the all-to-all combine.
    """
    reduced_output = allocate_output([1, 1, T_total_local, H])
    
    for token_idx in unique_tokens:
        # Find all expert outputs for this token
        expert_outputs_for_token = get_outputs_for_token(token_idx, expert_outputs)
        weights_for_token = expert_weights_per_token[token_idx]
        
        # Weighted sum of expert outputs
        weighted_sum = sum(w * out for w, out in zip(weights_for_token, expert_outputs_for_token))
        reduced_output[token_idx] = weighted_sum
    
    return reduced_output
```

**Expert Output After Local Reduction**: `[1, 1, T_unique_local, H]` where `T_unique_local` ≤ `T_total_local`

---

## 6. All-to-All Combine with Position Restoration

### Purpose
Route expert outputs back to originating token devices and **restore original token positions**.

### Combine Operation

```python
all_to_all_combine_output_tensors = ttnn.all_to_all_combine_dense(
    locally_reduced_output,           # [1, 1, T_unique_local, H]
    reverse_mapping_metadata,         # Maps each output back to (orig_device, orig_position)
    expert_mapping_tensors,           # [1, 1, E, D]
    cluster_axis=0,
)
```

### Data Movement Pattern with Position Restoration

```
ALL-TO-ALL COMBINE WITH POSITION RESTORATION (cluster_axis=0):
┌─────────────────────────────────────────────────────────────────────────────────┐
│ OPTIMIZED combine reverses the reordering from dispatch:                        │
│                                                                                 │
│ Device[0,0] has locally reduced outputs:                                        │
│   Output for Token 7 (originally from Device[0,0])   → Send to Device[0,*]      │
│   Output for Token 40 (originally from Device[1,0])  → Send to Device[1,*]      │
│   Output for Token 88 (originally from Device[2,0])  → Send to Device[2,*]      │
│   ... etc                                                                       │
│                                                                                 │
│ Each output is placed back at its ORIGINAL position in the destination tensor  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Output Format

After combine, each device has:
- **Shape**: `[K, B_local, S, H]` - K expert contributions per original token position
- Tokens restored to original positions
- Ready for final aggregation

---

## 7. Final Aggregation and TP Reduction

### Weighted Sum Across Remaining Experts

After the local reduction on expert-owning devices, some tokens may still have multiple expert outputs (from experts on **different** devices). These are aggregated:

```python
# Reshape combined output
post_combine_output_tensor = ttnn.reshape(
    all_to_all_combine_output_tensors,
    shape=(num_experts_per_tok, 1, batch_size_per_device * seq_len, hidden_size),
)  # [K=8, 1, B_local*S, H] - note: many positions may be zeros due to local reduction

# Apply remaining expert weights (for experts on other devices)
post_combine_output_tensor = ttnn.mul(
    post_combine_output_tensor, remaining_topk_weights, ...
)

# Sum across experts
post_combine_output_tensor = ttnn.sum(post_combine_output_tensor, dim=0, keepdim=True)
# Result: [1, 1, B_local*S, H]
```

### Reduce-Scatter for TP Reduction

Same as functional:

```python
return ttnn.experimental.reduce_scatter_minimal_async(
    post_combine_output_tensor, 
    cluster_axis=1,  # TP dimension (columns)
    dim=3,           # hidden dimension
    ...
)
```

**Final Data Movement:**
```
Before reduce-scatter (replicated across columns):
  Device[row,0]: [B_local, H]  |  Device[row,1]: [B_local, H]  | ... | Device[row,7]: [B_local, H]
                    ↓ reduce-scatter along cluster_axis=1, dim=3 ↓
After reduce-scatter (TP sharded):
  Device[row,0]: [B_local, H/8] |  Device[row,1]: [B_local, H/8] | ... | Device[row,7]: [B_local, H/8]
```

---

## Summary: Complete Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ INPUT: TP-sharded activations [B_local, H/8] per device                      │
│        (4 rows × 8 columns for TG)                                           │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 1. ALL-GATHER (cluster_axis=1, TP dimension)                                 │
│    Replicate hidden dimension across columns                                  │
│    Result: [B_local, H] replicated on each row                               │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 2. MOE GATE                                                                   │
│    Compute routing: topk_weights [B_local, 8], topk_indices [B_local, 8]     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 3. ALL-TO-ALL DISPATCH WITH DENSE REORDERING (cluster_axis=0)                │
│    Route tokens to devices owning their selected experts                      │
│    ★ REORDER tokens to pack densely by expert                                │
│    Input:  [B_local, 1, S, H] per device                                     │
│    Output: [1, 1, T_local, H] per device (DENSE, variable height)            │
│    + metadata: expert offsets, token counts, reverse mapping                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. DENSE EXPERT COMPUTATION (local to each device)                           │
│    ★ Load only experts with assigned tokens (lazy loading)                   │
│    ★ Use standard dense matmul (not sparse_matmul)                           │
│    For each active expert e:                                                 │
│      x_e = input[offset[e] : offset[e] + count[e]]                          │
│      out_e = FF2(SiLU(FF1(x_e)) * FF3(x_e))                                 │
│    Output: Expert outputs grouped by expert                                  │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. LOCAL REDUCTION                                                           │
│    ★ Weighted-sum outputs for same tokens routed to multiple local experts   │
│    Reduces computation in combine phase                                      │
│    Output: [1, 1, T_unique_local, H] per device                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. ALL-TO-ALL COMBINE WITH POSITION RESTORATION (cluster_axis=0)             │
│    Route expert outputs back to token-owning devices                          │
│    ★ RESTORE original token positions (revert dispatch reordering)           │
│    Input:  [1, 1, T_unique_local, H] per device                              │
│    Output: [K, B_local, S, H] per device (partially reduced)                 │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. FINAL WEIGHTED SUM                                                         │
│    Apply remaining routing weights, sum across K expert slots                │
│    (Many slots zeros due to local reduction on expert devices)               │
│    Result: [1, B_local, S, H] per device (replicated across columns)         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 8. REDUCE-SCATTER (cluster_axis=1, TP dimension)                             │
│    Reduce across columns, scatter hidden dimension                            │
│    Result: [B_local, H/8] per device (TP-sharded)                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: TP-sharded activations [B_local, H/8] per device                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparison: Functional vs Optimized

| Aspect | Functional | Optimized |
|--------|------------|-----------|
| **Dispatch Output** | Sparse, fixed size `[1, B_global, S, H]` | Dense, variable height `[1, 1, T_local, H]` |
| **Token Layout** | Original positions, garbage elsewhere | Reordered, densely packed by expert |
| **Expert Loading** | All local experts loaded | Only active experts loaded |
| **MatMul Type** | `sparse_matmul` with sparsity mask | Standard `dense_matmul` |
| **Local Reduction** | None (all reduction in combine) | Weighted-sum same-token outputs |
| **Memory Efficiency** | Wastes memory on garbage positions | Minimal allocation for actual tokens |
| **Compute Efficiency** | Sparsity mask overhead | Direct dense computation |
| **Combine Input** | Fixed shape `[E_local, B_global, S, H]` | Variable shape `[1, 1, T_unique_local, H]` |

---

## Key Implementation Details

| Aspect | Value |
|--------|-------|
| Total Experts | 256 |
| Experts per Token | 8 |
| Expert Groups | 16 (for gating) |
| All-to-All Dispatch Axis | 0 (DP/rows) |
| All-Gather/Reduce-Scatter Axis | 1 (TP/columns) |
| Topology | Linear (configurable) |
| **Dispatch Mode** | Dense with reordering |
| **MatMul Mode** | Dense (no sparsity mask) |
| **Expert Loading** | Lazy (on-demand) |
| **Local Reduction** | Enabled |

---

## Benefits of Optimized Implementation

1. **Better Memory Utilization**: No wasted memory on sparse/garbage positions
2. **More Efficient Compute**: Dense matmul is more efficient than masked sparse matmul
3. **Reduced Memory Bandwidth**: Only load expert weights that are actually used
4. **Lower Communication Volume** (potentially): Local reduction reduces data sent in combine
5. **Better Hardware Utilization**: Dense operations map better to accelerator tensor cores

---

## Challenges and Considerations

1. **Dynamic Shapes**: Variable tensor heights require careful memory management
2. **Reordering Overhead**: Dispatch must compute and apply token reordering
3. **Metadata Tracking**: Need to track reverse mapping for position restoration
4. **Load Imbalance**: Some devices may receive more tokens than others (mitigated by aux loss in training)
5. **Implementation Complexity**: More complex logic than fixed sparse approach

# Notes on Performance
## A2A Dispatch
New dispatch now does a sparse mcast. If a token is sent to multiple devices, it will pass it along. So if the token starts on device 0 and hits experts and device 2/3/7, then it will send to 2, write to NoC there, 2 will pass it off to 3, write to NoC there, who will pass it off to 7 and write to NoC there
- *We send the token in 2 directions (ring) and split it in half when it hits the device on the exact other side of the ring*
- This scales sublinearly now. As more experts along our cluster axis are selected, we teeter off. The number of experts selected on that axis is less of a problem than the average distance those experts live away from that token. If every token is selecting experts 8 hops away, there's more traffic interference than if you just selected next door neighbours
- If the same token goes to one device (rows) it's only sent once, not twice

Perf table in us
||E=1|E=2|E=3|E=4|E=5|E=6|E=7|E=8|
|Best case|26.868|96.183|33.005|36.073|39.141|42.209|45.277|48.345|
|Avergae case|55.258|66.713|78.169|89.624|101.079|112.535|123.990|135.445|
|Worst case|88.238|96.183|104.129|112.074|120.020|127.965|135.911|143.856|

Where E=1 - E=8 represents the maximum number of active experts within a column. 
Best case is the respective number of max. active experts within that column where the hop count (rows) is minimal, average case is the respective number of max. active experts within that column where the hop count (rows) is the average, and worst case is the respective number of max. active experts within that column where the hop count (rows) is maximal.


## A2A Combine
Combine is still p2p, but it reduces locally in case one token does to multiple experts on the same device and is sent back only once per device.
- There is currently no reduction along the way, i.e. the reduction is done with devices along the path from furthest device to origin device; currently they are sent back separately (one per device) and then reduced on the origin device
- In the case where the token selects only 1 expert per device column, this reduces to just a p2p


## Compute
MoE compute scales stepwise.
- When each expert has <= 32 tokens: 235 us
- For each additional set of (up to) 32 tokens per expert it adds on around 130 us
Example:
E0 has 17, E1 has 17 -> ~235 us
E0 has 17, E1 has 38 -> ~365 us
E0 has 38, E1 has 38 -> ~495 us
E0 has 38, E1 has 70 -> ~625 us
