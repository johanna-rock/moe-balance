# DeepSeek V3 MoE Module - Data Movement and Expert Sharding Report

## Overview

The DeepSeek V3 MoE (Mixture of Experts) module operates on a 2D mesh of devices with shape `(rows, cols)` where typically:
- **TG/T3K**: `(4, 8)` = 32 devices
- **DUAL**: `(8, 8)` = 64 devices  
- **QUAD**: `(16, 8)` = 128 devices

The MoE module uses **256 routed experts** with `num_experts_per_tok=8` (8 experts selected per token).

---

## 1. Activation Replication Before All-to-All Dispatch

### Input State
The MoE receives input as a **Data Parallel (DP) + Tensor Parallel (TP)** sharded tensor:
- **Shape per device**: `(1, 1, batch_size_per_device, hidden_size / TP_size)`
- **DP dimension** = `mesh.shape[0]` (rows)
- **TP dimension** = `mesh.shape[1]` (columns) = 8

### Step 1: All-Gather to Replicate Across TP Dimension

From `moe.py` line 379:

```python
x = ttnn.experimental.all_gather_async(x, **cfg["ccl"].populate_all_gather_runtime_args(cfg["revert_tp"]))
```

This performs an **all-gather along cluster_axis=1** (TP/column dimension) to replicate the full hidden dimension across all 8 devices in a row:

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

The gate computes which 8 experts each token should route to:

From `moe_gate.py`:

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

## 3. All-to-All Dispatch - Inter-Chip Data Movement

### Purpose
Route tokens to the devices that own the selected experts.

### Expert Mapping Tensor
From `moe.py` lines 84-94:

```python
expert_mapping_tensors = ttnn.from_torch(
    torch.eye(num_devices, dtype=torch.int32)
    .repeat_interleave(num_experts_per_device, dim=0)  # [256, 32] for TG
    .unsqueeze(0).unsqueeze(0),  # [1, 1, 256, 32]
    ...
)
```

This creates a one-hot mapping: **Expert → Device**. For TG (32 devices), each device owns 8 experts:
- Device 0: Experts 0-7
- Device 1: Experts 8-15
- ...
- Device 31: Experts 248-255

### Dispatch Operation

From `moe.py` lines 318-323:

```python
all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors = ttnn.all_to_all_dispatch(
    x_rm,                        # [B_local, 1, seq_len, H]
    topk_experts_indices_rm,     # [B_local, 1, seq_len, K=8]
    expert_mapping_tensors,      # [1, 1, E=256, D=32]
    cluster_axis=0,              # Dispatch along row dimension (DP axis)
)
```

### Detailed Data Movement

**Cluster Axis = 0** means dispatch operates along the DP (row) dimension.

From the C++ documentation in `all_to_all_dispatch_nanobind.cpp`:

```
Inputs:
  - input_tensor: [B, S, 1, H] per device, sharded along batch or sequence
  - expert_indices: [B, S, 1, K] - which experts each token routes to
  - expert_mapping: [1, 1, E, D] - one-hot expert-to-device map

Output:
  - output_tensor: [1, B*D[A], S, H] per device - tokens from ALL DP devices
  - metadata_tensor: [1, B*D[A], S, K] - gathered expert indices
```

**Example for TG (4 rows, 8 cols):**

```
Input: Each device has B_local=32 tokens (total 32*4=128 tokens across DP)

ALL-TO-ALL DISPATCH ACROSS ROWS (cluster_axis=0):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Row 0: Tokens 0-31   → Sends tokens to rows owning selected experts         │
│ Row 1: Tokens 32-63  → Sends tokens to rows owning selected experts         │
│ Row 2: Tokens 64-95  → Sends tokens to rows owning selected experts         │
│ Row 3: Tokens 96-127 → Sends tokens to rows owning selected experts         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
After dispatch, each device has:
  - All 128 tokens (sparse - only tokens routing to this device's experts filled)
  - Shape: [1, B*D_rows, S, H] = [1, 128, 1, 7168] per device
```

**Key Points:**
1. Tokens are sent to devices based on expert selection
2. Output is **sparse** - only relevant tokens are populated, others are garbage
3. Metadata tensor tracks which expert indices were selected (needed for combine)

---

## 4. Expert Sharding Strategy

### Weight Distribution

From `experts.py` lines 39-41:

```python
def _get_num_experts_per_device(cls, hf_config, mesh_device):
    return even_int_div(hf_config.n_routed_experts, mesh_device.get_num_devices())
```

**Expert Distribution:**
- Total experts: `n_routed_experts = 256`
- TG (32 devices): `256 / 32 = 8` experts per device
- DUAL (64 devices): `256 / 64 = 4` experts per device
- QUAD (128 devices): `256 / 128 = 2` experts per device

### Weight Shapes

From `experts.py` lines 58-90, each expert has 3 weight matrices:
- `w1_experts` (gate_proj): `[1, E_local, H, intermediate_size]` 
- `w2_experts` (down_proj): `[1, E_local, intermediate_size, H]`
- `w3_experts` (up_proj): `[1, E_local, H, intermediate_size]`

Where `E_local = num_experts_per_device`.

**Sharding**: Weights are sharded along dimension `(1, 1)` meaning experts are distributed across devices in blocks.

---

## 5. Sparsity and Expert Computation

### Token-Expert Remapping

From `moe.py` lines 331-340:

```python
remap_topk_mask = ttnn.repeat(cfg["remap_topk_mask"], ttnn.Shape((1, batch_size_per_device, 1, 1)))

_, sparsity_t = ttnn.moe_expert_token_remap(
    remap_topk_mask,
    expert_mapping_tensors,
    all_to_all_dispatch_metadata_tensors,
    reduction_size=SPARSITY_BLOCK_SIZE,  # 32
)
```

**Purpose**: Convert global expert indices to local indices and create a **sparsity mask** for efficient computation.

**Sparsity Tensor Shape**: `[D, 1, B*S/reduction_size, E_local]`
- Indicates which blocks of tokens have non-zero computation for each expert
- Enables `sparse_matmul` to skip empty blocks

### Expert Forward Pass

From `experts.py` lines 212-244:

```python
def _forward(cls, x: ttnn.Tensor, sparsity: ttnn.Tensor, cfg: RunDecodeConfig):
    # Reshape for sparse computation
    x = ttnn.reshape(x, shape=(1, num_sparse_blocks, SPARSITY_BLOCK_SIZE, hidden_size))
    
    # FF1: Gate projection (sparse matmul)
    w1_out = ttnn.sparse_matmul(x, sparsity=sparsity, **cfg["w1_experts"])
    
    # FF3: Up projection (sparse matmul)
    w3_out = ttnn.sparse_matmul(x, sparsity=sparsity, **cfg["w3_experts"])
    
    # SiLU activation and element-wise multiply
    activated = ttnn.mul(w1_out, w3_out, **cfg["mul_experts"])  # includes SILU
    
    # FF2: Down projection (sparse matmul)
    output = ttnn.sparse_matmul(activated, sparsity=sparsity, **cfg["w2_experts"])
    
    # Reshape output
    output = ttnn.reshape(output, shape=(1, E_local, num_tokens, hidden_size))
    return output
```

**Expert Output Shape**: `[1, E_local, B_global*S, H]` per device

---

## 6. All-to-All Combine - Return Data Movement

### Purpose
Route expert outputs back to originating token devices.

From `moe.py` lines 350-355:

```python
all_to_all_combine_output_tensors = ttnn.all_to_all_combine(
    experts_output,                          # [E_local, B_global, S, H]
    all_to_all_dispatch_metadata_tensors,    # [D, B_global, S, K]
    expert_mapping_tensors,                  # [1, 1, E, D]
    cluster_axis=0,
)
```

### Data Movement Pattern

From `all_to_all_combine_nanobind.cpp`:

```
Input:
  - input_tensor: [E_local, B_global, S, H] per device - expert outputs
  - metadata: Expert indices from dispatch
  - mapping: Expert-to-device mapping

Output:
  - output_tensor: [K, B_local, S, H] per device - outputs for local tokens
```

**Reverse Routing:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Row 0: Has outputs for experts 0-63   → Sends outputs back to token owners  │
│ Row 1: Has outputs for experts 64-127 → Sends outputs back to token owners  │
│ Row 2: Has outputs for experts 128-191→ Sends outputs back to token owners  │
│ Row 3: Has outputs for experts 192-255→ Sends outputs back to token owners  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
After combine, each device has:
  - Expert outputs for its local tokens only
  - Shape: [K=8, B_local, S, H] - K expert outputs per token
```

---

## 7. Final Aggregation

### Weighted Sum Across Experts

From `moe.py` lines 357-368:

```python
# Reshape combined output
post_combine_output_tensor = ttnn.reshape(
    all_to_all_combine_output_tensors,
    shape=(num_experts_per_tok, 1, batch_size_per_device * seq_len, hidden_size),
)  # [K=8, 1, B_local*S, H]

# Multiply by expert weights
post_combine_output_tensor = ttnn.mul(
    post_combine_output_tensor, topk_experts_weights, ...
)

# Sum across experts
post_combine_output_tensor = ttnn.sum(post_combine_output_tensor, dim=0, keepdim=True)
# Result: [1, 1, B_local*S, H]
```

### Reduce-Scatter for TP Reduction

From `moe.py` lines 374-376:

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
│ 3. ALL-TO-ALL DISPATCH (cluster_axis=0, DP/row dimension)                    │
│    Route tokens to devices owning their selected experts                      │
│    Input:  [B_local, 1, S, H] per device                                     │
│    Output: [1, B_global, S, H] per device (sparse)                           │
│    + metadata: [1, B_global, S, K] expert indices                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 4. EXPERT COMPUTATION (local to each device)                                 │
│    Each device computes its E_local=8 experts on received tokens             │
│    Uses sparse_matmul with sparsity mask                                     │
│    Output: [E_local, B_global, S, H]                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 5. ALL-TO-ALL COMBINE (cluster_axis=0, DP/row dimension)                     │
│    Route expert outputs back to token-owning devices                          │
│    Input:  [E_local, B_global, S, H] per device                              │
│    Output: [K=8, B_local, S, H] per device                                   │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 6. WEIGHTED SUM                                                               │
│    Multiply by routing weights, sum across K=8 experts                       │
│    Result: [1, B_local, S, H] per device (replicated across columns)         │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ 7. REDUCE-SCATTER (cluster_axis=1, TP dimension)                             │
│    Reduce across columns, scatter hidden dimension                            │
│    Result: [B_local, H/8] per device (TP-sharded)                            │
└──────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: TP-sharded activations [B_local, H/8] per device                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Implementation Details

| Aspect | Value |
|--------|-------|
| Total Experts | 256 |
| Experts per Token | 8 |
| Expert Groups | 16 (for gating) |
| Sparsity Block Size | 32 tokens |
| All-to-All Dispatch Axis | 0 (DP/rows) |
| All-Gather/Reduce-Scatter Axis | 1 (TP/columns) |
| Topology | Linear (configurable) |
| Memory Config (decode) | L1_MEMORY_CONFIG |
| Memory Config (prefill) | DRAM_MEMORY_CONFIG |

---

## File References

| File | Purpose |
|------|---------|
| `tt/moe.py` | Main MoE module orchestration |
| `tt/moe_gate.py` | Expert routing/gating logic |
| `tt/experts.py` | Expert weight loading and forward pass |
| `tt/ccl.py` | Collective communication (semaphores, links) |
| `utils/config_dataclass.py` | Configuration dataclasses for ops |
| `utils/config_helpers.py` | Constants (SPARSITY_BLOCK_SIZE, etc.) |

