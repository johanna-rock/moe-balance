# Initial Placement Configs

This folder contains initial placement configurations used by `sim/search.py`. These configs control how expert replicas are placed **before** any search steps (e.g., local/anneal/sequence). Shared expert replicas are placed first (when enabled), then the chosen placement strategy is applied for all remaining replicas.

## Common Concepts

- **Replica**: One expert instance (original expert or its replication).
- **Shared expert**: The global shared expert (default id 256) replicated a fixed number of times (16 total instances when `num_shared_experts=1` and shared replication is enabled).
- **Axis**: Placement strategies operate along either **rows** or **columns**. The axis is controlled by `placement_axis`.
- **Device load**: A device’s load is the sum of frequencies of expert replicas placed on that device. The placement strategies try to spread replicas to reduce load hotspots.
- **Row/column load**: Sum of replica frequencies across all devices in a row/column (depending on axis).

## Shared Expert Placement (common step)

When `replica_spread_by_distance` is true, shared expert replicas are placed first using round‑robin over **rows and columns**:

- Replica `i` is placed at `row = i % rows`, `col = i % cols`.
- This spreads shared replicas evenly across both dimensions.
- The shared expert placements reserve capacity so other replicas do not take those device slots.

If disabled, shared replicas are still included but placed by the main placement strategy like any other replica.

---

## Placement Strategies

### 1) `balance`
**Goal**: Evenly spread total load across the chosen axis.

**Algorithm (conceptually):**
1. **Row/column assignment**: Each expert (and its replicas) is assigned to the **least loaded rows/columns** based on total frequency already placed on that axis.
2. **Within-axis placement**: For each replica assigned to a row/column, pick the **least loaded device** in that row/column (frequency‑weighted device load).
3. **Capacity handling**: If a row/column is full (max replicas/device reached), the algorithm falls back to the next least‑loaded row/column.

**Effect**: Minimizes per‑axis load imbalance; useful when communication cost scales with that axis.

---

### 2) `balance_load_coactivation`
**Goal**: Similar to `balance`, but uses **co‑activation** information to reduce co‑located hotspots.

**Algorithm (conceptually):**
1. **Row/column assignment**: Same as `balance` (place onto least loaded rows/columns).
2. **Within-axis placement**: When choosing a device inside a row/column, the strategy **penalizes devices that already host experts that often co‑activate** with the current expert. This attempts to reduce contention on the same device.
3. **Capacity handling**: Same as `balance`.

**Effect**: Spreads co‑activating experts across devices while keeping global load balanced.

---

### 3) `hot-tier`
**Goal**: Place top‑frequency experts in a way that spreads them across a chosen axis, then fill the rest normally.

**Parameters:**
- `top_n`: How many of the highest‑frequency experts are considered “hot”.
- `hot_tier_axis`: Which axis to spread hot experts across (row or col).

**Algorithm (conceptually):**
1. **Hot experts**: Select top‑`N` experts by frequency.
2. **Hot placement**: Assign hot experts across the `hot_tier_axis` to distribute their load.
3. **Remaining experts**: Place all remaining replicas using the standard row/column balancing logic.

**Effect**: Prevents the most frequently used experts from clustering on the same axis, reducing worst‑case hotspots.

---

## Config Files in This Folder

- `row_balance_example.jsonc`: balance with `placement_axis = row`
- `col_balance_example.jsonc`: balance with `placement_axis = col`
- `row_balance_load_coactivation_example.jsonc`: balance_load_coactivation with `placement_axis = row`
- `col_balance_load_coactivation_example.jsonc`: balance_load_coactivation with `placement_axis = col`
- `hot_tier_example.jsonc`: hot‑tier placement, with `hot_tier_axis` controlling the axis for hot experts

You can run all initial placements with the `none` search (to inspect initial placements only) using:

```bash
bash scripts/run_search_sweep.sh configs/system/system_example.jsonc
```
