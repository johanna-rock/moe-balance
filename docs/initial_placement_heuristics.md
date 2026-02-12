# Initial Placement Heuristics (Optimized MoE)

These are **candidate initialization strategies** for expert placement and replication, designed to align with the optimized dense‑dispatch MoE flow.

---

## 1) Row‑First + Column‑Balance

**Goal:** Minimize cross‑row traffic (dispatch/combine) first, then balance per‑row device load.

**Approach**
- Assign experts (and replicas) to rows using a locality‑first objective.
- Within each row, place replicas on least‑loaded devices (columns).

---

## 2) Co‑Activation Graph Partitioning

**Goal:** Keep co‑activated experts on the same row.

**Approach**
- Build a co‑activation matrix (edge weight = co‑selected frequency).
- Partition experts into `rows` to maximize intra‑row affinity.
- Place replicas within each row using least‑loaded device.

---

## 3) Two‑Tier Hot‑Expert Placement

**Goal:** Spread hot experts across rows while keeping cold experts local.

**Approach**
- Identify top‑N hottest experts by frequency.
- Distribute their replicas evenly across rows to avoid hot rows.
- Assign cold experts using co‑activation affinity within rows.

---

## 4) Shared Expert Row Replication

**Goal:** Reduce shared‑expert row pressure.

**Approach**
- Ensure shared expert has at least one replica per row.
- Spread remaining shared replicas across columns to reduce contention.

---

## 5) Greedy Marginal‑Cost Placement

**Goal:** Directly minimize the dispatch/compute/combine proxy during placement.

**Approach**
- Start with empty placement.
- For each replica, place on the device that gives the smallest increase in:
  ```
  ΔT = Δdispatch + Δcompute + Δcombine
  ```

---

## 6) Balanced‑Capacity Initialization

**Goal:** Avoid row hotspots.

**Approach**
- Estimate per‑expert expected tokens.
- Target a uniform per‑row load.
- Assign replicas to keep rows near the target.

