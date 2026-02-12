# Focused MoE Cost Model (Balancing-Only)

This model captures **only the components affected by expert replication and placement**:

- **A2A Dispatch (DP rows)**
- **Expert Compute**
- **A2A Combine (DP rows)**

Everything else (gate, all‑gather, reduce‑scatter, etc.) is treated as constant and ignored for balancing.

---

## 1) A2A Dispatch (DP rows)

Dispatch cost is driven by the amount of data that must cross **rows** to reach the chosen expert replicas.

```
T_dispatch = bytes_dispatched_across_rows / BW_row
```

Placement and replication reduce this term when replicas are co‑located on the same row as origin tokens.

---

## 2) Expert Compute (Dense)

Compute cost is dominated by the **most loaded device** (or expert group) after routing, replication, and placement.

Let `t_e` be the number of tokens assigned to expert replica `e` on a device. Then:

```
T_expert = max_device ( Σ_e t_e * cost_ffn / F_matmul )
```

In practice:

```
T_expert ∝ max_device ( total_tokens_on_device )
```

Replication reduces hot‑expert load; placement controls where those loads land.

---

## 3) A2A Combine (DP rows)

Combine cost is again driven by row‑crossing traffic (returning expert outputs to token‑origin rows).

```
T_combine = bytes_returned_across_rows / BW_row
```

This is typically similar in shape to `T_dispatch`, and benefits from the same locality.

---

## Balance‑Sensitive Objective

The balancing‑only latency proxy is:

```
T_balance = T_dispatch + T_expert + T_combine
```

Where:
- The **comm term** is determined by row‑crossing traffic.
- The **compute term** is determined by the **max per‑device load**.

This is the part of the model that replication/placement directly optimizes.

