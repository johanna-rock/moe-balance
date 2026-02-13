# Focused MoE Cost Model (Balancing-Only)

This model captures **only the components affected by expert replication and placement**:

- **A2A Dispatch (DP rows)**
- **Expert Compute**
- **A2A Combine (DP rows)**

Everything else is treated as constant and ignored for balancing.

---

## 1) A2A Dispatch (DP rows)

Dispatch uses the optimized sparse‑multicast ring. For each token we compute:

- `E_col`: maximum number of **distinct experts** active in any column.
- `avg_hop`: average hop distance (rows) for the token.
- `max_hop`: maximum hop distance (rows) for the token.

We model dispatch time (µs) as:

```
T_dispatch = A + B*E_col + C*avg_hop + D*max_hop
```

The coefficients `(A,B,C,D)` are **calibrated** from the hardware table for best/avg/worst hop conditions.
The calibration is performed in `sim/cost_model.py::calibrate_dispatch_coeffs`.

---

## 2) Expert Compute

Compute is per device and **sums per‑expert stepwise costs**:

```
T_expert_device = Σ_e ( 235 + 130 * floor((t_e - 1)/32) )
```

Where `t_e` is the number of tokens sent to expert `e` on that device.

We use:

```
T_compute = max_device(T_expert_device)
```

This reflects that the slowest device dominates latency.

---

## 3) A2A Combine (DP rows)

Combine is p2p with **per‑device reduction** (one return per device). For each token:

- `U`: number of **distinct devices** that received experts.
- `avg_hop`, `max_hop` as above.

We model combine time (µs) using the same calibrated dispatch coefficients:

```
T_combine = A + B*U + C*avg_hop + D*max_hop
```

(We will update coefficients once a combine‑specific calibration table is available.)

---

## Balance‑Sensitive Objective

The balancing‑only latency proxy is:

```
T_balance = T_dispatch + T_compute + T_combine
```

This is the part of the model directly influenced by replication and placement.
