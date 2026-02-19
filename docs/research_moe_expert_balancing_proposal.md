# MoE Expert Balancing Research Proposal

## Scope
- Target: DeepSeek V3/R1-style MoE deployment with expert replication and placement on a `16x8` cluster.
- Levers:  
  1. Replicate hot experts (including shared expert).  
  2. Place expert replicas to reduce row/node traffic and avoid compute stragglers.
- Objective: low tail latency (`p95/p99`) under changing request mixes.

## Runtime Categories
- `online` (sub-ms to ~10 ms): per-batch/per-step routing decisions only.
- `nearline` (~10 ms to seconds): periodic rebalancing during serving.
- `offline-heavy` (minutes to hours): global search for high-quality baseline placement plans.

## Methods Catalogue (Replication + Placement)
| Method | Core idea | Replication | Placement | Runtime class | Best use |
|---|---|---|---|---|---|
| DeepSeek V3 auxiliary-loss-free balancing | Per-expert routing bias updated from batch load stats; avoids heavy aux-loss coupling | Indirect | Indirect | online | Production routing stability |
| DeepSeek V2/V3 limited routing | Constrain token fanout to limited devices/nodes to cap communication | No | Constrains effective placement | online | Production comm control |
| EPLB (DeepSeek) hierarchical/global balancing | Replicate heavy experts, then greedily pack for load balance; hierarchical variant preserves group locality | Yes (primary) | Yes (primary) | nearline | Production periodic rebalance |
| Load-aware replica routing | Pick least-loaded (or same-row) replica instead of round-robin | Uses existing replicas | No (runtime only) | online | Production tail reduction |
| GShard/Switch capacity + auxiliary loss | Capacity factors + auxiliary loss to reduce overload/drop | Optional | No | online/training-time | Router baseline/ablation |
| BASE layers (linear assignment) | Global balanced token-to-expert assignment (equalized load by construction) | No | No | nearline/offline-heavy | Oracle/reference for balancing quality |
| Expert Choice routing | Experts choose top tokens (fixed bucket), giving near-perfect load balance | No | No | online (with implementation work) | Alternative router family |
| MegaBlocks dropless execution | Avoid token dropping/padding tradeoff via sparse kernels | No | No | online kernel-level | Improves robustness when load spikes |
| Topology-aware ILP placement | Solve placement as ILP minimizing expected transmissions over topology | Yes (if modeled) | Yes (primary) | offline-heavy | High-quality initial placement |
| Trace-driven metaheuristics (SA/Tabu/LNS) | Optimize trace-derived latency proxy under capacity/topology constraints | Yes | Yes | offline-heavy / nearline (small neighborhoods) | Best global plans + warm-start updates |

## Best Ideas For Your 16x8 System

### 1) EPLB-style rebalance + row-aware routing (highest practical value)
- Use estimated per-expert load (moving average over serving window).
- Recompute replica counts + placement every `N` steps (nearline).
- During dispatch, prefer same-row least-loaded replica first, then spill cross-row.
- Why high potential: aligns with DeepSeek deployment strategy and is computationally feasible in production.

### 2) Two-stage offline planner with trace objective (highest quality)
- Stage A (replication): allocate extra slots by minimizing max expected device load.
- Stage B (placement): graph/hypergraph partition or ILP warm-start, then local search (swap/move).
- Objective: your `T_dispatch + T_compute + T_combine` proxy on recorded traces.
- Why high potential: directly optimizes your simulator objective and hardware topology.

### 3) ILP/CP-SAT oracle for nightly planning + heuristic distillation
- Run expensive solver nightly on sampled traces to produce “gold” plans.
- Distill solver behavior into fast heuristics for daytime nearline updates.
- Why high potential: provides a defensible upper bound and improves heuristic design over time.

## Suggested Initial Experiment Matrix
1. Baseline: current heuristic + round-robin replicas.
2. Nearline: EPLB-style replication/packing + least-loaded replica routing.
3. Offline: two-stage planner (replicate -> placement local search).
4. Oracle: small-scale ILP (subset of experts/devices) to benchmark search quality.

Metrics:
- `p50/p95/p99` MoE latency proxy.
- max device compute load and row imbalance.
- cross-row/cross-node traffic proxy.
- rebalance overhead (time + frequency).

## Notes Specific to DeepSeek V3/R1
- Shared expert should be treated as a persistent hot expert in replication budgeting.
- Node-/device-limited routing should be modeled as hard constraints in placement search.
- Keep a strict separation between:  
  1. fast online routing controls, and  
  2. slower replica/placement optimization loops.

## Primary Sources
- DeepSeek-V3 Technical Report (arXiv, v2, Feb 18, 2025): https://arxiv.org/abs/2412.19437  
- DeepSeek-V2 Technical Report (arXiv, v5, Jun 19, 2024): https://arxiv.org/abs/2405.04434  
- DeepSeek EPLB repo: https://github.com/deepseek-ai/EPLB  
- Switch Transformers (arXiv): https://arxiv.org/abs/2101.03961  
- GShard (arXiv): https://arxiv.org/abs/2006.16668  
- Sparsely-Gated MoE (arXiv): https://arxiv.org/abs/1701.06538  
- BASE Layers (arXiv): https://arxiv.org/abs/2103.16716  
- Expert Choice Routing (arXiv): https://arxiv.org/abs/2202.09368  
- MegaBlocks (arXiv): https://arxiv.org/abs/2211.15841  
- Topology-driven ILP placement for MoE inference (arXiv): https://arxiv.org/abs/2508.09229
