# Search Configs

## Search Types

**random**  
Evaluates a series of random placements and keeps the best. Useful as a baseline and for sanity‑checking the objective; scales linearly with the number of iterations.

**row-balance**  
Greedy row‑first placement that balances replicas across rows (and then columns). Fast deterministic baseline focused on reducing row‑crossing traffic.

**row-aware**  
Starts from a row‑partition that clusters co‑activated experts, then applies local swap search to improve the objective. Good when co‑activation structure matters.

**hot-tier**  
Places the most‑used experts in a “hot tier” (few rows) and the rest elsewhere. Useful when a small set of experts dominate traffic.

**anneal**  
Simulated annealing over swap moves. Allows temporary worse moves to escape local minima; control with `anneal_t0`, `anneal_t1`, and `anneal_iters`.

**hybrid**  
Row‑balance initialization followed by local search, annealing, and a final local search refinement. A pragmatic default for better local optima.
