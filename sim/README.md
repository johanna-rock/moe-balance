# Minimal MoE placement simulator

Quick start:

1) Generate a synthetic trace:
```bash
python3 sim/generate_trace.py --out /tmp/trace.jsonl --layers 1 --batches 50 --tokens-per-batch 1024 --rows 16 --experts 256 --k 8
```

2) Run a simple random search:
```bash
python3 sim/search.py \
  --system-config configs/system/system_example.jsonc \
  --search-config configs/search/random_example.jsonc \
  --initial-placement-config configs/initial_placement/row_balance_example.jsonc
```

3) Run balance_load_coactivation placement + local search:
```bash
python3 sim/search.py \
  --system-config configs/system/system_example.jsonc \
  --search-config configs/search/local_example.jsonc \
  --initial-placement-config configs/initial_placement/row_balance_load_coactivation_example.jsonc
```

4) Run balance_load_coactivation placement + simulated annealing:
```bash
python3 sim/search.py \
  --system-config configs/system/system_example.jsonc \
  --search-config configs/search/anneal_example.jsonc \
  --initial-placement-config configs/initial_placement/row_balance_load_coactivation_example.jsonc
```

5) Write a co-activation CSV (experts x experts):
```bash
python3 sim/search.py --trace /tmp/trace.jsonl --layer 0 --coact-csv /tmp/coact.csv
```

6) Plot co-activation CSV to a heatmap PNG:
```bash
python3 plots/plot_coact.py --csv /tmp/coact.csv --out /tmp/coact.png
```

Convert NPZ routing data to trace JSONL:
```bash
# Full trace with per-row duplication (used for placement/scheduling sims)
python3 sim/convert_npz_to_trace.py \
  --data-dir data/math-ai_aime25 \
  --out data/math-ai_aime25/processed/moe_trace.jsonl

# Smaller trace for analysis (no origin rows / no duplication)
python3 sim/convert_npz_to_trace.py \
  --data-dir data/math-ai_aime25 \
  --out data/math-ai_aime25/processed/trace_no_origin_rows.jsonl

# Limit layers to keep output small
python3 sim/convert_npz_to_trace.py \
  --data-dir data/math-ai_aime25 \
  --out data/math-ai_aime25/processed/moe_trace_layer0.jsonl \
  --layers 0
```

Trace format: see `sim/trace_format.md`.

Notes:
- The cost model is intentionally simple: max compute + max row comm + a queueing penalty.
- Replica routing uses least-loaded replica per batch; replace with other policies as needed.
