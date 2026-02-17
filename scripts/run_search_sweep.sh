#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

SYSTEM_CONFIG="${1:-configs/system/system_example.jsonc}"
INIT_CONFIG="${2:-}"
RUNS_RANDOM="${3:-3}"
RUNS_HYBRID="${4:-3}"

EXTRA_INIT=()
if [[ -n "${INIT_CONFIG}" ]]; then
  EXTRA_INIT=(--initial-placement-config "${INIT_CONFIG}")
fi

echo "System config: ${SYSTEM_CONFIG}"
if [[ -n "${INIT_CONFIG}" ]]; then
  echo "Initial placement config: ${INIT_CONFIG}"
fi
echo "Random runs: ${RUNS_RANDOM}"
echo "Hybrid runs: ${RUNS_HYBRID}"
echo ""

run_search () {
  local search_cfg="$1"
  echo "==> ${search_cfg}"
  ${PY} "${ROOT_DIR}/sim/search.py" \
    --system-config "${SYSTEM_CONFIG}" \
    --search-config "${search_cfg}" \
    "${EXTRA_INIT[@]}"
  echo ""
}

# Multiple random runs
for i in $(seq 1 "${RUNS_RANDOM}"); do
  run_search "configs/search/random_example.jsonc"
done

# Multiple hybrid runs
for i in $(seq 1 "${RUNS_HYBRID}"); do
  run_search "configs/search/hybrid_example.jsonc"
done

# Single runs for other search types
run_search "configs/search/row_balance_example.jsonc"
run_search "configs/search/row_aware_example.jsonc"
run_search "configs/search/hot_tier_example.jsonc"
run_search "configs/search/anneal_example.jsonc"
