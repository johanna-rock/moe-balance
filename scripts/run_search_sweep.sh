#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

SYSTEM_CONFIG="${1:-configs/system/system_example.jsonc}"
DEFAULT_SMALL="/Users/jrock/repos/moe-balancing/configs/system/system_example_small_seq_len.jsonc"
SYSTEM_CONFIG_SMALL="${2:-$DEFAULT_SMALL}"
INIT_CONFIG="${3:-}"
RUNS_RANDOM="${4:-3}"
RUNS_HYBRID="${5:-3}"

# Backward-compatible arg handling: if arg2 looks like an initial placement config,
# shift parameters and use default small system config.
if [[ -n "${2:-}" && -f "${2:-}" && "${2:-}" == *initial_placement* ]]; then
  SYSTEM_CONFIG_SMALL="$DEFAULT_SMALL"
  INIT_CONFIG="${2:-}"
  RUNS_RANDOM="${3:-3}"
  RUNS_HYBRID="${4:-3}"
fi

EXTRA_INIT=()
if [[ -n "${INIT_CONFIG}" ]]; then
  EXTRA_INIT=(--initial-placement-config "${INIT_CONFIG}")
fi

echo "System config: ${SYSTEM_CONFIG}"
echo "Hybrid system config: ${SYSTEM_CONFIG_SMALL}"
if [[ -n "${INIT_CONFIG}" ]]; then
  echo "Initial placement config: ${INIT_CONFIG}"
else
  echo "Initial placement config: (none)"
fi
echo "Random runs: ${RUNS_RANDOM}"
echo "Hybrid runs: ${RUNS_HYBRID}"
echo ""

run_search () {
  local search_cfg="$1"
  local system_cfg="$2"
  echo "==> ${search_cfg}"
  ${PY} "${ROOT_DIR}/sim/search.py" \
    --system-config "${system_cfg}" \
    --search-config "${search_cfg}" \
    "${EXTRA_INIT[@]}"
  echo ""
}

# # Multiple random runs
# for i in $(seq 1 "${RUNS_RANDOM}"); do
#   run_search "configs/search/random_example.jsonc" "${SYSTEM_CONFIG}"
# done

# # Multiple hybrid runs
# for i in $(seq 1 "${RUNS_HYBRID}"); do
#   run_search "configs/search/hybrid_example.jsonc" "${SYSTEM_CONFIG_SMALL}"
# done

# Single runs for other search types
# run_search "configs/search/row_balance_example.jsonc" "${SYSTEM_CONFIG}"
# run_search "configs/search/row_aware_example.jsonc" "${SYSTEM_CONFIG}"
run_search "configs/search/hot_tier_example.jsonc" "${SYSTEM_CONFIG}"
run_search "configs/search/anneal_example.jsonc" "${SYSTEM_CONFIG}"
