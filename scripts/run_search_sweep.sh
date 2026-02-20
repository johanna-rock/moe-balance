#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"

SYSTEM_CONFIG="${1:-configs/system/system_example.jsonc}"
INIT_CONFIG="${2:-}"

if [[ -n "${INIT_CONFIG}" ]]; then
  INIT_CONFIGS=("${INIT_CONFIG}")
else
  INIT_CONFIGS=("${ROOT_DIR}"/configs/initial_placement/*.jsonc)
fi

SEARCH_CONFIG="configs/search/none_example.jsonc"

echo "System config: ${SYSTEM_CONFIG}"
if [[ -n "${INIT_CONFIG}" ]]; then
  echo "Initial placement config: ${INIT_CONFIG}"
else
  echo "Initial placement configs: ${#INIT_CONFIGS[@]}"
fi
echo "Search config: ${SEARCH_CONFIG}"
echo ""

run_search () {
  local init_cfg="$1"
  echo "==> ${SEARCH_CONFIG} (init: $(basename "${init_cfg}"))"
  ${PY} "${ROOT_DIR}/sim/search.py" \
    --system-config "${SYSTEM_CONFIG}" \
    --search-config "${SEARCH_CONFIG}" \
    --initial-placement-config "${init_cfg}"
  echo ""
}

for init_cfg in "${INIT_CONFIGS[@]}"; do
  run_search "${init_cfg}"
done
