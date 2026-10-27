#!/usr/bin/env bash
# Run one training phase sequentially for every model config in configs/.
#
# Usage:
#   ./run_all.sh {pretrain|fulltraining} [filters...] [--fail-fast] [extra train flags]
#
# Examples:
#   ./run_all.sh pretrain                  # all models, in alphabetical order
#   ./run_all.sh fulltraining SwinNO FNO   # only model_*.json matching a substring
#   ./run_all.sh pretrain --fail-fast      # stop at the first failure (default: continue)
#   ./run_all.sh fulltraining --device cpu # extra flags passed to pynop.training.train

if [ "$#" -lt 1 ] || [[ "$1" == --* ]]; then
    echo "Usage: $0 {pretrain|fulltraining} [filters...] [--fail-fast] [extra train flags]" >&2
    exit 2
fi

PHASE="$1"; shift
case "$PHASE" in
    pretrain) PHASE_FLAG=(--pretrain) ;;
    fulltraining) PHASE_FLAG=() ;;
    *) echo "Unknown phase: $PHASE (expected 'pretrain' or 'fulltraining')" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$ROOT/configs"
TRAINING_CONFIG="$CONFIG_DIR/training_PDEBench.json"
PYTHON="${PYTHON:-python}"

FAIL_FAST=0
FILTERS=()
EXTRA_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --fail-fast) FAIL_FAST=1 ;;
        --*) EXTRA_ARGS+=("$arg") ;;
        *) FILTERS+=("$arg") ;;
    esac
done

ok=()
failed=()
total=0
for cfg in "$CONFIG_DIR"/model_*.json; do
    name="$(basename "$cfg" .json)"
    if [ "${#FILTERS[@]}" -gt 0 ]; then
        match=0
        for f in "${FILTERS[@]}"; do
            case "$name" in *"$f"*) match=1; break ;; esac
        done
        [ "$match" -eq 1 ] || continue
    fi
    total=$((total + 1))
    echo "=============================================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ($total) $name -> $cfg"
    echo "=============================================================="
    if "$PYTHON" -m pynop.training.train "$cfg" "$TRAINING_CONFIG" \
            "${PHASE_FLAG[@]}" "${EXTRA_ARGS[@]}"; then
        ok+=("$name")
    else
        failed+=("$name")
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $name" >&2
        [ "$FAIL_FAST" -eq 1 ] && break
    fi
done

echo
echo "==================== SUMMARY ===================="
echo "OK (${#ok[@]}):     ${ok[*]:-none}"
echo "FAILED (${#failed[@]}): ${failed[*]:-none}"
[ "${#failed[@]}" -eq 0 ]
