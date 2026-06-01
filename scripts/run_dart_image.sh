#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$ROOT_DIR/data/test2014}"
LOG_DIR="${2:-$ROOT_DIR/logs/robust_inr}"

python "$ROOT_DIR/train_robust_inr.py" \
  --config "$ROOT_DIR/configs/config_robust_train_diff_epsilon.txt" \
  --test_dir "$DATA_DIR" \
  --base_log_dir "$LOG_DIR"
