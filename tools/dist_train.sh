#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: tools/dist_train.sh CONFIG GPUS [TRAIN_ARGS...]}
GPUS=${2:?Usage: tools/dist_train.sh CONFIG GPUS [TRAIN_ARGS...]}
shift 2

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
PORT=${PORT:-28509}

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" \
  -m torch.distributed.launch \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  "$ROOT_DIR/tools/train.py" \
  "$CONFIG" \
  --launcher pytorch \
  "$@"
