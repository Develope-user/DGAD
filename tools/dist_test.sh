#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?Usage: tools/dist_test.sh CONFIG CHECKPOINT GPUS [TEST_ARGS...]}
CHECKPOINT=${2:?Usage: tools/dist_test.sh CONFIG CHECKPOINT GPUS [TEST_ARGS...]}
GPUS=${3:?Usage: tools/dist_test.sh CONFIG CHECKPOINT GPUS [TEST_ARGS...]}
shift 3

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON:-python}
PORT=${PORT:-29503}

PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}" "$PYTHON_BIN" \
  -m torch.distributed.launch \
  --nproc_per_node="$GPUS" \
  --master_port="$PORT" \
  "$ROOT_DIR/tools/test.py" \
  "$CONFIG" \
  "$CHECKPOINT" \
  --launcher pytorch \
  "$@"
