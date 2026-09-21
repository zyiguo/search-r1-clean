#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec tensorboard --logdir outputs --host 127.0.0.1 --port "${1:-6006}"
