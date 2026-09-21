#!/usr/bin/env bash
# Source from setup/run wrappers. Keep environment, models and caches on the data disk.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
: "${POSTTRAIN_DATA_ROOT:?Set POSTTRAIN_DATA_ROOT to the actual expanded data-disk mount directory}"
data_root="$(cd "$POSTTRAIN_DATA_ROOT" && pwd -P)"
if [[ "$data_root" == / ]]; then
  echo 'Choose the data disk mount, not /' >&2
  return 1
fi
case "$project_root/" in
  "$data_root/"*) ;;
  *) echo 'Extract the project under POSTTRAIN_DATA_ROOT before running setup.' >&2; return 1 ;;
esac
export HF_HOME="$project_root/.cache/huggingface"
export PIP_CACHE_DIR="$project_root/.cache/pip"
export TMPDIR="$project_root/.cache/tmp"
export TORCH_HOME="$project_root/.cache/torch"
export TRITON_CACHE_DIR="$project_root/.cache/triton"
export CUDA_CACHE_PATH="$project_root/.cache/cuda"
export POSTTRAIN_ALLOW_CLOUD=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
cd "$project_root"
