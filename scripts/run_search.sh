#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_6000d.sh"
stage="${1:-help}"
config="${2:-search/pilot.json}"
mkdir -p reports/logs
exec > >(tee -a "reports/logs/search-${stage}-$(date +%Y%m%d-%H%M%S).log") 2>&1
export PYTHONUNBUFFERED=1
case "$stage" in
  prepare)
    python search_train.py validate --config "$config"
    python search_train.py download --config "$config"
    python search_train.py lengths --config "$config"
    python search_train.py eval --config "$config" --variant B0 --mode none
    python search_train.py eval --config "$config" --variant B0 --mode fixed
    python search_train.py eval --config "$config" --variant B0 --mode adaptive
    ;;
  sft)
    python search_train.py sft --config "$config"
    python search_train.py eval --config "$config" --variant B1
    python search_train.py merge --config "$config"
    python search_train.py eval --config "$config" --variant B1-init
    ;;
  grpo)
    python search_train.py grpo --config "$config"
    python search_train.py eval --config "$config" --variant B3
    ;;
  final)
    python search_train.py eval --config "$config" --variant B0 --mode none --split test --final-test
    python search_train.py eval --config "$config" --variant B0 --mode fixed --split test --final-test
    for variant in B0 B1 B1-init B3; do
      python search_train.py eval --config "$config" --variant "$variant" --split test --final-test
    done
    ;;
  smoke)
    for command in validate download sft merge grpo; do
      python search_train.py "$command" --config search/config.json --allow-demo
    done
    python search_train.py eval --config search/config.json --allow-demo --variant B3 --output reports/search/demo-B3.json
    ;;
  feedback) python cloud.py feedback ;;
  *) echo 'Usage: bash scripts/run_search.sh {prepare|sft|grpo|final|smoke|feedback} [config]'; exit 2 ;;
esac
