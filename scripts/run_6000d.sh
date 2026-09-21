#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_6000d.sh"
source .venv/bin/activate
stage="${1:-help}"
case "$stage" in
  probe) python hardware_probe.py ;;
  tensorboard) bash scripts/tensorboard.sh "${2:-6006}" ;;
  *) bash scripts/run_search.sh "$stage" "${2:-search/pilot.json}" ;;
esac
