#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env_6000d.sh"
mkdir -p reports/logs
exec > >(tee -a reports/logs/setup-6000d.log) 2>&1
python3 -c 'import shutil; from pathlib import Path; n=shutil.disk_usage(Path.cwd()).free; print("Free project disk GiB:", n/1024**3); assert n >= 60*1024**3, "Expand data disk: setup requires at least 60 GiB free"'
df -h .
python3 -c 'import sys; assert sys.version_info[:2] in ((3,11),(3,12)), "Use Python 3.11 or 3.12"'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.2
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-train.txt
python -m pip check
python -m pip freeze > reports/pip-freeze.txt
python hardware_probe.py
