"""Local-safe checks for the search pipeline. Never imports a model or downloads weights."""
import ast
import os
import subprocess
import sys
from datetime import datetime, timezone
from common import ROOT, read_json, write_json


def main():
    checked = []
    for p in list(ROOT.glob('*.py')) + list((ROOT / 'tests').glob('*.py')):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
        checked.append(p.relative_to(ROOT).as_posix())
    for name in ('search/pilot.json', 'search/config.json', 'search/demo/manifest.json'):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(name)
        read_json(name)
    env = {**os.environ, 'PYTHONUTF8': '1'}
    commands = [
        [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
        [sys.executable, 'search_train.py', 'validate'],
        [sys.executable, 'search_train.py', '--help'],
        [sys.executable, 'prepare_search_data.py', '--help'],
        [sys.executable, 'prepare_grpo_experiment.py', '--help'],
        [sys.executable, 'prepare_sft_experiment.py', '--help'],
        [sys.executable, 'expand_search_data.py', '--help'],
        [sys.executable, 'estimate_search_resources.py', '--help'],
        [sys.executable, 'scripts/eval_grpo_checkpoint.py', '--help'],
        [sys.executable, 'diagnose_signal.py', '--help'],
        [sys.executable, 'search_compare.py', '--help'],
        [sys.executable, 'cloud.py', '--help'],
        [sys.executable, 'hardware_probe.py', '--help'],
    ]
    records = []
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        record = {'command': command[1:], 'returncode': result.returncode}
        if 'unittest' in command or result.returncode:
            record['stdout'], record['stderr'] = result.stdout, result.stderr
        records.append(record)
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {' '.join(command[1:])}")
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version,
        'status': 'passed' if all(r['returncode'] == 0 for r in records) else 'failed',
        'syntax_checked': checked,
        'checks': records,
        'not_verified': [
            'Linux dependency installation',
            'CUDA and bitsandbytes',
            'real tokenizer lengths',
            'SFT/GRPO training',
            'checkpoint resume',
            'model reload and metrics',
        ],
        'hotel_legacy_removed': True,
    }
    write_json('reports/local-verification.json', report)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
