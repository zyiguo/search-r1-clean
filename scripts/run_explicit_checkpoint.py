"""Resume an existing GRPO run with its implicit reentrant=True made explicit.

Does not edit experiment identity or suppress warnings. Records the launcher separately.
"""
import functools
import hashlib
import json
from pathlib import Path
import sys
import time


def explicit_checkpoint_prepare(original):
    @functools.wraps(original)
    def prepare(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
        options = dict(gradient_checkpointing_kwargs or {})
        options.setdefault('use_reentrant', True)
        return original(model, use_gradient_checkpointing=use_gradient_checkpointing,
                        gradient_checkpointing_kwargs=options)
    return prepare


def main():
    args = sys.argv[1:]
    if not args or args[0] != 'grpo' or '--resume' not in args:
        raise SystemExit('Usage: python scripts/run_explicit_checkpoint.py grpo <original arguments> --resume <checkpoint>')
    index = args.index('--resume')
    if index + 1 >= len(args):
        raise SystemExit('--resume requires a checkpoint directory')
    checkpoint = Path(args[index+1]).resolve()
    if not (checkpoint / 'checkpoint-manifest.json').is_file():
        raise SystemExit('Expected a complete GRPO checkpoint with checkpoint-manifest.json')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import peft
    import torch
    import search_train
    original = peft.prepare_model_for_kbit_training
    record = {'mode': 'explicit use_reentrant=True; preserves the prior implicit default',
              'checkpoint': str(checkpoint), 'arguments': args,
              'python': sys.executable, 'torch': torch.__version__, 'peft': peft.__version__,
              'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    audit = checkpoint.parent / f'checkpoint-launch-{time.time_ns()}.json'
    with audit.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(record, f, indent=2)
        f.write('\n')
    print(f'Explicit gradient checkpoint mode: use_reentrant=True; audit: {audit}', flush=True)
    peft.prepare_model_for_kbit_training = explicit_checkpoint_prepare(original)
    try:
        search_train.main()
    finally:
        peft.prepare_model_for_kbit_training = original


if __name__ == '__main__':
    main()
