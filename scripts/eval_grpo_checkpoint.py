"""Evaluate a verified GRPO checkpoint on val without editing training code or adapters."""
import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace


@contextmanager
def checkpoint_adapter(train_module, checkpoint, configured_output):
    """Route only the configured final adapter lookup to a checkpoint in that run."""
    original = train_module.path
    checkpoint = original(checkpoint).resolve()
    output = original(configured_output).resolve()
    if checkpoint.parent != output:
        raise ValueError('Checkpoint must be a direct child of the configured GRPO output')
    expected = output / 'adapter'
    def resolve(value):
        p = original(value)
        return checkpoint if p.resolve() == expected else p
    train_module.path = resolve
    try:
        yield
    finally:
        train_module.path = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--eval-batch-size', type=int, default=8)
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import search_train as train
    from common import path, read_json, write_json, sha256
    from search_task import load_bundle
    cp = path(args.checkpoint).resolve()
    output = path(args.output).resolve()
    if output.is_relative_to(cp):
        raise ValueError('Evaluation report must be outside the checkpoint')
    if output.exists():
        raise FileExistsError(output)
    train.verify_checkpoint(cp, cp.parent)
    saved = read_json(cp / 'run-identity.json')
    config = saved['config']
    if path(config['grpo_output']).resolve() != cp.parent:
        raise ValueError('Checkpoint directory differs from recorded GRPO run')
    files, data_identity = load_bundle(path(config.get('grpo_data_dir', config['data_dir'])))
    if data_identity['manifest'].get('demo'):
        raise ValueError('This checkpoint evaluator is for formal validation data')
    merged = config.get('grpo_start') != 'sft_adapter'
    if train.identity(config, data_identity, merged) != saved:
        raise ValueError('Current code/config/data/model identity differs from checkpoint')
    if merged:
        raise ValueError('This launcher currently supports unmerged SFT GRPO checkpoints only')
    evaluation = SimpleNamespace(variant='B3', mode='adaptive', split='val', output=str(output), eval_batch_size=args.eval_batch_size)
    print(f'Evaluating {cp.name} on val; training state is unchanged.', flush=True)
    with checkpoint_adapter(train, cp, config['grpo_output']):
        train.evaluate(config, files, data_identity, evaluation)
    report = read_json(output)
    report['evaluated_checkpoint'] = {
        'path': str(cp), 'name': cp.name,
        'manifest_sha256': sha256(cp / 'checkpoint-manifest.json'),
        'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(output, report)
    print(f"EM: {report['exact_match']}; report: {output}", flush=True)


if __name__ == '__main__':
    main()
