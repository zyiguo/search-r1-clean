"""Evaluate frozen SFT/GRPO weights on a separate validation bundle; retain training identity checks."""
import argparse
from contextlib import nullcontext
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import path, read_json, write_json, sha256
from search_task import load_bundle
import search_train as train
from scripts.eval_grpo_checkpoint import checkpoint_adapter


def verify_dataset(original, identity, files, expanded_identity):
    m = expanded_identity['manifest']
    if m.get('evaluation_only') is not True or m.get('source_training_identity') != identity:
        raise ValueError('Evaluation dataset must derive from the exact training bundle')
    if any(expanded_identity['sha256'][k] != identity['sha256'][k] for k in ('train','test')):
        raise ValueError('Evaluation bundle changed original train/test')
    n = len(original['val'])
    if (files['val'][:n] != original['val'] or len(files['val']) != m['validation_size']
            or m['original_val_ids'] != [r['id'] for r in original['val']]):
        raise ValueError('Original validation subset or target size changed')
    docs = {d['id']: d for d in files['corpus']}
    if any(docs.get(d['id']) != d for d in original['corpus']):
        raise ValueError('Evaluation corpus must preserve all original documents')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--eval-data', default='search/eval-val300')
    p.add_argument('--variant', choices=['B1','B3'], required=True)
    p.add_argument('--checkpoint-step', type=int)
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if a.eval_batch_size < 1 or (a.variant == 'B3' and (a.checkpoint_step is None or a.checkpoint_step < 1)):
        p.error('Positive batch size required; B3 requires a positive --checkpoint-step')
    if a.variant == 'B1' and a.checkpoint_step is not None:
        p.error('B1 does not use a GRPO checkpoint')
    c = read_json(a.config)
    if c.get('grpo_start') != 'sft_adapter':
        raise ValueError('Only original-base SFT / unmerged GRPO supported')
    original, training_identity = load_bundle(path(c.get('grpo_data_dir', c['data_dir'])))
    files, evaluation_identity = load_bundle(path(a.eval_data))
    verify_dataset(original, training_identity, files, evaluation_identity)
    if evaluation_identity['manifest'].get('demo'):
        raise ValueError('Formal evaluation cannot use demo data')
    train.verify_sft_source(c, training_identity)
    checkpoint = None
    current_identity = train.identity(c, training_identity, False)
    if a.variant == 'B3':
        checkpoint = path(c['grpo_output'])/f'checkpoint-{a.checkpoint_step}'
        train.verify_checkpoint(checkpoint, path(c['grpo_output']))
        if read_json(checkpoint/'run-identity.json') != current_identity:
            raise ValueError('Original training identity changed; do not overwrite root training code or config')
    output = path(a.output).resolve()
    protected = [path(c['sft_output']).resolve(), path(c['grpo_output']).resolve(), path(a.eval_data).resolve(),
                 path(c['data_dir']).resolve(), path(c.get('grpo_data_dir',c['data_dir'])).resolve()]
    if any(output.is_relative_to(directory) for directory in protected):
        raise ValueError('Write reports outside datasets and training output directories')
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix='validation-report-') as d:
        args = SimpleNamespace(variant=a.variant, mode='adaptive', split='val',
                               output=str(Path(d)/'report.json'), eval_batch_size=a.eval_batch_size)
        redirect = checkpoint_adapter(train, checkpoint, c['grpo_output']) if checkpoint else nullcontext()
        # Validate adapter against ORIGINAL training data, execute only the supplied eval files.
        with redirect:
            train.evaluate(c, files, training_identity, args)
        report = read_json(args.output)
    report['training_identity'] = report['identity']
    report['identity'] = copy.deepcopy(report['identity'])
    report['identity']['data'] = evaluation_identity
    report['evaluation_data_identity'] = evaluation_identity
    report['evaluation_protocol'] = 'expanded_validation_only; training identity checked against original bundle'
    report['evaluation_launcher_sha256'] = sha256(Path(__file__))
    if checkpoint:
        report['evaluated_checkpoint'] = {'path': str(checkpoint), 'step': a.checkpoint_step,
                                          'manifest_sha256': sha256(checkpoint/'checkpoint-manifest.json')}
    old_ids = set(evaluation_identity['manifest']['original_val_ids'])
    report['subsets'] = {}
    for name, rows in [('original', [r for r in report['rows'] if r['id'] in old_ids]),
                       ('new', [r for r in report['rows'] if r['id'] not in old_ids])]:
        report['subsets'][name] = dict(questions=len(rows), exact_match=sum(r['correct'] for r in rows)/len(rows) if rows else None)
    write_json(output, report)
    print({'EM': report['exact_match'], 'subsets': report['subsets'], 'report': str(output)}, flush=True)


if __name__ == '__main__':
    main()
