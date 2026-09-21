"""Derive a fresh SFT run from verified teacher data; no model loading."""
import argparse
import copy
import json
import math
from common import path, read_json
from search_task import Retriever, load_bundle, sft_steps
from search_train import split_sft_actions


def make_sft_config(source, destination, data_dir=None, rank=32, epochs=3, micro_batch=16):
    if any(type(x) is not int or x < 1 for x in (rank, epochs, micro_batch)):
        raise ValueError('rank, epochs and micro_batch must be positive integers')
    c = copy.deepcopy(read_json(source))
    # A fresh SFT run must not inherit RL-only data, warm-start declarations or an old plan.
    for key in list(c):
        if key.startswith('grpo_') and key not in ('grpo_steps', 'grpo_lr', 'grpo_output'):
            c.pop(key)
    c.pop('sft_plan', None)
    if data_dir is not None:
        c['data_dir'] = str(data_dir)
    c['lora'].update(r=rank, lora_alpha=2*rank, lora_dropout=0.0)
    c.update(max_context=2048, max_action_tokens=512, max_searches=3,
             sft_micro_batch=micro_batch, sft_accumulation=1,
             sft_gradient_checkpointing=True, sft_eval_steps=5, sft_save_total_limit=2)
    files, data_identity = load_bundle(path(c['data_dir']))
    retriever = Retriever(files['corpus'], c['document_chars'])
    actions = []
    for row in files['train']:
        steps = sft_steps(row, retriever, c['max_searches'], c['top_k'])
        actions.extend({'question_id': row['id']} for _ in steps)
    training, validation, report = split_sft_actions(actions, c.get('sft_val_ratio', .1), c['seed'])
    per_epoch = math.ceil(len(training) / micro_batch)
    c['sft_steps'] = epochs * per_epoch
    c['sft_eval_steps'] = min(5, c['sft_steps'])
    tag = f'r{rank}-e{epochs}-b{micro_batch}-ctx2048'
    c.update(sft_output='outputs/search-sft-' + tag,
             grpo_output='outputs/search-grpo-' + tag,
             merged_model_path='models/search-sft-' + tag + '-merged')
    c['sft_plan'] = dict(epochs=epochs, steps_per_epoch=per_epoch, total_steps=c['sft_steps'],
                         split=report, data_identity=data_identity,
                         note='Single GPU, accumulation1; best held-out action-loss checkpoint may precede the final epoch.')
    for key in ('sft_output', 'grpo_output'):
        directory = path(c[key])
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('Fresh run output already contains files: ' + str(directory))
    destination = path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(c, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='search/pilot-sft-3epoch-b16.json')
    p.add_argument('--output', default='search/sft-r32-e3-b16-ctx2048.json')
    p.add_argument('--data-dir')
    p.add_argument('--rank', type=int, default=32)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--micro-batch', type=int, default=16)
    a = p.parse_args()
    c = make_sft_config(a.source, a.output, a.data_dir, a.rank, a.epochs, a.micro_batch)
    print(json.dumps({'config': str(path(a.output)), 'sft_output': c['sft_output'],
                      'lora': c['lora'], 'sft_steps': c['sft_steps'], 'plan': c['sft_plan']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
