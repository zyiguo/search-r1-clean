"""Derive a multi-group GRPO config from the actual cloud SFT config; no model execution."""
import argparse

from common import path, read_json
from grpo_sampling import sampling_options
import json


def make_config(source, destination, steps=100, groups=4, attempts=16, question_batch_size=1, grpo_data_dir=None, profile=None, checkpoint_every=None):
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer')
    c = read_json(source)
    if profile not in (None, 'rtx6000-80gb'):
        raise ValueError('Unknown GRPO profile')
    if profile == 'rtx6000-80gb':
        groups, attempts, question_batch_size = 48, 512, 16
        overrides = dict(max_action_tokens=512, max_context=2048, max_searches=3,
                         grpo_lr=1e-6, beta=0.001)
        c['grpo_runtime_overrides'] = {k: {'source': c[k], 'value': v} for k, v in overrides.items()}
        c.update(overrides)
    c.update(grpo_steps=steps, grpo_groups_per_update=groups,
             grpo_max_group_attempts=attempts, grpo_filter_zero_variance=True,
             grpo_evidence_weight=0.1, grpo_protocol_weight=0.02,
             grpo_output=c['grpo_output'] + f'-multi{groups}-a{attempts}-n{steps}-aux')
    if question_batch_size < 1:
        raise ValueError('question_batch_size must be positive')
    if question_batch_size > 1 or grpo_data_dir:
        c.update(grpo_question_batch_size=question_batch_size, grpo_question_sampling='epoch')
        c.update(grpo_start='sft_adapter', group_size=5 if profile else 8,
                 rollout_batch_size=min(16, question_batch_size * (5 if profile else 8)),
                 logprob_batch_size=2 if profile else 4)
        c['grpo_output'] += f'-crossq{question_batch_size}-g{c["group_size"]}-r{c["rollout_batch_size"]}-batch{c["logprob_batch_size"]}-unmerged'
    if grpo_data_dir:
        c['grpo_data_dir'] = grpo_data_dir
        c['grpo_output'] += '-expanded'
    if checkpoint_every is not None or profile:
        every = checkpoint_every if checkpoint_every is not None else 5
        if type(every) is not int or every < 1:
            raise ValueError('checkpoint_every must be a positive integer')
        c['grpo_checkpoint_every'] = every
    if profile:
        c['grpo_compress_rollouts'] = True
        c['grpo_output'] += '-ctx2048-act512-s3-lr1e6-kl001'
    sampling_options(c)
    target = path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(c, f, ensure_ascii=False, indent=2)
        f.write('\n')
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='search/pilot-sft-3epoch-b16.json')
    p.add_argument('--output', default='search/pilot-sft3epoch-grpo100-multi-aux.json')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--attempts', type=int, default=16)
    p.add_argument('--question-batch-size', type=int, default=1)
    p.add_argument('--grpo-data-dir')
    p.add_argument('--profile', choices=['rtx6000-80gb'],
                   help='48 target prompts, 512 attempts, G=5, 16 generation slots, action512/total-context2048, 3 searches, lr1e-6, KL0.001; overrides groups/attempts/question-batch-size')
    p.add_argument('--checkpoint-every', type=int, help='Save every N rounds and the final round; profile default5, legacy default1')
    a = p.parse_args()
    c = make_config(a.source, a.output, a.steps, a.groups, a.attempts, a.question_batch_size, a.grpo_data_dir, a.profile, a.checkpoint_every)
    print(json.dumps({'config': str(path(a.output)), 'sft_source': c['sft_output'],
                      'grpo_output': c['grpo_output'], 'rounds': c['grpo_steps'],
                      'target_prompt_batch': c['grpo_groups_per_update'], 'rollouts_per_question': c['group_size'],
                      'max_context': c['max_context'], 'max_action_tokens': c['max_action_tokens'], 'max_searches': c['max_searches'],
                      'learning_rate': c['grpo_lr'], 'kl_coefficient': c['beta']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
