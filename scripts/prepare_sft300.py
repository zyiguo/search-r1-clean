"""Select complete teacher trajectories without changing existing run identities."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import path, read_json, write_json, sha256
from search_task import load_bundle, normalize, Retriever, sft_steps
from search_train import split_sft_actions


def prepare(config, source, evaluation, output, destination, count=300, candidate_source=None):
    if type(count) is not int or count < 2:
        raise ValueError('At least two complete trajectories required')
    c = copy.deepcopy(read_json(config))
    source, evaluation, output, destination = map(path, (source, evaluation, output, destination))
    if output.exists() or destination.exists():
        raise FileExistsError('Dataset/config already exists; use new paths')
    teachers, teacher_identity = load_bundle(source)
    heldout, eval_identity = load_bundle(evaluation)
    if not eval_identity['manifest'].get('evaluation_only'):
        raise ValueError('Use the frozen evaluation-only validation bundle')
    for key in list(c):
        if key.startswith('grpo_') and key not in ('grpo_steps', 'grpo_lr', 'grpo_output'):
            c.pop(key)
    c.pop('sft_plan', None)
    c.update(data_dir=str(output), max_context=2048, max_action_tokens=512, max_searches=3,
             sft_micro_batch=16, sft_accumulation=1, sft_val_ratio=.1,
             sft_gradient_checkpointing=True, sft_save_total_limit=2, sft_load_best=True)
    c['lora'].update(r=32, lora_alpha=64, lora_dropout=0.0)
    tag = f't{count}-r32-e3-b16-ctx2048'
    c.update(sft_output='outputs/search-sft-'+tag, grpo_output='outputs/search-grpo-'+tag,
             merged_model_path='models/search-sft-'+tag+'-merged')
    if any(path(c[k]).exists() for k in ('sft_output','grpo_output','merged_model_path')):
        raise FileExistsError('New experiment output already exists')
    excluded = [r for bundle in (teachers, heldout) for split in ('val','test') for r in bundle[split]]
    ids = {r['id'] for r in excluded}
    questions = {normalize(r['question']) for r in excluded}
    corpus_ids = {r['id'] for r in heldout['corpus']}
    docs = {r['id']: r['contents'] for r in heldout['corpus']}
    retriever = Retriever(heldout['corpus'], c['document_chars'])
    candidates = list(teachers['train'])
    random.Random(c['seed']).shuffle(candidates)
    original_ids = {r['id'] for r in teachers['train']}
    candidate_identity = None
    if candidate_source is not None:
        extra, candidate_identity = load_bundle(path(candidate_source))
        if eval_identity['manifest'].get('source_training_identity') != candidate_identity:
            raise ValueError('Candidate bundle must match the frozen evaluation source training identity')
        extra_rows = list(extra['train'])
        random.Random(c['seed']).shuffle(extra_rows)
        for row in extra_rows:
            if row['id'] in original_ids:
                continue
            support = row['support_ids']
            if not 1 <= len(support) <= c['max_searches'] or any(d not in docs for d in support):
                continue
            titles = [docs[d].split('\n', 1)[0] for d in support]
            if any(not title.strip() for title in titles):
                continue
            item = copy.deepcopy(row)
            item['actions'] = [{'search': title} for title in titles] + [{'answer':row['answers'][0]}]
            candidates.append(item)
    selected, actions, rejected = [], [], {'overlap': 0, 'invalid': 0, 'evidence': 0}
    for row in candidates:
        if len(selected) == count:
            break
        if row['id'] in ids or normalize(row['question']) in questions:
            rejected['overlap'] += 1
            continue
        try:
            steps = sft_steps(row, retriever, c['max_searches'], c['top_k'])
        except (ValueError, TypeError, KeyError):
            rejected['invalid'] += 1
            continue
        hits = {hit['id'] for a in row['actions'] if 'search' in a
                for hit in retriever.search(a['search'], c['top_k'])}
        if not row['support_ids'] or not set(row['support_ids']) <= hits & corpus_ids:
            rejected['evidence'] += 1
            continue
        selected.append(row)
        ids.add(row['id'])
        questions.add(normalize(row['question']))
        actions.extend({'question_id': row['id']} for _ in steps)
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)}/{count} eligible complete trajectories; rejected={rejected}. More eligible candidate questions are required.')
    training, validation, split = split_sft_actions(actions, .1, c['seed'])
    output.mkdir(parents=True)
    for name in ('val','test','corpus'):
        (output/(name+'.jsonl')).write_bytes((evaluation/(name+'.jsonl')).read_bytes())
    with (output/'train.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = dict(source=teacher_identity['manifest']['source'], license=teacher_identity['manifest']['license'],
                    demo=teacher_identity['manifest'].get('demo', False),
                    teacher_source_identity=teacher_identity, evaluation_source_identity=eval_identity,
                    candidate_source_identity=candidate_identity,
                    trajectory_selection={'requested':count, 'selected':len(selected), 'rejected':rejected, 'seed':c['seed']},
                    teacher='Original teachers retained first; new training-only supporting-title queries followed by gold answer. Document hits verified, semantic sufficiency not verified.',
                    sha256={k:sha256(output/(k+'.jsonl')) for k in ('train','val','test','corpus')})
    write_json(output/'manifest.json', manifest)
    _, identity = load_bundle(output)
    per_epoch = math.ceil(len(training)/16)
    c.update(sft_steps=3*per_epoch, sft_eval_steps=min(5,3*per_epoch))
    c['sft_plan'] = dict(epochs=3, steps_per_epoch=per_epoch, total_steps=c['sft_steps'],
                         split=split, data_identity=identity,
                         note='Complete-trajectory subset; 10% held out by question; best val-loss adapter saved.')
    write_json(destination, c)
    return {'config':str(destination), 'source_trajectories':len(teachers['train']),
            'retained_original':sum(r['id'] in original_ids for r in selected),
            'added_rule_trajectories':sum(r['id'] not in original_ids for r in selected),
            'selected_trajectories':count, 'split':split, 'steps':c['sft_steps'], 'output':c['sft_output']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='search/sft-r32-e3-b16-ctx2048.json')
    p.add_argument('--source', default='search/hotpot-pilot')
    p.add_argument('--evaluation', default='search/eval-val300')
    p.add_argument('--output', default='search/sft-trajectories300')
    p.add_argument('--destination', default='search/sft-t300-r32-e3-b16-ctx2048.json')
    p.add_argument('--count', type=int, default=300)
    p.add_argument('--candidates', default='search/hotpot-grpo-2000')
    a = p.parse_args()
    print(json.dumps(prepare(a.config,a.source,a.evaluation,a.output,a.destination,a.count,a.candidates),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
