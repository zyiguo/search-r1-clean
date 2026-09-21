"""Diagnose GRPO all-zero groups and corpus answer visibility. CPU-only, no model."""
import argparse
import json
from collections import Counter
from pathlib import Path

from common import path, read_json, write_json
from search_task import Retriever, load_bundle, normalize, reward


def answer_visible(contents, answers):
    text = normalize(contents)
    return any(normalize(a) in text for a in answers if a and a.strip())


def docs_by_id(files):
    return {d['id']: d for d in files['corpus']}


def visible_from_ids(doc_map, ids, answers, max_chars):
    for i in ids:
        d = doc_map.get(i)
        if d and answer_visible(d['contents'][:max_chars], answers):
            return True
    return False


def static_split(files, split, retriever, doc_map, max_chars, top_k, max_searches):
    rows = []
    for row in files[split]:
        support = list(row.get('support_ids') or [])
        answers = row['answers']
        q_hits = retriever.search(row['question'], top_k)
        q_ids = [h['id'] for h in q_hits]
        q_support = set(q_ids) & set(support)
        q_vis = visible_from_ids(doc_map, q_ids, answers, max_chars)
        q_vis_full = visible_from_ids(doc_map, q_ids, answers, 10**9)
        support_vis_trunc = visible_from_ids(doc_map, support, answers, max_chars)
        support_vis_full = visible_from_ids(doc_map, support, answers, 10**9)

        teacher_ids = []
        teacher_vis = None
        teacher_support = None
        if split == 'train' and row.get('actions'):
            for act in row['actions'][:-1]:
                if isinstance(act, dict) and act.get('search'):
                    teacher_ids.extend(h['id'] for h in retriever.search(act['search'], top_k))
            teacher_ids = list(dict.fromkeys(teacher_ids))
            teacher_support = set(teacher_ids) & set(support)
            teacher_vis = visible_from_ids(doc_map, teacher_ids, answers, max_chars)

        if not support_vis_full:
            category = 'gold_absent_from_support'
        elif not support_vis_trunc:
            category = 'gold_lost_to_truncation_in_support'
        elif q_support and q_vis:
            category = 'question_bm25_ok'
        elif q_support and not q_vis:
            category = 'support_retrieved_but_gold_invisible'
        elif teacher_vis:
            category = 'question_bm25_miss_teacher_query_ok'
        elif not q_support and q_vis:
            category = 'gold_visible_without_support_hit'
        else:
            category = 'question_bm25_miss'

        rows.append({
            'id': row['id'],
            'split': split,
            'question': row['question'],
            'answers': answers,
            'support_ids': support,
            'question_top_ids': q_ids,
            'question_support_hit': len(q_support),
            'question_support_total': len(support),
            'question_answer_visible_trunc': q_vis,
            'question_answer_visible_full': q_vis_full,
            'support_answer_visible_trunc': support_vis_trunc,
            'support_answer_visible_full': support_vis_full,
            'teacher_top_ids': teacher_ids,
            'teacher_support_hit': len(teacher_support) if teacher_support is not None else None,
            'teacher_answer_visible_trunc': teacher_vis,
            'category': category,
        })
    return rows


def classify_trace(trace, doc_map, answers, support, max_chars, stored_reward=None):
    status = trace.get('status', '?')
    ans = (trace.get('answer') or '').strip()
    ids = list(trace.get('document_ids') or [])
    hit_support = set(ids) & set(support)
    vis = visible_from_ids(doc_map, ids, answers, max_chars)
    r = reward(trace, answers) if stored_reward is None else float(stored_reward)
    if status != 'answered':
        if status == 'search_limit':
            failure = 'search_limit'
        elif status == 'invalid_action':
            failure = 'invalid_action'
        elif status == 'truncated':
            failure = next((s['truncation_reason'] for s in reversed(trace.get('steps', []))
                            if s.get('truncation_reason')), 'truncated')
        else:
            failure = status
    elif r == 1.0:
        failure = None
    elif not ans:
        failure = 'empty_answer'
    elif ans.upper() == 'UNKNOWN':
        failure = 'unknown_answer'
    elif not hit_support and not vis:
        failure = 'retrieval_miss'
    elif hit_support and not vis:
        failure = 'evidence_truncated_or_weak'
    elif vis:
        failure = 'policy_miss_gold_visible'
    else:
        failure = 'wrong_answer'
    return {
        'status': status,
        'answer': ans,
        'reward': r,
        'searches': trace.get('searches'),
        'document_ids': ids,
        'support_hit': len(hit_support),
        'support_total': len(support),
        'answer_visible_in_retrieved_trunc': vis,
        'failure': failure,
    }


def analyze_rollouts(rollout_dir, files, doc_map, max_chars):
    rollout_dir = path(rollout_dir)
    paths = sorted([*rollout_dir.glob('rollout-*.json'), *rollout_dir.glob('rollout-*.json.gz')],
                   key=lambda p: int(p.name.split('-')[1].split('.')[0]))
    if not paths:
        return {'available': False, 'rollout_dir': str(rollout_dir), 'steps': []}
    split_rows = {row['id']: row for row in files['train']}
    steps = []
    group_kinds = Counter()
    failure_all0 = Counter()
    failure_all1 = Counter()
    sample_stats = {}

    def payloads():
        for p in paths:
            payload = read_json(p)
            if 'groups' in payload:
                for group in payload['groups']:
                    yield p, {**group, 'metrics': {
                        'reward_std': group['reward_std'],
                        'has_signal': group.get('has_signal', group['reward_std'] > 0),
                        'optimizer_update': payload.get('metrics', {}).get('optimizer_update', 0)}}
            else:
                yield p, payload
    for p, payload in payloads():
        sid = payload.get('sample_id')
        rewards = payload.get('rewards') or []
        traces = payload.get('traces') or []
        row = split_rows.get(sid, {})
        answers = row.get('answers') or []
        support = list(row.get('support_ids') or [])
        details = [
            classify_trace(t, doc_map, answers, support, max_chars,
                           rewards[i] if i < len(rewards) else None)
            for i, t in enumerate(traces)
        ]
        if not rewards:
            kind = 'empty'
        else:
            s = set(rewards)
            if s == {0.0} or s == {0}:
                kind = 'all_zero'
            elif s == {1.0} or s == {1}:
                kind = 'all_one'
            else:
                kind = 'mixed'
        group_kinds[kind] += 1
        for d in details:
            key = d['failure'] or 'em_ok'
            if kind == 'all_zero':
                failure_all0[key] += 1
            elif kind == 'all_one':
                failure_all1[key] += 1
        steps.append({
            'file': p.name,
            'sample_id': sid,
            'question': row.get('question'),
            'answers': answers,
            'rewards': rewards,
            'group_kind': kind,
            'selected_for_update': payload.get('selected', True),
            'reward_mean': (sum(rewards) / len(rewards)) if rewards else None,
            'reward_std': payload.get('metrics', {}).get('reward_std'),
            'signal_update': bool(payload.get('selected', True) and
                                  payload.get('metrics', {}).get('optimizer_update', 1) and
                                  payload.get('metrics', {}).get('has_signal',
                                      (payload.get('metrics', {}).get('reward_std') or 0) > 0)),
            'reward_components': payload.get('reward_components'),
            'details': details,
        })
        st = sample_stats.setdefault(sid, {
            'sample_id': sid,
            'question': row.get('question'),
            'answers': answers,
            'visits': 0,
            'group_kinds': Counter(),
            'failures': Counter(),
            'mean_rewards': [],
        })
        st['visits'] += 1
        st['group_kinds'][kind] += 1
        st['mean_rewards'].append(sum(rewards) / len(rewards) if rewards else None)
        for d in details:
            st['failures'][d['failure']] += 1

    samples = []
    for sid, st in sample_stats.items():
        samples.append({
            'sample_id': sid,
            'question': st['question'],
            'answers': st['answers'],
            'visits': st['visits'],
            'group_kinds': dict(st['group_kinds']),
            'failures': dict(st['failures']),
            'mean_reward_avg': (sum(x for x in st['mean_rewards'] if x is not None) /
                                max(1, sum(1 for x in st['mean_rewards'] if x is not None))),
        })
    samples.sort(key=lambda x: (x['group_kinds'].get('all_zero', 0), -x['mean_reward_avg']), reverse=True)

    return {
        'available': True,
        'rollout_dir': str(rollout_dir),
        'n_rollout_files': len(paths),
        'group_kind_counts': dict(group_kinds),
        'failure_all_zero': dict(failure_all0),
        'failure_all_one': dict(failure_all1),
        'steps': steps,
        'samples': samples,
    }


def summarize(report):
    lines = []
    lines.append('# GRPO 信号 / 可检索性诊断')
    lines.append('')
    lines.append(f"data_dir: {report['data_dir']}  document_chars: {report['document_chars']}  top_k: {report['top_k']}")
    lines.append('')
    for split, rows in report['static'].items():
        cats = Counter(r['category'] for r in rows)
        n = len(rows)
        vis_q = sum(1 for r in rows if r['question_answer_visible_trunc'])
        hit_q = sum(1 for r in rows if r['question_support_hit'] > 0)
        lines.append(f'## Static {split} (n={n})')
        lines.append(f'- question BM25 support hit >0: {hit_q}/{n}')
        lines.append(f'- gold visible in question top-{report["top_k"]} after trunc: {vis_q}/{n}')
        for k, v in cats.most_common():
            lines.append(f'- {k}: {v}')
        hard = [r for r in rows if r['category'] in (
            'gold_absent_from_support', 'gold_lost_to_truncation_in_support',
            'support_retrieved_but_gold_invisible', 'question_bm25_miss')]
        if hard:
            lines.append(f'- hard / weak-evidence examples ({min(5, len(hard))}):')
            for r in hard[:5]:
                lines.append(f"  - {r['id']}: {r['category']} answers={r['answers']!r} q={r['question'][:80]!r}")
        lines.append('')

    ro = report['rollouts']
    lines.append('## GRPO rollouts')
    if not ro.get('available'):
        lines.append(f"- no rollouts under {ro.get('rollout_dir')}")
    else:
        lines.append(f"- files: {ro['n_rollout_files']}  dir: {ro['rollout_dir']}")
        lines.append(f"- group kinds: {ro['group_kind_counts']}")
        lines.append(f"- failures in all-zero groups: {ro['failure_all_zero']}")
        lines.append(f"- failures in all-one groups: {ro['failure_all_one']}")
        mixed = ro['group_kind_counts'].get('mixed', 0)
        total = ro['n_rollout_files'] or 1
        lines.append(f"- mixed (learnable) group ratio: {mixed}/{total} = {mixed/total:.3f}")
        lines.append('- top all-zero samples:')
        for s in ro['samples'][:8]:
            if s['group_kinds'].get('all_zero'):
                lines.append(
                    f"  - {s['sample_id']} visits={s['visits']} kinds={s['group_kinds']} "
                    f"fail={s['failures']} ans={s['answers']!r}"
                )
    lines.append('')
    lines.append('## Suggested filter flags (data side)')
    for split, rows in report['static'].items():
        drop = [r['id'] for r in rows if r['category'] in (
            'gold_absent_from_support', 'gold_lost_to_truncation_in_support')]
        weak = [r['id'] for r in rows if r['category'] in (
            'support_retrieved_but_gold_invisible', 'question_bm25_miss')]
        lines.append(f'- {split}: drop_or_relabel={len(drop)} weak_retrieval={len(weak)}')
        if drop[:10]:
            lines.append(f'  drop ids sample: {drop[:10]}')
        if weak[:10]:
            lines.append(f'  weak ids sample: {weak[:10]}')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='search/pilot.json')
    p.add_argument('--data-dir', default=None)
    p.add_argument('--rollout-dir', default=None)
    p.add_argument('--output', default='reports/search/diagnose-signal.json')
    p.add_argument('--splits', default='train,val')
    p.add_argument('--skip-rollouts', action='store_true')
    a = p.parse_args()

    c = read_json(a.config)
    data_dir = a.data_dir or c.get('grpo_data_dir') or c.get('data_dir') or 'search/hotpot-pilot'
    rollout_dir = a.rollout_dir or c.get('grpo_output') or 'outputs/search-grpo-g8-r8-batch4-unmerged'
    max_chars = int(c.get('document_chars', 800))
    top_k = int(c.get('top_k', 2))
    max_searches = int(c.get('max_searches', 2))

    files, identity = load_bundle(data_dir)
    doc_map = docs_by_id(files)
    retriever = Retriever(files['corpus'], max_chars)
    splits = [s.strip() for s in a.splits.split(',') if s.strip()]
    static = {}
    for split in splits:
        if split not in files or not files[split]:
            continue
        static[split] = static_split(files, split, retriever, doc_map, max_chars, top_k, max_searches)

    rollouts = ({'available': False, 'rollout_dir': rollout_dir, 'steps': []}
                if a.skip_rollouts else analyze_rollouts(rollout_dir, files, doc_map, max_chars))

    report = {
        'config_path': a.config,
        'data_dir': str(data_dir),
        'rollout_dir': str(rollout_dir),
        'document_chars': max_chars,
        'top_k': top_k,
        'max_searches': max_searches,
        'data_identity_sha256': identity.get('sha256'),
        'static': static,
        'rollouts': rollouts,
    }
    out = path(a.output)
    write_json(out, report)
    summary = summarize(report)
    summary_path = out.with_suffix('.md')
    summary_path.write_text(summary + '\n', encoding='utf-8')
    print(summary)
    print(f'\nWrote {out}')
    print(f'Wrote {summary_path}')


if __name__ == '__main__':
    main()
