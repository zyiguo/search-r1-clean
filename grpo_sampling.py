"""Bounded, question-local GRPO collection. No model/runtime imports."""
import random
import math

from search_task import advantages, reward


def sampling_options(config):
    target = config.get('grpo_groups_per_update', 1)
    budget = config.get('grpo_max_group_attempts', target)
    filtering = config.get('grpo_filter_zero_variance', False)
    if type(target) is not int or type(budget) is not int or not 1 <= target <= budget:
        raise ValueError('Require integer 1 <= grpo_groups_per_update <= grpo_max_group_attempts')
    if type(filtering) is not bool:
        raise ValueError('grpo_filter_zero_variance must be a boolean')
    qbatch = config.get('grpo_question_batch_size', 1)
    if type(qbatch) is not int or qbatch < 1:
        raise ValueError('grpo_question_batch_size must be a positive integer')
    if config.get('grpo_question_sampling', 'random') not in ('random', 'epoch'):
        raise ValueError('Unknown grpo_question_sampling')
    every = config.get('grpo_checkpoint_every', 1)
    if type(every) is not int or every < 1:
        raise ValueError('grpo_checkpoint_every must be a positive integer')
    return target, budget, filtering


def reward_weights(config):
    values = tuple(config.get(key, 0.) for key in ('grpo_evidence_weight', 'grpo_protocol_weight'))
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Auxiliary reward weights must be finite numbers in [0, 1]')
    return values


def collect_groups(rows, sample, target, budget, filtering, rng=random, group_size=None,
                   evidence_weight=0., protocol_weight=0., sample_batch=None, question_batch_size=1, sampler=None):
    reward_weights({'grpo_evidence_weight': evidence_weight, 'grpo_protocol_weight': protocol_weight})
    sampling_options({'grpo_groups_per_update': target, 'grpo_max_group_attempts': budget,
                      'grpo_filter_zero_variance': filtering})
    if not rows:
        raise ValueError('Cannot sample an empty training split')
    if type(question_batch_size) is not int or question_batch_size < 1:
        raise ValueError('question_batch_size must be positive')
    pool = list(rows)
    groups, pending = [], []
    selected = 0
    drawn = set()
    for _ in range(min(budget, len(pool))):
        if not pending:
            n = min(question_batch_size, budget - len(groups), len(pool) - len(drawn))
            batch = []
            for _ in range(n):
                if sampler is not None:
                    row = sampler.draw(drawn)
                else:
                    available = [r for r in pool if r['id'] not in drawn]
                    row = available[rng.randrange(len(available))]
                drawn.add(row['id'])
                batch.append(row)
            results = sample_batch(batch) if sample_batch else [sample(r) for r in batch]
            if len(results) != len(batch):
                raise ValueError('Question batch result count mismatch')
            pending = list(zip(batch, results))
        row, traces = pending.pop(0)
        if len(traces) < 2 or (group_size is not None and len(traces) != group_size):
            raise ValueError('Sampler returned an invalid trajectory group size')
        rewards = [reward(t, row['answers']) for t in traces]
        em_adv, std = advantages(rewards)
        support = set(row.get('support_ids', []))
        evidence = [len(set(t.get('document_ids', [])) & support) / len(support)
                    if support else 0. for t in traces]
        protocol = [-1. if t['status'] in ('invalid_action', 'search_limit') else 0. for t in traces]
        evidence_adv, evidence_std = advantages(evidence)
        protocol_adv, protocol_std = advantages(protocol)
        adv = [a + evidence_weight * e + protocol_weight * p
               for a, e, p in zip(em_adv, evidence_adv, protocol_adv)]
        # Do not renormalize the mixture: doing so erases weak weights in all-wrong groups.
        signal = any(abs(a) > 1e-12 for a in adv)
        keep = selected < target and (not filtering or signal)
        groups.append({'sample_id': row['id'], 'rewards': rewards, 'advantages': adv,
                       'has_signal': signal,
                       'reward_components': {'em': rewards, 'evidence': evidence, 'protocol': protocol},
                       'component_advantages': {'em': em_adv, 'evidence': evidence_adv, 'protocol': protocol_adv},
                       'component_stds': {'em': std, 'evidence': evidence_std, 'protocol': protocol_std},
                       'reward_std': std, 'selected': keep, 'traces': traces})
        selected += int(keep)
        if selected >= target and not pending:
            break
    effective = sum(g['has_signal'] for g in groups)
    metrics = {
        'sampled_groups': len(groups), 'selected_groups': selected,
        'effective_groups': effective, 'effective_group_fraction': effective / len(groups),
        'em_effective_groups': sum(g['reward_std'] > 0 for g in groups),
        'evidence_effective_groups': sum(evidence_weight > 0 and g['component_stds']['evidence'] > 0 for g in groups),
        'protocol_effective_groups': sum(protocol_weight > 0 and g['component_stds']['protocol'] > 0 for g in groups),
        'auxiliary_only_groups': sum(g['has_signal'] and g['reward_std'] == 0 for g in groups),
        'filtered_groups': sum(filtering and not g['has_signal'] for g in groups),
        'surplus_groups': sum(not g['selected'] and (not filtering or g['has_signal']) for g in groups),
        'all_correct_groups': sum(all(r == 1 for r in g['rewards']) for g in groups),
        'all_wrong_groups': sum(all(r == 0 for r in g['rewards']) for g in groups),
        'sampled_trajectories': sum(len(g['traces']) for g in groups),
        'selected_trajectories': sum(len(g['traces']) for g in groups if g['selected']),
        'collection_target_met': int(selected >= target),
    }
    for component in ('evidence', 'protocol'):
        values = [v for g in groups for v in g['reward_components'][component]]
        metrics[component + '_reward'] = sum(values) / len(values)
    return groups, metrics


def training_work(groups):
    """Normalize each trajectory by its tokens, then average selected trajectories."""
    work = []
    trajectories = 0
    for group in groups:
        if not group['selected']:
            continue
        for advantage, trace in zip(group['advantages'], group['traces']):
            trajectories += 1
            count = sum(len(s['action_ids']) for s in trace['steps'])
            work.extend((s, advantage, count) for s in trace['steps'] if s['action_ids'])
    return work, trajectories


class QuestionSampler:
    """Shuffle each epoch; checkpoint cursor and RNG independently of model sampling."""
    def __init__(self, rows, seed=42):
        self.rows = rows
        self.ids = [r['id'] for r in rows]
        if not rows or len(set(self.ids)) != len(rows):
            raise ValueError('Unique nonempty training questions required')
        self.rng = random.Random(seed)
        self.order = list(range(len(rows)))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.visits = [0] * len(rows)

    def draw(self, exclude=()):
        if len(set(exclude) & set(self.ids)) >= len(self.ids):
            raise ValueError('No distinct question remains')
        while True:
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
            eligible = next((j for j in range(self.cursor, len(self.order))
                             if self.ids[self.order[j]] not in exclude), None)
            if eligible is None:
                raise ValueError('No eligible question in remaining epoch')
            self.order[self.cursor], self.order[eligible] = self.order[eligible], self.order[self.cursor]
            index = self.order[self.cursor]
            self.cursor += 1
            self.visits[index] += 1
            return self.rows[index]

    def state_dict(self):
        return dict(ids=self.ids, order=self.order[:], cursor=self.cursor,
                    visits=self.visits[:], rng=self.rng.getstate())

    def load_state_dict(self, state):
        if state['ids'] != self.ids or sorted(state['order']) != list(range(len(self.rows))):
            raise ValueError('Question sampler dataset/order mismatch')
        if not 0 <= state['cursor'] <= len(self.rows) or len(state['visits']) != len(self.rows):
            raise ValueError('Question sampler cursor/visits mismatch')
        self.order, self.cursor, self.visits = state['order'][:], state['cursor'], state['visits'][:]
        self.rng.setstate(state['rng'])
