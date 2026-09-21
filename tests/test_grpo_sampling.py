import random
import unittest
import tempfile
from pathlib import Path
from common import write_json, read_json

from grpo_sampling import collect_groups, sampling_options, training_work


def trace(correct, tokens=2):
    return {'status': 'answered', 'answer': 'yes' if correct else 'no',
            'searches': 1, 'steps': [{'action_ids': list(range(tokens))}]}


class SamplingTests(unittest.TestCase):
    def collect(self, patterns, target=2, budget=4, filtering=True):
        rows = [{'id': str(i), 'answers': ['yes']} for i in range(len(patterns))]
        draws = iter(patterns)
        return collect_groups(rows, lambda row: [trace(x) for x in next(draws)],
                              target, budget, filtering, random.Random(42), 2)

    def test_filter_and_per_question_advantages(self):
        groups, metrics = self.collect([[1, 1], [0, 0], [1, 0], [0, 1]])
        kept = [g for g in groups if g['selected']]
        self.assertEqual(len(kept), 2)
        self.assertEqual(metrics['sampled_groups'], 4)
        self.assertEqual(metrics['sampled_trajectories'], 8)
        self.assertEqual(metrics['effective_group_fraction'], .5)
        self.assertEqual(metrics['all_correct_groups'], 1)
        self.assertEqual(metrics['all_wrong_groups'], 1)
        self.assertEqual(len({g['sample_id'] for g in groups}), 4)
        for g in kept:
            self.assertAlmostEqual(sum(g['advantages']), 0)
            self.assertGreater(g['advantages'][g['rewards'].index(1)], 0)

    def test_all_zero_variance_is_bounded_and_empty_work(self):
        groups, metrics = self.collect([[0, 0]] * 6, budget=3)
        self.assertEqual(len(groups), 3)
        self.assertEqual(metrics['selected_groups'], 0)
        self.assertEqual(training_work(groups), ([], 0))

    def test_partial_and_small_dataset(self):
        groups, metrics = self.collect([[1, 0], [0, 0]])
        self.assertEqual(metrics['selected_groups'], 1)
        self.assertEqual(metrics['sampled_groups'], 2)
        self.assertEqual(training_work(groups)[1], 2)

    def test_legacy_keeps_zero_variance_for_kl(self):
        groups, metrics = self.collect([[1, 1]], target=1, budget=1, filtering=False)
        self.assertTrue(groups[0]['selected'])
        self.assertEqual(training_work(groups)[1], 2)

    def test_normalization_counts_actual_selected_trajectories(self):
        groups, _ = self.collect([[1, 0], [0, 1]])
        groups[0]['traces'][0]['steps'].append({'action_ids': [3, 4, 5]})
        work, denominator = training_work(groups)
        self.assertEqual(denominator, 4)
        self.assertEqual(len(work), 5)
        self.assertEqual(work[0][2], 5)
        self.assertEqual(work[1][2], 5)

    def test_defaults_and_invalid_budgets(self):
        self.assertEqual(sampling_options({}), (1, 1, False))
        for c in ({'grpo_groups_per_update': 0},
                  {'grpo_groups_per_update': 4, 'grpo_max_group_attempts': 3},
                  {'grpo_groups_per_update': 1.5},
                  {'grpo_filter_zero_variance': 'true'}):
            with self.subTest(c=c), self.assertRaises(ValueError):
                sampling_options(c)

    def test_rng_checkpoint_reproduces_collection(self):
        rows = [{'id': str(i), 'answers': ['yes']} for i in range(10)]
        rng = random.Random(9)
        state = rng.getstate()
        sample = lambda row: [trace(1), trace(0)]
        first = collect_groups(rows, sample, 4, 8, True, rng)
        rng.setstate(state)
        self.assertEqual(first, collect_groups(rows, sample, 4, 8, True, rng))

    def test_multi_group_diagnostics_and_config_derivation(self):
        from diagnose_signal import analyze_rollouts
        from prepare_grpo_experiment import make_config
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            groups, _ = self.collect([[1, 1], [0, 0], [1, 0]], target=1)
            write_json(root/'rollout-1.json', {'groups': groups, 'metrics': {'optimizer_update': 1}})
            files = {'train': [{'id': g['sample_id'], 'answers': ['yes']} for g in groups]}
            report = analyze_rollouts(root, files, {}, 800)
            self.assertEqual(report['group_kind_counts'], {'all_one': 1, 'all_zero': 1, 'mixed': 1})
            self.assertEqual(sum(s['signal_update'] for s in report['steps']), 1)
            source = {'sft_steps': 39, 'sft_output': 'actual-cloud-sft', 'grpo_output': 'old'}
            write_json(root/'source.json', source)
            c = make_config(root/'source.json', root/'new.json')
            self.assertEqual(c['sft_steps'], 39)
            self.assertEqual(c['sft_output'], source['sft_output'])
            self.assertEqual(sampling_options(c), (4, 16, True))
            self.assertEqual(c['grpo_evidence_weight'], .1)
            self.assertEqual(c['grpo_protocol_weight'], .02)
            self.assertEqual(read_json(root/'new.json'), c)
            with self.assertRaises(FileExistsError):
                make_config(root/'source.json', root/'new.json')
