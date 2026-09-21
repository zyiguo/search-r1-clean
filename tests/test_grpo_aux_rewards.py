import unittest

from grpo_sampling import collect_groups, reward_weights


class AuxiliaryRewardsTests(unittest.TestCase):
    def collect(self, traces, weights=(.1, .02), support=('a', 'b')):
        row = {'id': 'q', 'answers': ['gold'], 'support_ids': list(support)}
        return collect_groups([row], lambda r: traces, 1, 1, True,
                              evidence_weight=weights[0], protocol_weight=weights[1])

    def trace(self, ids=(), status='answered', answer='wrong'):
        return {'document_ids': list(ids), 'status': status, 'answer': answer, 'steps': []}

    def test_all_wrong_evidence_difference_is_retained_and_weak(self):
        groups, m = self.collect([self.trace(['a', 'a']), self.trace([])])
        g = groups[0]
        self.assertTrue(g['selected'])
        self.assertEqual(g['reward_std'], 0)
        self.assertEqual(g['reward_components']['evidence'], [.5, 0])
        self.assertGreater(g['advantages'][0], 0)
        self.assertLess(g['advantages'][0], .1)
        self.assertEqual(m['auxiliary_only_groups'], 1)
        self.assertEqual(m['em_effective_groups'], 0)

    def test_protocol_penalty_distinguishes_budget_violation(self):
        groups, m = self.collect([self.trace(), self.trace(status='search_limit')])
        g = groups[0]
        self.assertEqual(g['reward_components']['protocol'], [0., -1.])
        self.assertGreater(g['advantages'][0], 0)
        self.assertLess(g['advantages'][0], .02)
        self.assertEqual(m['protocol_effective_groups'], 1)

    def test_constant_auxiliary_is_filtered_and_zero_weights_preserve_em(self):
        self.assertFalse(self.collect([self.trace(['a']), self.trace(['a', 'a'])])[0][0]['selected'])
        self.assertFalse(self.collect([self.trace(['a']), self.trace(status='invalid_action')], (0, 0))[0][0]['selected'])
        g = self.collect([self.trace(answer='gold'), self.trace()], (0, 0))[0][0]
        self.assertEqual(g['advantages'], g['component_advantages']['em'])

    def test_missing_support_and_truncation_are_not_protocol_violations(self):
        g = self.collect([self.trace(['a']), self.trace(status='truncated')], support=())[0][0]
        self.assertEqual(g['reward_components']['evidence'], [0., 0.])
        self.assertEqual(g['reward_components']['protocol'], [0., 0.])
        self.assertFalse(g['selected'])

    def test_weight_validation(self):
        self.assertEqual(reward_weights({}), (0., 0.))
        for v in (-1, float('nan'), float('inf'), True, '0.1'):
            with self.subTest(v=v), self.assertRaises(ValueError):
                reward_weights({'grpo_evidence_weight': v})
