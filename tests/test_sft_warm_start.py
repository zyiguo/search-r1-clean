import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import read_json, sha256, write_json
from search_train import prepare_output, verify_sft_source


class SftWarmStartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'base'
        self.source = self.root / 'sft'
        self.config = read_json('search/pilot-sft-metrics.json')
        self.config['sft_output'] = str(self.source)
        write_json(self.base / 'origin.json', {'revision': 'test'})
        write_json(self.source / 'adapter/origin.json', {
            'stage': 'sft', 'task': 'search',
            'model_id': self.config['model_id'],
            'model_revision': self.config['model_revision'],
        })
        self.data = {'dataset': 'fixed'}
        self.saved = {'config': copy.deepcopy(self.config), 'data': self.data,
                      'model_origin': sha256(self.base / 'origin.json'),
                      'requirements': sha256('requirements-train.txt')}
        write_json(self.source / 'identity.json', self.saved)
        mock = patch('search_train.model_directory', return_value=self.base)
        mock.start()
        self.addCleanup(mock.stop)

    def test_new_grpo_budget_accepts_same_sft(self):
        self.config.update(grpo_steps=100, grpo_output='new-run', grpo_start='sft_adapter',
                           grpo_groups_per_update=4, grpo_max_group_attempts=16,
                           grpo_filter_zero_variance=True,
                           grpo_evidence_weight=.1, grpo_protocol_weight=.02)
        self.assertEqual(verify_sft_source(self.config, self.data), self.source / 'adapter')

    def test_sft_and_shared_parameters_still_rejected(self):
        for key, value in [('sft_steps', 999), ('sft_lr', 0.01), ('max_context', 4096)]:
            with self.subTest(key=key):
                config = {**self.config, key: value, 'grpo_steps': 100}
                with self.assertRaises(ValueError):
                    verify_sft_source(config, self.data)

    def test_data_base_and_dependencies_still_rejected(self):
        with self.assertRaises(ValueError):
            verify_sft_source(self.config, {'dataset': 'other'})
        for key in ('model_origin', 'requirements'):
            with self.subTest(key=key):
                write_json(self.source / 'identity.json', {**self.saved, key: 'changed'})
                with self.assertRaises(ValueError):
                    verify_sft_source(self.config, self.data)

    def test_resume_still_rejects_new_budget(self):
        out = self.root / 'grpo'
        write_json(out / 'identity.json', self.saved)
        changed = copy.deepcopy(self.saved)
        changed['config']['grpo_steps'] = 100
        with self.assertRaisesRegex(ValueError, 'cannot resume'):
            prepare_output(out, changed, out / 'checkpoint-20')

    def test_expanded_dataset_warm_start_requires_original_holdout_hashes(self):
        parent = {'manifest': {}, 'sha256': {'val': 'v', 'test': 't'}}
        saved = {**self.saved, 'data': parent}
        write_json(self.source / 'identity.json', saved)
        c = {**self.config, 'grpo_data_dir': 'expanded', 'grpo_question_batch_size': 8,
             'grpo_question_sampling': 'epoch', 'grpo_start': 'sft_adapter'}
        expanded = {'manifest': {'parent_identity': parent}, 'sha256': {'val': 'v', 'test': 't'}}
        with patch('search_train.load_bundle', return_value=({}, parent)):
            self.assertEqual(verify_sft_source(c, expanded), self.source / 'adapter')
            expanded['sha256']['val'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'held-out'):
                verify_sft_source(c, expanded)

    def test_large_prompt_profile_keeps_sft_lineage(self):
        from prepare_grpo_experiment import make_config
        src, dst = self.root/'source.json', self.root/'derived.json'
        write_json(src, self.config)
        c = make_config(str(src), str(dst), profile='rtx6000-80gb')
        self.assertEqual(verify_sft_source(c, self.data), self.source/'adapter')
        self.assertEqual((c['grpo_groups_per_update'], c['group_size']), (48, 5))
        self.assertEqual((c['max_searches'], c['max_action_tokens'], c['max_context']), (3, 512, 2048))
        self.assertEqual((c['grpo_lr'], c['beta']), (1e-6, .001))
        self.assertEqual((c['rollout_batch_size'], c['logprob_batch_size']), (16, 2))
        self.assertEqual(c['sft_steps'], self.config['sft_steps'])
        c['grpo_runtime_overrides']['max_context']['source'] = 1
        with self.assertRaisesRegex(ValueError, 'source/value mismatch'):
            verify_sft_source(c, self.data)

    def test_runtime_override_cannot_change_sft_training(self):
        c = copy.deepcopy(self.config)
        c.update(sft_lr=.5, grpo_runtime_overrides={'sft_lr': {'source': self.config['sft_lr'], 'value': .5}})
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            verify_sft_source(c, self.data)
