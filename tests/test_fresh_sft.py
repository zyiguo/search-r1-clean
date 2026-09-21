import math
import tempfile
import unittest
from pathlib import Path
from common import read_json, write_json
from prepare_sft_experiment import make_sft_config
from prepare_grpo_experiment import make_config


class FreshSftTests(unittest.TestCase):
    def test_fresh_sft_rank_epochs_and_grpo_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            src, dest, grpo = [Path(d)/n for n in ('source.json', 'sft.json', 'grpo.json')]
            c = read_json('search/pilot-sft-400-b16.json')
            c.update(data_dir='search/demo', sft_val_ratio=0, grpo_data_dir='wrong-old-data', grpo_start='sft_adapter',
                     grpo_runtime_overrides={'max_context': {'source': 2048, 'value': 4096}})
            write_json(src, c)
            fresh = make_sft_config(src, dest)
            self.assertEqual((fresh['lora']['r'], fresh['lora']['lora_alpha']), (32, 64))
            self.assertNotIn('grpo_data_dir', fresh)
            self.assertNotIn('grpo_start', fresh)
            self.assertNotIn('grpo_runtime_overrides', fresh)
            self.assertEqual(fresh['sft_steps'], 3*math.ceil(fresh['sft_plan']['split']['actions_train']/16))
            self.assertEqual((fresh['max_context'], fresh['max_action_tokens'], fresh['max_searches']), (2048, 512, 3))
            self.assertTrue(fresh['sft_gradient_checkpointing'])
            rl = make_config(dest, grpo, profile='rtx6000-80gb')
            self.assertEqual(rl['lora'], fresh['lora'])
            self.assertEqual(rl['sft_output'], fresh['sft_output'])
            self.assertEqual(rl['sft_plan'], fresh['sft_plan'])
            self.assertEqual(rl['grpo_checkpoint_every'], 5)
            with self.assertRaises(FileExistsError):
                make_sft_config(src, dest)

    def test_rl_only_questions_cannot_be_used_as_sft(self):
        from unittest.mock import patch
        files = {'corpus': [], 'train': [dict(id='x', question='Q', answers=['A'])]}
        with tempfile.TemporaryDirectory() as d, patch('prepare_sft_experiment.load_bundle', return_value=(files, {})):
            with self.assertRaisesRegex(ValueError, 'Incomplete SFT'):
                make_sft_config('search/pilot-sft-400-b16.json', Path(d)/'sft.json')

    def test_epoch_count_uses_training_actions_after_question_holdout(self):
        import copy
        from unittest.mock import patch
        from search_task import load_bundle
        files, identity = load_bundle('search/demo')
        extra = copy.deepcopy(files['train'][0])
        extra.update(id='extra', question='Another teacher question')
        files['train'].append(extra)
        with tempfile.TemporaryDirectory() as d, patch('prepare_sft_experiment.load_bundle', return_value=(files, identity)):
            source, output = Path(d)/'source.json', Path(d)/'new.json'
            c = read_json('search/pilot-sft-400-b16.json')
            c['sft_val_ratio'] = .5
            write_json(source, c)
            result = make_sft_config(source, output, micro_batch=2)
            plan = result['sft_plan']
            self.assertEqual(plan['split']['questions_train'], 1)
            self.assertEqual(plan['split']['questions_val'], 1)
            self.assertEqual(plan['split']['actions_train'], 3)
            self.assertEqual(plan['steps_per_epoch'], 2)
            self.assertEqual(result['sft_steps'], 6)
