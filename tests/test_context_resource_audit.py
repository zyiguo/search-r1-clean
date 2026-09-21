import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from common import read_json
from prepare_grpo_experiment import make_config
from estimate_search_resources import estimate
from search_task import Retriever, batched_rollouts


class AuditTests(unittest.TestCase):
    def test_generation_never_sends_over_budget_prompt(self):
        import torch
        from search_train import generate_action, generate_action_batch
        class Tok:
            pad_token_id, eos_token_id = 0, 9
            def apply_chat_template(self, messages, **kw): return messages
            def decode(self, tokens, **kw): return str(tokens)
        class Model:
            device = 'cpu'
            calls = 0
            def generate(self, input_ids, attention_mask, generation_config):
                self.calls += 1
                n = generation_config['max_new_tokens']
                self_context = input_ids.shape[1] + n
                if self_context > 2048:
                    raise AssertionError('Exceeded total context')
                return torch.cat([input_ids, torch.full((len(input_ids), n), 4)], dim=1)
        c = dict(max_context=2048, max_action_tokens=512, rollout_batch_size=16)
        m = Model()
        with patch.dict('sys.modules', {'transformers': types.SimpleNamespace(GenerationConfig=lambda **kw: kw)}):
            result = generate_action_batch(m, Tok(), c)([[1]*2000, [1]*2020, [1]*2049, [1]*100])
            self.assertEqual(m.calls, 3)
            self.assertEqual([len(r['action_ids']) for r in result], [48, 28, 0, 512])
            self.assertEqual([r['truncation_reason'] for r in result],
                             ['context_limit', 'context_limit', 'context_exhausted', 'action_limit'])
            single = generate_action(m, Tok(), c, False)([1]*2048)
            self.assertEqual(m.calls, 3)
            self.assertEqual(single['truncation_reason'], 'context_exhausted')

    def test_three_searches_leave_a_fourth_answer_turn(self):
        def generate(batch):
            return [dict(text='{"search":"doc"}' if len(m) < 8 else '{"answer":"done"}',
                         truncated=False, prefix_ids=[1], action_ids=[2]) for m in batch]
        trace = batched_rollouts(['q'], generate, Retriever([{'id':'d', 'contents':'doc'}]), 3, 1)[0]
        self.assertEqual((trace['searches'], trace['status'], len(trace['steps'])), (3, 'answered', 4))

    def test_compressed_rollout_diagnostics_preserve_trace(self):
        from search_train import write_rollout
        from diagnose_signal import analyze_rollouts
        payload = dict(sample_id='a', rewards=[1., 0.], advantages=[1., -1.],
                       traces=[dict(answer='yes', status='answered', searches=0, document_ids=[]),
                               dict(answer='no', status='answered', searches=0, document_ids=[])])
        with tempfile.TemporaryDirectory() as d:
            write_rollout(Path(d), 1, payload, {'grpo_compress_rollouts': True})
            self.assertFalse(Path(d, 'rollout-1.json').exists())
            self.assertEqual(read_json(Path(d, 'rollout-1.json.gz')), payload)
            report = analyze_rollouts(d, {'train':[dict(id='a', answers=['yes'])]}, {}, 800)
            self.assertEqual(report['group_kind_counts'], {'mixed': 1})

    def test_resource_plan_counts_microbatch_and_full_snapshots(self):
        with tempfile.TemporaryDirectory() as d:
            c = make_config('search/pilot-sft-metrics.json', str(Path(d)/'c.json'), profile='rtx6000-80gb')
        r = estimate(c)
        self.assertEqual(c['max_context'], 2048)
        self.assertEqual(r['lora_parameters'], 33030144)
        self.assertEqual(r['trajectories']['target_per_update'], 240)
        self.assertEqual(r['trajectories']['maximum_per_round'], 2560)
        self.assertEqual(r['gpu_gib']['kv_cache_bf16_at_full_context'], 4.5)
        self.assertAlmostEqual(r['disk_gib']['all_checkpoints_weights_adam'],
                               r['disk_gib']['one_checkpoint_weights_adam'] * 20)
        bigger = estimate({**c, 'grpo_groups_per_update': 96})
        self.assertEqual(bigger['gpu_gib'], r['gpu_gib'])
        self.assertEqual(bigger['trajectories']['target_per_update'], 480)

    def test_checkpoint_frequency_and_nonmultiple_final_round(self):
        from search_train import save_checkpoint
        from common import write_json
        from unittest.mock import MagicMock
        class Saver:
            def save_pretrained(self, directory):
                Path(directory, 'adapter_model.safetensors').write_bytes(b'fixture')
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            write_json(out/'identity.json', {})
            config = dict(grpo_checkpoint_every=5, grpo_steps=12)
            with patch('search_train.adapter_origin', return_value={}), patch('torch.save'), \
                 patch('torch.cuda.get_rng_state_all', return_value=[]):
                for step in range(1, 13):
                    save_checkpoint(Saver(), Saver(), MagicMock(), MagicMock(), out, step, 0, config)
            self.assertEqual({p.name for p in out.glob('checkpoint-*')},
                             {'checkpoint-5', 'checkpoint-10', 'checkpoint-12'})
