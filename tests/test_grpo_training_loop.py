"""CPU integration of real GRPO control flow with tiny differentiable policy scores."""
import contextlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from common import read_json
import search_train


class GrpoLoopTests(unittest.TestCase):
    def run_round(self, answers, auxiliary=False, cross=False):
        try:
            import torch
            import torch._dynamo  # Initialize lazy optimizer imports before patch.dict restores sys.modules.
        except ImportError:
            self.skipTest('Optional torch CPU checks')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.2))
                self.config = types.SimpleNamespace()
            def load_adapter(self, *args, **kwargs): pass
            def set_adapter(self, *args, **kwargs): pass
            def save_pretrained(self, *args, **kwargs): pass
        model = Model()
        config = read_json('search/pilot-sft-metrics.json')
        config.update(grpo_steps=1, group_size=2, grpo_start='sft_adapter',
                      grpo_groups_per_update=2, grpo_max_group_attempts=3,
                      grpo_filter_zero_variance=True)
        if cross:
            config.update(grpo_question_batch_size=3, grpo_question_sampling='epoch', rollout_batch_size=3)
        if auxiliary:
            config.update(grpo_evidence_weight=.1, grpo_protocol_weight=.02)
        samples = iter(answers)
        def generate(messages):
            answer = next(samples)
            return dict(text='{}' if answer == 'INVALID' else '{"answer":"'+answer+'"}', truncated=False,
                        prefix_ids=[1], action_ids=[2 if answer in ('yes', 'wrong') else 3])
        def logps(model, steps, pad):
            return [model.weight.reshape(1) * (1 if s['action_ids'][0] == 2 else -1) for s in steps]
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as stack:
            out = Path(d)/'out'
            config['grpo_output'] = str(out)
            fake_peft = types.SimpleNamespace(
                get_peft_model=lambda m, c: m, prepare_model_for_kbit_training=lambda m, **kw: m,
                PeftModel=types.SimpleNamespace(from_pretrained=lambda *a, **kw: model))
            stack.enter_context(patch.dict('sys.modules', {
                'peft': fake_peft, 'transformers': types.SimpleNamespace(set_seed=lambda seed: None)}))
            mocks = dict(require_cloud=lambda: torch, verify_sft_source=lambda *a: Path(d),
                         identity=lambda *a: {'test': 1}, load_tokenizer=lambda *a: MagicMock(pad_token_id=0),
                         check_lengths=lambda *a: ([], {}), load_model=lambda *a: model,
                         sha256=lambda *a: 'test', read_json=lambda *a: {},
                         compute_dtype=lambda *a: torch.bfloat16,
                         generate_action=lambda *a: generate,
                         generate_action_batch=lambda *a: lambda ms: [generate(m) for m in ms],
                         batch_token_logps=logps,
                         reference_policy=lambda *a: contextlib.nullcontext(),
                         adapter_origin=lambda *a: {'test': 1})
            for name, value in mocks.items():
                stack.enter_context(patch.object(search_train, name, value))
            checkpoints = stack.enter_context(patch.object(search_train, 'save_checkpoint'))
            stack.enter_context(patch('training_visualization.Telemetry'))
            stack.enter_context(patch.object(torch.cuda, 'max_memory_allocated', return_value=0))
            files = {'corpus': [], 'train': [{'id': str(i), 'question': 'q', 'answers': ['yes']} for i in range(3)]}
            before = model.weight.item()
            search_train.train(config, files, {}, 'grpo', None)
            self.assertEqual(checkpoints.call_count, 1)
            if cross:
                self.assertEqual(sum(checkpoints.call_args.args[-1]['visits']), 3)
            return read_json(out/'result.json'), read_json(out/'rollout-1.json'), before, model.weight.item()

    def test_all_filtered_skips_optimizer_and_saves_round(self):
        result, log, before, after = self.run_round(['no'] * 6)
        self.assertEqual(before, after)
        self.assertEqual(result['sampling_totals']['optimizer_updates'], 0)
        self.assertEqual(result['sampling_totals']['skipped_rounds'], 1)
        self.assertEqual(log['metrics']['sampled_groups'], 3)
        self.assertEqual(log['metrics']['optimizer_update'], 0)

    def test_two_effective_groups_one_optimizer_update(self):
        result, log, before, after = self.run_round(['yes', 'yes', 'yes', 'no', 'no', 'yes'])
        self.assertGreater(after, before)
        self.assertEqual(result['sampling_totals']['optimizer_updates'], 1)
        self.assertEqual(result['signal_updates'], 1)
        self.assertEqual(log['metrics']['selected_trajectories'], 4)
        self.assertEqual(log['metrics']['sampled_trajectories'], 6)
        for group in log['groups']:
            for trace in group['traces']:
                self.assertNotIn('old', trace['steps'][0])

    def test_auxiliary_only_round_updates_and_counts_signal(self):
        result, log, before, after = self.run_round(['wrong', 'INVALID'] * 2, auxiliary=True)
        self.assertGreater(after, before)
        self.assertEqual(result['signal_updates'], 1)
        self.assertEqual(log['metrics']['reward_std'], 0)
        self.assertEqual(result['sampling_totals']['auxiliary_only_groups'], 2)
        self.assertEqual(result['sampling_totals']['em_effective_groups'], 0)

    def test_cross_question_training_updates_and_saves_sampler(self):
        result, log, before, after = self.run_round(['yes', 'no', 'yes', 'no', 'yes', 'no'], cross=True)
        self.assertGreater(after, before)
        self.assertEqual(result['sampling_totals']['optimizer_updates'], 1)
        self.assertEqual(log['metrics']['unique_questions_seen'], 3)
        self.assertEqual(log['metrics']['selected_trajectories'], 4)
        self.assertEqual(log['metrics']['surplus_groups'], 1)
