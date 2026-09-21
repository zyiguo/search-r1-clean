import copy
import json
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace
import sys
from scripts.train_compact import grpo_console, SFTConsole


class CompactConsoleTests(unittest.TestCase):
    def test_launcher_restores_callbacks_after_failure(self):
        from scripts.train_compact import main
        import search_train
        callback = ModuleType('transformers.trainer_callback')
        class Progress:
            def on_log(self, *args, **kwargs):
                pass
        class Printer(Progress):
            pass
        callback.ProgressCallback, callback.PrinterCallback = Progress, Printer
        original = Progress.on_log
        captured = []
        tqdm = ModuleType('tqdm.auto')
        tqdm.tqdm = SimpleNamespace(write=captured.append)
        logs = {'loss':.2,'total_flos':123}
        def run():
            Progress().on_log(None,SimpleNamespace(is_world_process_zero=True,global_step=2),None,logs)
            raise RuntimeError('training failure')
        with patch.dict(sys.modules, {'transformers.trainer_callback':callback,'tqdm.auto':tqdm}), patch.object(search_train,'main',run), patch.object(sys,'argv',['train_compact.py','sft']):
            with self.assertRaisesRegex(RuntimeError,'training failure'):
                main()
        self.assertIs(Progress.on_log,original)
        self.assertIs(Printer.on_log,original)
        self.assertNotIn('print',vars(search_train))
        self.assertEqual(logs,{'loss':.2,'total_flos':123})
        self.assertEqual(len(json.loads(captured[0])),5)

    def test_grpo_retains_full_source_and_five_console_fields(self):
        metrics = dict(step=15, loss=1e-7, reward=.6, effective_group_fraction=.26,
                       kl_per_token=.0003, signal_updates=15, seconds=800)
        text = json.dumps(metrics)
        result = json.loads(grpo_console(text))
        self.assertEqual(len(result),5)
        self.assertEqual(result['effective_group_fraction'],.26)
        self.assertIn('seconds',json.loads(text))
        self.assertEqual(grpo_console('WARNING: failed'), 'WARNING: failed')
        self.assertEqual(grpo_console('{"status":"failed"}'), '{"status":"failed"}')

    def test_sft_does_not_mutate_logs_and_retains_last_val_loss(self):
        console = SFTConsole()
        logs = dict(loss=.4,grad_norm=.2,learning_rate=.0001,epoch=1,total_flos=100)
        original = copy.deepcopy(logs)
        first = console.update(1,logs)
        self.assertEqual(logs,original)
        self.assertEqual(len(first),5)
        self.assertIsNone(first['last_val_loss'])
        self.assertEqual(console.update(5,{'eval_loss':.5})['last_val_loss'],.5)
        self.assertEqual(console.update(6,{'loss':.3})['last_val_loss'],.5)
        self.assertEqual(console.update(7,{})['loss'],.3)
