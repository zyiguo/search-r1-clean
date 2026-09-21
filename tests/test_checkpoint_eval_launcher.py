import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from scripts.eval_grpo_checkpoint import checkpoint_adapter


class CheckpointEvalTests(unittest.TestCase):
    def test_redirect_only_adapter_restore_even_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'run'
            cp = out/'checkpoint-10'
            original = lambda p: Path(p)
            module = SimpleNamespace(path=original)
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                with checkpoint_adapter(module, cp, out):
                    self.assertEqual(module.path(out/'adapter'), cp)
                    self.assertEqual(module.path(out/'identity.json'), out/'identity.json')
                    self.assertEqual(module.path(Path(d)/'sft/adapter'), Path(d)/'sft/adapter')
                    raise RuntimeError('failure')
            self.assertIs(module.path, original)

    def test_different_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            module = SimpleNamespace(path=Path)
            with self.assertRaises(ValueError):
                with checkpoint_adapter(module, Path(d)/'other/checkpoint-10', Path(d)/'run'):
                    self.fail('should not evaluate another run')
