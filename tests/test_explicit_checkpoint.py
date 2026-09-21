import unittest
import warnings
from scripts.run_explicit_checkpoint import explicit_checkpoint_prepare


class ExplicitCheckpointTests(unittest.TestCase):
    def test_missing_option_is_explicit_and_existing_options_preserved(self):
        calls = []
        def original(model, **kwargs):
            calls.append(kwargs)
            return model
        wrapped = explicit_checkpoint_prepare(original)
        model = object()
        self.assertIs(wrapped(model), model)
        self.assertEqual(calls[-1]['gradient_checkpointing_kwargs'], {'use_reentrant': True})
        options = {'use_reentrant': False, 'preserve_rng_state': True}
        wrapped(model, False, options)
        self.assertEqual(calls[-1]['gradient_checkpointing_kwargs'], options)
        self.assertIsNot(calls[-1]['gradient_checkpointing_kwargs'], options)
        self.assertFalse(calls[-1]['use_gradient_checkpointing'])

    def test_explicit_true_matches_implicit_loss_and_gradient_without_warning(self):
        try:
            import torch
            from torch.utils.checkpoint import checkpoint
        except ImportError:
            self.skipTest('Optional local torch test')
        def run(explicit):
            x = torch.tensor([1., 2.], requires_grad=True)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                y = checkpoint(lambda t: t.square().sum(), x, **({'use_reentrant': True} if explicit else {}))
                y.backward()
            return y.item(), x.grad, captured
        old_loss, old_grad, _ = run(False)
        loss, grad, captured = run(True)
        self.assertEqual(loss, old_loss)
        self.assertTrue(torch.equal(grad, old_grad))
        self.assertFalse(any('use_reentrant' in str(w.message) for w in captured))
