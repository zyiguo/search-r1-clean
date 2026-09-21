import importlib.util
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from training_visualization import Telemetry, audit_scalars, parse_gpu_metrics, trainer_callback


class VisualizationTests(unittest.TestCase):
    def test_gpu_timeout_recovers_and_closed_writer_is_not_reopened(self):
        import subprocess
        records = []
        writer = types.SimpleNamespace(add_scalar=lambda *a: records.append(a), flush=lambda: None, close=lambda: None)
        with tempfile.TemporaryDirectory() as temp:
            telemetry = Telemetry(Path(temp), writer_factory=lambda **kw: writer, monitor_gpu=False, interval=0.001)
            calls = []
            def sample(*args, **kwargs):
                calls.append(1)
                if len(calls) == 1:
                    raise subprocess.TimeoutExpired('nvidia-smi', 3)
                telemetry.stop.set()
                return types.SimpleNamespace(stdout='0, 100, 1000, 50')
            with patch('training_visualization.subprocess.run', side_effect=sample), self.assertWarns(UserWarning):
                telemetry._monitor()
            self.assertEqual([r[1] for r in records if r[0] == 'diagnostics/gpu_sampler_available'], [0, 1])
            telemetry.close()
            count = len(records)
            telemetry.scalars({'late': 1}, 5)
            self.assertEqual(len(records), count)

    def test_gpu_parser_preserves_physical_device_and_ignores_na(self):
        result = parse_gpu_metrics('2, 1024, 49152, 80\n3, N/A, 24576, N/A\ninvalid')
        self.assertEqual(result['gpu/2/memory_used_mib'], 1024)
        self.assertEqual(result['gpu/2/utilization_percent'], 80)
        self.assertNotIn('gpu/3/utilization_percent', result)

    def test_effective_fraction_includes_skipped_updates(self):
        history = [{'signal_bearing_update': True}, {'signal_bearing_update': False,
            'optimizer_skipped': True, 'gradient_norm_after_clip': 0,
            'microbatches': [{'task_loss': -1, 'kl_loss': 0.2, 'completion_tokens': 10},
                             {'task_loss': 1, 'kl_loss': 0.4, 'completion_tokens': 12}]}]
        result = audit_scalars(history)
        self.assertEqual(result['audit/signal_update_fraction'], 0.5)
        self.assertEqual(result['audit/task_loss'], 0)
        self.assertEqual(result['audit/completion_tokens'], 22)

    def test_callback_routes_validation_and_training(self):
        module = types.ModuleType('transformers')
        module.TrainerCallback = object
        captured = []
        telemetry = types.SimpleNamespace(scalars=lambda values, step: captured.append((values, step)))
        with patch.dict('sys.modules', {'transformers': module}):
            callback = trainer_callback(telemetry)
        state = types.SimpleNamespace(global_step=12, is_world_process_zero=True)
        callback.on_log(None, state, None, logs={'loss': 0.5, 'eval_loss': 0.4, 'reward': 2})
        self.assertEqual(captured[0], ({'train/loss': 0.5, 'eval/loss': 0.4, 'train/reward': 2}, 12))

    @unittest.skipUnless(importlib.util.find_spec('tensorboard') and importlib.util.find_spec('torch'), 'Optional real TensorBoard round trip')
    def test_real_event_roundtrip_and_unique_resume_session(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        with tempfile.TemporaryDirectory() as temp:
            first = Telemetry(Path(temp), monitor_gpu=False)
            first.scalars({'train/loss': 0.25, 'bad': float('nan'), 'text': 'ignore'}, 7)
            first.close()
            first.close()
            event = EventAccumulator(str(first.log_dir)).Reload()
            scalar = event.Scalars('train/loss')[0]
            self.assertEqual((scalar.step, scalar.value), (7, 0.25))
            self.assertNotIn('bad', event.Tags()['scalars'])
            self.assertEqual(event.Scalars('diagnostics/nonfinite/bad')[0].value, 1)
            second = Telemetry(Path(temp), monitor_gpu=False)
            self.assertNotEqual(first.log_dir, second.log_dir)
            second.close()


if __name__ == '__main__': unittest.main()
