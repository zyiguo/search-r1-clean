"""TensorBoard telemetry; imports no training dependencies until activated."""
import math
import numbers
import subprocess
import threading
import time
import warnings
from datetime import datetime, timezone
from uuid import uuid4


def parse_gpu_metrics(output):
    """Each row is physical GPU index, used MiB, total MiB, utilization %."""
    metrics = {}
    for line in output.strip().splitlines():
        fields = [s.strip() for s in line.split(',')]
        if len(fields) != 4:
            continue
        index = fields[0]
        if not index.isdigit():
            continue
        for name, value in zip(('memory_used_mib', 'memory_total_mib', 'utilization_percent'), fields[1:]):
            try:
                value = float(value)
                if math.isfinite(value):
                    metrics[f'gpu/{index}/{name}'] = value
            except ValueError:
                pass
    return metrics


def audit_scalars(history):
    last = history[-1]
    batches = last['microbatches']
    result = {
        'audit/signal_bearing_update': int(last['signal_bearing_update']),
        'audit/signal_update_count': sum(int(h['signal_bearing_update']) for h in history),
        'audit/signal_update_fraction': sum(int(h['signal_bearing_update']) for h in history) / len(history),
        'audit/optimizer_skipped': int(last['optimizer_skipped']),
        'audit/gradient_norm_after_clip': last['gradient_norm_after_clip'],
    }
    if batches:
        for key in ('task_loss', 'kl_loss'):
            result[f'audit/{key}'] = sum(b[key] for b in batches) / len(batches)
        result['audit/completion_tokens'] = sum(b['completion_tokens'] for b in batches)
    return result


class Telemetry:
    def __init__(self, output_dir, writer_factory=None, monitor_gpu=True, interval=5):
        if writer_factory is None:
            from torch.utils.tensorboard import SummaryWriter
            writer_factory = SummaryWriter
        session = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
        self.log_dir = output_dir / 'tensorboard' / session
        self.writer = writer_factory(log_dir=str(self.log_dir), flush_secs=5)
        self.stop = threading.Event()
        self.thread = None
        self.started = time.monotonic()
        self.closed = False
        self.lock = threading.RLock()
        self.interval = interval
        if monitor_gpu:
            self.thread = threading.Thread(target=self._monitor, daemon=True)
            self.thread.start()

    def scalars(self, values, step):
        with self.lock:
            if self.closed:
                return
            for name, value in values.items():
                if isinstance(value, numbers.Real):
                    if math.isfinite(float(value)):
                        self.writer.add_scalar(name, float(value), int(step))
                    else:
                        warnings.warn(f'Non-finite telemetry metric: {name} at step {step}')
                        self.writer.add_scalar('diagnostics/nonfinite/' + name, 1, int(step))
            self.writer.flush()

    def _monitor(self):
        warned = False
        while not self.stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=3)
                metrics = parse_gpu_metrics(result.stdout)
                if not metrics:
                    raise ValueError('nvidia-smi returned no numeric metrics')
                self.scalars(metrics, int(time.monotonic() - self.started))
                self.scalars({'diagnostics/gpu_sampler_available': 1}, int(time.monotonic() - self.started))
                warned = False
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.scalars({'diagnostics/gpu_sampler_available': 0}, int(time.monotonic() - self.started))
                if not warned:
                    warnings.warn(f'TensorBoard GPU sampler unavailable: {exc}; retrying, training metrics remain enabled')
                    warned = True
            self.stop.wait(self.interval)

    def close(self):
        if self.closed:
            return
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        with self.lock:
            self.writer.flush()
            self.writer.close()
            self.closed = True


def trainer_callback(telemetry):
    from transformers import TrainerCallback

    class MetricsCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not state.is_world_process_zero:
                return
            metrics = {}
            for key, value in (logs or {}).items():
                name = 'eval/' + key[5:] if key.startswith('eval_') else 'train/' + key
                metrics[name] = value
            telemetry.scalars(metrics, state.global_step)

    return MetricsCallback()
