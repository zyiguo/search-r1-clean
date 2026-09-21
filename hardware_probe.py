"""RTX 6000D preflight. CUDA kernels execute only on the authorized cloud host."""
import argparse
import platform
import shutil
from pathlib import Path
from common import ROOT, write_json


def available_ram(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup')):
    values = {}
    for line in (proc / 'meminfo').read_text().splitlines():
        key, rest = line.split(':', 1)
        values[key] = int(rest.split()[0]) * 1024
    available = values['MemAvailable']
    # Container quota can be smaller than the host memory reported in /proc.
    for limit_name, used_name in [('memory.max', 'memory.current'),
                                 ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes')]:
        limit_file, used_file = cgroup / limit_name, cgroup / used_name
        if limit_file.exists() and used_file.exists():
            limit = limit_file.read_text().strip()
            if limit != 'max':
                available = min(available, max(0, int(limit) - int(used_file.read_text())))
    return available


def profile_errors(vram_bytes, ram_bytes, disk_bytes, capability, cuda_version, bf16):
    gib = 1024**3
    errors = []
    if vram_bytes < 75 * gib:
        errors.append('6000D profile requires >=75 GiB visible GPU memory; verify full-card allocation')
    if ram_bytes < 32 * gib:
        errors.append('CPU merge requires >=32 GiB available RAM within the container quota')
    if disk_bytes < 40 * gib:
        errors.append('Need >=40 GiB free project disk before model download/merge; expand data disk')
    if capability[0] >= 10 and tuple(map(int, cuda_version.split('.')[:2])) < (12, 8):
        errors.append('Blackwell requires this project CUDA 12.8 runtime or newer')
    if not bf16:
        errors.append('Expected bf16 support on the selected 6000D instance')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='reports/hardware-6000d.json')
    args = p.parse_args()
    report = {'status': 'started', 'profile': 'RTX6000D-84GB', 'platform': platform.platform()}
    try:
        from training_common import require_cloud
        torch = require_cloud()
        import bitsandbytes as bnb
        report.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
            vram_bytes=torch.cuda.get_device_properties(0).total_memory, available_ram_bytes=available_ram(),
            free_project_disk_bytes=shutil.disk_usage(ROOT).free, cuda_runtime=torch.version.cuda,
            bf16=torch.cuda.is_bf16_supported(), compiled_architectures=torch.cuda.get_arch_list())
        errors = profile_errors(report['vram_bytes'], report['available_ram_bytes'], report['free_project_disk_bytes'],
                                report['capability'], report['cuda_runtime'], report['bf16'])
        if errors:
            raise RuntimeError('; '.join(errors))
        # Small synthetic kernel checks, before downloading model weights.
        x = torch.randn(8, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        layer = bnb.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16,
                                 quant_type='nf4', compress_statistics=True).to('cuda')
        y = layer(x)
        y.float().square().mean().backward()
        if x.grad is None or not torch.isfinite(y).all() or not torch.isfinite(x.grad).all():
            raise RuntimeError('NF4 forward/backward returned non-finite values or missing gradient')
        q = torch.randn(1, 2, 16, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        attention = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        attention.float().square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(attention).all() or q.grad is None or not torch.isfinite(q.grad).all():
            raise RuntimeError('bf16 SDPA forward/backward failed')
        report.update(status='passed', nf4_forward_backward=True, bf16_sdpa_forward_backward=True,
                      note='Synthetic kernels passed; real model SFT/GRPO smoke is still required')
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(args.output, report)
    print('6000D hardware preflight passed; proceed to model download and one-step smoke.')


if __name__ == '__main__': main()
