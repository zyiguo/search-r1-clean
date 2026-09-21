"""Cloud preflight, pinned download, SFT merge, feedback export for the search pipeline."""
import argparse
import json
import platform
import shutil
import subprocess
import zipfile
from common import ROOT, path, provenance, read_json, sha256, write_json
from training_common import compute_dtype, environment, load_tokenizer, model_directory, require_cloud


def download(c):
    if platform.system() != 'Linux':
        raise RuntimeError('Download model weights on the cloud server, not locally')
    from huggingface_hub import snapshot_download
    directory = path(c['model_path'])
    if (directory / 'origin.json').exists():
        model_directory(c)
        print(f'Verified existing pinned model at {directory}')
        return
    snapshot_download(repo_id=c['model_id'], revision=c['model_revision'], local_dir=str(directory))
    index = read_json(directory / 'model.safetensors.index.json')
    weights = sorted(set(index['weight_map'].values()))
    for name in weights:
        if not (directory / name).is_file():
            raise FileNotFoundError(name)
    write_json(directory / 'origin.json', {'model_id': c['model_id'], 'model_revision': c['model_revision'],
        'merged': False, 'files': {name: {'bytes': (directory / name).stat().st_size,
                                           'sha256': sha256(directory / name)} for name in weights}})
    print(f'Downloaded pinned revision to {directory}')


def merge(c, adapter):
    torch = require_cloud()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    directory = path(c['merged_model_path'])
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Merged directory exists; use a new path')
    adapter = path(adapter)
    info = read_json(adapter / 'origin.json')
    if info.get('stage') != 'sft' or info.get('task') != 'search' or info.get('model_revision') != c['model_revision']:
        raise ValueError('Merge requires the search SFT adapter for the pinned base')
    from hardware_probe import available_ram
    if available_ram() < 32 * 1024**3:
        raise RuntimeError('CPU merge requires >=32 GiB available host RAM; choose a suitable cloud instance')
    directory.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory.parent).free < 20 * 1024**3:
        raise RuntimeError('Merge requires >=20 GiB free disk')
    model = AutoModelForCausalLM.from_pretrained(str(model_directory(c)), torch_dtype=compute_dtype(c),
        device_map={'': 'cpu'}, local_files_only=True, trust_remote_code=False, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter))
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(str(directory), safe_serialization=True, max_shard_size='4GB')
    load_tokenizer(c).save_pretrained(str(directory))
    write_json(directory / 'origin.json', {'model_id': c['model_id'], 'model_revision': c['model_revision'],
        'merged': True, 'task': 'search', 'sft_adapter': str(adapter),
        'adapter_sha256': sha256(adapter / 'adapter_model.safetensors')})
    del model
    torch.cuda.empty_cache()
    print(f'Merged SFT base: {directory}. Prefer unmerged GRPO unless evaluating B1-init.')


def feedback():
    names = []
    for directory in ('reports', 'outputs'):
        root = path(directory)
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if p.is_file() and (
                p.suffix in ('.json', '.jsonl', '.log', '.txt', '.md')
                or p.name.startswith('events.out.tfevents.')
            ) and 'checkpoint-' not in p.as_posix():
                names.append(p)
    dest = path('dist/cloud-feedback.zip')
    dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in names:
            z.write(p, p.relative_to(ROOT).as_posix())
    dest.with_suffix('.zip.sha256').write_text(sha256(dest) + '  ' + dest.name + '\n', encoding='ascii')
    print(f'{dest}: {len(names)} files, {dest.stat().st_size} bytes')


def doctor():
    require_cloud()
    report = environment(None)
    report['assets'] = provenance()
    report['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    report['free_disk_bytes'] = shutil.disk_usage(ROOT).free
    write_json('reports/environment.json', report)
    print(json.dumps({k: report[k] for k in report if k != 'pip_freeze'}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['doctor', 'download', 'merge', 'feedback'])
    p.add_argument('--config', default='search/pilot.json')
    p.add_argument('--adapter', default=None)
    args = p.parse_args()
    if args.command == 'feedback':
        feedback()
        return
    c = read_json(args.config)
    if args.command == 'download':
        download(c)
    elif args.command == 'merge':
        merge(c, args.adapter or (c['sft_output'] + '/adapter'))
    else:
        doctor()


if __name__ == '__main__':
    main()
