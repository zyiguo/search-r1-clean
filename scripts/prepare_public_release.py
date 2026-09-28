"""Prepare verified GitHub Release assets from completed cloud experiments.

This script creates local files only. Publish them after inspecting RELEASE-NOTES.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import path, read_json, sha256, write_json
from search_task import load_bundle

ASSET_LIMIT = 2 * 1024**3  # GitHub Releases require each asset to be under 2 GiB.
DATA_ATTRIBUTION = """# Dataset attribution

Derived from HotpotQA by Yang et al., EMNLP 2018:
https://hotpotqa.github.io/
HotpotQA and its processed Wikipedia content are distributed under CC BY-SA 4.0:
https://creativecommons.org/licenses/by-sa/4.0/

Changes: selected and deduplicated questions; constructed a shared retrieval corpus
from supplied contexts; created rule-based SFT search actions from training-only
supporting titles; assigned project-specific train/validation/test roles.
This is a project pilot, not the official HotpotQA leaderboard protocol.
Do not treat the project test split as the official hidden test set.
"""
MODEL_ATTRIBUTION = """# Adapter attribution

These files contain LoRA adapters only. The base model is not included.
Base: Qwen/Qwen3-4B-Instruct-2507, Apache-2.0:
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
Use the exact model revision recorded in experiment-provenance.json.
"""


def _zip_asset(destination, files, text_files=None):
    """Write selected named files and verify their bytes and size."""
    text_files = text_files or {}
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for arcname, source in files.items():
            archive.write(source, arcname)
        for arcname, contents in text_files.items():
            archive.writestr(arcname, contents)
    if destination.stat().st_size >= ASSET_LIMIT:
        destination.unlink()
        raise ValueError(f'{destination.name} exceeds GitHub Release 2 GiB per-asset limit')
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'Corrupt archive: {destination}')
        for arcname, source in files.items():
            digest = hashlib.sha256()
            with archive.open(arcname) as member:
                for block in iter(lambda: member.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != sha256(source):
                raise ValueError(f'Archive content differs: {arcname}')


def _bundle_files(directory, prefix):
    return {f'{prefix}/{name}': directory / name
            for name in ('manifest.json', 'train.jsonl', 'val.jsonl', 'test.jsonl', 'corpus.jsonl')}


def _adapter_files(directory, prefix):
    required = ('adapter_model.safetensors', 'adapter_config.json', 'origin.json')
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing adapter files in {directory}: {missing}')
    result = {f'{prefix}/{name}': directory / name for name in required}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'):
        if (directory / name).is_file():
            result[f'{prefix}/{name}'] = directory / name
    return result


def prepare(sft_report, grpo_report, sft_data, grpo_data, sft_output,
            grpo_checkpoint, output_dir):
    paths = list(map(path, (sft_report, grpo_report, sft_data, grpo_data,
                            sft_output, grpo_checkpoint, output_dir)))
    sft_report, grpo_report, sft_data, grpo_data, sft_output, cp, dest = paths
    if dest.exists():
        raise FileExistsError(f'Use a new release directory: {dest}')
    sft_rows, sft_data_identity = load_bundle(sft_data)
    grpo_rows, grpo_data_identity = load_bundle(grpo_data)
    if grpo_data_identity['manifest'].get('parent_identity') != sft_data_identity:
        raise ValueError('GRPO dataset must derive from this exact SFT dataset')
    if any(sft_data_identity['sha256'][key] != grpo_data_identity['sha256'][key]
           for key in ('val', 'test')):
        raise ValueError('SFT and GRPO data changed held-out questions')
    sft = read_json(sft_report)
    grpo = read_json(grpo_report)
    if sft.get('variant_base') != 'B1' or grpo.get('variant_base') != 'B3':
        raise ValueError('Expected a B1 SFT report and B3 GRPO report')
    if sft.get('split') != 'val' or grpo.get('split') != 'val':
        raise ValueError('Publish validated val reports before final test results')
    sft_identity = read_json(sft_output / 'identity.json')
    if sft_identity['data'] != sft_data_identity:
        raise ValueError('SFT output identity differs from selected dataset')
    sft_adapter = sft_output / 'adapter'
    sft_weight_sha = sha256(sft_adapter / 'adapter_model.safetensors')
    if sft['adapter_sha256'] != sft_weight_sha:
        raise ValueError('SFT report used different adapter weights')
    if grpo['adapter_sha256'] != sha256(cp / 'adapter_model.safetensors'):
        raise ValueError('GRPO report used different checkpoint weights')
    cp_identity = read_json(cp / 'run-identity.json')
    if cp_identity['data'] != grpo_data_identity:
        raise ValueError('GRPO checkpoint used different dataset')
    if cp_identity.get('sft_source_sha256') != sft_weight_sha:
        raise ValueError('GRPO did not start from this SFT adapter')
    if grpo['identity'] != cp_identity:
        raise ValueError('GRPO report and checkpoint training identities differ')
    if cp.parent.resolve() != path(cp_identity['config']['grpo_output']).resolve():
        raise ValueError('Checkpoint is outside its recorded GRPO output')
    if not cp.name.startswith('checkpoint-') or not cp.name[11:].isdigit():
        raise ValueError('Expected a numbered GRPO checkpoint')
    checkpoint_manifest = read_json(cp / 'checkpoint-manifest.json')
    for name, digest in checkpoint_manifest.items():
        if Path(name).name != name or sha256(cp / name) != digest:
            raise ValueError(f'GRPO checkpoint manifest mismatch: {name}')
    selected = grpo.get('evaluated_checkpoint', {})
    if selected and selected.get('manifest_sha256') != sha256(cp / 'checkpoint-manifest.json'):
        raise ValueError('GRPO report refers to another checkpoint')
    sft_ids = [row['id'] for row in sft.get('rows', [])]
    grpo_ids = [row['id'] for row in grpo.get('rows', [])]
    expected_ids = [row['id'] for row in grpo_rows['val']]
    if not sft_ids or sft_ids != grpo_ids or grpo_ids != expected_ids:
        raise ValueError('Reports must cover the same held-out questions in the same order')
    if sft['identity']['data'] not in (sft_data_identity, grpo_data_identity):
        raise ValueError('SFT report used an unrelated evaluation dataset')
    corpus_equal = sft['identity']['data']['sha256']['corpus'] == grpo_data_identity['sha256']['corpus']
    provenance = {
        'sft_report_em': sft['exact_match'], 'grpo_report_em': grpo['exact_match'],
        'questions': len(grpo_ids), 'checkpoint_step': int(cp.name[11:]),
        'same_questions': True, 'same_retrieval_corpus': corpus_equal,
        'sft_training_identity': sft_identity, 'grpo_training_identity': cp_identity,
        'sft_dataset_identity': sft_data_identity, 'grpo_dataset_identity': grpo_data_identity,
        'published_weights': 'LoRA adapters only; no optimizer state or base weights',
    }
    dest.mkdir(parents=True)
    _zip_asset(dest / 'reports.zip',
               {'reports/sft.json': sft_report, 'reports/grpo.json': grpo_report},
               {'experiment-provenance.json': json.dumps(provenance, ensure_ascii=False, indent=2)})
    _zip_asset(dest / 'sft-dataset.zip', _bundle_files(sft_data, 'sft-dataset'),
               {'DATASET-ATTRIBUTION.md': DATA_ATTRIBUTION})
    _zip_asset(dest / 'grpo-dataset.zip', _bundle_files(grpo_data, 'grpo-dataset'),
               {'DATASET-ATTRIBUTION.md': DATA_ATTRIBUTION})
    _zip_asset(dest / 'sft-adapter.zip', _adapter_files(sft_adapter, 'sft-adapter'),
               {'MODEL-ATTRIBUTION.md': MODEL_ATTRIBUTION,
                'sft-training-identity.json': json.dumps(sft_identity, ensure_ascii=False, indent=2)})
    _zip_asset(dest / f'grpo-{cp.name}-adapter.zip', _adapter_files(cp, 'grpo-adapter'),
               {'MODEL-ATTRIBUTION.md': MODEL_ATTRIBUTION,
                'grpo-training-identity.json': json.dumps(cp_identity, ensure_ascii=False, indent=2)})
    assets = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size}
              for p in sorted(dest.glob('*.zip'))}
    write_json(dest / 'SHA256-MANIFEST.json', assets)
    (dest / 'SHA256SUMS').write_text(
        ''.join(f"{info['sha256']}  {name}\n" for name, info in assets.items()),
        encoding='ascii', newline='\n')
    notes = f'''# SFT300 and GRPO {cp.name} experimental artifacts

SFT EM: {sft['exact_match']:.4f}; GRPO EM: {grpo['exact_match']:.4f}; questions: {len(grpo_ids)}.
Same question IDs and order: yes. Same retrieval corpus: {'yes' if corpus_equal else 'no'}.
{'The two EM values use different retrieval corpora and are not a controlled model-only comparison.' if not corpus_equal else 'The two reports use the same retrieval corpus.'}

Artifacts: processed SFT and GRPO data bundles, full evaluation reports,
SFT and GRPO LoRA adapters, provenance and SHA256 manifest. Optimizer states,
raw downloaded files and Qwen base weights are not included.

HotpotQA-derived data: CC BY-SA 4.0, Yang et al., EMNLP 2018:
https://hotpotqa.github.io/
Qwen3-4B-Instruct-2507 base: Apache-2.0:
https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507

The dataset is a selected-context pilot, not the official HotpotQA leaderboard.
'''
    (dest / 'RELEASE-NOTES.md').write_text(notes, encoding='utf-8', newline='\n')
    return {'directory': str(dest), 'assets': assets, 'same_retrieval_corpus': corpus_equal,
            'release_command': f'gh release create <TAG> --repo zyiguo/search-r1-clean --target main --title <TITLE> --notes-file {dest / "RELEASE-NOTES.md"} {dest / "*.zip"} {dest / "SHA256-MANIFEST.json"} {dest / "SHA256SUMS"}'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('sft-report', 'grpo-report', 'output-dir'):
        parser.add_argument('--' + key, required=True)
    for key in ('sft-data', 'grpo-data', 'sft-output', 'grpo-checkpoint'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    sft = read_json(args.sft_report)
    grpo = read_json(args.grpo_report)
    sft_config = sft['identity']['config']
    grpo_config = grpo['identity']['config']
    checkpoint = args.grpo_checkpoint or grpo.get('evaluated_checkpoint', {}).get('path')
    if not checkpoint:
        parser.error('GRPO report has no checkpoint path; supply --grpo-checkpoint')
    result = prepare(args.sft_report, args.grpo_report,
                     args.sft_data or sft_config['data_dir'],
                     args.grpo_data or grpo_config.get('grpo_data_dir', grpo_config['data_dir']),
                     args.sft_output or sft_config['sft_output'],
                     checkpoint, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
