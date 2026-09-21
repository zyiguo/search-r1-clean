"""Shared path/hash helpers for the search pipeline. Importing never loads a model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read_json(value):
    p = path(value)
    if p.suffix == '.gz':
        import gzip
        with gzip.open(p, 'rt', encoding='utf-8') as f:
            return json.load(f)
    return json.loads(p.read_text(encoding='utf-8'))


def write_json(value, data):
    p = path(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(p)


def sha256(value):
    h = hashlib.sha256()
    with path(value).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def provenance():
    """Digests of search-relevant code and requirements (no hotel assets)."""
    assets = ['requirements-train.txt']
    assets += [p.relative_to(ROOT).as_posix() for p in ROOT.glob('*.py')]
    assets += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'search').glob('*.json')]
    missing = [name for name in assets if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing provenance assets: {missing}')
    return {name: sha256(name) for name in sorted(assets)}


def write_json_gzip(value, data):
    """Stream compact JSON to an atomic compressed artifact without a giant string copy."""
    import gzip
    p = path(value)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + '.tmp')
    with gzip.open(temp, 'wt', encoding='utf-8', compresslevel=3) as f:
        json.dump(data, f, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    temp.replace(p)
