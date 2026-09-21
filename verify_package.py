"""Verify extracted package hashes without downloading or running models."""
import json
from common import ROOT, sha256


def main():
    manifest = json.loads((ROOT / 'PACKAGE-MANIFEST.json').read_text(encoding='utf-8'))
    for name, digest in manifest.items():
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or sha256(p) != digest:
            raise ValueError(f'Package digest mismatch: {name}')
    print(f'Package verified: {len(manifest)} files')


if __name__ == '__main__': main()
