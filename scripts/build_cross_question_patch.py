"""Build an overlay containing code/tests/docs only; never overwrite cloud configs or data."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted({p for folder, pattern in [('', '*.py'), ('scripts', '*.py'), ('tests', '*.py'),
                     ('docs/plans', '2026-09-21-cross-question-*.md')]
                    for p in (ROOT / folder).glob(pattern)})
    files.append(ROOT / 'reports/local-verification.json')
    files.append(ROOT / 'reports/search/resource-estimate-ctx2048.json')
    output = ROOT / 'dist/grpo-cross-question-patch.zip'
    output.parent.mkdir(exist_ok=True)
    manifest = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr('PACKAGE-MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(p)).hexdigest() == h for p, h in manifest.items())
        assert not any(p.startswith(('search/', 'outputs/', 'models/')) for p in z.namelist())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n', encoding='ascii', newline='\n')
    print(json.dumps({'path': str(output), 'files': len(files), 'bytes': output.stat().st_size,
                      'sha256': digest, 'internal_hashes_verified': True}))


if __name__ == '__main__':
    main()
