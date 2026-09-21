"""Build a portable search-pipeline archive with an internal digest manifest."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    paths = ['README.md', 'requirements-train.txt']
    paths += [p.relative_to(ROOT).as_posix() for p in ROOT.glob('*.py')]
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'search').glob('*.json')]
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / 'search/demo').glob('*') if p.is_file()]
    for folder, pattern in [('tests', '*.py'), ('scripts', '*.sh'), ('scripts', '*.py'), ('docs/plans', '*.md')]:
        paths += [p.relative_to(ROOT).as_posix() for p in (ROOT / folder).glob(pattern)]
    for name in ('reports/local-verification.json', 'reports/implementation-status.md'):
        if (ROOT / name).exists():
            paths.append(name)
    paths = sorted(set(paths))
    for name in paths:
        if name.endswith('.sh'):
            data = (ROOT / name).read_bytes()
            if b'\r' in data or data.startswith(b'\xef\xbb\xbf'):
                raise ValueError(f'Bash script must use LF and UTF-8 without BOM: {name}')
    dest = ROOT / 'dist/search-r1-qwen3-4b.zip'
    dest.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as archive:
        for relative in paths:
            archive.write(ROOT / relative, relative)
        manifest = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
        archive.writestr('PACKAGE-MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(dest) as archive:
        if archive.testzip() is not None:
            raise ValueError('Invalid output archive')
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix('.zip.sha256').write_text(f'{digest}  {dest.name}\n', encoding='ascii', newline='\n')
    print(f'{dest}\nfiles={len(paths)} bytes={dest.stat().st_size}\nsha256={digest}')


if __name__ == '__main__':
    main()
