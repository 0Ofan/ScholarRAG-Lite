"""Export existing runtime artifacts without loading indexes or calling any API."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUNS = ('phase33_large_fixed', 'phase33_large_sentence')


def runtime_files(root):
    files = set()
    for run in RUNS:
        index = root / 'data/indexes' / run
        files.update(index / name for name in ('faiss.index', 'chunk_ids.json', 'index_metadata.json'))
        files.add(root / 'data/processed' / f'{run}_chunks.jsonl')
        metadata = json.loads((index / 'index_metadata.json').read_text(encoding='utf-8'))
        snapshot = root / 'models/models--BAAI--bge-small-en-v1.5/snapshots' / metadata['model_snapshot']
        files.update(p for p in snapshot.rglob('*') if p.is_file())
        if not (snapshot / 'model.safetensors').is_file():
            raise FileNotFoundError(f'Missing model weights: {snapshot}')
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        path.relative_to(root)
    return sorted(files)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(root, output):
    files = runtime_files(root)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / 'scholarrag_runtime.tar.gz'
    manifest = output / 'runtime_manifest.json'
    if archive.exists() or manifest.exists():
        raise FileExistsError('Runtime export already exists; choose another output directory.')
    rows = []
    with tarfile.open(archive, 'w:gz', compresslevel=1, dereference=True) as tar:
        for path in files:
            name = path.relative_to(root).as_posix()
            rows.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
            tar.add(path, arcname=name, recursive=False)
    manifest.write_text(json.dumps({
        'created_at': datetime.now(timezone.utc).isoformat(),
        'files': rows, 'archive': archive.name, 'archive_sha256': sha256(archive),
        'notes': 'Existing immutable artifacts; no embeddings, indexes or answers regenerated.',
    }, indent=2), encoding='utf-8')
    print(f'Exported {len(rows)} files; archive {archive.stat().st_size / 1024**3:.3f} GiB')


def verify(root, manifest_path):
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    for row in data['files']:
        path = root / row['path']
        path.resolve().relative_to(root.resolve())
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError(f'Checksum mismatch: {row["path"]}')
    print(f'Verified {len(data["files"])} runtime files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'deployment/dist')
    args = parser.parse_args()
    if args.verify:
        verify(args.root, args.verify)
    elif args.build:
        build(args.root, args.output)
    else:
        files = runtime_files(args.root)
        print(json.dumps({'files': len(files), 'bytes': sum(p.stat().st_size for p in files),
                          'paths': [p.relative_to(args.root).as_posix() for p in files]}, indent=2))
