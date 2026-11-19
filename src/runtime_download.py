"""Retrieve unchanged runtime artifacts from the same project's GitHub Release."""
import hashlib
import json
from pathlib import Path
import tarfile
from threading import Lock
from urllib.request import Request, urlopen

LOCK = Lock()
ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_path(root, name):
    path = (root / name).resolve()
    path.relative_to(root.resolve())
    return path


def extract_verified(archive, root, files):
    expected = {row['path']: row for row in files}
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError('Unexpected runtime archive member')
            row = expected[member.name]
            path = checked_path(root, member.name)
            if member.size != row['bytes']:
                raise ValueError('Runtime file size mismatch')
            if path.exists():
                if path.stat().st_size != row['bytes'] or file_hash(path) != row['sha256']:
                    raise ValueError(f'Existing file differs; refusing to overwrite: {member.name}')
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(path.name + '.download')
                digest = hashlib.sha256()
                try:
                    with tar.extractfile(member) as source, partial.open('wb') as output:
                        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
                            digest.update(block)
                            output.write(block)
                    if digest.hexdigest() != row['sha256']:
                        raise ValueError('Runtime file SHA-256 mismatch')
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Runtime archive has missing files')


def ensure_runtime(root=ROOT, progress=None):
    root = Path(root)
    manifest = json.loads((root / 'deployment/cloud_manifest.json').read_text(encoding='utf-8'))
    files = manifest['files']
    cache = root / '.cache/cloud-runtime'
    marker = cache / 'ready.json'
    def complete_sizes():
        return all(checked_path(root, row['path']).is_file() and
                   checked_path(root, row['path']).stat().st_size == row['bytes'] for row in files)
    with LOCK:
        if marker.exists() and json.loads(marker.read_text()) == manifest['archive_sha256'] and complete_sizes():
            return
        cache.mkdir(parents=True, exist_ok=True)
        if complete_sizes():
            for row in files:
                if file_hash(checked_path(root, row['path'])) != row['sha256']:
                    raise ValueError('Existing runtime data differs from the release manifest')
        else:
            archive = cache / 'runtime.tar.gz.download'
            archive_digest = hashlib.sha256()
            try:
                with archive.open('wb') as output:
                    for position, part in enumerate(manifest['parts'], 1):
                        if progress:
                            progress(f'Downloading full 12,000-paper runtime: {position}/{len(manifest["parts"])}')
                        request = Request(manifest['base_url'] + '/' + part['name'],
                                          headers={'User-Agent': 'ScholarRAG-Lite'})
                        digest = hashlib.sha256()
                        size = 0
                        with urlopen(request, timeout=60) as source:
                            while block := source.read(4 * 1024 * 1024):
                                output.write(block)
                                digest.update(block)
                                archive_digest.update(block)
                                size += len(block)
                        if size != part['bytes'] or digest.hexdigest() != part['sha256']:
                            raise ValueError('Runtime download checksum mismatch')
                if archive_digest.hexdigest() != manifest['archive_sha256']:
                    raise ValueError('Combined runtime archive SHA-256 mismatch')
                if progress:
                    progress('Checking and unpacking original indexes, model and paper chunks…')
                extract_verified(archive, root, files)
            finally:
                archive.unlink(missing_ok=True)
        marker.write_text(json.dumps(manifest['archive_sha256']), encoding='utf-8')


if __name__ == '__main__':
    ensure_runtime(progress=print)
    print('Full runtime is ready.')
