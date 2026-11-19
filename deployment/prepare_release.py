"""Split the verified runtime export into GitHub Release assets (<2 GiB each)."""
import hashlib
import json
from pathlib import Path

from deployment.package_runtime import ROOT, sha256


def prepare(root=ROOT, part_bytes=512 * 1024 * 1024):
    dist = root / 'deployment/dist'
    original = json.loads((dist / 'runtime_manifest.json').read_text(encoding='utf-8'))
    archive = dist / original['archive']
    if sha256(archive) != original['archive_sha256']:
        raise ValueError('Runtime archive differs from its verified manifest')
    target = dist / 'github-release'
    target.mkdir(exist_ok=True)
    parts = []
    with archive.open('rb') as source:
        number = 1
        while block := source.read(min(part_bytes, 4 * 1024 * 1024)):
            name = f'runtime.part{number:02d}'
            digest = hashlib.sha256()
            size = 0
            with (target / name).open('wb') as output:
                while block:
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                    if size >= part_bytes:
                        break
                    block = source.read(min(4 * 1024 * 1024, part_bytes - size))
            parts.append({'name': name, 'bytes': size, 'sha256': digest.hexdigest()})
            number += 1
    manifest = {**original, 'release_tag': 'runtime-v1',
                'base_url': 'https://github.com/0Ofan/ScholarRAG-Lite/releases/download/runtime-v1',
                'parts': parts}
    # This small checksum manifest is committed with the source; data parts are not.
    (root / 'deployment/cloud_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {len(parts)} release assets in {target}')


if __name__ == '__main__':
    prepare()
