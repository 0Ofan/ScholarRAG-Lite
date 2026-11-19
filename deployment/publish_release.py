"""Publish verified runtime assets using existing Git credentials (never stored)."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.github.com/repos/0Ofan/ScholarRAG-Lite'


def main():
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n',
                            capture_output=True, text=True, env=env, timeout=20, check=True)
    credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if not credential.get('password'):
        raise RuntimeError('No saved GitHub credential; sign in using Git first')
    token = credential['password']
    proxy = os.getenv('RELEASE_PROXY', '')

    def api(method, url, payload=None, file=None):
        # Authorization is supplied on stdin, not exposed in process arguments or files.
        config = 'header = ' + json.dumps('Authorization: Bearer ' + token) + '\n'
        command = ['curl.exe' if os.name == 'nt' else 'curl', '--config', '-', '--silent', '--show-error',
                   '--connect-timeout', '20', '--max-time', '1800', '--request', method,
                   '--header', 'Accept: application/vnd.github+json', '--write-out', '\n%{http_code}']
        if proxy:
            command += ['--proxy', proxy]
        if payload is not None:
            command += ['--header', 'Content-Type: application/json', '--data-binary', json.dumps(payload)]
        if file is not None:
            command += ['--header', 'Content-Type: application/octet-stream', '--upload-file', str(file)]
        command.append(url)
        for attempt in range(3):
            response = subprocess.run(command, input=config, text=True, capture_output=True, timeout=1820)
            if response.returncode not in (6, 7, 35):
                break  # Only retry failures before an HTTP request can be sent.
        if response.returncode:
            # Do not print request config or credentials on errors.
            raise RuntimeError(f'GitHub transfer failed, curl exit {response.returncode}')
        body, status = response.stdout.rsplit('\n', 1)
        if not 200 <= int(status) < 300:
            raise RuntimeError(f'GitHub API returned HTTP {status}')
        return json.loads(body)

    manifest = json.loads((ROOT / 'deployment/cloud_manifest.json').read_text(encoding='utf-8'))
    releases = api('GET', API + '/releases')
    release = next((row for row in releases if row['tag_name'] == 'runtime-v1'), None)
    if release is None:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        release = api('POST', API + '/releases', {
            'tag_name': 'runtime-v1', 'target_commitish': commit, 'name': 'Complete 12000-paper runtime',
            'draft': True,
            'body': 'Complete runtime for both 12,000-paper retrieval variants. Five binary parts are '
                    'automatically downloaded and SHA-256 verified by community/app.py. Includes unchanged '
                    'BGE-small weights, FAISS indexes, original chunks and source metadata. No API key, '
                    'report or presentation. Checksums: deployment/cloud_manifest.json. Third-party '
                    'materials retain their original licenses.',
        })
    assets_url = API + f'/releases/{release["id"]}/assets'
    existing = {row['name']: row for row in api('GET', assets_url)}
    for part in manifest['parts']:
        name = part['name']
        asset = existing.get(name)
        if asset is None:
            print(f'Uploading {name} ({part["bytes"] / 1024**2:.1f} MiB)', flush=True)
            url = release['upload_url'].split('{')[0] + '?name=' + name
            asset = api('POST', url, file=ROOT / 'deployment/dist/github-release' / name)
        if asset['size'] != part['bytes'] or asset.get('digest') != 'sha256:' + part['sha256']:
            raise RuntimeError(f'{name}: remote asset differs; refusing to replace it')
        print(f'{name}: GitHub SHA-256 verified', flush=True)
    if release['draft']:
        release = api('PATCH', API + f'/releases/{release["id"]}', {'draft': False})
    print('Published:', release['html_url'], flush=True)


if __name__ == '__main__':
    main()
