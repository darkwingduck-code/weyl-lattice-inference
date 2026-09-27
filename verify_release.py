"""Check the exact public payload and run its bounded numerical checks."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile


def check_hashes(root, manifest):
    root = root.resolve()
    for name, record in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing or unsafe release path: ' + name)
        data = path.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise ValueError('Release checksum mismatch: ' + name)
    return len(manifest['files'])


def self_test():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'data.csv').write_bytes(b'1,2\n')
        entry = {'sha256': hashlib.sha256(b'1,2\n').hexdigest(), 'bytes': 4}
        record = {'files': {'data.csv': entry}}
        assert check_hashes(root, record) == 1
        (root / 'data.csv').write_bytes(b'1,3\n')
        for invalid in (record, {'files': {'../outside': entry}}):
            try:
                check_hashes(root, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('Mutation or path escape was accepted')
    print('PASS: checksum and path-boundary self-check')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--tests', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = args.root.resolve()
    manifest = json.loads((root / 'release_manifest.json').read_text(encoding='utf-8'))
    count = check_hashes(root, manifest)
    report = {'status': 'PASS', 'paper': manifest['paper'], 'checked_files': count,
              'test_results': []}
    environment = os.environ.copy()
    environment.update(PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if args.tests:
        commands = [[sys.executable, '-B', '-m', 'unittest', 'discover', '-s', folder, '-p', name]
                    for folder, name in manifest['tests']]
        commands += [[sys.executable, '-B', *command] for command in manifest['self_checks']]
        for command in commands:
            result = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                                    text=True, encoding='utf-8', errors='replace', timeout=180)
            if result.returncode:
                raise RuntimeError(' '.join(command) + '\n' + result.stdout + result.stderr)
            report['test_results'].append({'command': command[1:],
                                           'output': (result.stdout + result.stderr).strip()})
        check_hashes(root, manifest)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
