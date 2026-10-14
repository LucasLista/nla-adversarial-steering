"""Check released model access and runtime readiness without experiment content."""
import argparse
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent


def credential():
    token = os.environ.get('HF_TOKEN')
    if token:
        return token
    directory = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    path = Path(os.environ.get('HF_TOKEN_PATH', str(directory / 'token')))
    return path.read_text().strip() if path.is_file() else None


def check(profile, token=None):
    access = {}
    for role, model in profile['roles'].items():
        url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/config.json"
        headers = {'Authorization': 'Bearer ' + token} if token else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as response:
                access[role] = {'accessible': response.status == 200, 'http_status': response.status}
        except urllib.error.HTTPError as error:
            # An expired cached token can reject otherwise public checkpoints.
            # Retry public access; gated repositories still enforce their access rules.
            if token and error.code == 401:
                try:
                    with urllib.request.urlopen(url, timeout=20) as response:
                        access[role] = {'accessible': response.status == 200,
                                        'http_status': response.status, 'public_access': True}
                except urllib.error.HTTPError as public_error:
                    access[role] = {'accessible': False, 'http_status': public_error.code}
                except OSError:
                    access[role] = {'accessible': False, 'reason': 'network_unavailable'}
            else:
                access[role] = {'accessible': False, 'http_status': error.code}
        except OSError:
            access[role] = {'accessible': False, 'reason': 'network_unavailable'}
    return {'model': profile['id'], 'layer': profile['layer'], 'dimension': profile['dimension'],
            'access': access, 'runtime': profile['runtime'],
            'runtime_blocker': profile['runtime_blocker'],
            'ready': all(item['accessible'] for item in access.values()) and bool(profile['runtime'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', help='Model ID; defaults to all released families')
    parser.add_argument('--output', type=Path, help='Optional metadata report outside response logs')
    args = parser.parse_args()
    profiles = json.loads((ROOT / 'evaluation_models.json').read_text())
    if args.model:
        unknown = set(args.model) - {profile['id'] for profile in profiles}
        if unknown:
            parser.error('Unknown model ID')
        profiles = [profile for profile in profiles if profile['id'] in args.model]
    report = {'models': [check(profile, credential()) for profile in profiles]}
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')
