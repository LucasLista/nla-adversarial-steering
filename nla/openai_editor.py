"""Sol editor through the OpenAI Responses API, billed to the key in .env (pay-per-use)."""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from nla import codex_editor
from nla.editor import MAX_PROPOSAL_CHARS, SCHEMA, SYSTEM, validate_request

MODEL = codex_editor.MODEL


def model():
    """Editor model for this run; override with NLA_OPENAI_MODEL."""
    return os.environ.get('NLA_OPENAI_MODEL', '').strip() or MODEL
LOCK = threading.Lock()


class EditorRefusal(RuntimeError):
    """The editor model declined the request."""


class EditorCreditExhausted(RuntimeError):
    """The configured API account needs credits before this run can resume."""


class EditorAccessDenied(RuntimeError):
    """API authentication, permissions, or model access needs attention."""


def api_key():
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    env = Path(__file__).resolve().parents[1] / '.env'
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith('OPENAI_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('"\'')
    if not key:
        raise RuntimeError('Put OPENAI_API_KEY in .env to use the API-billed Sol editor.')
    return key


def propose(instruction, text):
    validate_request(instruction, text)
    body = json.dumps({
        'model': model(), 'instructions': SYSTEM, 'reasoning': {'effort': 'low'},
        'input': json.dumps({'editing_instruction': instruction, 'original_description': text}),
        'text': {'format': {'type': 'json_schema', 'name': 'edit', 'strict': True,
                            'schema': SCHEMA}}}).encode()
    request = urllib.request.Request('https://api.openai.com/v1/responses', body,
                                     {'Authorization': 'Bearer ' + api_key(), 'Content-Type': 'application/json'})
    failures = []
    with LOCK:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    data = json.load(response)
                parts = [c for item in data.get('output', []) if item.get('type') == 'message'
                         for c in item.get('content', [])]
                if any(c.get('type') == 'refusal' for c in parts):
                    raise EditorRefusal('Sol editor refused this edit request; checkpoint preserved.')
                proposal = json.loads(''.join(c.get('text', '') for c in parts if c.get('type') == 'output_text'))['proposal'].strip()
                if proposal and len(proposal) <= MAX_PROPOSAL_CHARS:
                    return {'proposal': proposal, 'editor_model': model(), 'editor_provider': 'openai-api',
                            'instruction': instruction}
                failures.append({'status': data.get('status'), 'proposal_chars': len(proposal)})
            except urllib.error.HTTPError as exc:
                # Status code and OpenAI's error code only: messages can echo request content.
                try:
                    code = json.loads(exc.read() or b'{}').get('error', {}).get('code')
                except ValueError:
                    code = None
                failures.append({'http_status': exc.code, 'code': code})
                if code in ('insufficient_quota', 'credit_balance_exhausted'):
                    raise EditorCreditExhausted('OpenAI account is out of credit; top up billing. Checkpoint preserved.') from None
                if exc.code in (400, 401, 403, 404):
                    raise EditorAccessDenied('Sol API editor access failed; check credentials and model access. Checkpoint preserved.') from None
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                failures.append({'error_type': type(exc).__name__})
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError('Sol API editor failed; checkpoint preserved. ' + json.dumps(failures))


def preflight():
    """Verify the configured model answers one synthetic structured edit."""
    result = propose('Replace rabbit with mouse; keep the rest unchanged.',
                     'The next word describes a rabbit eating a carrot.')
    if 'mouse' not in result['proposal'].lower():
        raise RuntimeError(f"Editor {result['editor_model']} did not apply a synthetic edit")
    return result['editor_model']
