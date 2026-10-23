"""Subscription-authenticated Claude text editor via the Claude Code CLI (`claude -p`)."""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from nla.editor import MAX_PROPOSAL_CHARS, SCHEMA as PROPOSAL_SCHEMA, SYSTEM, validate_request

MODEL = 'claude-sonnet-5-5'
LOCK = threading.Lock()
SCHEMA = json.dumps(PROPOSAL_SCHEMA)
FATAL = ('usage limit', 'rate limit', 'not logged in', 'invalid api key', 'please run /login', 'authentication')


class ClaudeRefusal(RuntimeError):
    """The editor model declined the request."""


def propose(instruction, text):
    validate_request(instruction, text)
    executable = shutil.which('claude')
    if not executable:
        raise RuntimeError('Install Claude Code and sign in with your Claude subscription on this machine.')
    env = os.environ.copy()
    # Never fall back to API-key billing.
    for key in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL'):
        env.pop(key, None)
    cmd = [executable, '-p', '--model', MODEL, '--tools', '', '--strict-mcp-config', '--setting-sources', '',
           '--no-session-persistence', '--system-prompt', SYSTEM, '--output-format', 'json', '--json-schema', SCHEMA]
    payload = json.dumps({'editing_instruction': instruction, 'original_description': text})
    with LOCK, tempfile.TemporaryDirectory(prefix='nla-claude-editor-') as tmp:
        failures = []
        for attempt in range(3):
            try:
                result = subprocess.run(cmd, input=payload, capture_output=True, text=True, cwd=tmp, env=env, timeout=300)
            except subprocess.TimeoutExpired:
                result = None
            if result is not None:
                try:
                    status = json.loads(result.stdout)
                except json.JSONDecodeError:
                    status = {}
                # A refusal is deterministic for the same request; retrying will not help.
                if status.get('stop_reason') == 'refusal':
                    raise ClaudeRefusal('Claude editor refused this edit request (stop_reason=refusal); checkpoint preserved.')
            if result is not None and result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    proposal = (data.get('structured_output') or {}).get('proposal', '').strip()
                    if not data.get('is_error') and proposal and len(proposal) <= MAX_PROPOSAL_CHARS:
                        return {'proposal': proposal, 'editor_model': MODEL, 'editor_provider': 'claude-subscription',
                                'instruction': instruction}
                    failures.append({k: data.get(k) for k in ('is_error', 'subtype', 'api_error_status', 'stop_reason', 'terminal_reason')}
                                    | {'has_proposal': bool(proposal), 'proposal_chars': len(proposal)})
                except (json.JSONDecodeError, AttributeError):
                    failures.append({'unparseable_output': True})
            else:
                failures.append({'timeout': True} if result is None else {'returncode': result.returncode})
            # Account limits and authentication need a human; retrying will not help. Payloads are never echoed.
            if result is not None and any(m in (result.stdout + result.stderr).lower() for m in FATAL):
                raise RuntimeError('Claude editor unavailable: subscription limit or authentication. Checkpoint preserved.')
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    # Metadata only (status fields, lengths); never model text.
    raise RuntimeError('Claude editor failed after three attempts; checkpoint preserved. ' + json.dumps(failures))
