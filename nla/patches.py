"""Source patches for the pinned nla.cpp backend.

Activation capture and steering positions assumed a single microbatch, so long
contexts silently lost activations and interventions. A one-token raw completion
also need not parse as a chat message, which made some predictions fail.
"""
from pathlib import Path
import subprocess

CONTEXT = 'tools/server/server-context.cpp'
TASK = 'tools/server/server-task.h'


def patched_capture(s):
    old = '        nla_extract_values.resize(n_el);\n        nla_extract_n_tokens = t->ne[1];'
    new = ('        const size_t capture_offset = nla_extract_values.size();\n'
           '        nla_extract_values.resize(capture_offset + n_el);\n'
           '        nla_extract_n_tokens += t->ne[1];')
    if old not in s:
        raise RuntimeError('Unexpected pinned capture source')
    return s.replace(old, new).replace('nla_extract_values[i] = ggml_get_f32_1d',
                                       'nla_extract_values[capture_offset + i] = ggml_get_f32_1d')


# Steering positions were indexed from the start of the logical batch in every
# microbatch, so interventions beyond the first microbatch were silently dropped.
STEER = [
    ('    std::vector<llama_pos> nla_steering_batch_pos;\n',
     '    std::vector<llama_pos> nla_steering_batch_pos;\n'
     '    std::map<std::string, int64_t> nla_steering_ubatch_offset;\n'),
    ('        if ((int64_t) nla_steering_batch_pos.size() < n_tokens || (int64_t) nla_steering_batch_seq.size() < n_tokens) {\n'
     '            return;\n'
     '        }\n'
     '        const std::string name(t->name);\n',
     '        const std::string name(t->name);\n'
     '        int64_t & offset = nla_steering_ubatch_offset[name];\n'
     '        const int64_t base = offset;\n'
     '        offset += n_tokens;\n'
     '        if ((int64_t) nla_steering_batch_pos.size() < base + n_tokens || (int64_t) nla_steering_batch_seq.size() < base + n_tokens) {\n'
     '            return;\n'
     '        }\n'),
    ('                if (entry.seq_id != nla_steering_batch_seq[ib]) continue;\n'
     '                if (entry.token_pos != nla_steering_batch_pos[ib]) continue;\n',
     '                if (entry.seq_id != nla_steering_batch_seq[base + ib]) continue;\n'
     '                if (entry.token_pos != nla_steering_batch_pos[base + ib]) continue;\n'),
    ('            nla_steering_batch_seq.clear();\n'
     '            nla_steering_batch_pos.reserve(n_tokens);\n',
     '            nla_steering_batch_seq.clear();\n'
     '            nla_steering_ubatch_offset.clear();\n'
     '            nla_steering_batch_pos.reserve(n_tokens);\n'),
]


def patched_steering(s):
    for old, new in STEER:
        if s.count(old) != 1:
            raise RuntimeError('Unexpected pinned steering source')
        s = s.replace(old, new)
    return s


COMPLETION = '''    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);'''

RAW_COMPLETION = '''    virtual void update(task_result_state & state) override {
        is_updated = true;
        // Native /completion returns raw content, token IDs and probabilities.
        // A one-token result need not form a complete chat-parser production.
        if (res_type == TASK_RESPONSE_TYPE_NONE) {
            return;
        }
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);'''


def patched_completion(s):
    if s.count(COMPLETION) != 1:
        raise RuntimeError('Unexpected pinned native completion source')
    return s.replace(COMPLETION, RAW_COMPLETION, 1)


PATCHES = {CONTEXT: lambda s: patched_steering(patched_capture(s)), TASK: patched_completion}


def apply(source):
    """Patch a pinned checkout in place; rewrite only files whose content differs."""
    source = Path(source)
    git = lambda *args: subprocess.check_output(['git', '-C', str(source), *args], text=True)
    # Restore anything else an earlier version of this repository modified.
    changed = {line[3:] for line in git('status', '--porcelain', '--untracked-files=no').splitlines()}
    for name in sorted(changed - set(PATCHES)):
        git('checkout', 'HEAD', '--', name)
    for name, patch in PATCHES.items():
        expected = patch(git('show', 'HEAD:' + name))
        path = source / name
        if path.read_text() != expected:
            path.write_text(expected)


if __name__ == '__main__':
    apply(Path(__file__).resolve().parents[1] / '.runtime/nla.cpp')
