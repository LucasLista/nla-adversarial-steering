"""Keep native raw-completion results independent of the chat-message parser."""

ORIGINAL = '''    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);'''

REPLACEMENT = '''    virtual void update(task_result_state & state) override {
        is_updated = true;
        // Native /completion returns raw content, token IDs and probabilities.
        // A one-token result need not form a complete chat-parser production.
        if (res_type == TASK_RESPONSE_TYPE_NONE) {
            return;
        }
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);'''


def patched(source):
    if source.count(ORIGINAL) != 1:
        raise RuntimeError('Unexpected pinned native completion source')
    return source.replace(ORIGINAL, REPLACEMENT, 1)
