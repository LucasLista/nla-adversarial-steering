"""Load probe presets without logging their contents."""
import json

def load(path):
    raw = path.read_text()
    cleaned = []
    quoted = escape = False
    for i, char in enumerate(raw):
        if quoted:
            cleaned.append(char)
            if escape:
                escape = False
            elif char == chr(92):
                escape = True
            elif char == '"':
                quoted = False
        else:
            if char == '"':
                quoted = True
            if char == ',' and raw[i + 1:].lstrip().startswith((']', '}')):
                continue
            cleaned.append(char)
    try:
        entries = json.loads(''.join(cleaned))
    except ValueError:
        raise ValueError('Invalid prompt catalog JSON') from None
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) or not isinstance(item.get('id'), str)
        or not isinstance(item.get('prompt'), str) for item in entries
    ):
        raise ValueError('Prompt catalog must contain id and prompt strings')
    return entries
