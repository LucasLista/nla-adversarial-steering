"""Match fixed public native error constants to source locations, without echoing errors."""
import json
import re
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def error_sites(root):
    sites = []
    pattern = re.compile(r'throw\s+(?:std::runtime_error|nla_native_runtime_error)\(\s*(?:std::string\(|(?:string_)?format\()?\s*("(?:\\.|[^"\\])*")')
    for directory in ('tools/server', 'common', 'src'):
        for file in sorted((Path(root) / directory).rglob('*')):
            if file.suffix not in ('.cpp', '.h', '.hpp'):
                continue
            source = file.read_text(errors='replace')
            for matched in pattern.finditer(source):
                try:
                    constant = json.loads(matched[1])
                except ValueError:
                    continue
                prefix = constant.split('%', 1)[0]
                if len(prefix) < 12:
                    continue
                sites.append((prefix, {'file':file.name, 'function':'native_runtime_error',
                                       'line':source.count('\n', 0, matched.start()) + 1}))
    return sites


def source_frame(root, message):
    if not isinstance(message, str):
        return None
    matches = [(len(prefix), frame) for prefix, frame in error_sites(str(root))
               if message.startswith(prefix)]
    if not matches:
        return None
    longest = max(length for length, _ in matches)
    frames = [frame for length, frame in matches if length == longest]
    return frames[0] if len(frames) == 1 else None
