"""Exercise real extraction, explanation, reconstruction and steering."""
import json
import os
import math
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = os.environ.get('NLA_URL', 'http://127.0.0.1:18090')

def post(endpoint, data):
    start = time.monotonic()
    request = urllib.request.Request(URL + endpoint, json.dumps(data).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    print(endpoint, round(time.monotonic() - start, 2), 'seconds', flush=True)
    return result

def main():
    (ROOT / 'runs').mkdir(exist_ok=True)
    html = (ROOT / '.runtime/frontend.html').read_text()
    ids = [int(x) for x in re.search(r'NLA_ACTOR_INPUT_IDS = `([^`]+)`', html)[1].split()]
    assert ids[111] == 149705 and ids[110] == 29 and ids[112] == 522
    prompt = ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
              '<|im_start|>user\nComplete this rhyming couplet:\n'
              'He saw a carrot and had to grab it,\n<|im_end|>\n'
              '<|im_start|>assistant\n')
    tokens = post('/tokenize', {'content': prompt, 'add_special': False, 'parse_special': True})['tokens']
    base_request = {'prompt': tokens, 'n_predict': 32, 'temperature': 0,
                    'seed': 42, 'cache_prompt': False}
    baseline = post('/completion', base_request)
    extracted = post('/extract', {'prompt': tokens, 'layer': 20, 'token_pos': -1,
                                  'add_special': False})
    vector = extracted['activation']
    assert len(vector) == 3584 and all(math.isfinite(x) for x in vector)
    explained = post('/explain', {'activation': vector, 'input_ids': ids,
        'inject_pos': 111, 'scale': 150, 'n_predict': 200, 'temperature': 0, 'seed': 42})
    text = explained.get('explanation') or explained.get('text')
    assert text and len(text) > 10
    score = post('/score', {'activation': vector, 'explanation': text})
    reconstructed = post('/reconstruct', {'explanation': text})
    assert len(reconstructed['activation']) == 3584
    control = post('/edit-direction', {'original_explanation': text, 'edited_explanation': text})
    assert max(abs(x) for x in control['direction']) < 1e-5
    edited = text.replace('rabbit', 'mouse').replace('carrot', 'cheese')
    if edited == text:
        edited += '\nThe continuation should rhyme about a mouse and its house.'
    direction = post('/edit-direction', {'original_explanation': text, 'edited_explanation': edited})
    assert len(direction['direction']) == 3584
    zero = post('/completion', {**base_request, 'steering': [{'token_pos': len(tokens)-1,
        'layer': 20, 'alpha': 0, 'direction': direction['direction']}]})
    assert zero['content'] == baseline['content'], 'Zero-strength steering changed greedy output'
    steered = post('/completion', {**base_request, 'steering': [{'token_pos': len(tokens)-1,
        'layer': 20, 'alpha': .1, 'direction': direction['direction']}]})
    report = {'prompt': prompt, 'layer': 20, 'token_pos': len(tokens)-1,
        'baseline': baseline, 'explanation': text, 'score': score,
        'edited_explanation': edited, 'zero_strength': zero, 'steered_alpha_0_1': steered,
        'checks': ['3584-dimensional extraction', 'nonempty explanation', 'reconstruction',
                   'identical-text zero direction', 'zero-strength greedy-output equality'],
        'note': 'Functional smoke test; does not establish steering efficacy or BF16 equivalence.'}
    (ROOT / 'runs/smoke-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'baseline': baseline['content'], 'explanation': text,
                      'steered': steered['content'], 'score': score}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
