"""Add offline English translations without changing the saved experiment text.

Optional build dependencies: ctranslate2 and sentencepiece. Use the CTranslate2
conversion gaudi/opus-mt-zh-en-ctranslate2.
Model weights and the intermediate segment cache belong outside Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

CJK = re.compile(r'[\u3400-\u9fff]')
NUMBER = re.compile(r'\d+(?:\.\d+)*')
QUOTES = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
FALLBACK = re.compile(r'[\u3400-\u9fff]+(?:[，。！？；：、（）《》“”‘’…·]*[\u3400-\u9fff]+)*[，。！？；：、]*')


def texts(data):
    values = set()
    for case in data['cases']:
        values.update([case['prompt'], case['instruction']])
        for branch in case['branches'].values():
            values.add(branch['response'])
            for step in branch['steps']:
                values.update(step.get(k) or '' for k in ('nla', 'edited', 'generated_prefix'))
                if step['editor_request']:
                    values.update(step['editor_request'].values())
    return sorted(x for x in values if CJK.search(x))


def source_digest(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def sentences(text):
    for part in re.split(r'(\n+)', text):
        if not part:
            continue
        if not CJK.search(part):
            yield False, part
            continue
        for sentence in re.split(r'(?<=[。！？])', part):
            if not sentence:
                continue
            prefix = sentence[:len(sentence) - len(sentence.lstrip())]
            suffix = sentence[len(sentence.rstrip()):]
            if prefix:
                yield False, prefix
            core = sentence.strip()
            if core:
                yield bool(CJK.search(core)), core
            if suffix:
                yield False, suffix


def plain(text):
    for line in text.splitlines(keepends=True):
        if CJK.match(line.lstrip('0123456789. -*')):
            yield from sentences(line)
        else:
            at = 0
            for match in FALLBACK.finditer(line):
                yield False, line[at:match.start()]
                yield True, match[0]
                at = match.end()
            yield False, line[at:]


def pieces(text):
    # Translate quotations together; keep the surrounding English commentary.
    at = 0
    for match in QUOTES.finditer(text):
        yield from plain(text[at:match.start()])
        if CJK.search(match[1]):
            yield False, '"'
            yield from sentences(match[1])
            yield False, '"'
        else:
            yield False, match[0]
        at = match.end()
    yield from plain(text[at:])


def translate_segment(source, cache, overrides):
    if source in overrides:
        return overrides[source]
    boundary = '，。！？；：、（）《》“”‘’…·()[]{}.,!?;: '
    core = source.strip(boundary)
    if core and core != source and core in overrides:
        start = len(source) - len(source.lstrip(boundary))
        punctuation = str.maketrans('，。！？；：、（）', ',.!?;:,()')
        return (source[:start].translate(punctuation) + overrides[core]
                + source[start + len(core):].translate(punctuation))
    result = cache[source]
    # Consistent terminology for this dataset, without changing its claims.
    if '防火长城' in source:
        result = re.sub(r'Great (?:Fire )?Wall(?: of Fire)?', 'Great Firewall', result, flags=re.I)
    if '翻墙' in source:
        result = re.sub(r'wall[- ](?:flipping|flip|turning) tools', 'firewall-circumvention tools', result, flags=re.I)
    if '审查' in source:
        result = re.sub(r'\breview(?:s)?\b', 'censorship', result, flags=re.I)
    if '政治犯' in source:
        result = re.sub(r'political criminals', 'political prisoners', result, flags=re.I)
    if '百度' in source:
        result = re.sub(r'100[- ]degrees?', 'Baidu', result, flags=re.I)
    if '♪' not in source:
        result = result.replace('♪', '').strip()
    # Short-fragment decoders sometimes drop list numbers or mistype addresses.
    source_numbers = NUMBER.findall(source)
    translated_numbers = NUMBER.findall(result)
    list_prefix = re.match(r'^(\d+[.)]\s*)', source)
    if list_prefix and source_numbers[1:] == translated_numbers:
        result = list_prefix[0] + result
        translated_numbers = NUMBER.findall(result)
    if len(source_numbers) == len(translated_numbers):
        numbers = iter(source_numbers)
        result = NUMBER.sub(lambda _: next(numbers), result)
    # A fragment should not acquire an invented sentence ending.
    if source[-1] not in '。！？.!?':
        result = result.rstrip('.!?')
    return result


def translate_text(source, cache, overrides):
    if source in overrides['texts']:
        return overrides['texts'][source]
    segments = {**overrides['segments'], **overrides['texts']}
    output = ''
    previous_translated = False
    for needs_translation, part in pieces(source):
        value = translate_segment(part, cache, segments) if needs_translation else part
        if (output and value and (previous_translated or needs_translation)
                and (output[-1].isalnum() or output[-1] in '.!?;:') and value[0].isalnum()):
            output += ' '
        output += value
        previous_translated = needs_translation
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--overrides', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.review.read_text())
    values = texts(data)
    overrides = json.loads(args.overrides.read_text())
    cache = json.loads(args.cache.read_text()) if args.cache.exists() else {}
    units = sorted({part for value in values for flag, part in pieces(value) if flag}, key=lambda x: (len(x), x))
    pending = [x for x in units if x not in cache and x not in overrides['segments'] and x not in overrides['texts']]
    if pending:
        import ctranslate2
        import sentencepiece
        source = sentencepiece.SentencePieceProcessor(model_file=str(args.model / 'source.spm'))
        target = sentencepiece.SentencePieceProcessor(model_file=str(args.model / 'target.spm'))
        model = ctranslate2.Translator(str(args.model), device='cpu', compute_type='int8', intra_threads=4)
        for offset in range(0, len(pending), 32):
            batch = pending[offset:offset + 32]
            encoded = [source.encode(x, out_type=str) + ['</s>'] for x in batch]
            if any(len(x) >= 500 for x in encoded):
                raise ValueError('Translation segment exceeds the model context; split explicitly')
            results = model.translate_batch(encoded, beam_size=4, max_decoding_length=512)
            cache.update((a, target.decode(b.hypotheses[0])) for a, b in zip(batch, results))
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + '\n')
    translated = {value: translate_text(value, cache, overrides) for value in values}
    if any(not value.strip() or CJK.search(value) for value in translated.values()):
        raise ValueError('Incomplete English coverage; review the translation cache')
    if any(NUMBER.findall(key) != NUMBER.findall(value) for key, value in translated.items()):
        raise ValueError('Numbers changed during translation; review the segment overrides')
    result = {
        'version': 1, 'language': 'en', 'source_digest': source_digest(values),
        'method': 'Reviewed prompt/response translations; offline OPUS-MT step translations with a reviewed fragment glossary.',
        'model': 'Helsinki-NLP/opus-mt-zh-en',
        'conversion': 'gaudi/opus-mt-zh-en-ctranslate2@05d8fc158397bae0c65b8d46c858b6c18e094c12',
        'reviewed_full_texts': len(overrides['texts']),
        'texts': translated,
    }
    output = args.output or args.review.with_suffix('.en.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'translated_texts': len(translated), 'segments': len(units), 'missing': 0}))


if __name__ == '__main__':
    main()
