"""Blindly export authorized JBB text checkpoints; never display their contents."""
import argparse
import hashlib
import json
from pathlib import Path

VARIANTS = ('original', 'roundtrip', 'edited_raw', 'edited_corrected')


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def export(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    cases = []
    for config_path in sorted(source.glob('jbb-harmful*/*/config.json')):
        case_dir = config_path.parent
        if not case_dir.name.startswith('jbb-harmful-'):
            continue
        config = json.loads(config_path.read_text())
        name = case_dir.parent.name + '--' + case_dir.name
        target = destination / name
        metadata = {key: config[key] for key in (
            'prompt', 'instruction', 'max_tokens', 'editor_model', 'variants',
            'comparison_mode', 'policy', 'steering_multiplier') if key in config}
        metadata.update(id=case_dir.name, source_group=case_dir.parent.name)
        write(target / 'case.json', metadata)
        summary = {'id': case_dir.name, 'directory': name, 'branches': {}}
        for variant in VARIANTS:
            branch = case_dir / variant
            state_file = branch / 'state.json'
            state = json.loads(state_file.read_text()) if state_file.exists() else None
            steps = []
            for step in sorted(branch.glob('step-*')):
                record = {'index': int(step.name.split('-')[-1])}
                for filename, key, response_field in [('explain.json', 'nla_text', 'text'),
                                                      ('editor.json', 'edited_nla_text', 'proposal')]:
                    path = step / filename
                    if path.exists():
                        data = json.loads(path.read_text())
                        value = data['response'][response_field]
                        assert isinstance(value, str)
                        record[key] = value
                if len(record) > 1:
                    steps.append(record)
            count = len(state['generated']) if state else 0
            complete = bool(state and (state['ended'] or count >= config['max_tokens']))
            reason = (state.get('stop_reason', 'eos') if state and state['ended'] else
                      'token_limit' if complete else 'stopped_incomplete' if state else 'not_started')
            data = {'variant': variant, 'response': state['response'] if state else None,
                    'generated_token_ids': state['generated'] if state else [],
                    'tokens': count, 'complete': complete, 'stop_reason': reason,
                    'steps': steps,
                    'note': 'Only recorded text is exported. Original does not call the NLA; roundtrip does not call the editor. Missing fields mean no saved stage, not an unchanged edit.'}
            write(target / (variant + '.json'), data)
            summary['branches'][variant] = {'tokens': count, 'complete': complete,
                    'stop_reason': reason, 'nla_texts': sum('nla_text' in s for s in steps),
                    'edits': sum('edited_nla_text' in s for s in steps)}
        cases.append(summary)
    files = {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(destination.rglob('*.json'))}
    index = {'scope': 'Local preserved JBB harmful campaign checkpoints; no semantic grading',
             'cases': cases, 'sha256': files}
    write(destination / 'index.json', index)
    print(json.dumps({'cases': len(cases), 'files': len(files),
          'fully_complete': sum(all(v['complete'] for v in c['branches'].values()) for c in cases),
          'nla_texts': sum(v['nla_texts'] for c in cases for v in c['branches'].values()),
          'edits': sum(v['edits'] for c in cases for v in c['branches'].values())}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.output)
