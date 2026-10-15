"""Apply the reviewed microbatch patches (activation capture and steering) to the pinned backend."""
from pathlib import Path
import subprocess
FILE='tools/server/server-context.cpp'
def patched_capture(s):
 old='        nla_extract_values.resize(n_el);\n        nla_extract_n_tokens = t->ne[1];'
 new='        const size_t capture_offset = nla_extract_values.size();\n        nla_extract_values.resize(capture_offset + n_el);\n        nla_extract_n_tokens += t->ne[1];'
 if old not in s: raise RuntimeError('Unexpected pinned capture source')
 return s.replace(old,new).replace('nla_extract_values[i] = ggml_get_f32_1d','nla_extract_values[capture_offset + i] = ggml_get_f32_1d')
# Steering positions were indexed from the start of the logical batch in every
# microbatch, so interventions beyond the first microbatch were silently dropped.
STEER=[
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
 for old,new in STEER:
  if s.count(old)!=1: raise RuntimeError('Unexpected pinned steering source')
  s=s.replace(old,new)
 return s
def patched(s):
 return patched_steering(patched_capture(s))
def apply(source):
 original=subprocess.check_output(['git','-C',str(source),'show','HEAD:'+FILE],text=True)
 expected=patched(original)
 status=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
 if status not in ('','M '+FILE): raise RuntimeError('Unrecognized backend modifications')
 p=source/FILE
 if p.read_text() not in (original,patched_capture(original),expected):raise RuntimeError('Unrecognized backend modifications')
 p.write_text(expected)
if __name__=='__main__':apply(Path('.runtime/nla.cpp'))
