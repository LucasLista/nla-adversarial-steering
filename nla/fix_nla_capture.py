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
 from nla.fix_native_health import instrument
 from nla.fix_native_completion import patched as patched_completion
 original=subprocess.check_output(['git','-C',str(source),'show','HEAD:'+FILE],text=True)
 previous=patched(original)
 expected=instrument(previous,'context')
 http_file='tools/server/server.cpp'
 http_original=subprocess.check_output(['git','-C',str(source),'show','HEAD:'+http_file],text=True)
 http_expected=instrument(http_original,'http')
 status=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
 additional={'tools/server/server-common.cpp':'common','tools/server/server-task.cpp':'task'}
 header_file='tools/server/server-task.h'
 header_original=subprocess.check_output(['git','-C',str(source),'show','HEAD:'+header_file],text=True)
 header_expected=patched_completion(header_original)
 allowed={FILE,http_file,header_file,*additional}
 if any(line.strip() not in {'M '+f for f in allowed} for line in status.splitlines()): raise RuntimeError('Unrecognized backend modifications')
 p=source/FILE
 if p.read_text() not in (original,patched_capture(original),previous,instrument(previous,'context',runtime_frames=False),expected):raise RuntimeError('Unrecognized backend modifications')
 if (source/http_file).read_text() not in (http_original,http_expected):raise RuntimeError('Unrecognized backend modifications')
 if (source/header_file).read_text() not in (header_original,header_expected):raise RuntimeError('Unrecognized backend modifications')
 changes={}
 for filename,kind in additional.items():
  base=subprocess.check_output(['git','-C',str(source),'show','HEAD:'+filename],text=True)
  result=instrument(base,kind)
  if (source/filename).read_text() not in (base,result):raise RuntimeError('Unrecognized backend modifications')
  changes[filename]=result
 if p.read_text()!=expected:p.write_text(expected)
 if (source/http_file).read_text()!=http_expected:(source/http_file).write_text(http_expected)
 if (source/header_file).read_text()!=header_expected:(source/header_file).write_text(header_expected)
 for filename,result in changes.items():
  if (source/filename).read_text()!=result:(source/filename).write_text(result)
if __name__=='__main__':apply(Path('.runtime/nla.cpp'))
