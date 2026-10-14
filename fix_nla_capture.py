"""Apply only the reviewed microbatch capture patch to the pinned backend."""
from pathlib import Path
import subprocess
FILE='tools/server/server-context.cpp'
def patched(s):
 old='        nla_extract_values.resize(n_el);\n        nla_extract_n_tokens = t->ne[1];'
 new='        const size_t capture_offset = nla_extract_values.size();\n        nla_extract_values.resize(capture_offset + n_el);\n        nla_extract_n_tokens += t->ne[1];'
 if old not in s: raise RuntimeError('Unexpected pinned capture source')
 return s.replace(old,new).replace('nla_extract_values[i] = ggml_get_f32_1d','nla_extract_values[capture_offset + i] = ggml_get_f32_1d')
def apply(source):
 original=subprocess.check_output(['git','-C',str(source),'show','HEAD:'+FILE],text=True)
 expected=patched(original)
 status=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
 if status not in ('','M '+FILE): raise RuntimeError('Unrecognized backend modifications')
 p=source/FILE
 if p.read_text() not in (original,expected):raise RuntimeError('Unrecognized capture modifications')
 p.write_text(expected)
if __name__=='__main__':apply(Path('.runtime/nla.cpp'))
