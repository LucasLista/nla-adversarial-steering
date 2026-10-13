"""Real-model integration checks; run with a live localhost launcher."""
import json, math, urllib.request
from pathlib import Path

def post(action, **kwargs):
    req=urllib.request.Request('http://127.0.0.1:18090/stepper-api', json.dumps({'action':action,**kwargs}).encode(),{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1200) as r: out=json.load(r)
    print(action, 'complete',flush=True)
    return out

def probs(r):
    return {x['id']:math.exp(x['logprob']) for x in r['completion_probabilities'][0]['top_logprobs']}

def main():
    tokens=post('start',prompt='Complete this rhyme: He saw a carrot and had to grab it,')['tokens']
    state=post('inspect',tokens=tokens)
    assert len(state['h'])==3584 and len(probs(state['baseline']))==30
    av=post('explain',tokens=tokens,h=state['h'])
    assert max(abs(h-r-e) for h,r,e in zip(state['h'],av['reconstruction'],av['error']))<1e-10
    comparison=post('compare',tokens=tokens,edited=av['text'],reconstruction=av['reconstruction'])
    a,b=probs(comparison['original']),probs(comparison['edited_corrected'])
    diff=max(abs(a.get(k,0)-b.get(k,0)) for k in a.keys()|b.keys())
    assert diff<1e-5, diff
    proposal=post('propose',tokens=tokens,text=av['text'],instruction='Make the description emphasize a mouse and cheese. Preserve unrelated details.')
    assert proposal['proposal'].strip()
    changed=post('compare',tokens=tokens,edited='A short sentence about a mouse eating cheese.',reconstruction=av['reconstruction'])
    assert probs(changed['edited_corrected']) != probs(changed['original'])
    token=changed['edited_corrected']['completion_probabilities'][0]['id']
    nextstate=post('inspect',tokens=tokens+[token])
    assert len(nextstate['h'])==3584
    report={'tokens':tokens,**state,**av,'unchanged':comparison,'changed':changed,'identity_max_probability_error':diff,'next_token':token,'next_baseline':nextstate['baseline']}
    path=Path(__file__).parent/'runs/stepper-validation.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(report,indent=2))
    print('PASS: unchanged corrected distribution error',diff,flush=True)
if __name__=='__main__':main()
