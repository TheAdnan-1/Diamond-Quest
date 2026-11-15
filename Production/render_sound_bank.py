"""Offline production of the complete round-robin sound bank and sampled stings."""
import json,sys,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diamondquest import audio as A
from diamondquest.sound_design import VERSION,ambience
from compose_score import Ensemble

def sting(kind,variation):
    duration={'victory':4.5,'defeat':3.8,'unlock':2.0,'exit':2.8}[kind]
    events=[]
    for ch,program in [(0,48),(1,60),(2,46),(3,11),(4,47),(5,52)]:events.append((0,2,ch,program,0))
    def note(ch,pitch,at,length,velocity):
        events.append((at,1,ch,pitch,velocity));events.append((at+length,0,ch,pitch,0))
    if kind=='victory':
        for j,chord in enumerate([[62,65,69],[65,69,72],[67,70,74],[62,66,69,74]]):
            at=j*.52
            for pitch in chord:
                note(0,pitch,at,1.0,66);note(1,pitch,at,.49 if j<3 else 1.55,75)
            note(4,chord[0]-24,at,.48,82)
        for j,pitch in enumerate([74,76,77,81,79,81,86]):note(2,pitch,j*.17,.9,84)
    elif kind=='defeat':
        for j,pitch in enumerate([69,65,62,57]):
            for ch in [0,1]:note(ch,pitch,j*.43,1.0,56)
        for pitch in [38,50,53,57]:note(0,pitch,1.3,1.8,54)
        note(4,38,0,.8,70)
    else:
        notes=[74,77,81,86,89] if kind=='unlock' else [62,69,74,77,81,86]
        for j,pitch in enumerate(notes):
            note(2,pitch,j*.095,.85,76);note(3,pitch+12,j*.095,.62,53)
        for pitch in [62,65,69]:note(5,pitch,.12,1.30,48)
    sf=ROOT/'Production/resources/TimGM6mb.sf2'
    synth=Ensemble(sf);raw=synth.render(events,duration+1.5);synth.close()
    x=A.mix_reverb(raw,A.reverb_ir(1.8,8200,1700,seed=553),.23)
    x=A.trim(A.master(x,40,15000));x=A.level(x,A.LEVELS[kind])
    x*=min(1,.90/max(.001,float(np.abs(x).max())))
    return x

if __name__=='__main__':
    folder=ROOT/'diamondquest/sound_bank';folder.mkdir(exist_ok=True)
    bank=A.job_effects(44100)
    for name in ['victory','defeat','unlock','exit']:
        bank[name]=[sting(name,i) for i in range(2)]
    spec={'version':VERSION,'rate':44100,'sounds':{}}
    for name,variants in bank.items():
        files=[]
        for i,x in enumerate(variants):
            filename=f'{name}_{i}.wav';A._save_wav(str(folder/filename),x);files.append(filename)
        spec['sounds'][name]=files
        print('Rendered',name,len(variants),flush=True)
    for biome in ('jungle','frost','ember'):
        A._save_wav(str(folder/f'ambience_{biome}.wav'),A.master(ambience(biome,44100),40,14000))
    (folder/'index.json').write_text(json.dumps(spec,indent=2))
    print('Complete',len(bank),'families',sum(map(len,bank.values())),'variants',flush=True)
