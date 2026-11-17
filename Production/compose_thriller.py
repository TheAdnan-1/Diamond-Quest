"""Offline original expedition-thriller score: pulse, bowed tension and cave space.

No bright lead melody or fanfare. Each 32-bar theme has an exploration bed and
a synchronized tension stem. Only the resulting Ogg files are used at runtime.
"""
import json,math,sys,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diamondquest import audio as A
from compose_score import Ensemble,midi_file

RATE=44100;TAU=math.tau
CONFIG={
    'jungle':dict(root=38,bpm=84,title='The Buried Passage',seed=731),
    'frost':dict(root=37,bpm=76,title='Beneath Silent Ice',seed=732),
    'ember':dict(root=40,bpm=96,title='Vault of Ash',seed=733),
}

def hz(note):return 440*2**((note-69)/12)

def add(out,signal,seconds,gain=1,pan=0):
    if signal.ndim==1:signal=A.stereo(signal,pan,.18)
    start=round(seconds*RATE)%len(out);length=min(len(signal),len(out)-start)
    out[start:start+length]+=signal[:length]*gain
    tail=signal[length:]
    if len(tail):out[:len(tail)]+=tail*gain

def body(freq,dur,seed,kind='drum'):
    t=np.arange(round(dur*RATE),dtype=np.float32)/RATE
    rng=np.random.default_rng(seed);noise=rng.standard_normal(len(t)).astype(np.float32)
    attack=1-np.exp(-t/.002)
    if kind=='drum':
        phase=TAU*np.cumsum(freq*(1+.45*np.exp(-t/.027)))/RATE
        y=(np.sin(phase)+.32*np.sin(phase*1.59)+.13*np.sin(phase*2.17))*np.exp(-t/.22)*attack
        y+=A.fft_filter(noise,A.lp(1600,2))*np.exp(-t/.017)*.24
    elif kind=='wood':
        y=sum(np.sin(TAU*freq*ratio*t)*np.exp(-t/decay)*amp for ratio,decay,amp in [(1,.016,.55),(2.31,.012,.23),(4.13,.009,.09)])*attack
        y+=A.fft_filter(noise,lambda f:A.hp(800,2)(f)*A.lp(3600,2)(f))*np.exp(-t/.007)*.17
    else:
        y=np.zeros_like(t)
        for j,ratio in enumerate([1,1.414,2.756,4.11,5.87]):
            y+=np.sin(TAU*freq*ratio*t+rng.random()*TAU)*np.exp(-t/(.58/(1+j*.3)))*.30/(j+1)
        y*=attack
    return A.fft_filter(y,lambda f:A.hp(30,2)(f)*A.lp(5000,2)(f))

def pulse(note,duration,accent=1):
    t=np.arange(round(duration*RATE),dtype=np.float32)/RATE;f=hz(note)
    # Warm damped harmonic pluck. Most energy stays below 400 Hz.
    y=sum(np.sin(TAU*f*j*t)*np.exp(-t/(.18/j**.5))/j**1.65 for j in range(1,9))
    y*=(1-np.exp(-t/.005))*np.minimum(1,(duration-t)/.035)*accent
    return A.fft_filter(np.tanh(y*1.1)*.8,lambda f:A.hp(32,2)(f)*A.lp(620,2)(f))

def sampled_events(cfg,beat):
    root=cfg['root'];a=[];b=[]
    programs={0:42,1:44,2:45,3:0,4:43}
    for out in (a,b):
        for ch,program in programs.items():
            out.extend([(0,2,ch,program,0),(0,3,ch,10,[43,83,63,92,60][ch]),(0,3,ch,91,0),(0,3,ch,93,0)])
    def note(out,ch,p,at,length,vel):
        out.extend([(at*beat,1,ch,p,vel),((at+length)*beat,0,ch,p,0)])
    for bar in range(32):
        section=bar//8;at=bar*4
        # Pedal-based suspense with suspended seconds, no major chord cycle.
        color=[0,0,1,0,-2,0,1,0][bar%8]
        if bar%2==0:
            note(a,0,root+12+color,at,7.6,36+section*2)
            note(a,4,root,at,7.6,32)
        if bar%4==2:
            note(a,3,root+24+([7,1,5,1][bar//8]),at+1.5,2.5,37)
        if bar%8 in (5,7):note(a,0,root+19+(1 if bar%8==7 else 0),at+2,2.2,37)
        if section in (1,2):
            for q,degree in enumerate([0,0,7,1,0,0,7,1]):
                note(b,2,root+12+degree,at+q*.5,.18,45+(q%4==0)*11)
        else:
            for q,degree in enumerate([0,7,1,0]):note(b,2,root+12+degree,at+q,.20,41)
        if bar%2==1:
            note(b,1,root+24+color,at,3.8,32+section*3)
            note(b,1,root+25+color,at,3.8,23)
    return a,b

def render(biome):
    cfg=CONFIG[biome];beat=60/cfg['bpm'];length=32*4*beat;n=round(length*RATE)
    t=np.arange(n,dtype=np.float32)/RATE;rng=np.random.default_rng(cfg['seed'])
    base=np.zeros((n,2),np.float32);tension=np.zeros_like(base)
    root=cfg['root'];fundamental=round(hz(root)*length)/length
    # Periodic, slowly breathing stereo drone, sculpted out of dark harmonics.
    for ch in range(2):
        drone=np.zeros(n,np.float32)
        for j,(ratio,gain) in enumerate([(1,.030),(2,.015),(3,.007),(4,.004)]):
            f=round(fundamental*ratio*length)/length
            lfo=.82+.18*np.sin(TAU*t/length*(2+j)+ch*.7)
            phase=TAU*f*t+.06*np.sin(TAU*t/length*(3+j)+ch)
            drone+=np.sin(phase)*gain*lfo
        # Tension color floats softly above the pedal, rather than a tune.
        f=round(hz(root+13)*length)/length
        drone+=np.sin(TAU*f*t+ch*.55)*.003*(.5+.5*np.sin(TAU*t/length))
        base[:,ch]+=drone
        air=A.fft_filter(rng.standard_normal(n).astype(np.float32),lambda f:A.hp(460,2)(f)*A.lp(2200,3)(f))
        base[:,ch]+=air*(.0025+.0015*np.sin(TAU*t/length*5+ch))
    for bar in range(32):
        at=bar*4*beat;section=bar//8;energy=[.68,.84,1,.72][section]
        # Dotted, asymmetric pulse and a subdued double heartbeat.
        for q,step in enumerate([0,.75,1.5,2.5,3.25]):
            note=root+(7 if q==3 and bar%4==3 else 0)
            add(base,pulse(note,.48*beat),at+step*beat,.055*energy*(1 if q==0 else .62),-.10)
        for offset,gain in [(0,.048),(.31,.024)]:
            add(base,body(47 if biome!='ember' else 54,.75,cfg['seed']+bar,'drum'),at+offset*beat,gain*energy,-.06)
        if bar%2:
            add(base,body(430,.16,cfg['seed']+bar,'wood'),at+2.5*beat,.015,.42)
        for q in range(8):
            if q in (3,7) and bar%2==0:continue
            add(tension,pulse(root+(1 if q in (3,7) else 0),.30*beat),at+q*.5*beat,.045*energy,(-.2 if q%2 else .2))
            add(tension,body(560+q*17,.13,cfg['seed']+bar*10+q,'wood'),at+(q*.5+.25)*beat,.011*energy,(-.45 if q%2 else .45))
        for q in (0,2,3.5):add(tension,body(62,.8,cfg['seed']+bar*3+round(q),'drum'),at+q*beat,.032*energy,-.15)
        if bar%8 in (3,7):
            add(base,body(173 if biome!='frost' else 239,2.8,cfg['seed']+bar,'metal'),at+1.0*beat,.014,-.48)
        if bar%8==7:
            dur=3*beat;tt=np.arange(round(dur*RATE),dtype=np.float32)/RATE
            noise=rng.standard_normal(len(tt)).astype(np.float32)
            rise=A.fft_filter(noise,lambda f:A.hp(190,2)(f)*A.lp(2200,2)(f))*np.sin(np.pi*tt/dur)**2
            add(tension,rise,at+beat,.018,.15)
    events_a,events_b=sampled_events(cfg,beat)
    for tag,out,events in [('a',base,events_a),('b',tension,events_b)]:
        synth=Ensemble(ROOT/'Production/resources/TimGM6mb.sf2')
        samples=synth.render(events,length+3);synth.close()
        samples=A.fold_loop(samples,n)
        # Sampled cello/tremolo supplies bowed grain, behind the pulse texture.
        samples=np.stack([A.fft_filter(samples[:,ch],lambda f:A.hp(95,2)(f)*A.lp(2400,2)(f)) for ch in range(2)],1)
        out+=samples*(.48 if tag=='a' else .42)
        room=A.reverb_ir(2.4 if biome=='frost' else 1.9,4200,1350,seed=cfg['seed']+9)
        out=A.mix_reverb(out,room,.20 if tag=='a' else .12,loop_len=n)
        out=A.master(out,30,9200)
        # A short equal-power seam polish; all stems retain exact bar length.
        seam=round(.025*RATE)
        joined=(out[:seam]+out[-seam:])*.5
        blend=np.linspace(0,1,seam,dtype=np.float32)[:,None]
        out[:seam]=joined*(1-blend)+out[:seam]*blend
        out[-seam:]=out[-seam:]*(1-blend)+joined*blend
        peak=float(np.abs(out).max());out*=.88/max(.001,peak)
        folder=ROOT/'diamondquest/music_bank';folder.mkdir(exist_ok=True)
        source=ROOT/'Production/music_source';source.mkdir(exist_ok=True)
        wav=source/f'{biome}_{tag}.wav';A._save_wav(str(wav),out)
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-af',f'loudnorm=I={-24 if tag=="a" else -26}:TP=-3:LRA=10','-ar','44100','-c:a','libvorbis','-q:a','6',str(folder/f'{biome}_{tag}.ogg')],check=True)
        wav.unlink();midi_file(source/f'{biome}_{tag}.mid',events,cfg['bpm'])
        (source/f'{biome}_{tag}.json').write_text(json.dumps({'bpm':cfg['bpm'],'events':events,'style':'expedition thriller'}))
        print('Rendered',biome,tag,round(length,3),'seconds',flush=True)
    return dict(title=cfg['title'],bpm=cfg['bpm'],length=length,a=f'{biome}_a.ogg',b=f'{biome}_b.ogg')

if __name__=='__main__':
    A._set_rate(RATE)
    spec={'version':4,'style':'Expedition thriller','rate':RATE,'tracks':{}}
    for biome in CONFIG:spec['tracks'][biome]=render(biome)
    (ROOT/'diamondquest/music_bank/index.json').write_text(json.dumps(spec,indent=2))
