"""Original driving expedition score, rendered offline into synchronized stems.

Music only. The game, material effects and their playback settings are untouched.
The full groove is present in stem A; stem B adds intensity near hazards.
"""
from pathlib import Path
import sys,json,math,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from diamondquest import audio as A
from compose_score import Ensemble,midi_file

RATE=44100;TAU=math.tau
CONFIG={
 'jungle':dict(root=50,bpm=126,title='Danger in the Ruins',seed=1801),
 'frost':dict(root=52,bpm=130,title='Icebound Pursuit',seed=1802),
 'ember':dict(root=48,bpm=138,title='Cinder Run',seed=1803),
}

def hz(note):return 440*2**((note-69)/12)

def place(out,signal,at,gain=1,pan=0):
 if signal.ndim==1:signal=A.stereo(signal,pan,.12)
 start=round(at*RATE)%len(out);count=min(len(signal),len(out)-start)
 out[start:start+count]+=signal[:count]*gain
 tail=signal[count:]
 if len(tail):out[:len(tail)]+=tail*gain

def drum(kind,seed,pitch=65):
 rng=np.random.default_rng(seed)
 duration={'kick':.65,'snare':.50,'tom':.64,'hat':.12,'crash':1.8,'wood':.17}[kind]
 t=np.arange(round(duration*RATE),dtype=np.float32)/RATE
 noise=rng.standard_normal(len(t)).astype(np.float32);attack=1-np.exp(-t/.0015)
 if kind=='kick':
  f=46+102*np.exp(-t/.024)
  body=np.sin(TAU*np.cumsum(f)/RATE)*np.exp(-t/.17)
  punch=np.sin(TAU*106*t)*np.exp(-t/.036)*.42
  click=A.fft_filter(noise,lambda f:A.hp(900,2)(f)*A.lp(5700,2)(f))*np.exp(-t/.005)*.24
  y=(body+punch)*attack+click
 elif kind in ('snare','tom'):
  f=(154 if kind=='snare' else pitch)*(1+.38*np.exp(-t/.014))
  ph=TAU*np.cumsum(f)/RATE
  y=(np.sin(ph)+.41*np.sin(ph*1.59)+.18*np.sin(ph*2.17))*np.exp(-t/(.065 if kind=='snare' else .18))*attack
  snap=A.fft_filter(noise,lambda f:A.hp(750 if kind=='snare' else 240,2)(f)*A.lp(8300 if kind=='snare' else 3500,2)(f))
  y+=snap*np.exp(-t/(.070 if kind=='snare' else .018))*(.85 if kind=='snare' else .25)
  if kind=='snare':
   for delay,level in [(0,.38),(.012,.27),(.026,.18)]:
    q=np.maximum(0,t-delay);env=(t>=delay)*np.exp(-q/.025)
    y+=snap*env*level
 elif kind=='hat':
  y=A.fft_filter(noise,lambda f:A.hp(4300,3)(f)*A.lp(11800,2)(f))*np.exp(-t/.018)*attack
  y+=sum(np.sin(TAU*f*t)*np.exp(-t/.013)*.025 for f in (5351,7163,9631))
 elif kind=='crash':
  y=A.fft_filter(noise,lambda f:A.hp(1800,2)(f)*A.lp(11000,2)(f))*np.exp(-t/.43)*attack
  y*=.68+.21*np.sin(t*79)*np.sin(t*117)
 else:
  y=sum(np.sin(TAU*pitch*ratio*t)*np.exp(-t/decay)*gain for ratio,decay,gain in [(1,.026,.5),(2.31,.018,.19),(4.18,.01,.08)])*attack
  y+=A.fft_filter(noise,A.lp(4200,2))*np.exp(-t/.006)*.15
 return np.tanh(y*1.12).astype(np.float32)

def bass(note,duration,seed):
 t=np.arange(round(duration*RATE),dtype=np.float32)/RATE;f=hz(note)
 # A tight, bright harmonic bass speaks on small speakers as well as a sub.
 left=np.zeros_like(t);right=np.zeros_like(t)
 for j in range(1,19):
  gain=1/j**1.40
  decay=np.exp(-t/(.23/j**.25))
  left+=np.sin(TAU*f*j*t)*gain*decay
  right+=np.sin(TAU*f*j*1.0012*t+.004*j)*gain*decay
 env=(1-np.exp(-t/.003))*np.minimum(1,np.maximum(0,(duration-t)/.025))
 stereo=np.stack([left,right],1)*env[:,None]
 return np.tanh(stereo*1.3).astype(np.float32)*.67

def pluck(note,duration):
 t=np.arange(round(duration*RATE),dtype=np.float32)/RATE;f=hz(note)
 y=sum(np.sin(TAU*f*j*t)*np.exp(-t/(.15/j**.3))/j**1.6 for j in range(1,13))
 env=(1-np.exp(-t/.002))*np.minimum(1,np.maximum(0,(duration-t)/.025))
 return np.tanh(y*1.15)*env

def event_score(cfg):
 root=cfg['root'];beat=60/cfg['bpm'];rng=np.random.default_rng(cfg['seed']);a=[];b=[]
 # Pizzicato, short ensemble layers, horns, violin and cello. No flute or bell lead.
 programs={0:45,1:48,2:60,3:40,4:42,5:44}
 for out in (a,b):
  for ch,program in programs.items():
   out.extend([(0,2,ch,program,0),(0,3,ch,10,[36,88,62,70,48,95][ch]),(0,3,ch,91,0),(0,3,ch,93,0)])
 def note(out,ch,pitch,at,dur,vel):
  seconds=max(0,at*beat+float(rng.uniform(-.004,.004)))
  velocity=int(np.clip(vel+rng.integers(-4,5),28,120))
  out.extend([(seconds,1,ch,pitch,velocity),(seconds+dur*beat,0,ch,pitch,0)])
 progression=[(0,[0,3,7]),(0,[0,3,7]),(-4,[0,4,7]),(-4,[0,4,7]),(-7,[0,3,7]),(-7,[0,3,7]),(-5,[0,4,7]),(-5,[0,4,7])]
 motif=[(0,12,.70),(1,7,.45),(1.75,12,.65),(2.75,15,.8),(4,14,.70),(5,12,.45),(5.75,11,.65),(6.75,12,.95)]
 for bar in range(48):
  section=bar//8;offset,chord=progression[bar%8];r=root+offset;at=bar*4
  energy=[0,4,7,-3,9,4][section]
  pattern=[0,2,1,2,0,2,1,2]
  for q,idx in enumerate(pattern):
   pitch=r+12+chord[idx]
   note(a,0,pitch,at+q*.5,.22,84+energy+(q%4==0)*10)
   note(a,1,pitch,at+q*.5,.28,55+energy)
   if section in (1,2,4,5):note(b,0,pitch+12,at+q*.5+.25,.12,72)
  for pitch in [r,r+7]:note(a,4,pitch,at,3.78,62+energy)
  if bar%4 in (0,2):
   for interval in chord:note(a,2,r+interval,at,1.1 if bar%4==0 else .7,58+energy)
  if section in (1,2,4,5) and bar%4==0:
   for step,pitch,dur in motif:note(a,3,root+pitch,at+step,dur,77+energy)
  elif section==0 and bar%4==0:
   for step,pitch,dur in motif[:4]:note(a,3,root+pitch,at+step,dur,62)
  for q in range(16):
   degree=chord[[0,2,0,1][q%4]]
   note(b,0,r+12+degree+(12 if q%4==3 else 0),at+q*.25,.115,65+(q%4==0)*16)
  if bar%2==0:
   for interval in (0,7,12):note(b,2,r+interval,at+2.5,1.25,66+energy)
  if bar%8 in (5,7):
   note(b,5,root+25,at+2,1.8,49+energy)
 return a,b,progression

def render(biome):
 cfg=CONFIG[biome];beat=60/cfg['bpm'];length=48*4*beat;n=round(length*RATE)
 base=np.zeros((n,2),np.float32);tension=np.zeros_like(base)
 rng=np.random.default_rng(cfg['seed']);events_a,events_b,progression=event_score(cfg)
 kick=drum('kick',101);snare=drum('snare',102);hat=drum('hat',103);crash=drum('crash',104)
 for bar in range(48):
  at=bar*4*beat;section=bar//8;energy=[.85,.98,1.06,.81,1.12,.98][section]
  offset,chord=progression[bar%8];root=cfg['root']+offset
  # The full beat is in the always-audible exploration stem from bar one.
  for q in (0,1.5,2,2.75):place(base,kick,at+q*beat,.15*energy,-.04)
  for q in (1,3):place(base,snare,at+q*beat,.083*energy,.08)
  for q in range(16):
   gain=(.017 if q%2==0 else .010)*energy
   if q%4==3:gain*=.65
   place(base,hat,at+q*.25*beat,gain,(-.32 if q%2 else .32))
  for q,step in enumerate([0,.5,1,1.75,2,2.5,3,3.5]):
   interval=7 if q in (3,7) else 0
   place(base,bass(root-12+interval,.34*beat,cfg['seed']+q),at+step*beat,.078*energy)
  for q,idx in enumerate([0,2,1,2,0,2,1,2]):
   place(base,pluck(root+12+chord[idx],.33*beat),at+q*.5*beat,.018*energy,(-.22 if q%2 else .22))
  for q in (0,.75,1.5,2,2.5,3.5):
   place(tension,drum('tom',cfg['seed']+q.__hash__()%99,72 if q%1 else 58),at+q*beat,.075*energy,(-.27 if q%1 else .24))
  for q in (.5,1.5,2.5,3.5):place(tension,drum('wood',190,760),at+q*beat,.018*energy,.40)
  if bar%4==3:
   for j,q in enumerate([2.5,3,3.25,3.5,3.75]):
    place(base,drum('tom',500+j,78+j*13),at+q*beat,.040*(.7+j*.1)*energy,(-.35+j*.17))
  if bar%8==0:
   place(base,crash,at,.045*energy,-.12);place(tension,kick,at,.13)
  if bar%8==7:
   duration=2*beat;tt=np.arange(round(duration*RATE),dtype=np.float32)/RATE
   air=A.fft_filter(rng.standard_normal(len(tt)).astype(np.float32),lambda f:A.hp(1200,2)(f)*A.lp(9000,2)(f))
   air*=np.sin(np.pi*tt/(2*duration))**2
   place(base,air,at+2*beat,.026,-.12)
   place(tension,air,at+2*beat,.024,.2)
 for tag,out,events in [('a',base,events_a),('b',tension,events_b)]:
  synth=Ensemble(ROOT/'Production/resources/TimGM6mb.sf2');samples=synth.render(events,length+2.7);synth.close()
  samples=A.fold_loop(samples,n)
  samples=np.stack([A.fft_filter(samples[:,ch],lambda f:A.hp(140,2)(f)*A.lp(7800,2)(f)) for ch in range(2)],1)
  out+=samples*(.84 if tag=='a' else .62)
  room=A.reverb_ir(1.45,7200,1900,seed=cfg['seed'])
  out=A.mix_reverb(out,room,.13 if tag=='a' else .11,loop_len=n)
  out=A.master(out,32,15500)
  # Gentle saturation retains drum transients while adding harmonic body.
  out=np.tanh(out*1.12)/1.12
  # Folded releases preserve the bar line; only a 3ms edge polish is required.
  seam=round(.003*RATE);blend=np.linspace(0,1,seam,dtype=np.float32)[:,None]
  joined=(out[:seam]+out[-seam:])*.5
  out[:seam]=joined*(1-blend)+out[:seam]*blend
  out[-seam:]=out[-seam:]*(1-blend)+joined*blend
  out*=.90/max(.001,float(np.abs(out).max()))
  folder=ROOT/'diamondquest/music_bank';source=ROOT/'Production/music_source'
  wav=source/f'{biome}_{tag}.wav';A._save_wav(str(wav),out)
  target=-17 if tag=='a' else -22
  subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-af',f'loudnorm=I={target}:TP=-2:LRA=8','-ar','44100','-c:a','libvorbis','-q:a','7',str(folder/f'{biome}_{tag}.ogg')],check=True)
  wav.unlink();midi_file(source/f'{biome}_{tag}.mid',events,cfg['bpm'])
  (source/f'{biome}_{tag}.json').write_text(json.dumps({'bpm':cfg['bpm'],'style':'driving expedition action thriller','events':events}))
  print('Rendered',biome,tag,round(length,3),'seconds at',cfg['bpm'],'BPM',flush=True)
 return dict(title=cfg['title'],bpm=cfg['bpm'],length=length,a=f'{biome}_a.ogg',b=f'{biome}_b.ogg')

if __name__=='__main__':
 A._set_rate(RATE)
 spec={'version':5,'style':'Driving expedition action thriller','rate':RATE,'tracks':{}}
 for biome in CONFIG:spec['tracks'][biome]=render(biome)
 (ROOT/'diamondquest/music_bank/index.json').write_text(json.dumps(spec,indent=2))
