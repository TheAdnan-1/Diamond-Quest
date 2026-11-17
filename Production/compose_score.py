"""Render the original Expedition score offline with sampled instruments.

Build-time only: requires libfluidsynth, TimGM6mb.sf2, numpy and ffmpeg.
The game plays the bundled Ogg stems; none of these build tools are runtime dependencies.
"""
import ctypes as C
import ctypes.util
import json,math,struct,subprocess,sys,wave
from pathlib import Path
import numpy as np

RATE=44100
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diamondquest import audio as A


class Ensemble:
 def __init__(self,sf):
  self.lib=C.CDLL(ctypes.util.find_library('fluidsynth'))
  defs={'new_fluid_settings':(C.c_void_p,[]),'fluid_settings_setnum':(C.c_int,[C.c_void_p,C.c_char_p,C.c_double]),
        'fluid_settings_setint':(C.c_int,[C.c_void_p,C.c_char_p,C.c_int]),'new_fluid_synth':(C.c_void_p,[C.c_void_p]),
        'fluid_synth_sfload':(C.c_int,[C.c_void_p,C.c_char_p,C.c_int]),
        'fluid_synth_program_change':(C.c_int,[C.c_void_p,C.c_int,C.c_int]),
        'fluid_synth_cc':(C.c_int,[C.c_void_p,C.c_int,C.c_int,C.c_int]),
        'fluid_synth_noteon':(C.c_int,[C.c_void_p,C.c_int,C.c_int,C.c_int]),
        'fluid_synth_noteoff':(C.c_int,[C.c_void_p,C.c_int,C.c_int]),
        'fluid_synth_write_float':(C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_int]),
        'delete_fluid_synth':(None,[C.c_void_p]),'delete_fluid_settings':(None,[C.c_void_p])}
  for name,(ret,args) in defs.items():fn=getattr(self.lib,name);fn.restype=ret;fn.argtypes=args
  self.settings=self.lib.new_fluid_settings()
  for name,val in [('synth.sample-rate',RATE),('synth.gain',.48)]:self.lib.fluid_settings_setnum(self.settings,name.encode(),val)
  for name,val in [('synth.reverb.active',0),('synth.chorus.active',0),('synth.polyphony',256),('synth.cpu-cores',1)]:self.lib.fluid_settings_setint(self.settings,name.encode(),val)
  self.synth=self.lib.new_fluid_synth(self.settings)
  assert self.lib.fluid_synth_sfload(self.synth,str(sf).encode(),1)>=0
 def render(self,events,length):
  n=round(length*RATE);out=np.zeros((n,2),np.float32);cursor=0
  for event in sorted(events,key=lambda e:(e[0],e[1]))+[(n/RATE,9,0,0,0)]:
   position=min(n,max(cursor,round(event[0]*RATE)))
   while cursor<position:
    count=min(4096,position-cursor);block=np.empty((count,2),np.float32)
    ptr=block.ctypes.data_as(C.c_void_p)
    self.lib.fluid_synth_write_float(self.synth,count,ptr,0,2,ptr,1,2)
    out[cursor:cursor+count]=block;cursor+=count
   time,kind,ch,x,y=event
   if kind==0:self.lib.fluid_synth_noteoff(self.synth,ch,x)
   elif kind==1:self.lib.fluid_synth_noteon(self.synth,ch,x,y)
   elif kind==2:self.lib.fluid_synth_program_change(self.synth,ch,x)
   elif kind==3:self.lib.fluid_synth_cc(self.synth,ch,x,y)
  return out
 def close(self):self.lib.delete_fluid_synth(self.synth);self.lib.delete_fluid_settings(self.settings)


def vlq(n):
 out=[n&127];n>>=7
 while n:out.insert(0,(n&127)|128);n>>=7
 return bytes(out)


def midi_file(path,events,bpm):
 tick=lambda sec:round(sec*bpm/60*480)
 data=b'\x00\xff\x51\x03'+round(60_000_000/bpm).to_bytes(3,'big');last=0
 for t,kind,ch,x,y in sorted(events,key=lambda e:(e[0],e[1])):
  stamp=tick(t);data+=vlq(stamp-last);last=stamp
  if kind==0:data+=bytes([0x80+ch,x,0])
  elif kind==1:data+=bytes([0x90+ch,x,y])
  elif kind==2:data+=bytes([0xc0+ch,x])
  else:data+=bytes([0xb0+ch,x,y])
 data+=b'\x00\xff\x2f\x00'
 path.write_bytes(b'MThd'+struct.pack('>IHHH',6,0,1,480)+b'MTrk'+struct.pack('>I',len(data))+data)


CONFIG={
 'jungle':{'bpm':88,'root':50,'scale':[0,2,3,5,7,9,10],
  'prog':[(0,[0,3,7,14]),(5,[0,4,7,14]),(-2,[0,4,7,11]),(3,[0,4,7,11]),
          (0,[0,3,7,14]),(7,[0,3,7,10]),(5,[0,4,7,9]),(7,[0,5,7,10])],
  'lead':73,'plucked':46,'title':'The Emerald Descent'},
 'frost':{'bpm':72,'root':52,'scale':[0,2,3,5,7,8,10],
  'prog':[(0,[0,3,7,14]),(-4,[0,4,7,11]),(3,[0,4,7,11]),(-2,[0,5,7,14]),
          (7,[0,3,7,10]),(-4,[0,4,7,14]),(5,[0,3,7,14]),(7,[0,5,7,10])],
  'lead':75,'plucked':46,'title':'Cathedral of Ice'},
 'ember':{'bpm':104,'root':48,'scale':[0,2,3,5,7,8,11],
  'prog':[(0,[0,3,7,14]),(-4,[0,4,7,11]),(3,[0,4,7,14]),(-2,[0,5,7,10]),
          (0,[0,3,7,14]),(5,[0,3,7,10]),(-4,[0,4,7,14]),(7,[0,4,7,10])],
  'lead':69,'plucked':24,'title':'Ashes Beneath the Crown'}}


def score(biome):
 cfg=CONFIG[biome];bpm=cfg['bpm'];beat=60/bpm;bars=24;length=bars*4*beat
 rng=np.random.default_rng(600+len(biome));explore=[];tension=[]
 instruments={0:48,1:42,2:cfg['plucked'],3:cfg['lead'],4:52,5:11,6:60,7:47,8:116,10:45}
 pans={0:46,1:80,2:35,3:75,4:64,5:92,6:57,7:64,8:54,10:83}
 for out in [explore,tension]:
  for ch,program in instruments.items():
   out.append((0,2,ch,program,0));out.append((0,3,ch,10,pans[ch]));out.append((0,3,ch,91,0));out.append((0,3,ch,93,0))
 def note(out,ch,pitch,b,duration,velocity=70,human=True):
  seconds=max(0,b*beat+(float(rng.uniform(-.009,.009)) if human and b else 0))
  vel=int(np.clip(velocity+rng.integers(-5,6),20,118));duration*=beat
  out.append((seconds,1,ch,int(pitch),vel));out.append((seconds+max(.06,duration),0,ch,int(pitch),0))
 for bar in range(bars):
  section=bar//8;offset,intervals=cfg['prog'][bar%8];root=cfg['root']+offset
  b=bar*4;rise=.68+.25*math.sin((bar%8)/7*math.pi)
  # Sustained strings with inner-voice movement and long natural release.
  for j,interval in enumerate(intervals):
   pitch=root+12+interval
   while pitch>83:pitch-=12
   note(explore,0,pitch,b,3.85,44+section*5+j*3)
  note(explore,1,root-12,b,3.7,48+section*3)
  for q in range(8):
   at=b+q*.5
   target=root+12+intervals[(q+(bar%2))%len(intervals)]
   if target>84:target-=12
   if biome=='frost' and q in (1,3,5,7):continue
   note(explore,2,target,at,.95 if biome=='frost' else .60,48 if q%2 else 59)
  if bar%2==0:
   for pitch in [root+12,root+19]:note(explore,4,pitch,b,7.8,30+section*3)
  # A composed two-bar theme, with answer phrases rather than random pitches.
  if bar%2==0 and bar not in (0,16):
   motif=[0,2,4,3,2,1,4,2] if biome!='frost' else [0,3,2,1,4,2,1,0]
   rhythm=[0,.75,1.5,2.75,4,4.75,5.5,6.75]
   lengths=[.64,.64,1.05,.90,.64,.64,.95,1.0]
   if section==1:motif=[x+2 if j%3==0 else x for j,x in enumerate(motif)]
   for j,(degree,at,dur) in enumerate(zip(motif,rhythm,lengths)):
    pitch=cfg['root']+12+cfg['scale'][degree%7]+12*(degree//7)
    note(explore,3,pitch,b+at,dur,58+section*6-(j%3)*4)
  if bar%4==3:
   note(explore,5,root+24+intervals[1],b+2.5,1.5,40)
   note(explore,5,root+24+intervals[2],b+3.25,1.2,35)
  # A synchronized tension stem: bowed ostinato, horn answers, timpani and taiko.
  for q in range(8):
   degree=intervals[[0,2,1,2,0,3,1,2][q]]
   pitch=root+degree+(12 if q%2 else 0)
   note(tension,10,pitch,b+q*.5,.28,55+q%2*9+section*4)
  for at,vel in [(0,82),(1.5,57),(2,68),(3.5,48)]:
   note(tension,8,48 if at in (0,2) else 55,b+at,.42,vel)
  note(tension,7,root-12,b,.75,68)
  if bar%2==1:
   for pitch in [root,root+7,root+12]:note(tension,6,pitch,b+2,1.65,54+section*4)
  if bar%8==7:
   for j in range(8):note(tension,7,root,b+2+j*.25,.18,46+j*5)
  for out in [explore,tension]:
   for q in range(4):
    expression=int(78+15*math.sin((bar%8+q/4)/8*math.pi))
    out.append(((b+q)*beat,3,0,11,expression))
 return cfg,length,explore,tension


def main():
 soundfont=ROOT/'Production/resources/TimGM6mb.sf2'
 folder=ROOT/'diamondquest/music_bank';source=ROOT/'Production/music_source'
 folder.mkdir(exist_ok=True);source.mkdir(exist_ok=True)
 spec={'version':3,'rate':RATE,'tracks':{}}
 for biome in CONFIG:
  cfg,length,explore,tension=score(biome);parts={}
  for tag,events in [('a',explore),('b',tension)]:
   synth=Ensemble(soundfont);raw=synth.render(events,length+3.2);synth.close()
   room=A.reverb_ir(2.6 if biome=='frost' else 1.9,8500,1800,seed=171)
   n=round(length*RATE);audio=A.mix_reverb(raw,room,.24 if tag=='a' else .15,loop_len=n)
   audio=A.master(audio,34,15500)
   # Musical bar alignment with a very short boundary polish, no timing drift.
   fade=round(.020*RATE);audio[:fade]*=np.linspace(.0,1,fade)[:,None];audio[-fade:]*=np.linspace(1,.0,fade)[:,None]
   peak=float(np.abs(audio).max());audio*=.89/max(.00001,peak)
   wav=source/f'{biome}_{tag}.wav';A._save_wav(str(wav),audio)
   name=f'{biome}_{tag}.ogg'
   subprocess.run(['ffmpeg','-y','-v','error','-i',str(wav),'-af',f'loudnorm=I={-22 if tag=="a" else -24}:TP=-2:LRA=12','-ar','44100','-c:a','libvorbis','-q:a','6',str(folder/name)],check=True)
   midi_file(source/f'{biome}_{tag}.mid',events,cfg['bpm'])
   (source/f'{biome}_{tag}.json').write_text(json.dumps({'bpm':cfg['bpm'],'events':events}))
   wav.unlink();parts[tag]=name;print('Rendered',biome,tag,round(length,2),'seconds',flush=True)
  spec['tracks'][biome]={'title':cfg['title'],'bpm':cfg['bpm'],'length':length,**parts}
 (folder/'index.json').write_text(json.dumps(spec,indent=2))
if __name__=='__main__':main()
