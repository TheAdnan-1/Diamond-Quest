"""Expedition sound palette: material Foley, modal resonance and cinematic accents.

All samples are original deterministic synthesis. No downloaded recordings, runtime
network dependency, or gameplay decisions. Output is stereo floating-point PCM.
"""
import math
import numpy as np

NAMES=('step','step_ai','dig','wait','swing','clang','stun','crumble','push','hook_throw','hook_pull',
       'arm','gem_v','gem_r','unlock','burn','bite','crush','hurt','thud','splat','rattle','death',
       'exit','victory','defeat','hover','click','error','mine_place','explode','freeze','thaw','grind',
       'fire_loop','beam_loop')
VERSION=3


def build_bank(rate=44100):
    from . import audio as A
    A._set_rate(rate)
    ir=A.reverb_ir(.88,6700,1700,seed=431)
    ir_hall=A.reverb_ir(1.92,7800,1600,seed=432)
    bank={}

    def add(buf,sig,at=0,gain=1):
        start=int(at*rate);n=min(len(sig),len(buf)-start)
        if n>0:buf[start:start+n]+=sig[:n]*gain

    def tone(dur,freq,decay,amount=1):
        t=A.T(dur)
        return (np.sin(2*np.pi*np.cumsum(freq*(1+.025*np.exp(-t/.04)))/rate)*
                (1-np.exp(-t/.001))*np.exp(-t/decay)*amount).astype(np.float32)

    def band(dur,seed,low,high,decay=None,rise=.001):
        t=A.T(dur);s=A.fft_filter(A.noise(dur,seed),lambda f:A.hp(low,2)(f)*A.lp(high,2)(f))
        amp=(1-np.exp(-t/rise))
        if decay:amp*=np.exp(-t/decay)
        return s*amp

    def metal(dur,seed,f=380):
        rng=np.random.default_rng(seed);t=A.T(dur);y=np.zeros_like(t)
        for j,ratio in enumerate([1,2.756,5.404,8.933,13.34]):
            y+=tone(dur,f*ratio*rng.uniform(.997,1.003),.60/(1+j*.43),.38/(j+1)**.9)
        y+=band(dur,seed+1,2100,9100,.005)*.35
        return y

    def grains(dur,seed,count,material='stone'):
        rng=np.random.default_rng(seed);y=np.zeros(int(dur*rate),np.float32)
        for j in range(count):
            start=float(rng.uniform(0,max(.001,dur-.05)))
            length=float(rng.uniform(.014,.060));f=float(rng.uniform(360,2400))
            tap=band(length,seed*73+j,220,6500,.007)*.7
            tap+=tone(length,f,.012,.14)
            add(y,tap,start,float(rng.uniform(.2,.7))*math.exp(-start/max(.12,dur*.6)))
        return y

    def whoosh(dur,seed):
        t=A.T(dur);amp=np.sin(np.pi*np.clip(t/dur,0,1))**2
        air=band(dur,seed,180,5600,None,.004)
        rush=band(dur,seed+17,600,3000,None,.004)
        return (air*.8+rush*.55)*amp

    def crystal(dur,seed,root=760):
        rng=np.random.default_rng(seed);t=A.T(dur);y=np.zeros_like(t)
        for j,ratio in enumerate([1,2.04,2.83,4.16,5.39]):
            y+=tone(dur,root*ratio*rng.uniform(.992,1.008),.36/(1+j*.25),.28/(1+j))
        return y

    def render(name,i):
        seed=3109+sum((j+1)*ord(ch) for j,ch in enumerate(name))+i*197
        rng=np.random.default_rng(seed)
        dur={'explode':1.8,'crush':1.12,'crumble':.85,'freeze':1.12,'thaw':.78,'victory':2.3,'defeat':1.9,
             'exit':1.4,'unlock':1.2,'death':1.35,'push':.68,'grind':.82,'hook_pull':.62,'burn':1.25,
             'hook_throw':.38,'swing':.29,'rattle':.32,'bite':.60,'hover':.10,'click':.18,'error':.24,
             'step':.20,'step_ai':.24,'dig':.38,'mine_place':.24,'thud':.58,'fire_loop':3.0,'beam_loop':3.0}.get(name,.65)
        t=A.T(dur);y=np.zeros_like(t);pitch=rng.uniform(.95,1.05)
        wet=.10;hall=False;width=.3
        if name in ('step','dig'):
            y+=band(dur,seed,120,1900,.032)*1.4+tone(dur,92*pitch,.035,.65)
            y+=grains(dur,seed+2,7 if name=='step' else 23)*(.35 if name=='step' else 1.05)
            if name=='dig':y+=band(dur,seed+3,550,3300,.11)*.7
        elif name=='step_ai':
            # Soft martial-arts shoes and cloth replace the old robot clank.
            y+=tone(dur,104*pitch,.031,.58)+band(dur,seed+2,100,1700,.021)*1.25
            add(y,band(.13,seed+4,700,5100,.028)*.35,.052)
            y+=grains(dur,seed+6,5)*.20
        elif name=='swing':y+=whoosh(dur,seed)*1.9;wet=.06
        elif name=='clang':
            y+=metal(dur,seed,370*pitch)*1.0+tone(dur,105*pitch,.055,.55)
            y+=band(dur,seed+2,500,5000,.010)*.9;wet=.15
        elif name=='stun':
            y+=band(dur,seed,120,2200,.019)*1.2+tone(dur,83*pitch,.080,.9)
            add(y,metal(.50,seed+2,285)*.22,.015)
            add(y,band(.40,seed+3,1800,7600,.13,.01)*.25,.055)
        elif name in ('crumble','crush','thud'):
            force=1.5 if name=='crush' else (.75 if name=='thud' else .9)
            y+=tone(dur,62*pitch,.15,force)+tone(dur,108*pitch,.07,force*.50)
            y+=band(dur,seed,80,2200,.060)*1.5
            y+=grains(dur,seed+7,38 if name=='crush' else 24)*.95
            y+=band(dur,seed+3,180,800,.22,.015)*.65
            wet=.17;width=.70
        elif name in ('push','grind'):
            texture=band(dur,seed,90,1900,None,.006)
            scrape=.48+.20*np.sin(t*57+.5*np.sin(t*19))+.20*np.sin(t*97)
            y+=texture*scrape*np.sin(np.pi*t/dur)**.6*2.0
            y+=grains(dur,seed+5,23)*.32
            y+=tone(dur,77*pitch,.20,.25)
            wet=.08
        elif name=='hook_throw':
            y+=whoosh(dur,seed)*1.15
            add(y,metal(.14,seed+3,1220)*.18,.008)
            add(y,band(.08,seed+9,750,4200,.018)*.48,.015)
        elif name=='hook_pull':
            amp=np.sin(np.pi*t/dur)**.65
            y+=band(dur,seed,250,2700,None,.004)*amp*(.6+.22*np.sin(t*83))
            y+=tone(dur,139*pitch,.28,.30)+tone(dur,287*pitch,.16,.15)
            y+=grains(dur,seed+3,12)*.2
            wet=.055
        elif name in ('arm','mine_place','click','hover','wait','error'):
            if name=='hover':
                y+=band(dur,seed,900,4300,.012)*.28+tone(dur,1120,.018,.12)
            elif name=='error':
                for offset in (0,.075):add(y,tone(.09,240,.025,.6)+band(.09,seed,350,1900,.012)*.55,offset)
            elif name=='wait':
                y+=tone(dur,380,.025,.5)+band(dur,seed,500,2200,.018)*.4
                add(y,tone(.12,290,.02,.25),.13)
            else:
                y+=band(dur,seed,200,2700,.014)*.9+metal(dur,seed+3,720)*.28
                if name=='mine_place':
                    y+=tone(dur,143,.04,.35)
                    add(y,metal(.14,seed+11,1100)*.2,.07)
            wet=.04
        elif name in ('gem_v','gem_r','unlock'):
            root=(980 if name=='gem_v' else 1240)*pitch
            y+=crystal(dur,seed,root)*.85
            y+=band(dur,seed+4,4000,9000,.01)*.2
            if name in ('gem_r','unlock'):
                add(y,crystal(dur*.75,seed+2,root*1.5)*.45,.07)
            if name=='unlock':
                add(y,crystal(.6,seed+7,root*2)*.32,.17)
            wet=.22;hall=True;width=.7
        elif name=='burn':
            # Ignition, rising flame, then a cascade of embers and cloth swish.
            y+=band(dur,seed,60,2400,.31,.006)*1.45
            y+=tone(dur,55+80*np.exp(-t/.035),.16,.65)
            for j,at in enumerate((.03,.19,.43)):
                add(y,whoosh(.43,seed+30+j)*.55,at)
            y+=grains(dur,seed+3,64)*.63
            y+=band(dur,seed+5,3800,11000,.35,.025)*.28
            wet=.10;width=.75
        elif name=='bite':
            # Rapid inhalation, close jaw snap, leathery contact, venom flecks.
            y+=band(dur,seed,2800,11000,.080,.001)*.50
            add(y,band(.13,seed+3,700,7200,.005)*1.7+tone(.13,178,.019,.75),.032)
            add(y,band(.24,seed+4,100,1450,.040)*.8,.035)
            add(y,grains(.30,seed+9,16)*.18,.08)
            wet=.08;width=.22
        elif name=='rattle':
            y+=band(dur,seed,2800,7900,None)*np.sin(np.pi*t/dur)**.6*(.5+.5*np.sin(t*170))*.7
            wet=.07
        elif name in ('hurt','splat'):
            y+=tone(dur,128*pitch,.05,.7)+band(dur,seed,150,2400,.026)*1.3
            if name=='splat':y+=grains(dur,seed+4,10)*.3
            wet=.07
        elif name=='explode':
            # Initial crack, a falling low body, combustion, then falling debris.
            y+=band(dur,seed,1600,8000,.010)*1.3
            frequency=43+83*np.exp(-t/.055)
            y+=np.sin(2*np.pi*np.cumsum(frequency)/rate)*(1-np.exp(-t/.002))*np.exp(-t/.30)*1.6
            body=band(dur,seed+4,38,1400,.29,.012)
            y+=np.tanh(body*2.2)*1.2
            y+=band(dur,seed+7,200,5300,.19,.005)*.6
            add(y,grains(1.25,seed+11,65)*.65,.15)
            wet=.21;hall=True;width=.9
        elif name in ('freeze','thaw'):
            y+=grains(dur,seed+7,52 if name=='freeze' else 40)*.6
            y+=crystal(dur,seed,1450*pitch)*.65
            frost=band(dur,seed+2,2600,9000,.18,.003)
            y+=frost*.52
            if name=='freeze':
                zap=np.sin(2*np.pi*np.cumsum(880+650*np.exp(-t/.055))/rate)*np.exp(-t/.10)*.38
                y+=zap
            else:y+=band(dur,seed+9,300,2400,.04)*.4
            wet=.22;hall=True;width=.80
        elif name in ('exit','death'):
            breath=whoosh(dur,seed)
            y+=breath*(.55 if name=='exit' else .75)
            for j,root in enumerate((260,390,520) if name=='exit' else (300,201,134)):
                add(y,crystal(dur*.65,seed+j,root)*.4,j*.12)
            y+=tone(dur,70,.30,.40)
            wet=.25;hall=True;width=.8
        elif name in ('victory','defeat'):
            notes=[392,494,587,784] if name=='victory' else [392,294,247,196]
            for j,f in enumerate(notes):
                add(y,crystal(1.25,seed+j,f)*.58,j*.16)
                add(y,tone(1.0,f/2,.35,.18),j*.16)
            if name=='victory':
                for f in [392,494,587]:add(y,tone(1.5,f,.65,.14),.55)
            wet=.23;hall=True;width=.8
        elif name in ('fire_loop','beam_loop'):
            # Periodic filtered noise and wrapped grains avoid clicks at the loop seam.
            y=band(dur,seed,75 if name=='fire_loop' else 1100,3800 if name=='fire_loop' else 6400,None,.002)
            mod=.65+.15*np.sin(TAU*t/dur*3)+.10*np.sin(TAU*t/dur*7+.4)
            y*=mod
            if name=='fire_loop':y+=grains(dur,seed+2,65)*.15
            else:y+=np.sin(TAU*717*t)*.017+np.sin(TAU*1063*t)*.011
            fade=int(.10*rate)
            seam=(y[:fade]+y[-fade:])*.5
            y[:fade]=seam;y[-fade:]=seam
            y-=np.linspace(0,1,len(y),dtype=np.float32)*(y[-1]-y[0])
            y-=y.mean()
            wet=.0;width=.70
        else:raise KeyError(name)
        # Every family has a new attack/body/tail treatment. Small material
        # details are separated in time so the effect reads at game volume.
        if name in ('step','step_ai','dig','push','grind','hook_throw','hook_pull','mine_place','arm'):
            add(y,band(min(.19,dur),seed+191,850,6400,.038,.004)*.22,min(.065,dur*.20))
        if name in ('clang','crumble','crush','thud','explode','stun','splat','hurt'):
            y+=tone(dur,(73 if name in ('crush','explode') else 119)*pitch,.075,.20)
            add(y,grains(min(.38,dur),seed+251,14)*.18,min(.08,dur*.15))
        if name in ('gem_v','gem_r','unlock','exit','freeze','thaw','death'):
            for j,at in enumerate((.045,.11,.23)):
                root=(1540 if name in ('freeze','thaw') else 1150)*pitch*(1+j*.25)
                add(y,metal(min(.66,dur),seed+301+j,root)*.12,at)
        if name in ('click','hover','error','wait'):
            # Felt-damped stone/metal detents rather than pure electronic beeps.
            y+=band(dur,seed+211,160,4600,.009)*.30+tone(dur,360*pitch,.012,.15)
        y=np.tanh(y*1.12)/1.12
        y=A.fft_filter(y,lambda f:A.hp(39 if name not in ('explode','crush') else 30,2)(f)*A.lp(14500,2)(f))
        dry=A.stereo(y,0,width)
        if not name.endswith('_loop'):
            # Irregular early wall reflections give depth without smearing the
            # direct transient. Diffuse tails remain quieter than the body.
            reflected=np.zeros_like(dry)
            for delay,gain in [(.017,.10),(.037,.075),(.061,.045)]:
                offset=round(delay*rate)
                if offset<len(dry):reflected[offset:,0]+=dry[:-offset,1]*gain;reflected[offset:,1]+=dry[:-offset,0]*gain*.86
            dry+=reflected
        result=A.mix_reverb(dry,ir_hall if hall else ir,wet) if wet else dry
        return result.astype(np.float32)

    TAU=math.tau
    for name in NAMES:
        count=1 if name.endswith('_loop') else (2 if name in ('victory','defeat','unlock','exit') else 6)
        bank[name]=[render(name,i) for i in range(count)]
    return bank


def ambience(biome,rate=44100,dur=24.0):
    """Quiet continuous cave Foley: wind/room tone and distant material details."""
    from . import audio as A
    A._set_rate(rate);n=int(dur*rate);t=A.T(dur)
    out=np.zeros((n,2),np.float32);rng=np.random.default_rng(912+len(biome))
    for ch in range(2):
        source=A.noise(dur,714+ch)
        if biome=='jungle':
            air=A.fft_filter(source,lambda f:A.hp(80,2)(f)*A.lp(900,2)(f))
            out[:,ch]+=air*(.020+.005*np.sin(math.tau*t/dur*3+ch))
        elif biome=='frost':
            air=A.fft_filter(source,lambda f:A.hp(100,2)(f)*A.lp(1400,2)(f))
            out[:,ch]+=air*(.024+.009*np.sin(math.tau*t/dur*2+ch))
        else:
            rumble=A.fft_filter(source,lambda f:A.hp(35,2)(f)*A.lp(210,3)(f))
            out[:,ch]+=rumble*.11
            ash=A.fft_filter(A.noise(dur,716+ch),lambda f:A.hp(1400,2)(f)*A.lp(5500,2)(f))
            out[:,ch]+=ash*.006*(1+.25*np.sin(math.tau*t/dur*4))
    for j in range(17 if biome!='ember' else 38):
        start=int(rng.uniform(.4,dur-1.8)*rate);length=.8 if biome!='ember' else .12
        tt=A.T(length);seed=9000+j;event=np.zeros_like(tt)
        if biome=='jungle':
            f=float(rng.uniform(850,1550))
            event=np.sin(math.tau*np.cumsum(f+f*.8*np.exp(-tt/.012))/rate)*np.exp(-tt/.045)*.06
            event+=A.fft_filter(A.noise(length,seed),A.bp(2000,1.0))*np.exp(-tt/.025)*.035
        elif biome=='frost':
            for f in [310,619,1420]:event+=np.sin(math.tau*f*tt)*np.exp(-tt/.08)*.025
            event+=A.fft_filter(A.noise(length,seed),lambda f:A.hp(650,2)(f)*A.lp(5000,2)(f))*np.exp(-tt/.025)*.13
        else:
            event=A.fft_filter(A.noise(length,seed),lambda f:A.hp(800,2)(f)*A.lp(6200,2)(f))*np.exp(-tt/.014)*.065
        stereo=A.stereo(event,float(rng.uniform(-.7,.7)),.65)
        count=min(len(stereo),n-start);out[start:start+count]+=stereo[:count]
    room=A.reverb_ir(1.5,4300,900,seed=721)
    out=A.mix_reverb(out,room,.20,loop_len=n)
    # Match the boundary quietly; bed and events remain restrained under music.
    out-=np.linspace(0,1,len(out),dtype=np.float32)[:,None]*(out[-1]-out[0])[None,:]
    peak=float(np.abs(out).max())
    if peak>.30:out*=.30/peak
    return out.astype(np.float32)
