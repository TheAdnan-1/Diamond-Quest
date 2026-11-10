"""Weighted action and hazard choreography. Never changes a logical World.

Authoritative events select the visuals; every position, hold and recoil in this
module is a render offset. The rules engine and event payloads remain untouched.
"""
import math
from . import gfx
from .gpu import ADD,BLEND
from .fx_art import guardian_anchor
from .art_director import character_source

TAU=math.tau
DURATION={'hammer':.72,'push':.70,'hook':1.02}


def sat(x):return max(0.0,min(1.0,x))
def ease(x):return gfx.ease_in_out(sat(x))


class CinematicFX:
    def __init__(self,view):
        self.v=view;self.actions=[];self.clips=[];self.marks=[];self.bites={};self.burns={};self.reactions={};self.rock_hits={};self.ghostwalls=[];self.freeze_times={};self.source_positions={}
        self.eye=guardian_anchor(view.biome)
        self.stun_times={};self.waking={};self.venom=[]

    def update(self,now):
        self.actions=[a for a in self.actions if now<a['t']+a['dur']]
        self.clips=[a for a in self.clips if now<a['t']+a['dur']]
        self.marks=[a for a in self.marks if now<a['t']+a['dur']]
        self.ghostwalls=[a for a in self.ghostwalls if now<a[1]]
        self.bites={k:a for k,a in self.bites.items() if now<a['t']+.90}
        self.burns={k:t for k,t in self.burns.items() if now<t+1.35}
        self.waking={k:t for k,t in self.waking.items() if now<t+.58}
        self.venom=[a for a in self.venom if now<a['t']+.65]
        self.reactions={k:a for k,a in self.reactions.items() if now<a[1]+a[2]}
        self.rock_hits={k:t for k,t in self.rock_hits.items() if now<t+.55}

    def clip(self,name,pos,now,dur,size,frames=24):
        self.clips.append(dict(name=name,pos=pos,t=now,dur=dur,size=size,frames=frames))

    def mark(self,name,pos,now,size=1.5,dur=12):
        self.marks.append(dict(name=name,pos=pos,t=now,dur=dur,size=size))

    def dust(self,x,y,count=6,speed=1.0,size=.8):
        v=self.v;r=v.rng
        for i in range(count):
            a=r.random()*TAU;sp=r.uniform(.3,1)*speed
            v.part(x,y,math.cos(a)*sp,math.sin(a)*sp*.30-.08,r.uniform(.6,1.15),'fx_dust',size*r.uniform(.6,1.1),
                   (185,172,146),-.12,r.uniform(-30,30),r.uniform(-18,18),BLEND,True,.8,125,.28)

    def splinters(self,x,y,kind,count=14,speed=4):
        v=self.v;r=v.rng
        for i in range(count):
            a=r.random()*TAU;sp=r.uniform(.3,1)*speed
            color=(255,255,255)
            v.part(x,y,math.cos(a)*sp,math.sin(a)*sp*.65-sp*.45,r.uniform(.45,.9),'fx_'+kind,
                   r.uniform(.07,.16),color,8,r.uniform(0,360),r.uniform(-480,480),BLEND,True,.4,245)

    def sparks(self,x,y,count=12,color=(255,201,114),speed=4):
        v=self.v;r=v.rng
        for i in range(count):
            a=r.random()*TAU;sp=r.uniform(.2,1)*speed
            v.part(x,y,math.cos(a)*sp,math.sin(a)*sp-sp*.25,r.uniform(.15,.42),'fx_spark',r.uniform(.10,.24),
                   color,5,math.degrees(a),r.uniform(-30,30),ADD,False,.4,235)

    def begin(self,kind,pid,event,t0):
        v=self.v;pr=self.source_positions.get(pid,v.player_cell_f(pid,t0))
        target=event.get('pos',event.get('from'))
        dr,dc=target[0]-pr[0],target[1]-pr[1]
        fac=('right' if dc>0 else 'left') if abs(dc)>=abs(dr) else ('down' if dr>0 else 'up')
        a=dict(kind=kind,pid=pid,t=t0,dur=DURATION[kind],fac=fac,target=target,event=event,source=pr)
        self.actions.append(a)
        return a

    def hammer(self,pid,event,t0):
        a=self.begin('hammer',pid,event,t0);v=self.v;target=event['pos']
        if event['target']=='wall':self.ghostwalls.append((target,t0+.34))
        else:self.stun_times[event['sid']]=t0+.34
        def hit():
            x,y=target[1]+.5,target[0]+.52
            if event['target']=='wall':self.splinters(x,y,'stone',18,4.2)
            self.dust(x,y,8,1.6,.72);self.sparks(x,y,12)
            self.mark('fracture',target,v._now,1.20,5)
            v.light_flash(x,y,2.0,(255,212,145),.16,.7)
            v.shake=max(v.shake,.13);v.invalidate(target)
            if event['target']=='snake':
                self.clip('stun_contact',target,v._now,.32,1.25,1)
                v.light_flash(x,y,1.3,(161,218,252),.22,.55)
        v.at(t0+.34,hit)

    def push(self,pid,event,t0):
        self.begin('push',pid,event,t0);v=self.v
        for j in range(5):
            def scrape(j=j):
                q=j/4;r=event['from'][0];c=gfx.lerp(event['from'][1],event['to'][1],q)
                self.dust(c+.5,r+.85,2,.65,.46)
                if j in (1,3):self.sparks(c+.5,r+.82,3,(205,177,126),1.5)
            v.at(t0+.24+j*.065,scrape)

    def hook(self,pid,event,key,t0):
        a=self.begin('hook',pid,event,t0);a['key']=key
        def catch():
            src=event['from'];self.sparks(src[1]+.5,src[0]+.45,9,(209,228,231),2.4)
            self.v.light_flash(src[1]+.5,src[0]+.45,1.1,(157,222,227),.13,.35)
        self.v.at(t0+.38,catch)

    def recoil(self,pid,kind,now,dur=.55,sign=1):self.reactions[pid]=(kind,now,dur,sign)

    def reaction(self,pid,now):
        a=self.reactions.get(pid)
        if not a:return 0,0,0,1
        kind,t,dur,sign=a;k=sat((now-t)/dur)
        kick=math.sin(k*math.pi)*math.exp(-k*2)
        if kind=='crush':return sign*.035*math.sin(k*22)*(1-k),.12*kick,sign*3*kick,1-.22*math.sin(k*math.pi)**2
        if kind=='burn':
            shake=math.sin(k*34)*math.exp(-k*3)*.04
            return shake,-.13*math.sin(k*math.pi)**2,shake*105,1-.06*math.sin(k*math.pi*2)
        if kind=='bite':return sign*.13*math.sin(k*math.pi)*math.exp(-k),-.06*kick,sign*15*kick,1-.09*kick
        return -sign*.16*kick,-.035*math.sin(k*math.pi),-sign*9*kick,1-.04*kick

    def actor(self,pid,now):
        return next((a for a in reversed(self.actions) if a['pid']==pid and a['t']<=now<a['t']+a['dur']),None)

    def draw_actor(self,p,pos,now):
        a=self.actor(p.pid,now)
        if not a:return False
        v=self.v;g=v.g;ts=v.ts;k=sat((now-a['t'])/a['dur']);kind=a['kind'];fac=a['fac']
        phase=[.0,.18,.45,.73,1]
        if kind=='push':phase=[0,.22,.42,.80,1]
        if kind=='hook':phase=[0,.17,.40,.78,1]
        idx=next((i for i in range(4) if phase[i]<=k<phase[i+1]),3)
        pose=fac if kind=='hammer' and fac in ('up','down') else 'side'
        readable_left=p.pid==0 and fac=='left' and pose=='side'
        if readable_left:pose='left'
        key=f'action_{p.pid}_{kind}_{pose}/{idx}'
        tex=v.A.get(key,ts*2)
        if tex is None:return False
        x,y=v.P(pos[1]+.5,pos[0]+.94)
        lean=math.sin(k*math.pi)*(.08 if kind!='hook' else -.045)
        x+=(-1 if fac=='left' else 1)*lean*ts if pose in ('side','left') else 0
        # A slight continuous weight shift bridges authored contact poses.
        hh=ts*2*(1-.015*math.sin(k*TAU));ww=ts*2
        if kind=='push':hh*=.84 if idx==2 else (.94 if idx==1 else 1)
        g.draw(tex,x-ww/2,y-hh,ww,hh,flip=fac=='left' and not readable_left)
        return True

    def lens(self,st,pos=None,face=None):
        v=self.v
        if pos is None:pos=v.tw.pos(('statue',st.sid),(st.r,st.c),v._now)
        if face is None:face=v.tw.pos(('sface',st.sid),st.d,v._now)
        return pos[1]+.5+self.eye[0]*face,pos[0]+self.eye[1]

    def draw_statue(self,data,now,glows):
        st,pos=data;v=self.v;g=v.g;ts=v.ts
        face=v.tw.pos(('sface',st.sid),st.d,now)
        front=max(0,1-abs(face))**.65
        x,y=v.P(pos[1]+.5,pos[0]+1.98);W,H=ts*1.1,ts*2.2
        side=v.A.get(f'{v.biome}/guardian/right',W)
        mid=v.A.get(f'{v.biome}/guardian/front',W)
        if side:g.draw(side,x-W/2,y-H,W,H,flip=face<0)
        if mid and front>.01:g.draw(mid,x-W/2,y-H,W,H,alpha=255*front)
        ex,ey=v.P(*self.lens(st,pos,face));pulse=.75+.15*math.sin(now*3.5+st.sid)
        glows.append(((ex,ey),ts*.23,(117,214,244),.55*pulse))
        g.disc(ex,ey,ts*.022,(222,255,255),200)

    def draw_beams(self,now):
        v=self.v;g=v.g;ts=v.ts
        if now>=v.beam_hold or v.beam_segs is None:v.beam_segs=v._beam_segments()
        busy={key[1] for key in v.tw.seg if key[0] in ('statue','sface') and v.tw.active(key,now)}
        statues={s.sid:s for s in v.world.statues.values()}
        for sid,r,cells,d in v.beam_segs:
            if not cells or sid in busy or sid not in statues:continue
            st=statues[sid];ox,oy=self.lens(st);x1,y=v.P(ox,oy)
            x2,_=v.P(cells[-1][1]+(1 if d>0 else 0),oy)
            length=abs(x2-x1)
            if length<1:continue
            tex=v.A.get(f'fx_beam/{int(now*24+sid*3)%24}')
            h=ts*.44
            if tex:g.draw(tex,min(x1,x2),y-h/2,length,h,alpha=245,mode=ADD,flip=d<0)
            # A focused emitter and a frosty contact spot, not a giant white bar.
            g.glow(x1,y,ts*.30,(125,218,249),.48)
            g.glow(x2,y,ts*.22,(136,218,240),.30)
            for j in range(max(1,int(length/ts/1.8))):
                q=(now*.9+j*.618+sid*.13)%1
                xx=x1+(x2-x1)*q;yy=y+math.sin(now*5+j*3)*ts*.025
                shard=v.A.get('fx_ice',ts*.07)
                if shard:g.draw(shard,xx-ts*.04,yy-ts*.04,ts*.08,ts*.08,alpha=160,angle=now*70+j*40)

    def explode(self,event):
        v=self.v;now=v._now;pos=event['pos'];x,y=pos[1]+.5,pos[0]+.66
        self.clip('fx_explosion',pos,now,1.45,3.8)
        self.mark('scorch',pos,now,2.2,25)
        self.splinters(x,y,'stone',22,6.0);self.sparks(x,y,34,(255,187,83),7)
        v.light_flash(x,y,6,(255,178,89),.36,1.0);v.shake=max(v.shake,.34)
        if event.get('pid') is not None:self.recoil(event['pid'],'blast',now,.65)
        # A low, elliptical pressure front is anchored to the floor plane.
        self.clip('pressure',pos,now,.45,3.7,1)

    def bite(self,event):
        v=self.v;now=v._now;sid=event['sid'];pos=event['pos']
        entry=self.source_positions.get(event['pid'],pos)
        face=-1 if entry[1]<pos[1] else (1 if entry[1]>pos[1] else v.snake_face.get(sid,1))
        v.snake_face[sid]=face
        self.bites[sid]=dict(t=now,pos=pos,pid=event['pid'],face=face)
        def contact():
            x,y=pos[1]+.5+face*.12,pos[0]+.40
            self.splinters(x,y,'venom',12,2.0);self.recoil(event['pid'],'bite',v._now,.60,face)
            self.venom.append(dict(t=v._now,pos=pos,face=face))
            v.light_flash(x,y,1.0,(181,224,91),.16,.35)
            v.hit_time[event['pid']]=v._now;v.shake=max(v.shake,.06)
        v.at(now+.32,contact)

    def draw_bite(self,sid,pos,now):
        a=self.bites.get(sid)
        if not a:return False
        v=self.v;g=v.g;ts=v.ts;k=sat((now-a['t'])/.90)
        idx=min(41,int(k*42))
        tex=v.A.get(f'{v.biome}/bite/{idx}',ts*1.14)
        if not tex:return False
        lunge=math.sin(sat((k-.20)/.56)*math.pi)**3
        side=.38*ease(k/.20)-.14*lunge
        x,y=v.P(pos[1]+.5-a['face']*side,pos[0]+.91)
        w,h=ts*1.12*(1+lunge*.08),ts*1.10*(1-lunge*.08)
        if .28<k<.45:
            for j in (2,1):g.draw(tex,x-w/2-a['face']*ts*.07*j,y-h,w,h,flip=a['face']<0,alpha=24/j)
        g.draw(tex,x-w/2,y-h,w,h,flip=a['face']<0,alpha=255*(1-sat((k-.76)/.24)))
        return True

    def wake(self,event):
        self.stun_times.pop(event['sid'],None);self.waking[event['sid']]=self.v._now

    def draw_stun(self,sid,pos,stunned,owners,now):
        v=self.v;g=v.g;ts=v.ts
        if not stunned and sid not in self.stun_times and sid not in self.waking:return False
        if sid not in self.stun_times and sid not in self.waking:self.stun_times[sid]=now-.7
        t=self.stun_times.get(sid,now)
        if now<t:return False
        recovering=sid in self.waking
        age=now-self.waking[sid] if recovering else now-t
        idx=min(39,20+int(age/.58*20)) if recovering else min(19,int(age/.60*20))
        tex=v.A.get(f'{v.biome}/stun/{idx}',ts*1.08)
        if not tex:return False
        face=v.snake_face.get(sid,1);x,y=v.P(pos[1]+.5,pos[0]+.92)
        breathing=1+.012*math.sin(now*2.8+sid) if not recovering and age>.6 else 1
        w,h=ts*1.04,ts*1.04*breathing
        g.draw(g.prims['shadow'],x-w*.44,y-ts*.1,w*.88,ts*.18,alpha=160)
        # Engraved, pulsing ground sigil and intermittent filaments replace
        # floating cartoon stars. The creature itself slumps onto its coils.
        fade=1-sat(age/.58) if recovering else min(1,age/.18)
        for radius in (.43,.36):
            pts=[]
            for j in range(37):
                a=j*TAU/36;pts.append((x+math.cos(a)*ts*radius,y-ts*.08+math.sin(a)*ts*radius*.28))
            for j,(p,q) in enumerate(zip(pts,pts[1:])):
                if j%6!=0:g.line(*p,*q,max(1,ts*.009),(132,196,225),int((50+25*math.sin(now*3))*fade))
        g.draw(tex,x-w/2,y-h,w,h,flip=face<0,alpha=255)
        if not recovering and age<.42:
            for j in range(3):
                a=j*TAU/3+age*3;px=x+math.cos(a)*ts*.22;py=y-ts*.30+math.sin(a)*ts*.12
                g.line(px,py,px+ts*.06,py-ts*.12,ts*.018,(203,240,255),170*(1-age/.42),ADD)
        label=' + '.join('YOU' if owner==0 else 'AI' for owner in sorted(owners))
        if label and not recovering:g.text('STUNNED · '+label,x,y-h-ts*.05,max(9,ts*.135),(192,222,235),True,'center',shadow=True)
        return True

    def burn(self,pos,pid=None):
        v=self.v;now=v._now
        if pid is None:
            pid=next((p.pid for p in v.world.players if (p.r,p.c)==tuple(pos)),None)
        if pid is not None:
            self.burns[pid]=now;self.recoil(pid,'burn',now,1.1)
            for j in range(5):
                def flare(j=j):
                    p=v.player_cell_f(pid,v._now);self.sparks(p[1]+.5,p[0]+.40-j*.09,5,(255,173,74),1.6)
                    if j>1:self.dust(p[1]+.5,p[0]+.15-j*.06,1,.22,.34)
                v.at(now+.12+j*.15,flare)
        x,y=pos[1]+.5,pos[0]+.55
        self.sparks(x,y,22,(255,149,43),3.8);self.dust(x,y,4,.5,.48)
        v.light_flash(x,y,2.3,(255,139,46),.42,.6)

    def draw_burn(self,pid,pos,now,front=False):
        t=self.burns.get(pid)
        if t is None:return
        v=self.v;g=v.g;ts=v.ts;k=sat((now-t)/1.35)
        index=min(15,int(k*16));tex=v.A.get(f'burn_{"front" if front else "back"}/{index}',ts*1.7)
        x,y=v.P(pos[1]+.5,pos[0]+.98);rx,ry,_,_=self.reaction(pid,now);x+=rx*ts;y+=ry*ts
        w,h=ts*1.68,ts*1.90
        if tex:g.draw(tex,x-w/2,y-h,w,h,alpha=255 if front else 230)

    def freeze(self,event):
        v=self.v;now=v._now;pos=event['pos'];x,y=pos[1]+.5,pos[0]+.4
        self.freeze_times[event['pid']]=now
        self.splinters(x,y,'ice',26,3.6);self.dust(x,y,5,1.1,.65)
        v.light_flash(x,y,3.0,(117,211,243),.45,.65)
        self.clip('freeze',pos,now,.64,1.7,1)
        self.recoil(event['pid'],'freeze',now,.4)
        v.float_text(x,y-.65,'FROZEN',(179,233,248),now,18,.95)
        st=next((s for s in v.world.statues.values() if s.sid==event.get('sid')),None)
        if st:
            ox,oy=self.lens(st)
            self.clips.append(dict(name='discharge',pos=pos,t=now,dur=.24,size=0,frames=1,origin=(ox,oy)))

    def thaw(self,event):
        self.freeze_times.pop(event['pid'],None)
        p=event['pos'];self.splinters(p[1]+.5,p[0]+.4,'ice',22,4.7);self.dust(p[1]+.5,p[0]+.6,3,.7,.6)

    def land(self,pos,heavy=False,event=None):
        v=self.v;now=v._now;x,y=pos[1]+.5,pos[0]+.85
        self.dust(x,y,12 if heavy else 6,2.5 if heavy else 1.1,1.0 if heavy else .60)
        self.splinters(x,pos[0]-.15 if heavy else y,'stone',22 if heavy else 7,5.5 if heavy else 2.2)
        self.mark('fracture',pos,now,1.8 if heavy else .8,12 if heavy else 4)
        if heavy:
            self.clip('pressure',pos,now,.4,2.1,1);v.shake=max(v.shake,.28)
            v.light_flash(x,y,2.4,(220,196,154),.15,.6)
            if event:
                self.recoil(event['pid'],'crush',now,.70);self.rock_hits[event['rock']]=now
                self.sparks(x,y,9,(209,191,149),3)

    def draw_ground(self,now):
        v=self.v;g=v.g;ts=v.ts
        for a in self.marks:
            k=sat((now-a['t'])/a['dur']);size=a['size']*ts;x,y=v.P(a['pos'][1]+.5,a['pos'][0]+.80)
            t=v.A.get('fx_'+a['name'],size)
            if t:g.draw(t,x-size/2,y-size*.24,size,size*.48,alpha=220*(1-k)**.4)
        for pos,end in self.ghostwalls:
            r,c=pos;x,y=v.P(c,r);i=r%4*4+c%4
            for name in (f'{v.biome}/wall/f{i}',f'{v.biome}/crackov_0'):
                tex=v.A.get(name,ts)
                if tex:g.draw(tex,x,y,ts,ts)

    def draw_tools(self,now):
        v=self.v;g=v.g;ts=v.ts
        for a in self.actions:
            if not a['t']<=now<a['t']+a['dur']:continue
            k=sat((now-a['t'])/a['dur']);pr=v.player_cell_f(a['pid'],now);fac=a['fac'];sign=-1 if fac=='left' else 1
            if a['kind']=='hook':
                cuts=[0,.17,.40,.78,1]
                idx=next((j for j in range(4) if cuts[j]<=k<cuts[j+1]),3)
                source=character_source(a['pid'])
                hu,hv=([(.85,.50),(.93,.28),(.80,.44),(.75,.36)] if source==0 else
                       [(.81,.48),(.94,.34),(.81,.42),(.72,.41)])[idx]
                anchor=.49 if source==0 or idx!=2 else .45
                lean=-.045*math.sin(k*math.pi)
                hx,hy=v.P(pr[1]+.5+sign*(1.8*(hu-anchor)+lean),
                           pr[0]+.94-(1-.015*math.sin(k*TAU))*(1.786-1.8*hv))
                src=a['target']
                tx,ty=v.S(src)
                if k<.38:
                    q=gfx.ease_out(k/.38);ex,ey=gfx.lerp(hx,tx,q),gfx.lerp(hy,ty,q)-math.sin(q*math.pi)*ts*.28
                    slack=.20*math.sin(q*math.pi)
                else:
                    logical=v.tw.pos(a['key'],src,now);ex,ey=v.S(logical);slack=.03 if k<.82 else .16*ease((k-.82)/.18)
                if k>.92:
                    q=ease((k-.92)/.08);ex,ey=gfx.lerp(ex,hx,q),gfx.lerp(ey,hy,q)
                count=max(10,min(90,int(math.hypot(ex-hx,ey-hy)/ts*22)))
                points=[]
                for j in range(count+1):
                    q=j/count;points.append((gfx.lerp(hx,ex,q),gfx.lerp(hy,ey,q)+math.sin(q*math.pi)*ts*slack))
                for j,(p,q) in enumerate(zip(points,points[1:])):
                    g.line(*p,*q,max(1.5,ts*.037),(30,25,18),240)
                    g.line(*p,*q,max(1,ts*.018),(168+(j%2)*32,137+(j%2)*26,87),240)
                head=v.A.get('fx_hook',ts*.48)
                angle=math.degrees(math.atan2(ey-hy,ex-hx))
                if head:g.draw(head,ex-ts*.25,ey-ts*.25,ts*.50,ts*.50,angle=angle)
            elif a['kind']=='hammer' and .24<k<.52:
                direction={'right':0,'left':math.pi,'up':-math.pi/2,'down':math.pi/2}[fac]
                px,py=v.P(pr[1]+.5,pr[0]+.22)
                ang=direction-1.7+2.0*ease((k-.15)/.33)
                for j in range(6):
                    aa=ang-j*.10;bb=aa-.09;rad=ts*.82
                    p=(px+math.cos(aa)*rad,py+math.sin(aa)*rad)
                    q=(px+math.cos(bb)*rad,py+math.sin(bb)*rad)
                    g.line(*p,*q,ts*.023,(213,191,140),100*(1-j/6))

    def draw_after_light(self,now):
        v=self.v;g=v.g;ts=v.ts
        for a in self.clips:
            k=sat((now-a['t'])/a['dur']);pos=a['pos'];x,y=v.P(pos[1]+.5,pos[0]+.86)
            if a['name']=='fx_explosion':
                frame=min(a['frames']-1,int(k*a['frames']));size=a['size']*ts
                tex=v.A.get(f'fx_explosion/{frame}',size)
                if tex:g.draw(tex,x-size/2,y-size*.87,size,size,alpha=255*(1-sat((k-.88)/.12)))
            elif a['name']=='pressure':
                radius=ts*a['size']*gfx.ease_out(k)
                g.draw(g.prims['ring_soft'],x-radius/2,y-radius*.16,radius,radius*.32,
                       color=(174,151,115),alpha=150*(1-k)**2)
            elif a['name']=='freeze':
                radius=ts*a['size']*ease(k)
                g.draw(g.prims['ring'],x-radius/2,y-radius*.17,radius,radius*.34,
                       color=(130,216,245),alpha=160*(1-k),mode=ADD)
            elif a['name']=='discharge':
                ox,oy=v.P(*a['origin']);ex,_=v.P(pos[1]+.5,a['origin'][1])
                t=v.A.get(f'fx_beam/{int(now*24)%24}')
                if t:g.draw(t,min(ox,ex),oy-ts*.22,abs(ex-ox),ts*.44,alpha=240*(1-k),mode=ADD)
            elif a['name']=='stun_contact':
                for j in range(7):
                    angle=j*TAU/7+.2;rad=ts*(.10+.34*ease(k))
                    px=x+math.cos(angle)*rad;py=y-ts*.40+math.sin(angle)*rad*.7
                    g.line(px,py,px+math.cos(angle)*ts*.11*(1-k),py+math.sin(angle)*ts*.11*(1-k),ts*.025,(217,242,255),220*(1-k),ADD)
        for pid,t0 in self.burns.items():
            self.draw_burn(pid,v.player_cell_f(pid,now),now,True)
        for a in self.venom:
            k=sat((now-a['t'])/.65);p=a['pos'];x,y=v.P(p[1]+.5+a['face']*.10,p[0]+.42)
            # Two quick fang puncture glints with descending venom threads.
            for sign in (-1,1):
                xx=x+sign*ts*.045
                g.line(xx,y,xx+a['face']*ts*.05,y+ts*.19*k,ts*.016,(174,213,77),185*(1-k)**1.5,ADD)
