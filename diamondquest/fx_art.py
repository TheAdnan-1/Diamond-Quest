"""Presentation-only action atlases and material-aware effect textures."""
import math
from functools import lru_cache
import numpy as np
import pygame
from . import art_director as D
from .paint import Canvas,F,grid,rgb,Noise,smoothstep,ramp


def frame(name,col,row,cols=4,rows=3,tight=False):
    sheet=D.sheet(name);w,h=sheet.get_size();cw,ch=w//cols,h//rows
    s=sheet.subsurface((col*cw,row*ch,cw,ch)).copy()
    if tight:
        mask=pygame.mask.from_surface(s,70)
        components=mask.connected_components(40)
        if components:
            keep=max(components,key=lambda m:m.count())
            rect=keep.get_bounding_rects()[0]
            # Include soft edges around the connected sculpted object.
            rect.inflate_ip(8,8);rect.clamp_ip(s.get_rect())
            s=s.subsurface(rect).copy()
    return s


def action_frames(n,pid,kind,pose='side'):
    pid=D.character_source(pid)
    row={'hammer':0,'push':1,'hook':2}[kind]
    name='explorer_actions' if pid==0 else 'automaton_actions'
    rows=3
    if pose!='side':
        name='hammer_vertical';rows=4;row=pid*2+(pose=='up')
    W=H=n*2;scale=n*1.80
    anchors=[.49,.49,.39,.48] if kind=='hammer' else ([.50,.47,.38,.49] if kind=='push' else [.49,.49,.49,.49])
    if pid==1:anchors=[.49,.49,.42,.49] if kind=='hammer' else [.49,.49,.45,.49]
    if pose!='side':anchors=[.5]*4
    out=[]
    for k in range(4):
        s=frame(name,k,int(row),rows=rows)
        s=pygame.transform.smoothscale(s,(round(scale),round(scale)))
        dst=pygame.Surface((W,H),pygame.SRCALPHA)
        dst.blit(s,(round(W/2-scale*anchors[k]),round(H*.98-scale*.97)))
        out.append(D.array(dst))
    return out


def guardian_views(n,biome):
    row=D.BIOME_ROW[biome]
    return {key:D.fit(frame('statue_directions',i,row,2,3,True),n,n*2,.97,True)
            for i,key in enumerate(('right','front'))}


@lru_cache(maxsize=3)
def guardian_anchor(biome):
    im=guardian_views(160,biome)['right']
    h,w=im.shape[:2];x,y=grid(w,h)
    col=im[...,:3]/np.maximum(im[...,3:4],.001)
    score=np.maximum(0,col[...,2]-col[...,0])*col[...,1]*im[...,3]
    score*=((x>w*.64)&(x<w*.91)&(y>h*.07)&(y<h*.27))
    weight=np.maximum(0,score-score.max()*.78)
    u=float((weight*x).sum()/max(.00001,weight.sum()))/w
    v=float((weight*y).sum()/max(.00001,weight.sum()))/h
    # Statue destination: width 1.1 cells, height 2.2, bottom at row+1.98.
    return ((u-.5)*1.1, v*2.2-.22)


def explosion_frames(n):
    return [D.array(pygame.transform.smoothscale(frame('explosion',i%6,i//6,6,4),(n,n))) for i in range(24)]


def bite_frames(n,biome):
    return [D.array(pygame.transform.smoothscale(frame('baked/bite_'+biome,i%8,i//8,8,6),(n,n))) for i in range(42)]


def stun_frames(n,biome):
    return [D.array(pygame.transform.smoothscale(frame('baked/stun_'+biome,i%8,i//8,8,5),(n,n))) for i in range(40)]


def burn_frames(n,front=False):
    out=[]
    for i in range(16):
        s=frame('burn_mantle',i%4,i//4,4,4)
        im=D.array(pygame.transform.smoothscale(s,(n,n)))
        if front:
            # Foreground tongues cross the boots; upper flames sit behind the
            # actor. Keeping the face unobscured makes the hit readable.
            _,y=grid(n,n)
            im*=smoothstep(n*.57,n*.76,y)[...,None]
        out.append(im)
    return out


def beam_frames(n=256):
    W,H=n,64;x,y=grid(W,H);u=x/W;v=(y/H-.5)*2;out=[]
    for k in range(24):
        t=k/24*math.tau
        core=np.exp(-((v-.025*np.sin(u*35-t*3))/.055)**2)
        plasma=np.exp(-(v/.25)**2)*(.30+.12*np.sin(u*57-t*5))
        fil=np.zeros_like(v)
        for j in (-1,1):
            path=j*(.15+.04*np.sin(u*20-t*2))
            fil+=np.exp(-((v-path)/.025)**2)*(.4+.3*np.sin(u*40-t*3)**2)
        a=np.clip(core*.9+plasma*.34+fil*.36,0,1)
        col=np.stack([.28+.65*core,.73+.25*core,np.ones_like(core)],-1)
        out.append(np.concatenate((col*a[...,None],a[...,None]),-1).astype(F))
    return out


def shard(n,kind='ice',seed=0):
    c=Canvas(n);P=lambda x,y:(x*n,y*n)
    if kind=='spark':
        a=np.exp(-((c.Y/n-.5)/.065)**2)*np.clip(1-np.abs(c.X/n-.5)*2,0,1)**.7
        return np.concatenate((np.ones((n,n,3),F)*a[...,None],a[...,None]),-1)
    if kind=='venom':
        d=np.minimum(c.d_ellipse(n*.52,n*.58,n*.19,n*.24),c.d_poly([P(.35,.48),P(.51,.10),P(.67,.49)]))
        c.shape(d,rgb('#83bd36'),bevel=n*.13,ambient=.45,gloss=.8,outline=n*.014,outline_col=(18,42,10))
        c.over(rgb('#e0f7ab'),c.cov(c.d_ellipse(n*.45,n*.48,n*.03,n*.08))*.8)
        return c.px
    pts=[P(.15,.58),P(.42,.08),P(.83,.27),P(.88,.65),P(.57,.9)]
    if kind=='stone':colors=['#82715b','#b6a58a','#4c4940']
    else:colors=['#5499b9','#dcf8ff','#72cde9']
    mid=P(.53,.48)
    for i,col in enumerate(colors):
        group=[pts[0],pts[1],mid,pts[4]] if i==0 else ([pts[1],pts[2],mid] if i==1 else [mid,pts[2],pts[3],pts[4]])
        c.shape(c.d_poly(group),rgb(col),bevel=n*.025,ambient=.65,gloss=.45,opacity=.88 if kind=='ice' else 1)
    return c.px


def hook_head(n):
    c=Canvas(n);steel=rgb('#9badb2');dark=rgb('#29383d');brass=rgb('#b68c4e')
    d=c.d_capsule(n*.19,n*.50,n*.64,n*.50,n*.05)
    for side in (-1,1):
        pts=[(.39,.50),(.55,.5+side*.17),(.76,.5+side*.24),(.89,.5+side*.11)]
        for j,(a,b) in enumerate(zip(pts,pts[1:])):
            d=np.minimum(d,c.d_capsule(a[0]*n,a[1]*n,b[0]*n,b[1]*n,n*(.052-j*.014)))
    c.shape(d,steel,bevel=n*.045,ambient=.5,gloss=.9,outline=n*.016,outline_col=dark)
    collar=c.d_box(n*.30,n*.50,n*.045,n*.105,n*.025)
    c.shape(collar,brass,bevel=n*.03,ambient=.5,gloss=.75)
    eye=np.abs(c.d_circle(n*.12,n*.50,n*.083))-n*.028
    c.shape(eye,steel,bevel=n*.023,ambient=.65,gloss=.8)
    return c.px


def ice_cage(n):
    W,H=n,int(n*1.5);c=Canvas(W,H)
    P=lambda x,y:(x*W,y*H)
    # A translucent, irregular prism; the character remains visible through it.
    boundary=[P(.12,.84),P(.08,.37),P(.30,.07),P(.64,.025),P(.89,.29),P(.94,.80),P(.67,.96),P(.34,.98)]
    d=c.d_poly(boundary)
    c.over(rgb('#60b2d2'),c.cov(d)*.20)
    facets=[([boundary[0],boundary[1],boundary[2],P(.42,.46),P(.34,.98)],'#d5f6ff',.20),
            ([boundary[2],boundary[3],boundary[4],P(.42,.46)],'#8bd9f5',.30),
            ([P(.42,.46),boundary[4],boundary[5],boundary[6]],'#4d9cce',.20)]
    for pts,col,alpha in facets:
        c.over(rgb(col),c.cov(c.d_poly(pts))*alpha)
    lines=np.full((H,W),1e5,F)
    for a,b in zip(boundary,boundary[1:]+boundary[:1]):lines=np.minimum(lines,c.d_capsule(*a,*b,n*.008))
    for a,b in [(boundary[2],P(.42,.46)),(P(.42,.46),boundary[5]),(P(.42,.46),boundary[7])]:
        lines=np.minimum(lines,c.d_capsule(*a,*b,n*.004))
    c.over(rgb('#d5f6ff'),c.cov(lines)*.75)
    frost=Noise(17,64).fbm(c.X/W*17,c.Y/H*21,3)
    edge=smoothstep(n*.16,n*.005,np.abs(d))
    c.over(rgb('#e5faff'),edge*smoothstep(.52,.72,frost)*.55)
    return c.px


def ground_mark(n,kind='scorch'):
    c=Canvas(n);x,y=c.X/n-.5,c.Y/n-.5;r=np.hypot(x,y)
    noise=Noise(19,32).fbm(c.X/n*9,c.Y/n*9,4)
    a=smoothstep(.49,.18,r+(noise-.5)*.08)
    if kind=='scorch':c.over(rgb('#100d0b'),a*.82)
    else:
        for i in range(7):
            angle=i*math.tau/7+.12*math.sin(i*4)
            px=py=n*.5
            for j in range(4):
                a0=angle+math.sin(j*4+i)*.18
                nx,ny=px+math.cos(a0)*n*.095,py+math.sin(a0)*n*.095
                d=c.d_capsule(px,py,nx,ny,n*(.011-j*.002))
                c.over(rgb('#171d1c'),c.cov(d)*.85);px,py=nx,ny
    return c.px


def dust(n):
    a=D.array(pygame.transform.smoothscale(frame('explosion',2,3,6,4),(n,n)))
    a[...,:3]*=np.array([.82,.78,.70],F)
    return a
