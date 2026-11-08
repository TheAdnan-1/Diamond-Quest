"""Expedition art: authored atlases, animation preparation, and material composition.

This module is presentation-only. It never receives or mutates a World.
Source PNG atlases remain intact; frames are sampled and sized during asset loading.
All returned arrays follow paint.py's premultiplied RGBA contract.
"""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
import pygame
from .paint import F, Canvas, grid, mixc, ramp, smoothstep, cut_tiles, opaque, Noise, rgb

ROOT = Path(__file__).resolve().parent / 'art'
BIOME_ROW = {'jungle': 0, 'frost': 1, 'ember': 2}
# Identity is independent of the unchanged logical player IDs. Source atlas 0
# is the red-haired girl; source 1 is the boy wearing S.
CHARACTER_SOURCE = {0:1,1:0}


def character_source(pid):return CHARACTER_SOURCE[pid]


def character_name(pid):return 'explorer' if character_source(pid)==0 else 'automaton'

@lru_cache(maxsize=8)
def sheet(name):
    return pygame.image.load(str(ROOT / (name + '.png')))


def cell(name, col, row, tight=False):
    s = sheet(name)
    cw, ch = s.get_width() // 4, s.get_height() // 3
    rect=pygame.Rect(col*cw,row*ch,cw,ch)
    if name=='decor':
        # These authored props have intentionally unequal silhouettes. Explicit
        # vertical frame bounds exclude neighboring roots, crystals and plinths.
        bounds={(0,0):(0,.96),(1,0):(0,.88),(2,0):(0,.98),(3,0):(0,.93),
                (0,1):(.94,1.92),(1,1):(.96,1.92),(2,1):(1.05,1.82),(3,1):(.92,1.90),
                (0,2):(1.93,2.98),(1,2):(1.93,2.98),(2,2):(1.90,2.98),(3,2):(1.90,2.98)}
        y0,y1=bounds[col,row]
        rect=pygame.Rect(col*cw,round(y0*ch),cw,round((y1-y0)*ch))
    tile = s.subsurface(rect).copy()
    if tight:
        # Keep meaningful alpha-connected pieces. Neighboring atlas cells may
        # have a few antialiased pixels across the boundary; those must not
        # become floating horizontal slivers in the game.
        components=pygame.mask.from_surface(tile,30).connected_components(18)
        if components:
            largest=max(m.count() for m in components)
            keep=pygame.mask.Mask(tile.get_size())
            for m in components:
                if m.count()>=largest*.025:
                    keep.draw(m,(0,0))
            mask=keep.to_surface(setcolor=(255,255,255,255),unsetcolor=(0,0,0,0))
            tile.blit(mask,(0,0),special_flags=pygame.BLEND_RGBA_MULT)
        rect = tile.get_bounding_rect(min_alpha=30)
        if rect.w and rect.h:
            tile = tile.subsurface(rect).copy()
    return tile


def array(s):
    c = pygame.surfarray.array3d(s).transpose(1,0,2).astype(F)/255
    a = pygame.surfarray.array_alpha(s).T.astype(F)/255
    a=np.where(a<.018,0,a)
    return np.concatenate((c*a[...,None],a[...,None]),axis=-1)


def fit(s, w, h, fill=.92, bottom=False):
    w,h=max(1,int(w)),max(1,int(h))
    k=min(w*fill/s.get_width(),h*fill/s.get_height())
    sz=(max(1,round(s.get_width()*k)),max(1,round(s.get_height()*k)))
    s=pygame.transform.smoothscale(s,sz)
    dst=pygame.Surface((w,h),pygame.SRCALPHA)
    dst.blit(s,((w-sz[0])//2, h-sz[1]-round(h*.025) if bottom else (h-sz[1])//2))
    return array(dst)


def character_set(n, pid, pose):
    row={'down':0,'side':1,'up':2}[pose]
    name=character_name(pid)
    # A fixed scale per atlas preserves mass through the stride; independently
    # fitting every frame would make the character expand and shrink as it walks.
    frames=[cell(name,i,row,True) for i in range(4)]
    maxw=max(s.get_width() for s in frames)
    maxh=max(s.get_height() for s in frames)
    W,H=int(n*.92),int(n*1.28)
    k=min(W*.91/maxw,H*.94/maxh)
    out=[]
    for s in frames:
        sz=max(1,round(s.get_width()*k)),max(1,round(s.get_height()*k))
        dst=pygame.Surface((W,H),pygame.SRCALPHA)
        dst.blit(pygame.transform.smoothscale(s,sz),((W-sz[0])//2,H-sz[1]-round(H*.018)))
        out.append(array(dst))
    return out


def portrait(n,pid):
    s=identity(pid)
    # Bust framing, rather than shrinking a complete full-body sprite.
    s=s.subsurface((0,0,s.get_width(),round(s.get_height()*.53))).copy()
    return fit(s,n,n,1.0,True)


def prop(n,col,row,w=None,h=None,fill=.94):
    return fit(cell('relics',col,row,True),w or n,h or n,fill)


def gem(n,kind):return prop(n,0 if kind=='v' else 1,0,fill=.88)
def hammer(n):return prop(n,2,0,fill=.96)
def hook_head(n):return prop(n,3,0,fill=.96)
def mine(n):
    # Flat top-down trap: faceted ruby and engraved sectors, without a thick
    # cylinder or perspective side wall. Matches the gems' compact silhouette.
    c=Canvas(n);cx=cy=n*.5
    points=[(cx+math.cos(i*math.tau/8+math.pi/8)*n*.39,
             cy+math.sin(i*math.tau/8+math.pi/8)*n*.39) for i in range(8)]
    shape=c.d_poly(points)
    c.over(rgb('#15272b'),c.cov(shape-n*.025))
    c.shape(shape,rgb('#a8844f'),bevel=n*.018,ambient=.75,gloss=.25)
    inner=c.d_circle(cx,cy,n*.32)
    c.shape(inner,rgb('#35494b'),bevel=n*.015,ambient=.75,gloss=.15)
    for i in range(4):
        angle=math.pi/4+i*math.pi/2
        x,y=cx+math.cos(angle)*n*.24,cy+math.sin(angle)*n*.24
        c.over(rgb('#cdb47b'),c.cov(c.d_circle(x,y,n*.022)))
        ax,ay=cx+math.cos(i*math.pi/2)*n*.18,cy+math.sin(i*math.pi/2)*n*.18
        bx,by=cx+math.cos(i*math.pi/2)*n*.30,cy+math.sin(i*math.pi/2)*n*.30
        c.over(rgb('#91aaa3'),c.cov(c.d_capsule(ax,ay,bx,by,n*.008))*.65)
    ring=np.abs(c.d_circle(cx,cy,n*.15))-n*.013
    c.over(rgb('#c6a165'),c.cov(ring))
    P=lambda x,y:(x*n,y*n)
    for pts,color in [([P(.39,.46),P(.5,.34),P(.5,.51)],'#ffb5b1'),
                      ([P(.5,.34),P(.62,.46),P(.5,.51)],'#e65762'),
                      ([P(.39,.46),P(.5,.51),P(.5,.64)],'#b51f42'),
                      ([P(.5,.51),P(.62,.46),P(.5,.64)],'#e33853')]:
        c.over(rgb(color),c.cov(c.d_poly(pts)))
    return c.px


def boulder(n,seed,B,biome):
    s=cell('relics',BIOME_ROW[biome],1,True)
    # Three silhouettes without arbitrarily rotating the baked overhead light.
    if seed%3==1:s=pygame.transform.flip(s,True,False)
    return fit(s,n,n,.90 if seed%3==2 else .96)


def snake_frames(n,B,biome):
    src=prop(int(n*1.22),BIOME_ROW[biome],2,fill=.92)
    H,W=src.shape[:2];x,y=grid(W,H);out=[]
    # Smooth periodic deformation: the heavy coil stays planted; neck and head
    # sway progressively more. Precomputed once, no per-frame pixel work.
    for k in range(24):
        t=k/24*math.tau
        neck=(1-y/H)**1.8
        dx=np.sin(t+y/H*2.6)*W*.026*neck
        dy=np.cos(t)*H*.012*neck
        xx=np.clip(x-.5-dx,0,W-1);yy=np.clip(y-.5-dy,0,H-1)
        ix=np.floor(xx).astype(int);iy=np.floor(yy).astype(int)
        jx=np.minimum(ix+1,W-1);jy=np.minimum(iy+1,H-1)
        fx=(xx-ix)[...,None];fy=(yy-iy)[...,None]
        frame=(src[iy,ix]*(1-fx)+src[iy,jx]*fx)*(1-fy)+(src[jy,ix]*(1-fx)+src[jy,jx]*fx)*fy
        out.append(frame.astype(F))
    return out


def _material(n,biome,col,k=4):
    s=material_cell(biome,col)
    s=pygame.transform.smoothscale(s,(n*k,n*k))
    a=array(s)
    # Periodic-plus-smooth decomposition removes boundary discontinuities while
    # retaining the artwork's large forms. Unlike mirrored repeats, carvings,
    # roots and floor slabs keep their natural asymmetric silhouettes.
    u=a[...,:3]
    H,W=u.shape[:2]
    v=np.zeros_like(u)
    v[0]=u[-1]-u[0];v[-1]=u[0]-u[-1]
    v[:,0]+=u[:,-1]-u[:,0];v[:,-1]+=u[:,0]-u[:,-1]
    den=(2*np.cos(2*np.pi*np.arange(H)/H)[:,None]+
         2*np.cos(2*np.pi*np.arange(W)/W)[None,:]-4)
    den[0,0]=1
    smooth=np.fft.ifft2(np.fft.fft2(v,axes=(0,1))/den[...,None],axes=(0,1)).real
    smooth-=smooth.mean((0,1))
    a[...,:3]=np.clip(u-smooth,0,1)
    return a


def material_cell(biome,col):
    if biome!='frost':return cell('materials',col,BIOME_ROW[biome])
    s=sheet('frost_materials');w,h=s.get_size();cw,ch=w//2,h//2
    return s.subsurface((col%2*cw,col//2*ch,cw,ch)).copy()


def floor_atlas(B,n,seed,biome,k=4):
    a=_material(n,biome,0,k)
    # Floors have lower contrast than walls and props; critical for readability.
    a[...,:3]*=.94 if biome=='frost' else .83
    return cut_tiles(a,n,k)


def dirt_atlas(B,n,seed,biome,k=4):
    a=_material(n,biome,1,k);tops=cut_tiles(a,n,k);out={}
    for i,t in enumerate(tops):
        out['t'+str(i)]=t
        f=t.copy();y=np.linspace(0,1,n)[:,None]
        f[...,:3]*=(1-.36*smoothstep(.72,1,y))[...,None]
        # An excavated soil lip has a thin catching edge and a soft contact dark.
        f[int(n*.76):int(n*.78),:,:3]*=1.12
        out['f'+str(i)]=f
    return out


def wall_atlas(B,n,seed,biome,k=4):
    top=_material(n,biome,2,k);tops=cut_tiles(top,n,k);out={}
    face=material_cell(biome,3)
    depth=round(n*.49)
    face=array(pygame.transform.smoothscale(face,(n*k,depth)))
    for i,t in enumerate(tops):
        t=t.copy();t[...,:3]*=.74 if biome=='frost' else .72
        if biome=='frost':t=frost_block_rim(t,n)
        out['t'+str(i)]=t
        f=t.copy();c=i%k
        f[-depth:]=face[:,c*n:(c+1)*n]
        # Strong architectural cap, undercut, and dark foot anchor the wall.
        split=n-depth
        f[split-2:split,:,:3]*=1.28
        f[split:split+2,:,:3]*=.70
        fade=1-.40*np.linspace(0,1,depth)[:,None]**2
        f[-depth:,:,:3]*=fade[...,None]
        if biome=='frost':f=frost_block_rim(f,n)
        out['f'+str(i)]=np.clip(f,0,1)
    return out


def frost_block_rim(a,n):
    """Inset stone edging signals a raised, blocked cell even without color."""
    a=a.copy();x,y=grid(n,n);edge=np.minimum.reduce((x,y,n-x,n-y))
    border=1-smoothstep(n*.012,n*.030,edge)
    a[...,:3]*=(1-border*.64)[...,None]
    # A muted broken catching line and a deep lower foot distinguish wall
    # volume from the continuous, flat open path texture.
    cap=smoothstep(n*.020,n*.030,y)*(1-smoothstep(n*.045,n*.055,y))
    cap*=smoothstep(n*.04,n*.08,x)*(1-smoothstep(n*.92,n*.96,x))
    a[...,:3]=a[...,:3]*(1-cap[...,None]*.55)+rgb('#435968')*cap[...,None]*.55
    foot=smoothstep(n*.88,n*.98,y)
    a[...,:3]*=(1-foot*.35)[...,None]
    return a


def decor(n,col,row,fill=1.0):
    return fit(cell('decor',col,row,True),n,n,fill)


def fern(n,seed,B):return decor(n,0,0,.88)
def moss_patch(n,seed,B):return decor(n,1,0,.84)
def vines(n,seed,B):return decor(n,2,0,1.0)
def bones(n,seed):return decor(n,3,0,.52)
def crystal_cluster(n,seed,color):
    a=decor(n,0,1,.92);a[...,:3]*=.74;return a
def icicles(n,seed):return decor(n,1,1,1.0)
def snow_drift(n,seed):
    a=decor(n,2,1,.72);a[...,:3]*=.64;a*=.70;return a
def torch(n):return decor(n,0,2,.58)
def chains(n,seed):return decor(n,1,2,.94)


def portal_ring(n,seed,B):
    ring=decor(n,2,2,.98)
    # Keep the original additive rune pass alive with a restrained inner glow.
    c=Canvas(n);dist=np.hypot(c.X-n*.50,(c.Y-n*.46)*1.08)
    a=np.exp(-((dist-n*.28)/(n*.008))**2)*.16
    runes=np.concatenate((np.ones((n,n,3),F)*a[...,None],a[...,None]),-1)
    return ring,runes


def statue(n,B,biome):
    s=cell('relics',3,2,True) if biome=='jungle' else cell('decor',3,1 if biome=='frost' else 2,True)
    dst=pygame.Surface((n,n*2),pygame.SRCALPHA)
    dst.blit(pygame.transform.smoothscale(s,(int(n*.9),int(n*1.94))),(int(n*.05),int(n*.035)))
    return array(dst)


def key_art(n,pid):
    return fit(identity(pid),int(n*.8),n,.98,True)


def identity(pid):
    s=sheet('identity/boy_hd' if pid==0 else 'identity/girl_hd').copy()
    rect=s.get_bounding_rect(min_alpha=30)
    return s.subsurface(rect).copy()


def face_left(s):
    """Mirror the boy's pose while keeping the chest badge's S readable."""
    dst=pygame.transform.flip(s,True,False)
    a=pygame.surfarray.array3d(s).astype(np.int16)
    alpha=pygame.surfarray.array_alpha(s)
    w,h=s.get_size()
    cream=(a[:,:,0]>165)&(a[:,:,1]>150)&(a[:,:,2]>115)&((a.max(2)-a.min(2))<85)&(alpha>200)
    mask=pygame.Surface((w,h),pygame.SRCALPHA)
    pixels=pygame.surfarray.pixels_alpha(mask);pixels[:]=cream*255;del pixels
    for part in pygame.mask.from_surface(mask).connected_components(max(6,int(w*h*.000025))):
        rect=part.get_bounding_rects()[0]
        if not (.30*h<rect.centery<.80*h and .012*w<rect.w<.10*w and .012*h<rect.h<.10*h and .48<rect.w/rect.h<1.9):continue
        around=rect.inflate(round(w*.016),round(h*.016));around.clamp_ip(s.get_rect())
        region=a[around.left:around.right,around.top:around.bottom]
        blue=(region[:,:,2]>region[:,:,0]*1.35)&(region[:,:,2]>region[:,:,1]*1.15)
        if blue.mean()<.15:continue
        rect.inflate_ip(4,4);rect.clamp_ip(s.get_rect())
        dst.blit(s.subsurface(rect),(w-rect.right,rect.top))
    return dst


def icon(name,n):
    if name=='hammer':return hammer(n)
    if name=='hook':return hook_head(n)
    if name=='mine':return mine(n)
    from . import art_objects as O
    a=O.icon(name,n)
    if name in ('push','wait'):
        x,y=grid(n)
        metal=ramp(y/n,[(0,(255,235,180)),(.4,(218,172,92)),(.53,(139,98,47)),(.65,(240,197,119)),(1,(125,87,47))])
        a[...,:3]=metal*a[...,3:4]
    return a


def fire_frames(n,frames=24,seed=3,small=False):
    atlas=sheet('fire');cw,ch=atlas.get_width()//6,atlas.get_height()//4
    W,H=n,int(n*1.5);out=[]
    for k in range(frames):
        f=int(k*24/frames)%24
        tile=atlas.subsurface((f%6*cw,f//6*ch,cw,ch))
        dst=pygame.Surface((W,H),pygame.SRCALPHA)
        dst.blit(pygame.transform.smoothscale(tile,(W,H)),(0,round(H*.07)))
        out.append(array(dst))
    return out
