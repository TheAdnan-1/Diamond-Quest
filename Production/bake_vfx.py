"""Offline optical-flow inbetweens for authored snake animation keys.

Development dependencies: Pillow, numpy, OpenCV. The shipped game reads PNGs;
it does not import OpenCV or calculate optical flow while playing.
"""
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]/'diamondquest'/'art'
N=256

def keys(name,cols,rows,row):
    image=Image.open(ROOT/(name+'.png')).convert('RGBA')
    w,h=image.size
    return [np.asarray(image.crop((round(i*w/cols),round(row*h/rows),round((i+1)*w/cols),round((row+1)*h/rows))).resize((N,N),Image.Resampling.LANCZOS)).astype(np.float32)/255 for i in range(cols)]

def tween(a,b,count):
    # Premultiplied colors prevent transparent edge colors from leaking into
    # warped silhouettes. Flow sees both material detail and the silhouette.
    aa=a.copy();bb=b.copy();aa[:,:,:3]*=aa[:,:,3:4];bb[:,:,:3]*=bb[:,:,3:4]
    gray=lambda x:np.uint8(np.clip((x[:,:,:3].mean(2)*.7+x[:,:,3]*.3)*255,0,255))
    ga,gb=gray(aa),gray(bb)
    optical=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ab=optical.calc(ga,gb,None);ba=optical.calc(gb,ga,None)
    yy,xx=np.mgrid[:N,:N].astype(np.float32);base=np.stack([xx,yy],-1)
    out=[]
    for i in range(count):
        q=i/count;q=q*q*(3-2*q)
        ma=base-ab*q;mb=base-ba*(1-q)
        x=cv2.remap(aa,ma[:,:,0],ma[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        y=cv2.remap(bb,mb[:,:,0],mb[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        f=x*(1-q)+y*q
        f[:,:,:3]/=np.maximum(f[:,:,3:4],.001)
        out.append(np.uint8(np.clip(f*255,0,255)))
    return out

def strip(name,frames,cols=8):
    rows=(len(frames)+cols-1)//cols
    sheet=Image.new('RGBA',(cols*N,rows*N))
    for i,f in enumerate(frames):sheet.paste(Image.fromarray(f),(i%cols*N,i//cols*N))
    (ROOT/'baked').mkdir(exist_ok=True)
    sheet.save(ROOT/'baked'/(name+'.png'),optimize=True)
    print('Baked',name,len(frames),'frames',flush=True)

if __name__=='__main__':
    for row,biome in enumerate(('jungle','frost','ember')):
        k=keys('snake_bite',4,3,row)
        frames=[]
        for a,b,count in [(0,1,10),(1,2,7),(2,2,3),(2,3,12),(3,0,10)]:frames+=tween(k[a],k[b],count)
        strip('bite_'+biome,frames)
        k=keys('snake_stun',3,3,row)
        strip('stun_'+biome,tween(k[0],k[1],20)+tween(k[1],k[2],20))
