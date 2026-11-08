"""Read-only painted map overviews, built from the same art as the world."""
from functools import lru_cache
import pygame
from . import art_director as D
from . import paint
from .biomes import BIOMES
from .world import FLOOR,DIRT,WALL,CRACKED,FIRE,EXIT

@lru_cache(maxsize=9)
def _palette(biome,n):
    B=BIOMES[biome]
    floors=[paint.to_surface(x) for x in D.floor_atlas(B,n,10,biome)]
    walls={k:paint.to_surface(x) for k,x in D.wall_atlas(B,n,30,biome).items()}
    soils={k:paint.to_surface(x) for k,x in D.dirt_atlas(B,n,20,biome).items()}
    rock=paint.to_surface(D.boulder(n,50,B,biome))
    fire=paint.to_surface(D.fire_frames(n,1)[0])
    return floors,walls,soils,rock,fire


def map_surface(world,biome,n=16,objects=False):
    floors,walls,soils,rock,fire=_palette(biome,n)
    out=pygame.Surface((world.w*n,world.h*n),pygame.SRCALPHA)
    solid=(WALL,CRACKED)
    for r,row in enumerate(world.terrain):
        for c,t in enumerate(row):
            i=r%4*4+c%4
            front=r+1<world.h and world.terrain[r+1][c] not in solid+(DIRT,)
            if t in solid:
                tile=walls[('f' if front else 't')+str(i)]
                out.blit(tile,(c*n,r*n))
                shade=pygame.Surface((n,n),pygame.SRCALPHA)
                shade.fill((3,10,14,110 if front else 185));out.blit(shade,(c*n,r*n))
                if front:
                    pygame.draw.line(out,(133,132,103),(c*n,r*n+n//2),(c*n+n,r*n+n//2))
                if t==CRACKED:
                    pygame.draw.lines(out,(210,145,71),False,[(c*n+n//3,r*n+2),(c*n+n//2,r*n+n//2),(c*n+n//3,r*n+n-2)],max(1,n//10))
            elif t==DIRT:
                out.blit(soils[('f' if front else 't')+str(i)],(c*n,r*n))
            else:
                out.blit(floors[i],(c*n,r*n))
                if r and world.terrain[r-1][c] in solid:
                    pygame.draw.line(out,(18,29,28),(c*n,r*n),(c*n+n,r*n),max(1,n//6))
                if t==FIRE:out.blit(fire,(c*n,r*n-n//2))
    for r,c in world.rock_at:out.blit(rock,(c*n,r*n))
    if objects:
        for (r,c),kind in world.gems.items():
            x,y=c*n+n//2,r*n+n//2;s=max(3,n//3)
            col=(204,128,247) if kind=='v' else (255,77,110)
            pygame.draw.polygon(out,(27,13,36),[(x,y-s-1),(x+s+1,y),(x,y+s+1),(x-s-1,y)])
            pygame.draw.polygon(out,col,[(x,y-s),(x+s,y),(x,y+s),(x-s,y)])
            pygame.draw.line(out,(255,229,255),(x-s//2,y),(x,y-s//2))
        for r,c in world.snakes:
            x,y=c*n+n//2,r*n+n//2
            pygame.draw.arc(out,(115,191,104),(x-n//3,y-n//4,n*2//3,n//2),.2,5.6,max(1,n//7))
        for r,c in getattr(world,'statue_cells',{}):
            pygame.draw.rect(out,(139,166,157),(c*n+n//4,r*n+1,n//2,n-2),border_radius=1)
    return out
