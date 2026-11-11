"""World renderer (GPU): terrain layers, objects, characters, dynamic lighting, effects, camera."""
import math
import random

import numpy as np

from . import gfx
from . import roles
from .cinematics import CinematicFX
from .biomes import BIOMES
from .gpu import ADD, BLEND, MOD
from .world import WALL, CRACKED, FIRE, EXIT, DIRT, FLOOR

TILE_U = 56            # design units per cell at zoom 1
ZMIN = 0.42
PLAYER_COL = {0: gfx.HUMAN, 1: gfx.AI}
EYE_X, EYE_Y = 0.33, 0.15   # lens eye of a statue, relative to the centre-left / top of its head cell


def _h(r, c, k=0):
    return ((r * 73856093) ^ (c * 19349663) ^ (k * 83492791)) & 0x7FFFFFFF


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


class Tweens:
    def __init__(self):
        self.seg = {}

    def add(self, key, t0, t1, a, b, ease=gfx.ease_out):
        self.seg.setdefault(key, []).append((t0, t1, a, b, ease))

    def pos(self, key, logical, now):
        segs = self.seg.get(key)
        if not segs:
            return logical
        if now < segs[0][0]:
            return segs[0][2]
        cur = None
        for s in segs:
            if s[0] <= now < s[1]:
                cur = s
                break
            if now >= s[1]:
                cur = s
        t0, t1, a, b, ease = cur
        if now >= t1:
            if cur is segs[-1] and now > t1 + 0.05:
                del self.seg[key]
                return logical
            return b
        k = ease((now - t0) / (t1 - t0))
        if isinstance(a, tuple):
            return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
        return a + (b - a) * k

    def active(self, key, now):
        segs = self.seg.get(key)
        return bool(segs) and any(s[0] <= now < s[1] for s in segs)

    def busy_until(self):
        return max((s[1] for segs in self.seg.values() for s in segs), default=0)


class WorldView:
    def __init__(self, gpu, assets, world, biome, seed=0):
        self.g = gpu
        self.A = assets
        self.world = world
        self.biome = biome
        self.B = BIOMES[biome]
        self.rng = random.Random(seed)
        self.tw = Tweens()
        self.particles = []
        self.floats = []
        self.rings = []
        self.actives = []
        self.screen_fx = []
        self.pending = []
        self.temp_lights = []
        self.ghost_gems = {}
        self.ghost_snakes = {}
        self.ghost_mines = {}
        self.frozen_vis = {p.pid: bool(getattr(p, "frozen", 0)) for p in world.players}
        self.beam_hold = 0.0
        self.beam_segs = None
        self.facing = {0: "down", 1: "down"}
        self.hit_time = {0: -9.0, 1: -9.0}
        self.shake = 0.0
        self.flash = None
        self.cam_mode = "both"
        self.focus_pid = 0          # whose point of view: the human to move (0 in the original mode)
        self.free_cam = False
        self.zoom = 1.0
        self.audio = None
        self.on_gem_arrive = None
        self.hud_targets = {0: (200, 80), 1: (1000, 80)}
        self.timeline_end = 0.0
        self._now = 0.0
        self._sx = self._sy = 0.0
        self._visible_fires = []
        p0 = world.players[0] if world.players else None
        self.cam = [p0.c + 0.5, p0.r + 0.5] if p0 else [world.w / 2, world.h / 2]
        self.rock_angle = {rid: 0.0 for rid in world.rocks}
        self.snake_face = {s.sid: self.rng.choice((-1, 1)) for s in world.snakes.values()}
        self.layers = {}
        self.light_tgt = None
        self.light_size = None
        self.mini_tex = None
        self.mini_dirty = True
        self.ambient_t = 0.0
        self._decorate()
        self.fx = CinematicFX(self)
        self.show_grid=True
        self._route_key=None
        self._route_dist={}
        self._route_parent={}
        self._cell_info=None

    # ================================================================== static decoration
    def _decorate(self):
        w, b = self.world, self.biome
        rng = random.Random(_h(w.w, w.h, 99) + len(b) * 7919 + (w.exit_pos[0] if w.exit_pos else 0))
        self.front_decor, self.floor_decor, self.top_decor = {}, {}, {}
        self.torches = set()
        self.static_lights = []
        self.lava = set()

        def is_open(r, c):
            return w.inside(r, c) and w.terrain[r][c] in (FLOOR, FIRE, EXIT)
        taken = []
        for r in range(w.h):
            for c in range(w.w):
                t = w.terrain[r][c]
                hsh = _h(r, c, 5)
                if t == WALL and is_open(r + 1, c):
                    roll = rng.random()
                    if roll < 0.08 and all(abs(r - a) + abs(c - bb) > 4 for a, bb in taken):
                        taken.append((r, c))
                        if b == "frost":
                            self.top_decor[(r, c)] = f"{b}/crystal_{hsh % 2}"
                            self.static_lights.append((c + 0.5, r + 0.55, 2.2, self.B["crystal"], 0.42, hsh))
                        else:
                            self.torches.add((r, c))
                            self.static_lights.append((c + 0.5, r + 0.45, 3.6, self.B["torch"], 1.0, hsh))
                    elif roll < 0.42:
                        if b == "ember" and roll > 0.22:
                            continue
                        kind = {"jungle": "vines", "frost": "icicles", "ember": "chains"}[b]
                        self.front_decor[(r, c)] = f"{b}/{kind}_{hsh % 3}"
                elif t == FLOOR:
                    near_wall = any(w.inside(r + dr, c + dc) and w.terrain[r + dr][c + dc] in (WALL, CRACKED)
                                    for dr, dc in ((-1, 0), (0, -1), (0, 1), (1, 0)))
                    roll = rng.random()
                    if b == "jungle":
                        if near_wall and roll < 0.18:
                            self.floor_decor[(r, c)] = f"{b}/fern_{hsh % 3}"
                        elif roll < 0.12:
                            self.floor_decor[(r, c)] = f"{b}/moss_{hsh % 2}"
                    elif b == "frost":
                        if near_wall and roll < 0.15:
                            self.floor_decor[(r, c)] = f"{b}/snow_{hsh % 2}"
                    elif roll < 0.2:
                        self.lava.add((r, c))
                        self.static_lights.append((c + 0.5, r + 0.5, 1.7, (255, 110, 40), 0.6, hsh))
                    if (r, c) not in self.floor_decor and rng.random() < 0.025:
                        self.floor_decor[(r, c)] = f"bones_{hsh % 2}"

    # ================================================================== terrain layers
    def _cell_layers(self, r, c):
        w, b = self.world, self.biome
        t = w.terrain[r][c]
        hsh = _h(r, c)

        def ter(rr, cc):
            return w.terrain[rr][cc] if w.inside(rr, cc) else WALL
        below, above = ter(r + 1, c), ter(r - 1, c)
        left, right = ter(r, c - 1), ter(r, c + 1)
        flat = (FLOOR, FIRE, EXIT)
        solid = (WALL, CRACKED)
        L, E = [], []
        ai = (r % 4) * 4 + (c % 4)                      # position in the seamless 4x4 atlas
        if t in solid:
            front = below in flat
            L.append(f"{b}/wall/{'f' if front else 't'}{ai}")
            if t == CRACKED:
                L.append(f"{b}/crackov_{hsh % 2}")
                E.append(f"{b}/crackglow_{hsh % 2}")
            if above not in solid:
                L.append("shadows/rim")
            if right not in solid:
                L.append("shadows/side_r")
            if left not in solid:
                L.append("shadows/side_l")
            if t == WALL and front and (r, c) in self.front_decor:
                L.append(self.front_decor[(r, c)])
            if t == WALL and (r, c) in self.top_decor:
                L.append(self.top_decor[(r, c)])
            if t == WALL and front and (r, c) in self.torches:
                L.append("torch")
        elif t == DIRT:
            L.append(f"{b}/dirt/{'f' if below in flat else 't'}{ai}")
            if above in solid:
                L.append("shadows/n")
            if left in solid:
                L.append("shadows/w")
        else:
            L.append(f"{b}/floor/{ai}")
            if (r, c) in self.lava:
                E.append(f"{b}/lava_{hsh % 3}")
            if (r, c) in self.floor_decor:
                L.append(self.floor_decor[(r, c)])
            if above in solid or above == DIRT:
                L.append("shadows/n")
            if left in solid or left == DIRT:
                L.append("shadows/w")
            if right in solid:
                L.append("shadows/e")
            if t == FIRE:
                L.append(f"firepit_{hsh % 2}")
        return L, E

    def invalidate(self, cell):
        r, c = cell
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                self.layers.pop((r + dr, c + dc), None)
        self.mini_dirty = True

    def rebuild(self):
        self.layers.clear()
        self.mini_dirty = True

    def redraw_cell(self, cell):
        self.invalidate(cell)

    # ================================================================== minimap
    def minimap(self):
        """Terrain-only map texture (items are drawn on top by the HUD as icons)."""
        if self.mini_dirty or self.mini_tex is None:
            from .cartography import map_surface
            self.mini_tex = self.g.texture(map_surface(self.world,self.biome,10))
            self.mini_dirty = False
        return self.mini_tex

    # ================================================================== coords
    @property
    def ts(self):
        return TILE_U * self.g.u * self.zoom

    def S(self, cell):
        """(r, c) cell position (float ok) -> screen pixel centre."""
        ts = self.ts
        return ((cell[1] + 0.5 - self.cam[0]) * ts + self.g.W / 2 + self._sx,
                (cell[0] + 0.5 - self.cam[1]) * ts + self.g.H / 2 + self._sy)

    def P(self, x, y):
        """continuous coords (x = column, y = row) -> screen px."""
        ts = self.ts
        return ((x - self.cam[0]) * ts + self.g.W / 2 + self._sx,
                (y - self.cam[1]) * ts + self.g.H / 2 + self._sy)

    def screen_to_cell(self, sx, sy):
        ts = self.ts
        x = (sx - self.g.W / 2) / ts + self.cam[0]
        y = (sy - self.g.H / 2) / ts + self.cam[1]
        return int(math.floor(y)), int(math.floor(x))

    def player_cell_f(self, pid, now):
        p = self.world.players[pid]
        return self.tw.pos(("p", pid), (p.r, p.c), now)

    # ================================================================== effects api
    def part(self, x, y, vx, vy, life, tex, size, color=(255, 255, 255), grav=0.0, rot=0.0, vrot=0.0,
             mode=BLEND, lit=True, drag=1.5, alpha=255, grow=0.0):
        self.particles.append([x, y, vx, vy, life, life, tex, size, color, grav, rot, vrot, mode, lit,
                               drag, alpha, grow])
        if len(self.particles) > 1400:
            del self.particles[:200]

    def burst(self, x, y, color, n=18, speed=4.0, life=0.7, size=0.08, grav=6.0, mode=ADD, tex="disc",
              lit=False, up=0.3):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            sp = random.uniform(0.3, 1.0) * speed
            self.part(x, y, math.cos(a) * sp, math.sin(a) * sp - speed * up, life * random.uniform(0.6, 1.1),
                      tex, size * random.uniform(0.6, 1.3), color, grav, random.uniform(0, 360), 0, mode, lit)

    def puffs(self, x, y, n=5, color=(200, 200, 210), size=0.5, speed=1.2, life=0.9, alpha=170):
        for _ in range(n):
            a = random.uniform(0, math.tau)
            sp = random.uniform(0.2, 1.0) * speed
            self.part(x + math.cos(a) * 0.1, y + math.sin(a) * 0.1, math.cos(a) * sp,
                      math.sin(a) * sp * 0.6 - 0.3, life * random.uniform(0.7, 1.2),
                      f"smoke_{random.randint(0, 2)}", size * random.uniform(0.7, 1.2), color, -0.4,
                      random.uniform(0, 360), random.uniform(-90, 90), BLEND, True, 2.5, alpha, 0.9)

    def debris_burst(self, x, y, n=10, speed=4.0):
        for _ in range(n):
            a = random.uniform(math.pi * 1.1, math.pi * 1.9)
            sp = random.uniform(0.4, 1.0) * speed
            self.part(x, y, math.cos(a) * sp, math.sin(a) * sp, random.uniform(0.5, 0.9),
                      f"{self.biome}/debris_{random.randint(0, 2)}", random.uniform(0.08, 0.16),
                      (255, 255, 255), 14.0, random.uniform(0, 360), random.uniform(-500, 500), BLEND, True, 0.6)

    def float_text(self, x, y, s, color, now, size=22, dur=1.1):
        self.floats.append([x, y, s, color, now, dur, size])

    def ring(self, x, y, now, color, r0=0.1, r1=1.0, dur=0.45, soft=False):
        self.rings.append([x, y, now, dur, r0, r1, color, soft])

    def light_flash(self, x, y, radius, color, dur=0.3, inten=1.0):
        self.temp_lights.append([x, y, radius, color, self._now, dur, inten])

    def at(self, t, fn):
        self.pending.append((t, fn))

    def _snd(self, t, name, cell=None, vol=1.0):
        if self.audio is None:
            return

        def fn():
            pan, v = 0.5, vol
            if cell is not None:
                x, y = self.S(cell)
                pan = x / self.g.W
                if not (-80 < x < self.g.W + 80 and -80 < y < self.g.H + 80):
                    v *= 0.45
            self.audio.play(name, pan, v)
        self.at(t, fn)

    # ================================================================== event timeline
    def schedule(self, events, now):
        # Rules have already applied the entire turn. Tool choreography must use
        # the pre-move cell when the same turn also contains a movement event.
        self.fx.source_positions={e['pid']:e['from'] for e in events if e['type']=='move'}
        cur = now
        tool_end = now
        grav0 = None
        end = now + 0.05
        rock_t = {}                 # rock id -> when its fall so far ends (falls chain)
        last_fall = {}
        for i, e in enumerate(events):
            if e["type"] == "fall":
                last_fall[e["rock"]] = i
        crushed_rocks={e["rock"] for e in events if e["type"]=="crush"}
        crush_t = None
        for i, e in enumerate(events):
            ty = e["type"]
            pid = e.get("pid")
            if ty in ("move", "wait") and rock_t:
                cur = max(cur, max(rock_t.values()))     # walk on once the rocks have landed
            if ty == "hammer":
                t0 = cur
                self._fx_hammer(pid,e,t0)
                self._snd(t0+.16,'swing',e['pos'],.7)
                self._snd(t0+.34,'clang',e['pos'])
                self._snd(t0+.34,'crumble' if e['target']=='wall' else 'stun',e['pos'])
                self.beam_hold=max(self.beam_hold,t0+.34)
                cur=tool_end=t0+.72
                end=max(end,cur)
            elif ty == "push":
                t0=cur
                self.tw.add(('rock',e['rock']),t0+.23,t0+.58,e['from'],e['to'],gfx.ease_in_out)
                self._spin(e['rock'],t0+.23,t0+.58,75*(1 if e['to'][1]>e['from'][1] else -1))
                self._fx_push(pid,e,t0);self._snd(t0+.25,'push',e['from'])
                self.beam_hold=max(self.beam_hold,t0+.58)
                cur=tool_end=t0+.70;end=max(end,cur)
            elif ty == "hook" and e['obj']=='statue':
                t0=cur;key=('statue',e['sid']);bkey=('statue_body',e['sid'])
                fb,tb=(e['from'][0]+1,e['from'][1]),(e['to'][0]+1,e['to'][1])
                if e['to']!=e['from']:
                    self.tw.add(key,t0+.42,t0+.80,e['from'],e['to'],gfx.ease_in_out)
                    self.tw.add(bkey,t0+.42,t0+.80,fb,tb,gfx.ease_in_out)
                self.tw.add(('sface',e['sid']),t0+.56,t0+.98,-e['facing'],e['facing'],gfx.ease_in_out)
                self._fx_hook(pid,dict(e,**{'from':fb}),bkey,t0)
                self._snd(t0+.17,'hook_throw',fb);self._snd(t0+.42,'grind',fb)
                self.beam_hold=max(self.beam_hold,t0+.99)
                cur=tool_end=t0+1.02;end=max(end,cur)
            elif ty == "hook":
                t0=cur;key=('rock',e['rock']) if e['obj']=='rock' else ('snake',e['sid'])
                self.tw.add(key,t0+.42,t0+.80,e['from'],e['to'],gfx.ease_in_out)
                if e['obj']=='rock':
                    sign=1 if e['to'][1]>e['from'][1] else -1
                    self._spin(e['rock'],t0+.42,t0+.80,75*abs(e['to'][1]-e['from'][1])*sign)
                self._fx_hook(pid,e,key,t0)
                self._snd(t0+.17,'hook_throw',e['from']);self._snd(t0+.42,'hook_pull',e['from'])
                self.beam_hold=max(self.beam_hold,t0+.80)
                cur=tool_end=t0+1.02;end=max(end,cur)
            elif ty == "move":
                t0, t1 = cur, cur + 0.22
                self.tw.add(("p", pid), t0, t1, e["from"], e["to"], gfx.ease_in_out)
                self.facing[pid] = e["dir"]
                self._snd(t0 + 0.08, "step_ai" if pid == 0 else "step", e["to"], 0.75 if pid == 0 else 0.55)
                if e["dig"]:
                    self.beam_hold = max(self.beam_hold, t1)
                    self._snd(t0 + 0.05, "dig", e["to"], 0.8)
                    cell = e["to"]
                    self.at(t0 + 0.06, lambda c=cell: self._fx_dig(c))
                cur = t1
                end = max(end, cur)
            elif ty == "wait":
                p = self.world.players[pid]
                self._snd(cur, "wait", (p.r, p.c), 0.4)
                self.at(cur, lambda p=p: self.ring(p.c + 0.5, p.r + 0.8, self._now, PLAYER_COL[p.pid],
                                                    0.1, 0.55, 0.45))
                cur += 0.12
                end = max(end, cur)
            elif ty == "collect":
                self.ghost_gems[e["pos"]] = (e["kind"], cur)
                self._snd(cur, "gem_r" if e["kind"] == "r" else "gem_v", e["pos"])
                self.at(cur, lambda e=e: self._fx_collect(e))
            elif ty == "damage":
                col = (255, 90, 110) if e["why"] in ("snake", "fire", "boulder", "mine") else \
                    ((170, 225, 255) if e["why"] == "lens" else gfx.AI)
                txt = f"-{e['amount']}"
                tt = cur if e["why"] != "boulder" else (crush_t or cur) + 0.02
                if e["why"] in ("mine", "lens"):
                    tt = cur + 0.02
                if e["why"] in ("hammer", "push", "hook"):
                    tt = tool_end - 0.12
                if e['why']=='snake':tt=cur+.32
                pos = e["pos"]
                if e["why"] in ("fire", "snake", "boulder", "mine", "lens"):
                    self._snd(tt + 0.12, "hurt", pos, 0.6)
                    self.at(tt, lambda pid=pid: self.hit_time.__setitem__(pid, self._now))
                if e["why"] == "fire":
                    self._snd(tt, "burn", pos)
                    self.at(tt, lambda pos=pos, pid=pid: self.fx.burn(pos,pid))
                    end=max(end,tt+1.35)
                self.at(tt, lambda pos=pos, txt=txt, col=col: self.float_text(
                    pos[1] + 0.5, pos[0] + 0.05, txt + " life", col, self._now, 18))
            elif ty == "bite":
                s_cell = e["pos"]
                # The rules remove a viper immediately after a bite. Keep a
                # presentation ghost until the strike/recovery has finished.
                self.ghost_snakes[e['sid']]=(s_cell,cur+.90,False)
                # Bite is rendered once on the real snake, using authored strike frames.
                self._snd(cur, "rattle", s_cell,.55)
                self._snd(cur+.28, "bite", s_cell)
                self.at(cur, lambda e=e: self._fx_bite(e))
                end = max(end, cur + 0.90)
            elif ty == "wake":
                pos = e["pos"]
                self.at(cur,lambda e=e:self.fx.wake(e))
                end=max(end,cur+.58)
                self._snd(cur + 0.05, "rattle", pos, 0.7)
                self.at(cur + 0.05, lambda pos=pos: self.float_text(pos[1] + 0.5, pos[0], "!", gfx.SNAKE,
                                                                    self._now, 30, 0.7))
            elif ty == "fall":
                rid = e["rock"]
                t0 = max(cur + 0.02, rock_t.get(rid, 0.0))
                if grav0 is None:
                    grav0 = t0
                dur = 0.14 if "roll" in e else 0.09
                self.tw.add(("rock", rid), t0, t0 + dur, e["from"], e["to"], gfx.ease_in if rid not in rock_t
                            else (lambda k: k))
                if "roll" in e:
                    self._spin(rid, t0, t0 + dur, 90 * e["roll"])
                rock_t[rid] = t0 + dur
                self.beam_hold = max(self.beam_hold, t0 + dur)
                if last_fall.get(rid) == i and rid not in crushed_rocks:
                    to = e["to"]
                    self._snd(t0 + dur, "thud", to, 0.6)
                    self.at(t0 + dur, lambda to=to: self._fx_land(to))
                end = max(end, t0 + dur)
            elif ty == "crush":
                tt = rock_t.get(e["rock"], cur) + 0.01
                crush_t = tt
                if e['rock'] not in rock_t:
                    logical=tuple(self.world.rocks[e['rock']])
                    self.tw.add(('rock',e['rock']),max(now,tt-.12),tt+.06,
                                (logical[0]-.12,logical[1]),(logical[0]+.08,logical[1]),gfx.ease_in)
                self._snd(tt, "crush", e["pos"])
                self.at(tt, lambda e=e: self._fx_crush(e))
                self.at(tt, lambda pid=pid: self.hit_time.__setitem__(pid, self._now))
                end = max(end, tt + 0.80)
            elif ty == "snake_crushed":
                tt = rock_t.get(e["rock"], cur)
                self.ghost_snakes[e["sid"]] = (e["pos"], tt, False)
                self._snd(tt, "splat", e["pos"])
                self.at(tt, lambda pos=e["pos"]: self.burst(pos[1] + 0.5, pos[0] + 0.5, gfx.SNAKE, 24, 4, 0.7,
                                                            0.07, 8, BLEND, lit=True))
            elif ty == "death":
                tt = max(cur, (crush_t or cur) + 0.2)
                self.at(tt, lambda e=e: self._fx_death(e))
                self._snd(tt, "death", e["pos"])
                end = max(end, tt + 1.0)
            elif ty == "exit":
                self.at(cur, lambda e=e: self._fx_exit(e))
                self._snd(cur, "exit", e["pos"])
                end = max(end, cur + 0.9)
            elif ty == "mine_place":
                t0 = cur
                pos = e["pos"]
                self.tw.add(("mine", pos), t0, t0 + 0.3, 0.0, 1.0, gfx.ease_back)
                self._snd(t0 + 0.05, "mine_place", pos)
                self.at(t0 + 0.2, lambda pos=pos, pid=pid: self.ring(pos[1] + 0.5, pos[0] + 0.6, self._now,
                                                                     PLAYER_COL[pid], 0.1, 0.7, 0.4))
                cur = tool_end = t0 + 0.32
                end = max(end, cur)
            elif ty in ("mine", "mine_boom"):
                tt = cur if ty == "mine" else (max(rock_t.values()) if rock_t else cur)
                self.ghost_mines[e["pos"]] = (e["owner"], tt)
                self._snd(tt, "explode", e["pos"])
                self.at(tt, lambda e=e: self._fx_explode(e))
                if ty == "mine":
                    self.at(tt, lambda pid=pid: self.hit_time.__setitem__(pid, self._now))
                end = max(end, tt + 1.45)
            elif ty == "freeze":
                tt = cur
                self._snd(tt, "freeze", e["pos"])
                self.at(tt, lambda e=e: self._fx_freeze(e))
                self.at(tt, lambda pid=pid: self.frozen_vis.__setitem__(pid, True))
                self.at(tt, lambda pid=pid: self.hit_time.__setitem__(pid, self._now))
                end = max(end, tt + 0.5)
            elif ty == "thaw":
                self._snd(cur, "thaw", e["pos"])
                self.at(cur, lambda e=e: self._fx_thaw(e))
                self.at(cur, lambda pid=pid: self.frozen_vis.__setitem__(pid, False))
                end = max(end, cur + 0.3)
            elif ty == "game_over":
                end = max(end, cur + 0.6)
        self.timeline_end = max(self.timeline_end, end, self.tw.busy_until())
        return self.timeline_end

    def _spin(self, rid, t0, t1, deg):
        a0 = self.rock_angle.get(rid, 0)
        self.tw.add(("rspin", rid), t0, t1, a0, a0 + deg, gfx.ease_out)
        self.rock_angle[rid] = a0 + deg

    # ---- individual effects ------------------------------------------------
    def _fx_dig(self, cell):
        x, y = cell[1] + 0.5, cell[0] + 0.5
        self.debris_burst(x, y + 0.2, 8, 3.5)
        col = tuple(int(v * 255) for v in hexrgb(self.B["dirt"][1]))
        self.puffs(x, y + 0.2, 3, col, 0.45, 0.9, 0.7, 120)
        if self.biome == "jungle" and random.random() < 0.5:
            for _ in range(2):
                self.part(x, y, random.uniform(-1, 1), -random.uniform(1, 2), 1.2, "leaf", 0.14,
                          (255, 255, 255), 1.5, random.uniform(0, 360), random.uniform(-200, 200), BLEND, True, 1.2)
        self.invalidate(cell)

    def _fx_land(self, cell):
        self.fx.land(cell)

    def _fx_collect(self, e):
        x, y = e["pos"][1] + 0.5, e["pos"][0] + 0.5
        col = gfx.VIOLET if e["kind"] == "v" else gfx.RED
        light = gfx.VIOLET_LIGHT if e["kind"] == "v" else gfx.RED_LIGHT
        self.burst(x, y, col, 26, 5.5, 0.8, 0.07, 3)
        self.burst(x, y, light, 8, 3, 0.6, 0.18, 0, ADD, "flare")
        self.ring(x, y, self._now, light, 0.15, 1.2, 0.5)
        self.light_flash(x, y, 3.5, col, 0.5, 1.0)
        self.float_text(x, y - 0.5, f"+{e['points']}", light, self._now, 28)
        self.ghost_gems.pop(e["pos"], None)
        self.mini_dirty = True
        sx, sy = self.S(e["pos"])
        tx, ty = self.hud_targets[e["pid"]]
        name = "gem_r" if e["kind"] == "r" else "gem_v"
        t0 = self._now
        pid = e["pid"]
        g = self.g

        def draw(now, sx=sx, sy=sy, tx=tx, ty=ty, t0=t0, col=col):
            k = gfx.clamp((now - t0) / 0.8)
            kk = gfx.ease_in_out(k)
            arc = math.sin(k * math.pi)
            sz = g.u * 30 * (1 + 0.5 * arc)
            for i in range(6):
                kt = max(0, k - i * 0.025)
                ktt = gfx.ease_in_out(kt)
                px = gfx.lerp(sx, tx, ktt) - math.sin(kt * math.pi) * 80 * g.u
                py = gfx.lerp(sy, ty, ktt) - math.sin(kt * math.pi) * 140 * g.u
                g.glow(px, py, sz * (0.9 - i * 0.12), col, 0.7 - i * 0.1)
            cx = gfx.lerp(sx, tx, kk) - arc * 80 * g.u
            cy = gfx.lerp(sy, ty, kk) - arc * 140 * g.u
            t = self.A.get(name, sz)
            if t:
                g.draw(t, cx - sz / 2, cy - sz / 2, sz, sz, angle=k * 360)
        self.screen_fx.append({"t0": t0, "t1": t0 + 0.8, "fn": draw})
        self.at(t0 + 0.8, lambda pid=pid: self.on_gem_arrive and self.on_gem_arrive(pid))

    def _fx_explode(self, e):
        self.fx.explode(e)

    def _fx_freeze(self, e):
        self.fx.freeze(e)

    def _fx_thaw(self, e):
        self.fx.thaw(e)

    def _fx_burn(self, pos):
        self.fx.burn(pos)

    def _fx_bite(self, e):
        self.fx.bite(e)

    def _fx_crush(self, e):
        self.fx.land(e["pos"],heavy=True,event=e)

    def _fx_death(self, e):
        x, y = e["pos"][1] + 0.5, e["pos"][0] + 0.5
        col = PLAYER_COL[e["pid"]]
        self.burst(x, y, col, 70, 8, 1.2, 0.09, 2)
        self.burst(x, y, (255, 255, 255), 24, 5, 0.8, 0.05, 0)
        self.ring(x, y, self._now, col, 0.2, 3.0, 0.9)
        self.light_flash(x, y, 6, col, 0.9, 1.0)
        self.shake = max(self.shake, 0.4)

    def _fx_exit(self, e):
        x, y = e["pos"][1] + 0.5, e["pos"][0] + 0.5
        col = PLAYER_COL[e["pid"]]
        for i in range(3):
            self.rings.append([x, y, self._now + i * 0.12, 0.7, 0.1, 1.8 + i * 0.4, col, False])
        self.burst(x, y, (255, 255, 255), 40, 6, 1.0, 0.06, -1)
        self.light_flash(x, y, 6, col, 1.0, 1.0)

    def _fx_hammer(self, pid, e, t0):
        self.fx.hammer(pid,e,t0)

    def _fx_push(self, pid, e, t0):
        self.fx.push(pid,e,t0)

    def _fx_hook(self, pid, e, key, t0):
        self.fx.hook(pid,e,key,t0)

    # ================================================================== update
    def update(self, dt, now, focus_pid=None):
        self._now = now
        self.fx.update(now)
        due = [p for p in self.pending if p[0] <= now]
        if due:
            self.pending = [p for p in self.pending if p[0] > now]
            for _, fn in sorted(due, key=lambda p: p[0]):
                fn()
        alive = []
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += p[9] * dt
            k = max(0.0, 1 - p[14] * dt)
            p[2] *= k
            if p[9] == 0:
                p[3] *= k
            p[10] += p[11] * dt
            p[7] += p[16] * dt * p[7]
            p[4] -= dt
            if p[4] > 0:
                alive.append(p)
        self.particles = alive
        self.floats = [f for f in self.floats if now - f[4] < f[5]]
        self.rings = [r for r in self.rings if now - r[2] < r[3]]
        self.actives = [a for a in self.actives if now < a["t1"]]
        self.screen_fx = [a for a in self.screen_fx if now < a["t1"]]
        self.temp_lights = [t for t in self.temp_lights if now - t[4] < t[5]]
        self.ghost_gems = {k: v for k, v in self.ghost_gems.items() if v[1] > now}
        self.ghost_snakes = {k: v for k, v in self.ghost_snakes.items() if v[1] > now}
        self.ghost_mines = {k: v for k, v in self.ghost_mines.items() if v[1] > now}
        self._ambient(dt)
        self._camera(dt, now)
        self.shake = max(0.0, self.shake - dt * 0.8)

    def _ambient(self, dt):
        """Biome atmosphere: dust motes, snow, rising embers, plus embers from fires."""
        ts = self.ts
        vw, vh = self.g.W / ts, self.g.H / ts
        x0, y0 = self.cam[0] - vw / 2, self.cam[1] - vh / 2
        kind = self.B["particles"]
        self.ambient_t += dt * min(40, vw * vh * 0.06)
        while self.ambient_t > 1:
            self.ambient_t -= 1
            x = x0 + random.random() * vw
            y = y0 + random.random() * vh
            if kind == "snow":
                self.part(x, y0 - 0.5 + random.random() * vh * 0.4, random.uniform(-0.3, 0.3),
                          random.uniform(0.5, 1.0), random.uniform(4, 7), "snowflake", random.uniform(0.05, 0.1),
                          (235, 245, 255), 0, 0, random.uniform(-60, 60), BLEND, True, 0.0, 220)
            elif kind == "embers":
                self.part(x, y + vh * 0.3, random.uniform(-0.2, 0.2), -random.uniform(0.3, 0.8),
                          random.uniform(2.5, 5), "soft", random.uniform(0.04, 0.08),
                          random.choice([(255, 140, 50), (255, 90, 30), (255, 190, 90)]), -0.05, 0, 0, ADD,
                          False, 0.2, 220)
            else:
                self.part(x, y, random.uniform(-0.15, 0.15), random.uniform(-0.1, 0.1), random.uniform(3, 6),
                          "soft", random.uniform(0.025, 0.05), (255, 245, 210), 0, 0, 0, ADD, False, 0.0, 110)
        for (r, c) in self._visible_fires:
            if random.random() < dt * 1.7:
                self.part(c+.5, r+.20, random.uniform(-.15,.2), -.42,
                          1.65, f"smoke_{(r+c)%3}", .65, (96,102,112),
                          -.08, 0, 12, BLEND, True, .4, 54)
            if random.random() < dt * 6:
                self.part(c + 0.5 + random.uniform(-0.25, 0.25), r + 0.4, random.uniform(-0.2, 0.2),
                          -random.uniform(1.0, 2.0), random.uniform(0.6, 1.2), "soft", random.uniform(0.03, 0.06),
                          random.choice([(255, 170, 60), (255, 110, 30)]), -0.5, 0, 0, ADD, False, 0.5, 255)

    def _camera(self, dt, now):
        """Frame the explorers inside the part of the screen the HUD leaves free
        (self.insets: left, top, right, bottom in design units; the app widens the right
        inset while the minimap is showing so nobody hides under it)."""
        w = self.world
        g = self.g
        il, it, ir, ib = (v * g.u for v in getattr(self, "insets", (60, 170, 60, 150)))
        if not self.free_cam and w.players:
            pts = {p.pid: self.player_cell_f(p.pid, now) for p in w.players if p.alive and not p.exited}
            if not pts:
                pts = {p.pid: self.player_cell_f(p.pid, now) for p in w.players}
            base = TILE_U * g.u
            avail_w, avail_h = g.W - il - ir, g.H - it - ib
            if self.cam_mode == "both" and len(pts) == 2:
                (r0, c0), (r1, c1) = pts[0], pts[1]
                fp = getattr(self, 'focus_pid', 0)
                need_w = (abs(c0 - c1) + 2.5) * base
                need_h = (abs(r0 - r1) + 2.5) * base
                tz = max(ZMIN, min(1.0, avail_w / need_w, avail_h / need_h))
                cx, cy = (c0 + c1) / 2 + 0.5, (r0 + r1) / 2 + 0.5
                hx, hy = pts[fp][1] + 0.5, pts[fp][0] + 0.5
                tsz = base * tz
                # if both can't fit, keep YOU inside the free area
                mx = (avail_w / 2 - 40 * g.u) / tsz
                my = (avail_h / 2 - 30 * g.u) / tsz
                cx = min(max(cx, hx - mx), hx + mx)
                cy = min(max(cy, hy - my), hy + my)
            else:
                fp = getattr(self, "focus_pid", 0)
                pid = 1 if self.cam_mode == "ai" else 0
                if roles.mode() == "hvh":                  # "me" = whoever is moving, "ai" = the other
                    pid = 1 - fp if self.cam_mode == "ai" else fp
                if pid not in pts:
                    pid = next(iter(pts))
                tz = 1.0
                tsz = base
                cx, cy = pts[pid][1] + 0.5, pts[pid][0] + 0.5
            # centre of the free area, not of the screen
            cx += (ir - il) / 2 / tsz
            cy += (ib - it) / 2 / tsz
            self.zoom += (tz - self.zoom) * min(1.0, dt * 3)
            if abs(self.zoom - tz) < 0.002:
                self.zoom = tz
            k = min(1.0, dt * 5)
            self.cam[0] += (cx - self.cam[0]) * k
            self.cam[1] += (cy - self.cam[1]) * k
        ts = self.ts
        vw, vh = g.W / ts, g.H / ts
        ml, mr, top, bot = il / ts, ir / ts, it / ts, ib / ts
        if vw >= w.w + ml + mr:
            self.cam[0] = w.w / 2 + (mr - ml) / 2
        else:
            self.cam[0] = max(vw / 2 - ml, min(w.w - vw / 2 + mr, self.cam[0]))
        if vh >= w.h + top + bot:
            self.cam[1] = w.h / 2 + (bot - top) / 2
        else:
            self.cam[1] = max(vh / 2 - top, min(w.h - vh / 2 + bot, self.cam[1]))

    # ================================================================== draw
    def draw(self, now, overlays=None):
        g = self.g
        self._now = now
        sh = self.shake * self.ts
        self._sx = random.uniform(-1, 1) * sh if sh > 0.5 else 0
        self._sy = random.uniform(-1, 1) * sh if sh > 0.5 else 0
        w = self.world
        ts = self.ts
        g.clear(self.B["fog"])
        vw, vh = g.W / ts, g.H / ts
        x0 = self.cam[0] - vw / 2 - self._sx / ts
        y0 = self.cam[1] - vh / 2 - self._sy / ts
        vc0, vr0 = int(math.floor(x0)), int(math.floor(y0))
        vc1, vr1 = int(math.floor(x0 + vw)), int(math.floor(y0 + vh))
        c0, r0 = max(0, vc0), max(0, vr0)
        c1, r1 = min(w.w - 1, vc1), min(w.h - 1, vr1)
        ox = g.W / 2 - self.cam[0] * ts + self._sx
        oy = g.H / 2 - self.cam[1] * ts + self._sy
        A = self.A
        emissive = []
        fires = []
        layers = self.layers
        terrain = w.terrain
        level = 0 if ts >= A.ts * 0.66 else (1 if ts >= A.ts * 0.33 else 2)
        lm = A.lodmap(level)
        xs = [round(ox + c * ts) for c in range(vc0, vc1 + 2)]
        ys = [round(oy + r * ts) for r in range(vr0, vr1 + 2)]
        # ---------------- rock beyond the map edge (the cave continues)
        if vr0 < 0 or vc0 < 0 or vr1 >= w.h or vc1 >= w.w:
            used = []
            for r in range(vr0, vr1 + 1):
                ya, yb = ys[r - vr0], ys[r - vr0 + 1]
                for c in range(vc0, vc1 + 1):
                    if 0 <= r < w.h and 0 <= c < w.w:
                        continue
                    t = lm.get(f"{self.biome}/wall/t{(r % 4) * 4 + c % 4}")
                    if t is not None:
                        if t not in used:
                            t.blend_mode = BLEND
                            t.color = (150, 150, 150)
                            t.alpha = 255
                            used.append(t)
                        xa = xs[c - vc0]
                        t.draw(dstrect=(xa, ya, xs[c - vc0 + 1] - xa, yb - ya))
            for t in used:
                t.color = (255, 255, 255)
        # ---------------- terrain (terrain textures always keep the default blend state)
        for r in range(r0, r1 + 1):
            ya, yb = ys[r - vr0], ys[r - vr0 + 1]
            hgt = yb - ya
            row = terrain[r]
            for c in range(c0, c1 + 1):
                xa = xs[c - vc0]
                L = layers.get((r, c))
                if L is None:
                    L = self._cell_layers(r, c)
                    layers[(r, c)] = L
                rect = (xa, ya, xs[c - vc0 + 1] - xa, hgt)
                for name in L[0]:
                    t = lm.get(name)
                    if t is not None:
                        t.draw(dstrect=rect)
                if L[1]:
                    emissive.append((L[1], rect, r, c))
                if row[c] == FIRE:
                    fires.append((r, c))
        self._visible_fires = fires
        self._draw_cells(now,r0,r1,c0,c1)
        self.fx.draw_ground(now)
        # ---------------- objects (depth sorted)
        objs = []
        for (r, c), kind in list(w.gems.items()) + [(k, v[0]) for k, v in self.ghost_gems.items()]:
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                objs.append((r + 0.5, 0, (r, c, kind)))
        resting = getattr(w, "resting", {})
        for rid, (r, c) in w.rocks.items():
            pos = self.tw.pos(("rock", rid), (r, c), now)
            if r0 - 2 <= pos[0] <= r1 + 1 and c0 - 1 <= pos[1] <= c1 + 1:
                if resting.get(rid) and not self.tw.active(("rock", rid), now):
                    # pinned on an explorer's head: draw it on top, sitting on them
                    objs.append((pos[0] + 1.7, 1, (rid, (pos[0] + 0.2, pos[1]))))
                else:
                    objs.append((pos[0] + 0.55, 1, (rid, pos)))
        for (r, c), s in w.snakes.items():
            pos = self.tw.pos(("snake", s.sid), (r, c), now)
            if r0 - 1 <= pos[0] <= r1 + 1 and c0 - 1 <= pos[1] <= c1 + 1:
                objs.append((pos[0] + 0.5, 2, (s.sid, pos, bool(s.stuns), s.stuns, False)))
        for sid, (cell, until, _) in self.ghost_snakes.items():
            if r0 - 1 <= cell[0] <= r1 + 1:
                objs.append((cell[0] + 0.5, 2, (sid, cell, False, {}, False)))
        for (r, c) in fires:
            objs.append((r + 0.97, 3, (r, c)))
        for cell, owner in list(getattr(w, "mines", {}).items()) + [(k, v[0]) for k, v in self.ghost_mines.items()]:
            if r0 - 1 <= cell[0] <= r1 + 1 and c0 - 1 <= cell[1] <= c1 + 1:
                objs.append((cell[0] + 0.4, 7, (cell, owner)))
        for (r, c) in self.torches:
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1 and terrain[r][c] == WALL:
                objs.append((r + 0.99, 4, (r, c)))
        for (r, c), st in getattr(w, "statues", {}).items():
            pos = self.tw.pos(("statue", st.sid), (r, c), now)
            if r0 - 2 <= pos[0] <= r1 + 1 and c0 - 2 <= pos[1] <= c1 + 2:
                objs.append((pos[0] + 1.6, 8, (st, pos)))
        ex = w.exit_pos
        if ex and r0 - 2 <= ex[0] <= r1 + 2 and c0 - 2 <= ex[1] <= c1 + 2:
            objs.append((ex[0] + 0.3, 5, ex))
        for p in w.players:
            if p.exited:
                continue
            pos = self.player_cell_f(p.pid, now)
            objs.append((pos[0] + 0.62 + p.pid * 0.001, 6, (p, pos)))
        objs.sort(key=lambda o: o[0])
        glows = []
        shtex = g.prims["shadow"]
        for _, kind, data in objs:
            if kind == 1:
                pos = data[1]
                x, y = self.P(pos[1] + 0.55, pos[0] + 0.9)
                g.draw(shtex, x - ts * 0.42, y - ts * 0.12, ts * 0.84, ts * 0.26, alpha=150)
            elif kind == 6:
                pos = data[1]
                x, y = self.P(pos[1] + 0.52, pos[0] + 0.9)
                g.draw(shtex, x - ts * 0.34, y - ts * 0.1, ts * 0.68, ts * 0.22, alpha=160)
            elif kind == 8:
                pos = data[1]
                x, y = self.P(pos[1] + 0.5, pos[0] + 1.92)
                g.draw(shtex, x - ts * 0.5, y - ts * 0.12, ts * 1.0, ts * 0.26, alpha=170)
            elif kind == 0:
                r, c, _k = data
                x, y = self.P(c + 0.5, r + 0.86)
                g.draw(shtex, x - ts * 0.24, y - ts * 0.06, ts * 0.48, ts * 0.13, alpha=110)
        for _, kind, data in objs:
            if kind == 0:
                self._draw_gem(data, now, glows)
            elif kind == 1:
                self._draw_rock(data, now)
            elif kind == 2:
                self._draw_snake(data, now)
            elif kind == 3:
                self._draw_fire(data, now, glows)
            elif kind == 4:
                self._draw_torch(data, now, glows)
            elif kind == 5:
                self._draw_portal(data, now, glows)
            elif kind == 6:
                self._draw_player(data, now, glows)
            elif kind == 7:
                self._draw_mine(data, now, glows)
            elif kind == 8:
                self._draw_statue(data, now, glows)
        self._draw_particles(lit=True)
        for a in self.actives:
            if now >= a["t0"]:
                a["fn"](now)
        self.fx.draw_tools(now)
        # ---------------- lighting
        self._lighting(now, r0, r1, c0, c1)
        # ---------------- emissive / bloom
        emi = A.lodmap(level)
        for names, rect, r, c in emissive:
            pulse = 0.65 + 0.35 * math.sin(now * 2.2 + r * 0.7 + c)
            for name in names:
                t = emi.get(name)
                if t is not None:
                    g.draw(t, rect[0], rect[1], rect[2], rect[3], alpha=255 * pulse, mode=ADD)
        self._draw_beams(now)
        self.fx.draw_after_light(now)
        for (pos, rad, col, inten) in glows:
            g.glow(pos[0], pos[1], rad, col, inten)
        self._draw_particles(lit=False)
        for rr in self.rings:
            if now < rr[2]:
                continue
            k = gfx.clamp((now - rr[2]) / rr[3])
            rad = gfx.lerp(rr[4], rr[5], gfx.ease_out(k)) * ts
            x, y = self.P(rr[0], rr[1])
            g.ring(x, y, rad, rr[6], 255 * (1 - k), soft=rr[7], mode=ADD)
        if overlays:
            for fn in overlays:
                fn(self, now)
        self._draw_cell_info()
        self._tags(now)
        for f in self.floats:
            k = (now - f[4]) / f[5]
            x, y = self.P(f[0], f[1])
            y -= 60 * g.u * gfx.ease_out(k)
            sz = f[6] * g.u * (1 + 0.25 * max(0, 1 - k * 4))
            g.text(f[2], x, y, sz, f[3], True, "center", 255 * (1 - k ** 2), shadow=True)
        self._offscreen_markers(now)
        self._audio_scene()
        g.draw(g.prims["vignette"], 0, 0, g.W, g.H, color=(0, 0, 0), alpha=145)
        for a in self.screen_fx:
            if now >= a["t0"]:
                a["fn"](now)
        if self.flash:
            col, t0, dur = self.flash
            k = (now - t0) / dur
            if 0 <= k < 1:
                g.draw(g.prims["vignette"], 0, 0, g.W, g.H, color=col, alpha=255 * (1 - k))
                g.rect(0, 0, g.W, g.H, col, 40 * (1 - k))
            elif k >= 1:
                self.flash = None

    # ---------------------------------------------------------------- lighting
    def _lighting(self, now, r0, r1, c0, c1):
        g = self.g
        div = 3
        size = (max(1, g.W // div), max(1, g.H // div))
        if self.light_tgt is None or self.light_size != size:
            self.light_tgt = g.target(*size)
            self.light_size = size
        tgt = self.light_tgt
        if getattr(self, "capture", False):          # debugging aid: grab the unlit scene
            self.cap_scene = g.r.to_surface()
        g.r.target = tgt
        amb = self.B["ambient"]
        if g.overbright or getattr(self, "capture", False):
            gain = 0.42
            g.clear(tuple(min(255, int(v * 1.3)) for v in amb))
        else:
            gain = 1.0
            g.clear(tuple(min(255, int(v * 1.9)) for v in amb))
        if self.biome=='frost':gain*=.62
        light = g.prims["light"]
        light.blend_mode = ADD
        ts = self.ts / div
        sx0, sy0 = g.W / 2 + self._sx, g.H / 2 + self._sy
        cx, cy, tsf = self.cam[0], self.cam[1], self.ts

        def L(x, y, rad, col, inten):
            rr = rad * ts
            light.color = col
            light.alpha = int(255 * max(0.0, min(1.0, inten * gain)))
            light.draw(dstrect=(((x - cx) * tsf + sx0) / div - rr, ((y - cy) * tsf + sy0) / div - rr,
                                rr * 2, rr * 2))
        w = self.world
        for (x, y, rad, col, inten, hsh) in self.static_lights:
            if c0 - 5 <= x <= c1 + 5 and r0 - 5 <= y <= r1 + 5:
                fl = 0.85 + 0.15 * math.sin(now * 9 + hsh) * math.sin(now * 5.3 + hsh * 0.7)
                L(x, y, rad * (0.95 + 0.05 * fl), col, inten * fl)
        for (r, c) in self._visible_fires:
            fl = 0.8 + 0.2 * math.sin(now * 11 + r * 3 + c) * math.sin(now * 6.1 + c)
            L(c + 0.5, r + 0.4, 3.2 * (0.95 + 0.05 * fl), (255, 140, 60), fl)
        for (r, c), kind in w.gems.items():
            if c0 - 2 <= c <= c1 + 2 and r0 - 2 <= r <= r1 + 2:
                L(c + 0.5, r + 0.5, 1.4, gfx.VIOLET if kind == "v" else gfx.RED, 0.55)
        ex = w.exit_pos
        if ex:
            col = gfx.HUMAN if w.exit_open_for(self.focus_pid) else (150, 110, 230)
            L(ex[1] + 0.5, ex[0] + 0.5, 3.6, col, 0.9 + 0.1 * math.sin(now * 3))
        for sid, r, cells, d in (self.beam_segs or []):
            for (rr, cc) in cells[::2]:
                if c0 - 3 <= cc <= c1 + 3 and r0 - 3 <= rr <= r1 + 3:
                    L(cc + 0.5, rr + EYE_Y, 1.6, (150, 215, 255), 0.5)
        for (r, c), st in getattr(w, "statues", {}).items():
            if c0 - 3 <= c <= c1 + 3 and r0 - 3 <= r <= r1 + 3:
                L(*self.fx.lens(st), 1.4, (130, 216, 245), 0.55)
        for cell in getattr(w, "mines", {}):
            if (now * 1.6 + cell[0] * 0.3 + cell[1] * 0.7) % 1.0 < 0.35:
                L(cell[1] + 0.5, cell[0] + 0.45, 1.1, (255, 60, 50), 0.6)
        for p in w.players:
            if not p.alive or p.exited:
                continue
            pr = self.player_cell_f(p.pid, now)
            L(pr[1] + 0.5, pr[0] + 0.5, 4.8, (255, 244, 225) if p.pid == 0 else (255, 225, 190), 0.9)
            dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[self.facing[p.pid]]
            L(pr[1] + 0.5 + dc * 1.4, pr[0] + 0.5 + dr * 1.4, 2.6,
              (220, 250, 255) if p.pid == 0 else (255, 170, 110), 0.7)
        for (x, y, rad, col, t0, dur, inten) in self.temp_lights:
            k = (now - t0) / dur
            if k >= 0:
                L(x, y, rad, col, inten * (1 - k))
        if getattr(self, "capture", False):
            self.cap_light = g.r.to_surface()
        g.r.target = None
        tgt.blend_mode = g.MOD2X if g.overbright else MOD
        tgt.color = (255, 255, 255)
        tgt.alpha = 255
        tgt.draw(dstrect=(0, 0, g.W, g.H))

    # ---------------------------------------------------------------- particles
    def _draw_particles(self, lit):
        g = self.g
        ts = self.ts
        W, H = g.W, g.H
        prims = g.prims
        for p in self.particles:
            if p[13] != lit:
                continue
            x, y = self.P(p[0], p[1])
            if not (-60 < x < W + 60 and -60 < y < H + 60):
                continue
            k = p[4] / p[5]
            sz = p[7] * ts * (0.5 + 0.5 * k if p[16] == 0 else 1)
            name = p[6]
            t = prims.get(name) or self.A.get(name, sz)
            if t is None:
                continue
            a = p[15] * (k if p[16] == 0 else min(1, k * 1.6))
            g.draw(t, x - sz / 2, y - sz / 2, sz, sz, color=p[8], alpha=a, mode=p[12], angle=p[10])

    # ---------------------------------------------------------------- entities
    def _blit(self, name, cx, cy, w, h, **kw):
        ts = self.ts
        t = self.A.get(name, w * ts)
        if t is None:
            return
        x, y = self.P(cx, cy)
        self.g.draw(t, x - w * ts / 2, y - h * ts / 2, w * ts, h * ts, **kw)

    def _draw_gem(self, data, now, glows):
        r, c, kind = data
        bob = math.sin(now * 2.2 + r * 0.7 + c) * 0.035
        self._blit("gem_r" if kind == "r" else "gem_v", c + 0.5, r + 0.47 + bob, 0.66, 0.66)
        x, y = self.P(c + 0.5, r + 0.47 + bob)
        col = gfx.VIOLET if kind == "v" else gfx.RED
        glows.append(((x, y), self.ts * (0.5 if kind == "v" else 0.62), col,
                      0.35 + 0.15 * math.sin(now * 3 + r + c)))
        ph = (now * 0.7 + ((r * 7 + c * 3) % 10) * 0.37) % 3.2
        if ph < 0.4:
            k = math.sin(ph / 0.4 * math.pi)
            sz = self.ts * 0.55 * k
            t = self.A.get("flare", sz)
            if t and sz > 1:
                fx, fy = self.P(c + 0.38, r + 0.33 + bob)
                self.g.draw(t, fx - sz / 2, fy - sz / 2, sz, sz, mode=ADD, angle=ph * 60)

    def _draw_rock(self, data, now):
        rid, pos = data
        ang = self.tw.pos(("rspin", rid), self.rock_angle.get(rid, 0), now)
        hit=self.fx.rock_hits.get(rid)
        squash=.0 if hit is None else .08*math.sin(min(1,max(0,(now-hit)/.4))*math.pi)
        self._blit(f"{self.biome}/rock_{rid%3}",pos[1]+.5,pos[0]+.5+squash*.35,
                   .95*(1+squash*.4),.95*(1-squash),angle=ang)

    def _draw_snake(self, data, now):
        sid, pos, stunned, stuns, lunge = data
        g = self.g
        ts = self.ts
        if self.fx.draw_bite(sid,pos,now):
            return
        if self.fx.draw_stun(sid,pos,stunned,stuns,now):
            return
        face = self.snake_face.get(sid, 1)
        speed = 0.6 if stunned else 3.2
        cx, cy = pos[1] + 0.5, pos[0] + 0.55
        k = int(now * (9 if stunned else 19) + sid * 7) % 24
        sx, sy = self.P(cx, cy)
        sx += face * ts * 0.15 if lunge else 0
        sy -= ts * 0.08 if lunge else 0
        sw, sh = ts * .90, ts * .90
        seg = self.A.get(f"{self.biome}/snakecoil/{k}", sw)
        col = (150, 180, 220) if stunned else (255, 255, 255)
        if seg:
            g.draw(g.prims["shadow"], sx-sw*.48, sy+sh*.20, sw*.96, sh*.22, alpha=150)
            g.draw(seg, sx-sw/2, sy-sh*.62, sw, sh, flip=face<0, color=col)

    def _draw_mine(self, data, now, glows):
        cell, owner = data
        g = self.g
        ts = self.ts
        k = self.tw.pos(("mine", cell), 1.0, now)
        x, y = self.P(cell[1] + 0.5, cell[0] + 0.55)
        col = PLAYER_COL[owner]
        rw = ts * 0.65 * k
        g.draw(g.prims["ring"], x - rw / 2, y + ts * 0.05 - rw * 0.2, rw, rw * 0.4, color=col, alpha=65)
        sz = ts * 0.65 * k
        t = self.A.get("mine", sz)
        if t:
            g.draw(t, x - sz / 2, y - sz / 2 - ts * 0.04, sz, sz)
        blink = (now * 1.6 + cell[0] * 0.3 + cell[1] * 0.7) % 1.0 < 0.35
        lx, ly = x, y - ts * 0.04
        if blink:
            g.disc(lx, ly, ts * 0.035, (255, 60, 50))
            glows.append(((lx, ly), ts * 0.22, (255, 50, 40), 0.9))
        glows.append(((x, y + ts * 0.1), ts * 0.35, col, 0.25))

    def _beam_segments(self):
        """[(statue id, row, [lit cells in order], facing)] for the beams as they are now."""
        w = self.world
        out = []
        for (r, c), st in getattr(w, "statues", {}).items():
            cells = []
            cc = c + st.d
            while w.inside(r, cc):
                t = w.terrain[r][cc]
                if t in (WALL, CRACKED, DIRT) or (r, cc) in w.rock_at or (r, cc) in w.statue_cells:
                    break
                cells.append((r, cc))
                cc += st.d
            out.append((st.sid, r, cells, st.d))
        return out

    def _draw_beams(self, now):
        self.fx.draw_beams(now)

    def _draw_statue(self, data, now, glows):
        self.fx.draw_statue(data,now,glows)

    def _draw_fire(self, data, now, glows):
        r, c = data
        k = int(now * 22 + _h(r, c) % 24) % 24
        x, y = self.P(c + 0.5, r + 0.92)
        ts = self.ts
        t = self.A.get(f"fire/{k}", ts)
        if t:
            w, h = ts * 0.95, ts * 1.42
            self.g.draw(t, x - w / 2, y - h, w, h)
        # Low ground bounce gives each hazard a footprint on the stone.
        self.g.draw(self.g.prims["glow"], x-ts*.72, y-ts*.14, ts*1.44, ts*.32,
                    color=(255, 111, 24), alpha=125, mode=ADD)
        glows.append(((x, y - ts * 0.35), ts * 0.95, (255, 120, 40), 0.55 + 0.1 * math.sin(now * 13 + c)))

    def _draw_torch(self, data, now, glows):
        r, c = data
        k = int(now * 18 + _h(r, c) % 16) % 16
        x, y = self.P(c + 0.5, r + 0.45)
        ts = self.ts
        t = self.A.get(f"torchfire/{k}", ts * 0.35)
        if t:
            w, h = ts * 0.34, ts * 0.5
            self.g.draw(t, x - w / 2, y - h, w, h)
        glows.append(((x, y - ts * 0.12), ts * 0.55, self.B["torch"], 0.6))

    def _draw_portal(self, ex, now, glows):
        g = self.g
        ts = self.ts
        open_ = self.world.exit_open_for(self.focus_pid)
        col = gfx.HUMAN if open_ else (150, 110, 230)
        cx, cy = ex[1] + 0.5, ex[0] + 0.5
        x, y = self.P(cx, cy)
        size = 1.5 * ts
        for name, spd, alpha in (("vortex", 40, 230 if open_ else 140), ("vortex2", -65, 170 if open_ else 90)):
            t = self.A.get(name, size)
            if t:
                g.draw(t, x - size * 0.42, y - size * 0.42, size * 0.84, size * 0.84, color=col,
                       alpha=alpha, mode=ADD, angle=now * spd)
        ring = self.A.get(f"{self.biome}/portal/ring", size)
        if ring:
            g.draw(ring, x - size / 2, y - size / 2, size, size)
        runes = self.A.get(f"{self.biome}/portal/runes", size)
        if runes:
            pulse = 0.6 + 0.4 * math.sin(now * 2.5)
            g.draw(runes, x - size / 2, y - size / 2, size, size, color=col, alpha=255 * pulse, mode=ADD)
        glows.append(((x, y), ts * 1.0, col, 0.55 if open_ else 0.3))
        if not open_:
            lk = self.A.get("padlock", ts * 0.4)
            bob = math.sin(now * 2) * ts * 0.05
            if lk:
                g.draw(lk, x - ts * 0.2, y - ts * 0.95 + bob, ts * 0.4, ts * 0.4)
        if random.random() < 0.3:
            a = random.uniform(0, math.tau)
            self.part(cx + math.cos(a) * 0.7, cy + math.sin(a) * 0.7, -math.cos(a) * 1.2, -math.sin(a) * 1.2,
                      0.55, "soft", 0.06, col, 0, 0, 0, ADD, False, 0.0, 230)

    def _draw_player(self, data, now, glows):
        p, pos = data
        g = self.g
        ts = self.ts
        self.fx.draw_burn(p.pid,pos,now,False)
        if self.fx.draw_actor(p,pos,now):
            return
        fac = self.facing[p.pid]
        pose = "side" if fac in ("left", "right") else fac
        flip = fac == "left"
        if p.pid==0 and flip:pose='left';flip=False
        moving = self.tw.active(("p", p.pid), now)
        frame = (1, 2, 3, 2)[int(now * 12) % 4] if moving else 0
        cx = pos[1] + 0.5
        bottom = pos[0] + 0.5 + 0.44
        w, h = 1.04, 1.46
        breathe = 0 if moving else math.sin(now * 3 + p.pid) * 0.012
        col = PLAYER_COL[p.pid]
        x, yb = self.P(cx, bottom)
        rx,ry,react_angle,react_height=self.fx.reaction(p.pid,now)
        x+=rx*ts;yb+=ry*ts
        if self.world.current == p.pid and not self.world.over and p.alive:
            k = (now * 1.3) % 1.0
            rw = ts * (0.62 + 0.4 * k)
            g.draw(g.prims["ring"], x - rw / 2, yb - ts * 0.08 - rw * 0.16, rw, rw * 0.32, color=col,
                   alpha=230 * (1 - k))
        tex = self.A.get(f"chars_{p.pid}_{pose}/{frame}", w * ts)
        if tex is None:
            return
        frozen = self.frozen_vis.get(p.pid) and p.alive
        tint = (255, 255, 255) if p.alive else (90, 90, 100)
        if frozen:
            tint = (150, 200, 255)
            breathe = 0
        hh = h * ts * (1 + breathe)*react_height
        g.draw(tex, x - w * ts / 2, yb - hh, w * ts, hh, color=tint, flip=flip, angle=react_angle, origin=(w*ts/2,hh))
        if frozen:
            ice = self.A.get("ice", ts)
            if ice:
                iw, ih = w * ts * 1.12, hh * 1.04
                progress=gfx.clamp((now-self.fx.freeze_times.get(p.pid,now-.35))/.28)
                g.draw(ice,x-iw/2,yb-ih+ts*.02,iw,ih,alpha=235*gfx.ease_out(progress))
            glows.append(((x, yb - hh * 0.5), ts * 0.7, (150, 210, 255), 0.35))
        hk = now - self.hit_time.get(p.pid, -9)
        if 0 <= hk < 0.35:
            g.draw(tex, x - w * ts / 2, yb - hh, w * ts, hh, color=(255, 60, 60),
                   alpha=255 * (1 - hk / 0.35), mode=ADD, flip=flip)

    # ---------------------------------------------------------------- tags
    def _tags(self, now):
        g = self.g
        if self.free_cam:
            return
        ts = self.ts
        mines = getattr(self.world, "mines", {})
        for p in self.world.players:
            if not p.alive or p.exited:
                continue
            pos = self.player_cell_f(p.pid, now)
            x, y = self.P(pos[1] + 0.5, pos[0] + 0.5 + 0.44 - 1.46 - 0.04)
            above = (round(pos[0]) - 1, round(pos[1]))
            if above in mines:
                # the explorer's head hides a mine planted just above: show it through, move the tag aside
                mx, my = self.P(above[1] + 0.5, above[0] + 0.55)
                sz = ts * 0.62
                t = self.A.get("mine", sz)
                if t:
                    g.draw(t, mx - sz / 2, my - sz / 2 - ts * 0.04, sz, sz, alpha=170)
                x += ts * 0.75
            col = PLAYER_COL[p.pid]
            label = roles.tag(p.pid)
            if self.frozen_vis.get(p.pid):
                label += f"  · FROZEN {max(1, getattr(p, 'frozen', 0))}"
            px = max(11 * g.u, 12 * g.u * max(self.zoom, 0.7))
            tw, th = g.text_size(label, px, True)
            pw, ph = tw + px * 1.1, th + px * 0.25
            g.rrect(x - pw / 2, y - ph, pw, ph, ph / 2, col, 240)
            g.text(label, x, y - ph / 2, px, (10, 14, 26), True, "center")
            if self.zoom < 0.85:
                k = (now * 1.2 + p.pid * 0.5) % 1.0
                rr = ts * (0.3 + 0.6 * k)
                cx, cy = self.P(pos[1] + 0.5, pos[0] + 0.5)
                g.ring(cx, cy, rr, col, 200 * (1 - k))

    def _offscreen_markers(self, now):
        """Edge arrows pointing at the AI and at the exit when they are off screen."""
        if self.free_cam:
            return
        g = self.g
        u = g.u
        box = (40 * u, 150 * u, g.W - 40 * u, g.H - 120 * u)
        fp = self.focus_pid
        me = self.world.players[fp]
        targets = []
        for p in self.world.players:
            if not p.alive or p.exited:
                continue
            x, y = self.S(self.player_cell_f(p.pid, now))
            if p.pid == fp and 0 <= x <= g.W and 0 <= y <= g.H:
                continue
            label = roles.tag(p.pid) if p.pid == fp else \
                f"{roles.tag(p.pid)}  {abs(p.r - me.r) + abs(p.c - me.c)} cells"
            targets.append((x, y, PLAYER_COL[p.pid], label))
        ex = self.world.exit_pos
        if ex:
            x, y = self.S(ex)
            open_ = self.world.exit_open_for(fp)
            col = gfx.HUMAN if open_ else (180, 150, 255)
            targets.append((x, y, col, f"EXIT  {abs(ex[0] - me.r) + abs(ex[1] - me.c)} cells" +
                            ("" if open_ else "  (locked)")))
        for i, (x, y, col, label) in enumerate(targets):
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                continue
            self._edge_marker(x, y, col, label, now, box, i)

    def _edge_marker(self, x, y, col, label, now, box, i):
        g = self.g
        u = g.u
        bx0, by0, bx1, by1 = box
        cx, cy = g.W / 2, (by0 + by1) / 2
        ang = math.atan2(y - cy, x - cx)
        bob = math.sin(now * 5 + i) * 4 * u
        ex = min(max(x, bx0), bx1)
        ey = min(max(y, by0), by1)
        # keep clear of the HUD panels (minimap, tool dock, event log)
        if ex > g.W - 300 * u and ey > g.H - 330 * u:
            ey = g.H - 340 * u
        elif abs(ex - g.W / 2) < 280 * u and ey > g.H - 170 * u:
            ey = g.H - 175 * u
        elif ex < 370 * u and ey > g.H - 200 * u:
            ey = g.H - 205 * u
        ex += math.cos(ang) * bob
        ey += math.sin(ang) * bob
        g.glow(ex, ey, 34 * u, col, 0.6)
        ar = 16 * u
        tip = (ex + math.cos(ang) * ar, ey + math.sin(ang) * ar)
        for da in (2.5, -2.5):
            g.line(ex + math.cos(ang + da) * ar * 0.8, ey + math.sin(ang + da) * ar * 0.8, tip[0], tip[1],
                   4 * u, col)
        px = 12 * u
        tw, th = g.text_size(label, px, True)
        pw, ph = tw + 16 * u, th + 4 * u
        lx = ex - math.cos(ang) * (pw / 2 + 18 * u)
        ly = ey - math.sin(ang) * 26 * u
        lx = min(max(lx, pw / 2 + 6 * u), g.W - pw / 2 - 6 * u)
        g.rrect(lx - pw / 2, ly - ph / 2, pw, ph, ph / 2, col, 240)
        g.text(label, lx, ly, px, (10, 14, 26), True, "center")

    # ------------------------------------------------------------ tactical cell readability
    def _draw_cells(self,now,r0,r1,c0,c1):
        self._cell_info=None
        if not self.show_grid:return
        g=self.g;ts=self.ts;u=g.u
        colors={'jungle':(199,218,202),'frost':(185,222,244),'ember':(225,190,153)}
        col=colors[self.biome];weight=max(1,round(u*.7))
        x0,y0=self.P(c0,r0);x1,y1=self.P(c1+1,r1+1)
        # One continuous line per row/column, under all objects. This gives every
        # cell an unambiguous edge without filling or flattening the materials.
        for c in range(c0,c1+2):
            x,_=self.P(c,0)
            g.rect(round(x),round(y0),weight,max(1,y1-y0),(3,12,14),94)
            g.rect(round(x)+weight,round(y0),weight,max(1,y1-y0),col,42)
        for r in range(r0,r1+2):
            _,y=self.P(0,r)
            g.rect(round(x0),round(y),max(1,x1-x0),weight,(3,12,14),94)
            g.rect(round(x0),round(y)+weight,max(1,x1-x0),weight,col,42)
        if self.free_cam or now<self.timeline_end or self.world.over:return
        mx,my=g.mouse();il,it,ir,ib=(v*u for v in getattr(self,'insets',(60,145,60,120)))
        if not (il<mx<g.W-ir and it<my<g.H-ib):return
        target=self.screen_to_cell(mx,my);w=self.world;fp=self.focus_pid;p=w.players[fp]
        if not w.inside(*target):return
        start=(p.r,p.c)
        key=(w.t,w.tool_used,p.r,p.c,p.gems,w.over,fp)
        if key!=self._route_key:
            from collections import deque
            distances={start:0};parents={start:None};q=deque([start])
            while q:
                r,c=q.popleft()
                for dr,dc in ((-1,0),(0,1),(1,0),(0,-1)):
                    cell=(r+dr,c+dc)
                    if cell not in distances and w.can_enter(fp,*cell):
                        distances[cell]=distances[r,c]+1;parents[cell]=(r,c);q.append(cell)
            self._route_key=key;self._route_dist=distances;self._route_parent=parents
        distance=self._route_dist.get(target)
        route=[];at=target
        if distance is not None:
            while at!=start:route.append(at);at=self._route_parent[at]
            route.reverse()
        beam_cells=set(w.lit())
        hazard=any(w.terrain[r][c]==FIRE or (r,c) in w.snakes or
                   ((r,c) in w.mines and w.mines[r,c]!=0) or (r,c) in beam_cells for r,c in route)
        accent=(226,175,92) if hazard else gfx.HUMAN
        tr,tc=target;x,y=self.P(tc,tr)
        g.rrect(x+2,y+2,ts-4,ts-4,max(2,ts*.045),accent,34)
        g.rrect(x+2,y+2,ts-4,ts-4,max(2,ts*.045),accent,210,filled=False,stroke=max(1,u))
        for i,(r,c) in enumerate(route,1):
            if i==len(route):continue
            xx,yy=self.P(c+.5,r+.5)
            if len(route)<=18 or i%5==0:
                size=max(9*u,min(12*u,ts*.24))
                g.disc(xx,yy,size*.72,(6,24,28),170)
                g.text(str(i),xx,yy,size,accent,True,'center',210)
            else:g.disc(xx,yy,max(1.2,ts*.035),accent,130)
        surface={FLOOR:'Open path',DIRT:'Diggable ground',WALL:'Solid rock',CRACKED:'Cracked wall',FIRE:'Fire',EXIT:'Exit'}.get(w.terrain[tr][tc],'Cell')
        label=f'{surface}  /  R{tr+1} C{tc+1}  /  '+('Blocked' if distance is None else f'{distance} move'+('' if distance==1 else 's'))
        if hazard:label+='  /  hazard on route'
        self._cell_info=(label,mx,my,accent)

    def _draw_cell_info(self):
        if not self._cell_info:return
        label,mx,my,color=self._cell_info;g=self.g;u=g.u;px=12*u
        tw,th=g.text_size(label,px,True);ww,hh=tw+22*u,th+14*u
        x=min(g.W-ww-12*u,max(12*u,mx+16*u));y=max(12*u,my-hh-14*u)
        g.rrect(x,y,ww,hh,5*u,(7,22,27),245)
        g.rrect(x,y,ww,hh,5*u,color,125,filled=False,stroke=max(1,u))
        g.text(label,x+11*u,y+hh/2,px,gfx.INK,True,'midleft')


    def _audio_scene(self):
        if self.audio is None or not hasattr(self.audio,'set_scene'):return
        if not getattr(self,'scene_audio_enabled',True):
            self.audio.set_scene(0,0);return
        p=self.world.players[0]
        fire=0.0
        for r,c in self._visible_fires:
            dist=math.hypot(r-p.r,c-p.c)
            fire=max(fire,math.exp(-dist/4.0))
        beam=0.0
        for sid,r,cells,d in (self.beam_segs or []):
            if cells:
                dist=min(abs(r-p.r)+abs(c-p.c) for r,c in cells)
                beam=max(beam,math.exp(-dist/3.0)*.65)
        self.audio.set_scene(fire,beam)
