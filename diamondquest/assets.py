"""Builds every texture at the resolution the screen needs, with a disk cache and LOD copies.

Painting happens in a worker thread (numpy releases the GIL for most of the work);
the main thread uploads finished surfaces to the GPU while the loading bar animates.
"""
import math
import os
import queue
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial as P

import pygame

from . import art_objects as O
from . import art_terrain as T
from . import paint
from . import art_director as D
from . import fx_art as FX
from .biomes import BIOMES, ORDER

ART_VERSION = 62
CACHE_ROOT = os.path.join(os.path.expanduser("~"), ".diamond_quest", "cache")


def tile_size_for(u):
    """Texture size per tile: matches the tile's on-screen size at zoom 1 (56 design units)."""
    ts = int(math.ceil(56 * u / 16.0)) * 16
    return max(128, min(384, ts))


def _portal(n, seed, B):
    ring, runes = D.portal_ring(n, seed, B)
    return {"ring": ring, "runes": runes}


def _jobs_common(ts):
    j = [
        ("gem_v", P(D.gem, int(ts * 0.8), "v")),
        ("gem_r", P(D.gem, int(ts * 0.8), "r")),
        ("flare", P(O.flare, int(ts * 0.6))),
        ("@fire", P(D.fire_frames, ts, 24)),
        ("@torchfire", P(D.fire_frames, max(48, ts // 2), 16, 8)),
        ("firepit_0", P(O.fire_pit, ts, 1)),
        ("firepit_1", P(O.fire_pit, ts, 2)),
        ("vortex", P(O.vortex, int(ts * 1.2))),
        ("vortex2", P(O.vortex, int(ts * 1.2), 5, -10.0)),
        ("padlock", P(O.padlock, int(ts * 0.5))),
        ("portrait_0", P(D.portrait, 384, 0)),
        ("keyart_0", P(D.key_art, 1024, 0)),
        ("keyart_1", P(D.key_art, 1024, 1)),
        ("portrait_1", P(D.portrait, 384, 1)),
        ("@burn_back", P(FX.burn_frames, 384, False)),
        ("@burn_front", P(FX.burn_frames, 384, True)),
        ("hammer", P(D.hammer, ts)),
        ("hookhead", P(D.hook_head, int(ts * 0.45))),
        ("chain", P(O.chain_link, int(ts * 0.22))),
        ("slash", P(O.slash_arc, int(ts * 1.6))),
        ("snowflake", P(O.snowflake, max(24, ts // 6))),
        ("leaf", P(O.leaf, max(24, ts // 5), 1, (110, 170, 60))),
        ("torch", P(D.torch, ts)),
        ("mine", P(D.mine, int(ts * 0.8))),
        ("ice", P(FX.ice_cage, ts)),
        ("@fx_explosion", P(FX.explosion_frames, min(384,ts*2))),
        ("@fx_beam", P(FX.beam_frames, 256)),
        ("fx_hook", P(FX.hook_head, ts)),
        ("fx_dust", P(FX.dust, ts)),
        ("fx_scorch", P(FX.ground_mark, ts*2, "scorch")),
        ("fx_fracture", P(FX.ground_mark, ts*2, "fracture")),
        ("@shadows", P(T.shadows, ts)),
    ]
    for kind in ('ice','stone','spark','venom'):
        j.append((f'fx_{kind}',P(FX.shard,96,kind)))
    for pid in (0, 1):
        for kind in ('hammer','push','hook'):
            j.append((f'@action_{pid}_{kind}_side',P(FX.action_frames,256,pid,kind)))
        for pose in ('up','down'):
            j.append((f'@action_{pid}_hammer_{pose}',P(FX.action_frames,256,pid,'hammer',pose)))
        for pose in ("down", "up", "side"):
            j.append((f"@chars_{pid}_{pose}", P(D.character_set, 384, pid, pose)))
    for k in range(3):
        j.append((f"smoke_{k}", P(O.smoke, int(ts * 0.7), k)))
    for k in range(2):
        j.append((f"bones_{k}", P(D.bones, ts, k + 1)))
    for nm in ("hammer", "push", "hook", "wait", "heart", "lock", "exit", "music", "sound", "eye",
               "camera", "mine", "statue"):
        j.append((f"icon_{nm}", P(D.icon, nm, 128)))
    return j


def _jobs_biome(ts, b):
    B = BIOMES[b]
    j = []
    j.append((f"@{b}/floor", P(D.floor_atlas, B, ts, 10, b)))
    j.append((f"@{b}/dirt", P(D.dirt_atlas, B, ts, 20, b)))
    j.append((f"@{b}/wall", P(D.wall_atlas, B, ts, 30, b)))
    for k in range(2):
        j.append((f"{b}/crackov_{k}", P(T.crack_overlay, ts, 40 + k)))
        j.append((f"{b}/crackglow_{k}", P(T.cracked_glow, ts, 40 + k, False)))
    for k in range(3):
        j.append((f"{b}/rock_{k}", P(D.boulder, ts, 50 + k, B, b)))
        j.append((f"{b}/debris_{k}", P(O.debris, max(24, ts // 5), k, B)))
    j.append((f"@{b}/snake", P(O.snake_parts, ts, B)))
    j.append((f"@{b}/snakecoil", P(D.snake_frames, ts, B, b)))
    j.append((f"@{b}/bite", P(FX.bite_frames, 256, b)))
    j.append((f"@{b}/stun", P(FX.stun_frames, 256, b)))
    j.append((f"@{b}/guardian", P(FX.guardian_views, int(ts*1.2), b)))
    j.append((f"{b}/statue", P(D.statue, int(ts * 1.1), B, b)))
    j.append((f"@{b}/portal", P(_portal, int(ts * 1.25), 7, B)))
    if b == "jungle":
        for k in range(3):
            j.append((f"{b}/vines_{k}", P(D.vines, ts, k, B)))
            j.append((f"{b}/fern_{k}", P(D.fern, ts, k + 3, B)))
        for k in range(2):
            j.append((f"{b}/moss_{k}", P(D.moss_patch, ts, k + 7, B)))
    elif b == "frost":
        for k in range(3):
            j.append((f"{b}/icicles_{k}", P(D.icicles, ts, k)))
        for k in range(2):
            j.append((f"{b}/snow_{k}", P(D.snow_drift, ts, k + 2)))
            j.append((f"{b}/crystal_{k}", P(D.crystal_cluster, ts, k + 4, B["crystal"])))
    else:
        for k in range(3):
            j.append((f"{b}/chains_{k}", P(D.chains, ts, k)))
            j.append((f"{b}/lava_{k}", P(T.floor_emissive, B, ts, 60 + k)))
    return j


def _run_job(fn):
    """Runs in a worker process: returns uint8 straight-alpha RGBA bytes for each image."""
    res = fn()

    def pack(img):
        surf = paint.to_surface(img)
        return (pygame.image.tobytes(surf, "RGBA") if hasattr(pygame.image, "tobytes")
                else pygame.image.tostring(surf, "RGBA"), surf.get_size())
    if isinstance(res, dict):
        return {("_".join(map(str, k)) if isinstance(k, tuple) else str(k)): pack(v)
                for k, v in res.items()}
    if isinstance(res, list):
        return {str(i): pack(v) for i, v in enumerate(res)}
    return pack(res)


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        try:
            _POOL = ProcessPoolExecutor(max_workers=max(1, min(12, (os.cpu_count() or 2) - 1)))
        except Exception:
            _POOL = False
    return _POOL


class Assets:
    def __init__(self, gpu):
        self.g = gpu
        self.ts = None
        self.tex = {}           # name -> [full, half, quarter]
        self.size = {}          # name -> (w, h) of the full-size texture
        self.loaded_biomes = set()
        self._q = queue.Queue()
        self._thread = None
        self.total = 0
        self.done = 0
        self.status = ""
        self.gen = 0

    # ------------------------------------------------------------ loading
    def start(self, u, biomes=("jungle",)):
        ts = tile_size_for(u)
        if ts != self.ts:
            self.tex.clear()
            self.size.clear()
            self.loaded_biomes.clear()
            self.ts = ts
            self.total = self.done = 0
            self.gen += 1                    # results from older batches are ignored
            jobs = _jobs_common(ts)
        else:
            jobs = []
        for b in biomes:
            if b not in self.loaded_biomes:
                jobs += _jobs_biome(ts, b)
                self.loaded_biomes.add(b)
        if not jobs:
            return
        self.total += len(jobs)
        self._thread = threading.Thread(target=self._work, args=(jobs, ts, self.gen), daemon=True)
        self._thread.start()

    def busy(self):
        return self.done < self.total

    def _cache_dir(self, ts):
        return os.path.join(CACHE_ROOT, f"v{ART_VERSION}", str(ts))

    def _work(self, jobs, ts, gen):
        put = lambda item: self._q.put((gen, item))  # noqa: E731
        cdir = self._cache_dir(ts)
        try:
            os.makedirs(cdir, exist_ok=True)
        except OSError:
            cdir = None
        todo = []
        for name, fn in jobs:
            hit = self._from_cache(name, cdir)
            if hit is not None:
                put(hit)
            else:
                todo.append((name, fn))
        if not todo:
            return
        pool = _pool() if len(todo) > 4 else None
        if pool:
            try:
                futs = {pool.submit(_run_job, fn): name for name, fn in todo}
                for f in as_completed(futs):
                    name = futs[f]
                    try:
                        put(self._store(name, f.result(), cdir))
                    except Exception as ex:
                        put(("error", name, str(ex)))
                return
            except Exception:
                pass                      # fall back to painting in this thread
        for name, fn in todo:
            try:
                put(self._store(name, _run_job(fn), cdir))
            except Exception as ex:
                put(("error", name, str(ex)))

    @staticmethod
    def _fname(name):
        return name.lstrip("@").replace("/", "__")

    def _from_cache(self, name, cdir):
        if not cdir:
            return None
        key = name.lstrip("@")
        fname = self._fname(name)
        try:
            if name.startswith("@"):
                idx = os.path.join(cdir, fname + ".idx")
                if not os.path.exists(idx):
                    return None
                with open(idx) as f:
                    keys = [ln.strip() for ln in f if ln.strip()]
                return ("multi", key, {k: pygame.image.load(os.path.join(cdir, f"{fname}__{k}.png"))
                                       for k in keys})
            path = os.path.join(cdir, fname + ".png")
            if os.path.exists(path):
                return ("one", key, pygame.image.load(path))
        except Exception:
            return None
        return None

    def _store(self, name, packed, cdir):
        key = name.lstrip("@")
        fname = self._fname(name)

        def surf(p):
            data, size = p
            fb = pygame.image.frombytes if hasattr(pygame.image, "frombytes") else pygame.image.fromstring
            return fb(data, size, "RGBA")
        if name.startswith("@"):
            surfs = {k: surf(v) for k, v in packed.items()}
            if cdir:
                try:
                    for k, sf in surfs.items():
                        pygame.image.save(sf, os.path.join(cdir, f"{fname}__{k}.png"))
                    with open(os.path.join(cdir, fname + ".idx"), "w") as f:
                        f.write("\n".join(surfs))
                except Exception:
                    pass
            return ("multi", key, surfs)
        sf = surf(packed)
        if cdir:
            try:
                pygame.image.save(sf, os.path.join(cdir, fname + ".png"))
            except Exception:
                pass
        return ("one", key, sf)

    def pump(self, budget_ms=12):
        """Upload finished surfaces to the GPU (main thread). Returns progress 0..1."""
        t0 = pygame.time.get_ticks()
        while pygame.time.get_ticks() - t0 < budget_ms:
            try:
                gen, item = self._q.get_nowait()
            except queue.Empty:
                break
            if gen != self.gen:
                continue
            kind, key = item[0], item[1]
            if kind == "one":
                self._add(key, item[2])
            elif kind == "multi":
                for k, s in item[2].items():
                    self._add(f"{key}/{k}", s)
            self.status = key
            self.done += 1
        return self.progress()

    def progress(self):
        return 1.0 if not self.total else self.done / self.total

    def _add(self, name, surf):
        if name.startswith(('chars_0_side/','action_0_')) and '_side/' in name:
            self._add(name.replace('_side/','_left/'),D.face_left(surf))
        w, h = surf.get_size()
        lods = [self.g.texture(surf)]
        if max(w, h) >= 96:
            half = pygame.transform.smoothscale(surf, (max(1, w // 2), max(1, h // 2)))
            lods.append(self.g.texture(half))
            if max(w, h) >= 192:
                q = pygame.transform.smoothscale(half, (max(1, w // 4), max(1, h // 4)))
                lods.append(self.g.texture(q))
        self.tex[name] = lods
        self.size[name] = (w, h)

    # ------------------------------------------------------------ lookup
    def get(self, name, px=None):
        """Texture for name; px = on-screen width, picks the right level of detail."""
        lods = self.tex.get(name)
        if lods is None:
            return None
        if px is None or len(lods) == 1:
            return lods[0]
        w = self.size[name][0]
        # Characters retain at least two texture samples per screen pixel.
        # The previous general-purpose thresholds selected a 29px avatar LOD.
        if name.startswith(('chars_','action_','portrait_','keyart_')):
            if px * 2 <= w / 4 and len(lods)>2:return lods[2]
            if px * 2 <= w / 2 and len(lods)>1:return lods[1]
            return lods[0]
        if px < w * 0.33 and len(lods) > 2:
            return lods[2]
        if px < w * 0.66:
            return lods[1]
        return lods[0]

    def lodmap(self, level):
        """name -> texture at a fixed level of detail (cached until new textures arrive)."""
        key = (level, len(self.tex))
        cache = getattr(self, "_lodcache", None)
        if cache is None or cache[0] != key:
            m = {n: lods[min(level, len(lods) - 1)] for n, lods in self.tex.items()}
            self._lodcache = (key, m)
            return m
        return cache[1]

    def has(self, name):
        return name in self.tex


def all_biomes():
    return list(ORDER)
