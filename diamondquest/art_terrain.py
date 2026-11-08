"""Terrain textures: floors, soil, walls (top and front faces), cracked walls, shadows, decorations."""
import math

import numpy as np

from .paint import (F, Canvas, blur, cut_tiles, voronoi_grid, grid, jitter_points, lit_surface, mixc, opaque, rgb,
                    smoothstep, tile_fbm, voronoi, Noise)

K = 4          # atlas size: each surface is painted as one seamless 4x4-tile sheet
FACE_SEED = 777


def _brick_field(n, rows, cols, rng, stagger=0.5, gap=0.035):
    """Brick layout over an n x n tile. Returns edge distance (px), brick id, local u, v."""
    x, y = grid(n)
    bh = n / rows
    bw = n / cols
    row = np.floor(y / bh)
    off = (row % 2) * stagger
    bx = x / bw + off
    col = np.floor(bx) % cols
    u = bx - np.floor(bx)
    v = y / bh - row
    ex = np.minimum(u, 1 - u) * bw
    ey = np.minimum(v, 1 - v) * bh
    edge = np.minimum(ex, ey)
    bid = (row * cols + col).astype(np.int64)
    return edge, bid, u, v, bw, bh


def floor(B, n, seed, biome):
    rng = np.random.default_rng(seed)
    a, b, grout = (rgb(c) for c in B["floor"])
    if biome == "frost":
        detail = tile_fbm(n, 4, seed, 5)
        fine = tile_fbm(n, 16, seed + 3, 3)
        pts = jitter_points(n, n, 3, 3, rng, 0.9)
        f1, f2, idx = voronoi(n, n, pts)
        crack = 1 - smoothstep(0, n * 0.012, f2 - f1)
        height = detail * n * 0.06 + fine * n * 0.012 - crack * n * 0.01
        alb = mixc(a, b, detail[..., None] * 1.2 - 0.1)
        alb = mixc(alb, grout, crack[..., None] * 0.55)
        col = lit_surface(alb, height, 1.0, 0.55, 0.6, spec=0.5, power=40)
        spark = (Noise(seed + 9, 256).at(*(c * 0.9 for c in grid(n))) > 0.93).astype(F)
        col += spark[..., None] * 0.25
        return opaque(col)
    pts = jitter_points(n, n, 2, 2, rng, 0.75)
    f1, f2, idx = voronoi(n, n, pts)
    edge = (f2 - f1) * 0.5
    gap = n * (0.018 if biome == "jungle" else 0.024)
    stone = smoothstep(gap * 0.6, gap * 1.8, edge)
    tint = rng.uniform(-1, 1, len(pts)).astype(F)[idx]
    detail = tile_fbm(n, 4, seed, 5)
    fine = tile_fbm(n, 24, seed + 1, 3)
    bevel = smoothstep(0, n * 0.09, edge)
    height = (stone * (0.5 + 0.5 * bevel) + detail * 0.25 + fine * 0.08) * n * 0.05
    alb = mixc(a, b, (detail[..., None] - 0.25) * 1.4) * (1 + 0.09 * tint[..., None])
    alb = alb * (0.9 + 0.2 * fine[..., None])
    alb = mixc(grout, alb, stone[..., None])
    if biome == "jungle":
        m1, m2 = rgb(B["moss"][0]), rgb(B["moss"][1])
        mn = tile_fbm(n, 6, seed + 5, 4)
        moss = np.clip((1 - stone) * 1.3 + (mn - 0.62) * 3, 0, 1) * smoothstep(0.45, 0.7, mn)
        alb = mixc(alb, mixc(m1, m2, fine[..., None]), moss[..., None])
        height = height + moss * n * 0.01
    col = lit_surface(alb, height, 1.0, 0.5, 0.65, spec=0.12 if biome == "ember" else 0.06,
                      power=20, ao=0.6)
    return opaque(col)


def floor_emissive(B, n, seed):
    """Ember floor: glowing lava veins (drawn additively after lighting)."""
    rng = np.random.default_rng(seed)
    pts = jitter_points(n, n, 2, 2, rng, 0.9)
    f1, f2, _ = voronoi(n, n, pts)
    vein = 1 - smoothstep(0, n * 0.02, (f2 - f1) * 0.5)
    mask = tile_fbm(n, 3, seed + 2, 3)
    vein *= smoothstep(0.5, 0.65, mask)
    glow = blur(vein, n * 0.02) * 1.6
    inten = np.clip(vein + glow * 0.6, 0, 1)
    col = np.stack([np.ones_like(inten), 0.35 + 0.5 * vein, 0.1 * vein], -1)
    return np.concatenate([col * inten[..., None], inten[..., None]], -1).astype(F)


def dirt(B, n, seed, biome, front=False):
    """Diggable soil: soft clods, a few half-buried stones, roots / frost / embers by biome."""
    rng = np.random.default_rng(seed)
    dk, lt, deep = (rgb(c) for c in B["dirt"])
    x, y = grid(n)
    base = tile_fbm(n, 2, seed, 5, 0.55)
    mid = tile_fbm(n, 5, seed + 2, 3)
    fine = tile_fbm(n, 16, seed + 1, 2)
    lumps = np.clip((mid - 0.5) * 2.2, -1, 1)
    height = (base * 0.5 + lumps * 0.25 + fine * 0.1) * n * 0.07
    alb = mixc(dk, lt, (base[..., None] - 0.25) * 1.5)
    alb = alb * (0.94 + 0.1 * fine[..., None])
    alb = mixc(deep, alb, smoothstep(-0.9, 0.2, lumps)[..., None] * 0.35 + 0.65)
    c = Canvas(n)
    for _ in range(int(2 + rng.integers(0, 3))):
        px, py = rng.uniform(0, n, 2)
        r = rng.uniform(0.03, 0.055) * n
        d = c.d_ellipse(px, py, r * rng.uniform(1, 1.4), r)
        k = np.clip(-d / r, 0, 1)
        peb = c.cov(d)
        height = height + peb * np.sqrt(k) * r * 0.7
        pc = mixc(rgb(B["rock"][0]), rgb(B["rock"][1]), rng.random()) * 0.85
        alb = mixc(alb, np.broadcast_to(mixc(alb.mean((0, 1)), pc, 0.7), alb.shape), peb[..., None])
    if biome == "jungle":
        for _ in range(2):
            ax, ay = rng.uniform(0, n, 2)
            ang = rng.uniform(0, math.tau)
            pts_ = []
            for k in range(8):
                pts_.append((ax, ay))
                ang += rng.uniform(-0.6, 0.6)
                ax += math.cos(ang) * n * 0.07
                ay += math.sin(ang) * n * 0.07
            for (p0, p1) in zip(pts_, pts_[1:]):
                d = c.d_capsule(p0[0], p0[1], p1[0], p1[1], n * 0.011)
                cv = c.cov(d)
                height = height + cv * n * 0.01
                alb = mixc(alb, np.broadcast_to(rgb("#6f4d2e"), alb.shape), cv[..., None] * 0.8)
    if biome == "frost":
        fr = smoothstep(0.55, 0.75, tile_fbm(n, 4, seed + 4, 4))
        alb = mixc(alb, np.broadcast_to(rgb("#c9dcee"), alb.shape), fr[..., None] * 0.55)
        height = height + fr * n * 0.012
    col = lit_surface(alb, height, 1.0, 0.5, 0.7, spec=0.2 if biome == "frost" else 0.03,
                      power=16, ao=0.6)
    if biome == "ember":
        sp = (Noise(seed + 11, 256).at(x * 0.6, y * 0.6) > 0.95).astype(F)
        col = col + sp[..., None] * np.array([0.8, 0.3, 0.04], F)
    if front:
        col = _front_face(col, n, seed, dk * 0.7, deep, B, biome, depth=0.24, kind="dirt")
    return opaque(col)


def _front_face(top, n, seed, face_a, face_b, B, biome, depth=0.46, kind="wall"):
    """Replace the lower part of a tile with a vertical face lit from above."""
    x, y = grid(n)
    split = n * (1 - depth)
    rng = np.random.default_rng(seed + 99)
    t = np.clip((y - split) / (n - split), 0, 1)
    face = ((y >= split)).astype(F)
    if kind == "wall":
        rows = 2
        bh = (n - split) / rows
        row = np.floor((y - split) / bh)
        off = (row % 2) * 0.5
        cols = 2
        bx = x / (n / cols) + off
        u = bx - np.floor(bx)
        v = (y - split) / bh - row
        e = np.minimum(np.minimum(u, 1 - u) * n / cols, np.minimum(v, 1 - v) * bh)
        chip = tile_fbm(n, 12, seed + 3, 3)
        brick = smoothstep(n * 0.01, n * 0.045, e - (chip - 0.5) * n * 0.02)
        nz = tile_fbm(n, 6, seed + 7, 4)
        bid = (row * 3 + np.floor(bx)).astype(np.int64) % 7
        tint = np.array([0.0, 0.08, -0.06, 0.12, -0.1, 0.04, -0.02], F)[bid]
        alb = mixc(face_b, face_a * 1.35, brick[..., None] * (0.8 + 0.3 * nz[..., None]))
        alb = alb * (1 + tint[..., None]) * (0.9 + 0.2 * chip[..., None])
        hgt = brick * n * 0.035 + nz * n * 0.012 + chip * n * 0.006
    else:
        nz = tile_fbm(n, 6, seed + 7, 4)
        strata = 0.5 + 0.5 * np.sin((y - split) / n * 60 + nz * 6)
        alb = mixc(face_b, face_a, (strata * 0.6 + nz * 0.5)[..., None])
        hgt = strata * n * 0.01 + nz * n * 0.02
    sh = lit_surface(alb, hgt, 1.0, 0.62, 0.55, wrap=False)
    sh *= (1.15 - 0.55 * t)[..., None]                # lit from above, darker toward the floor
    lip = np.exp(-((y - split) / (n * 0.015)) ** 2)    # bright top edge
    out = top * (1 - face[..., None]) + sh * face[..., None]
    out = out + lip[..., None] * 0.22
    under = smoothstep(n * (1 - 0.06), n, y) * face
    out *= (1 - 0.5 * under)[..., None]
    if biome == "frost" and kind == "wall":
        out = out + (face * (0.08 + 0.1 * tile_fbm(n, 10, seed, 3)))[..., None] * \
            np.array([0.5, 0.75, 1.0], F)
    del rng
    return out


def wall(B, n, seed, biome, front=False, cracked=False):
    """Solid rock mass seen from above (dark, rough slabs); front=True adds a lit brick face."""
    rng = np.random.default_rng(seed)
    a, b, mortar = (rgb(c) for c in B["wall"])
    nz = tile_fbm(n, 4, seed, 5)
    fine = tile_fbm(n, 20, seed + 1, 3)
    pts = jitter_points(n, n, 2, 2, rng, 0.9)
    f1, f2, idx = voronoi(n, n, pts)
    edge = (f2 - f1) * 0.5
    slab = smoothstep(n * 0.01, n * 0.06, edge)
    tint = rng.uniform(-1, 1, len(pts)).astype(F)[idx]
    if biome == "frost":
        facet = rng.uniform(-1, 1, (len(pts), 2)).astype(F)
        x, y = grid(n)
        height = (facet[idx, 0] * x + facet[idx, 1] * y) * 0.08 * slab + nz * n * 0.04
        alb = mixc(a, b, nz[..., None]) * (0.75 + 0.12 * tint[..., None])
        alb = mixc(mortar, alb, slab[..., None])
        col = lit_surface(alb, height, 1.0, 0.42, 0.55, spec=0.35, power=40, wrap=True)
    else:
        height = (slab * 0.7 + nz * 0.35 + fine * 0.1) * n * 0.05
        alb = mixc(a, b, (nz[..., None] - 0.2) * 1.2) * (0.62 + 0.1 * tint[..., None])
        alb = alb * (0.85 + 0.3 * fine[..., None])
        alb = mixc(mortar * 0.8, alb, slab[..., None])
        if biome == "jungle":
            m1, m2 = rgb(B["moss"][0]), rgb(B["moss"][1])
            mn = tile_fbm(n, 5, seed + 5, 5)
            moss = smoothstep(0.5, 0.7, mn + (1 - slab) * 0.15)
            alb = mixc(alb, mixc(m1, m2, fine[..., None]) * 0.75, moss[..., None])
            height = height + moss * n * 0.02
        if biome == "ember":
            soot = smoothstep(0.4, 0.8, tile_fbm(n, 5, seed + 6, 4))
            alb = alb * (1 - 0.35 * soot[..., None])
        col = lit_surface(alb, height, 1.0, 0.42, 0.6, spec=0.06, power=18, ao=0.8)
    if front:
        fa, fb = (rgb(c) for c in B["wall_front"])
        col = _front_face(col, n, seed, fa, fb, B, biome)
    out = opaque(col)
    if cracked:
        out = _cracks(out, n, seed, front)
    return out


def _cracks(img, n, seed, front):
    rng = np.random.default_rng(seed + 50)
    c = Canvas(n)
    d_all = np.full((n, n), 1e9, F)
    for k in range(3):
        x, y = n * rng.uniform(0.3, 0.7), n * 0.08
        ang = math.pi / 2 + rng.uniform(-0.5, 0.5)
        w = n * 0.028
        while y < n * 0.92 and 0 < x < n:
            nx = x + math.cos(ang) * n * 0.09
            ny = y + math.sin(ang) * n * 0.09
            d_all = np.minimum(d_all, c.d_capsule(x, y, nx, ny, w, w * 0.8))
            if rng.random() < 0.3:
                bx = nx + math.cos(ang + rng.choice([-1, 1]) * 1.1) * n * 0.12
                by = ny + math.sin(ang + 0.6) * n * 0.08
                d_all = np.minimum(d_all, c.d_capsule(nx, ny, bx, by, w * 0.6, w * 0.2))
            x, y = nx, ny
            ang += rng.uniform(-0.6, 0.6)
            w *= 0.9
        if k == 0 and rng.random() < 0.6:
            break
    dark = c.cov(d_all - n * 0.02)
    core = c.cov(d_all)
    out = img.copy()
    out[..., :3] *= (1 - 0.75 * dark)[..., None]
    out[..., :3] += core[..., None] * np.array([0.55, 0.22, 0.04], F)
    return out


def cracked_glow(n, seed, front):
    """Emissive layer for cracked walls (pulses in the bloom pass)."""
    rng = np.random.default_rng(seed + 50)
    c = Canvas(n)
    d_all = np.full((n, n), 1e9, F)
    for k in range(3):
        x, y = n * rng.uniform(0.3, 0.7), n * 0.08
        ang = math.pi / 2 + rng.uniform(-0.5, 0.5)
        w = n * 0.028
        while y < n * 0.92 and 0 < x < n:
            nx = x + math.cos(ang) * n * 0.09
            ny = y + math.sin(ang) * n * 0.09
            d_all = np.minimum(d_all, c.d_capsule(x, y, nx, ny, w, w * 0.8))
            if rng.random() < 0.3:
                bx = nx + math.cos(ang + rng.choice([-1, 1]) * 1.1) * n * 0.12
                by = ny + math.sin(ang + 0.6) * n * 0.08
                d_all = np.minimum(d_all, c.d_capsule(nx, ny, bx, by, w * 0.6, w * 0.2))
            x, y = nx, ny
            ang += rng.uniform(-0.6, 0.6)
            w *= 0.9
        if k == 0 and rng.random() < 0.6:
            break
    core = c.cov(d_all)
    halo = blur(core, n * 0.03, wrap=False) * 1.5
    inten = np.clip(core + halo, 0, 1)
    col = np.stack([np.ones_like(inten), 0.45 + 0.4 * core, 0.12 * core], -1)
    return np.concatenate([col * inten[..., None], inten[..., None]], -1).astype(F)


# ------------------------------------------------------------------ shadows (overlays)
def shadows(n):
    x, y = grid(n)
    out = {}

    def mk(a):
        a = np.clip(a, 0, 1).astype(F)
        return np.concatenate([np.zeros((n, n, 3), F), a[..., None]], -1)
    out["n"] = mk(0.7 * (1 - smoothstep(0, n * 0.38, y)) ** 1.6)
    out["w"] = mk(0.45 * (1 - smoothstep(0, n * 0.22, x)) ** 1.6)
    out["e"] = mk(0.3 * (1 - smoothstep(0, n * 0.16, n - x)) ** 1.6)
    out["s"] = mk(0.25 * (1 - smoothstep(0, n * 0.12, n - y)) ** 1.6)
    out["side_l"] = mk(0.5 * (1 - smoothstep(0, n * 0.1, x)))
    out["side_r"] = mk(0.6 * (1 - smoothstep(0, n * 0.1, n - x)))
    hl = np.exp(-(y / (n * 0.03)) ** 2) * 0.9
    out["rim"] = np.concatenate([np.ones((n, n, 3), F) * hl[..., None] * 0.35,
                                 hl[..., None] * 0.35], -1).astype(F)
    return out


# ------------------------------------------------------------------ decorations
def vines(n, seed, B):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    m1, m2 = rgb(B["moss"][0]), rgb(B["moss"][1])
    for _ in range(rng.integers(2, 4)):
        x = rng.uniform(0.1, 0.9) * n
        y = n * 0.55
        length = rng.uniform(0.25, 0.48) * n
        steps = 10
        prev = (x, y)
        for k in range(steps):
            t = (k + 1) / steps
            nx = x + math.sin(t * 5 + seed) * n * 0.03
            ny = y + length * t
            d = c.d_capsule(prev[0], prev[1], nx, ny, n * 0.012)
            c.shape(d, mixc(m1 * 0.7, m2 * 0.7, t), bevel=n * 0.01, ambient=0.6)
            if k % 2 == 0:
                side = 1 if k % 4 == 0 else -1
                lx, ly = nx + side * n * 0.035, ny - n * 0.01
                dl = c.d_ellipse(lx, ly, n * 0.03, n * 0.016)
                c.shape(dl, mixc(m1, m2, rng.random()), bevel=n * 0.012, ambient=0.6, gloss=0.2)
            prev = (nx, ny)
    return c.px


def icicles(n, seed):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    x = n * 0.05
    while x < n * 0.95:
        w = rng.uniform(0.03, 0.06) * n
        ln = rng.uniform(0.08, 0.3) * n
        top = n * 0.58
        d = c.d_poly([(x - w, top), (x + w, top), (x + w * 0.1, top + ln)])
        col = np.broadcast_to(np.array([0.8, 0.92, 1.0], F), (n, n, 3)).copy()
        c.shape(d, col, bevel=w * 0.8, gloss=1.0, power=40, ambient=0.7, opacity=0.85)
        x += w * 2 + rng.uniform(0, 0.08) * n
    return c.px


def chains(n, seed):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    for _ in range(rng.integers(1, 3)):
        x = rng.uniform(0.2, 0.8) * n
        y = n * 0.58
        links = rng.integers(3, 6)
        for k in range(links):
            ly = y + k * n * 0.06
            if k % 2 == 0:
                d = np.abs(c.d_ellipse(x, ly, n * 0.022, n * 0.04)) - n * 0.008
            else:
                d = np.abs(c.d_ellipse(x, ly, n * 0.01, n * 0.04)) - n * 0.008
            c.shape(d, (0.42, 0.4, 0.42), bevel=n * 0.008, gloss=0.6, ambient=0.5)
    return c.px


def moss_patch(n, seed, B):
    m1, m2 = rgb(B["moss"][0]), rgb(B["moss"][1])
    nz = tile_fbm(n, 4, seed, 5)
    x, y = grid(n)
    rad = np.hypot(x / n - 0.5, y / n - 0.5)
    a = smoothstep(0.55, 0.7, nz + (0.5 - rad) * 0.6)
    fine = tile_fbm(n, 24, seed + 1, 2)
    col = lit_surface(mixc(m1, m2, fine[..., None]), (a + fine * 0.3) * n * 0.03, wrap=False,
                      ambient=0.55, diffuse=0.6)
    return np.concatenate([col * a[..., None], a[..., None]], -1).astype(F)


def fern(n, seed, B):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    m1, m2 = rgb(B["moss"][0]), rgb(B["moss"][1])
    cx, cy = n * rng.uniform(0.3, 0.7), n * rng.uniform(0.6, 0.8)
    for k in range(rng.integers(5, 8)):
        ang = -math.pi / 2 + rng.uniform(-1.3, 1.3)
        ln = rng.uniform(0.2, 0.36) * n
        for j in range(7):
            t = (j + 1) / 7
            px = cx + math.cos(ang) * ln * t
            py = cy + math.sin(ang) * ln * t + (t * t) * n * 0.08
            s = (1 - t) * n * 0.035 + n * 0.008
            d = c.d_ellipse(px, py, s, s * 0.55)
            c.shape(d, mixc(m1, m2, t * 0.8 + rng.random() * 0.2), bevel=s * 0.6, ambient=0.55,
                    gloss=0.15)
    return c.px


def snow_drift(n, seed):
    nz = tile_fbm(n, 4, seed, 4)
    x, y = grid(n)
    a = smoothstep(0.35, 0.55, nz + (0.3 - y / n) * 0.9)
    col = lit_surface(np.broadcast_to(np.array([0.92, 0.96, 1.0], F), (n, n, 3)).copy(),
                      a * n * 0.05 + nz * n * 0.01, wrap=False, ambient=0.65, diffuse=0.5,
                      spec=0.3)
    return np.concatenate([col * a[..., None], a[..., None]], -1).astype(F)


def bones(n, seed):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    bone = (0.88, 0.84, 0.72)
    # skull
    sx, sy = n * rng.uniform(0.35, 0.65), n * rng.uniform(0.45, 0.65)
    s = n * 0.12
    c.shape(c.d_ellipse(sx + s * 0.1, sy + s * 0.9, s * 1.0, s * 0.3), (0, 0, 0), opacity=0.35,
            soft=s * 0.4)
    d = np.minimum(c.d_circle(sx, sy, s), c.d_box(sx, sy + s * 0.75, s * 0.55, s * 0.35, s * 0.15))
    c.shape(d, bone, bevel=s * 0.6, gloss=0.25, ambient=0.5)
    for ex in (-0.38, 0.38):
        c.shape(c.d_ellipse(sx + ex * s, sy + s * 0.1, s * 0.25, s * 0.3), (0.12, 0.1, 0.08),
                bevel=s * 0.15, ambient=0.6)
    c.shape(c.d_poly([(sx - s * 0.1, sy + s * 0.52), (sx + s * 0.1, sy + s * 0.52), (sx, sy + s * 0.35)]),
            (0.15, 0.12, 0.1))
    # a bone
    bx, by = n * rng.uniform(0.15, 0.3), n * rng.uniform(0.7, 0.85)
    ex, ey = bx + n * 0.25, by - n * rng.uniform(-0.08, 0.08)
    r = n * 0.018
    d = np.minimum(c.d_capsule(bx, by, ex, ey, r),
                   np.minimum(np.minimum(c.d_circle(bx, by - r, r * 1.3), c.d_circle(bx, by + r, r * 1.3)),
                              np.minimum(c.d_circle(ex, ey - r, r * 1.3), c.d_circle(ex, ey + r, r * 1.3))))
    c.shape(d, bone, bevel=r, gloss=0.2, ambient=0.5)
    return c.px


def crystal_cluster(n, seed, color):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    base = np.array(color, F) / 255
    cx, cy = n * 0.5, n * 0.62
    for k in range(5):
        ang = -math.pi / 2 + rng.uniform(-0.9, 0.9)
        ln = rng.uniform(0.18, 0.4) * n
        w = rng.uniform(0.04, 0.07) * n
        tx, ty = cx + math.cos(ang) * ln, cy + math.sin(ang) * ln
        px, py = -math.sin(ang) * w, math.cos(ang) * w
        bx, by = cx + rng.uniform(-0.06, 0.06) * n, cy
        pts = [(bx + px, by + py), (tx + px * 0.6, ty + py * 0.6), (tx + math.cos(ang) * w * 1.5,
                                                                      ty + math.sin(ang) * w * 1.5),
               (tx - px * 0.6, ty - py * 0.6), (bx - px, by - py)]
        d = c.d_poly(pts)
        col = mixc(base, np.ones(3, F), 0.25 + 0.3 * rng.random())
        c.shape(d, col, bevel=w * 0.7, gloss=1.0, power=30, ambient=0.75, opacity=0.92,
                outline=n * 0.004, outline_col=base * 0.5)
    return c.px


def lava_crack(n, seed):
    return floor_emissive(None, n, seed)


def torch(n):
    """Wall-mounted torch bracket (flame added at runtime)."""
    c = Canvas(n)
    cx = n * 0.5
    # bracket plate
    c.shape(c.d_box(cx, n * 0.66, n * 0.07, n * 0.1, n * 0.02), (0.32, 0.3, 0.3), bevel=n * 0.02,
            gloss=0.5)
    # handle
    c.shape(c.d_capsule(cx, n * 0.72, cx, n * 0.48, n * 0.035, n * 0.05), (0.42, 0.27, 0.15),
            bevel=n * 0.03, gloss=0.2)
    # cup
    c.shape(c.d_poly([(cx - n * 0.09, n * 0.44), (cx + n * 0.09, n * 0.44), (cx + n * 0.05, n * 0.52),
                      (cx - n * 0.05, n * 0.52)]), (0.35, 0.33, 0.34), bevel=n * 0.02, gloss=0.7)
    return c.px


# ===================================================================== seamless atlases
def floor_atlas(B, n, seed, biome, k=K):
    """One continuous k x k tile sheet of floor; returns k*k tiles (row-major)."""
    N = n * k
    rng = np.random.default_rng(seed)
    a, b, grout = (rgb(c) for c in B["floor"])
    if biome == "frost":
        detail = tile_fbm(N, 4 * k, seed, 5)
        fine = tile_fbm(N, 16 * k, seed + 3, 3)
        f1, f2, idx, pts = voronoi_grid(N, N, 3 * k, 3 * k, rng, 0.9)
        crack = 1 - smoothstep(0, n * 0.012, f2 - f1)
        height = detail * n * 0.06 + fine * n * 0.012 - crack * n * 0.01
        alb = mixc(a, b, detail[..., None] * 1.2 - 0.1)
        alb = mixc(alb, grout, crack[..., None] * 0.55)
        col = lit_surface(alb, height, 1.0, 0.55, 0.6, spec=0.5, power=40)
        x, y = grid(N)
        spark = (Noise(seed + 9, 256).at(x * 0.9, y * 0.9) > 0.93).astype(F)
        col += spark[..., None] * 0.25
        return cut_tiles(opaque(col), n, k)
    f1, f2, idx, pts = voronoi_grid(N, N, 2 * k, 2 * k, rng, 0.75)
    edge = (f2 - f1) * 0.5
    gap = n * (0.018 if biome == "jungle" else 0.024)
    stone = smoothstep(gap * 0.6, gap * 1.8, edge)
    tint = rng.uniform(-1, 1, len(pts)).astype(F)[idx]
    detail = tile_fbm(N, 4 * k, seed, 5)
    fine = tile_fbm(N, 24 * k, seed + 1, 3)
    bevel = smoothstep(0, n * 0.09, edge)
    height = (stone * (0.5 + 0.5 * bevel) + detail * 0.25 + fine * 0.08) * n * 0.05
    alb = mixc(a, b, (detail[..., None] - 0.25) * 1.4) * (1 + 0.09 * tint[..., None])
    alb = alb * (0.9 + 0.2 * fine[..., None])
    alb = mixc(grout, alb, stone[..., None])
    if biome == "jungle":
        m1, m2 = rgb(B["moss"][0]), rgb(B["moss"][1])
        mn = tile_fbm(N, 6 * k, seed + 5, 4)
        moss = np.clip((1 - stone) * 1.3 + (mn - 0.62) * 3, 0, 1) * smoothstep(0.45, 0.7, mn)
        alb = mixc(alb, mixc(m1, m2, fine[..., None]), moss[..., None])
        height = height + moss * n * 0.01
    col = lit_surface(alb, height, 1.0, 0.5, 0.65, spec=0.12 if biome == "ember" else 0.06, power=20, ao=0.6)
    return cut_tiles(opaque(col), n, k)


def dirt_atlas(B, n, seed, biome, k=K):
    """Seamless soil sheet; returns {"t<i>": top tile, "f<i>": tile with a soil front face}."""
    N = n * k
    rng = np.random.default_rng(seed)
    dk, lt, deep = (rgb(c) for c in B["dirt"])
    x, y = grid(N)
    base = tile_fbm(N, 2 * k, seed, 5, 0.55)
    mid = tile_fbm(N, 5 * k, seed + 2, 3)
    fine = tile_fbm(N, 16 * k, seed + 1, 2)
    lumps = np.clip((mid - 0.5) * 2.2, -1, 1)
    height = (base * 0.5 + lumps * 0.25 + fine * 0.1) * n * 0.07
    alb = mixc(dk, lt, (base[..., None] - 0.25) * 1.5)
    alb = alb * (0.94 + 0.1 * fine[..., None])
    alb = mixc(deep, alb, smoothstep(-0.9, 0.2, lumps)[..., None] * 0.35 + 0.65)
    c = Canvas(N)
    mean = alb.mean((0, 1))
    for _ in range(int((2.5 + rng.random() * 1.5) * k * k * (0.4 if biome == "frost" else 1.0))):
        r = rng.uniform(0.03, 0.055) * n
        px, py = rng.uniform(r * 2, N - r * 2, 2)
        x0, x1 = int(max(0, px - r * 2)), int(min(N, px + r * 2))
        y0, y1 = int(max(0, py - r * 2)), int(min(N, py + r * 2))
        sub = (slice(y0, y1), slice(x0, x1))
        d = np.hypot((c.X[sub] - px) / (r * rng.uniform(1, 1.4)), (c.Y[sub] - py) / r) * r - r
        kk = np.clip(-d / r, 0, 1)
        peb = c.cov(d)
        height[sub] = height[sub] + peb * np.sqrt(kk) * r * 0.7
        pc = mixc(rgb(B["rock"][0]), rgb(B["rock"][1]), rng.random()) * 0.85
        alb[sub] = mixc(alb[sub], np.broadcast_to(mixc(mean, pc, 0.7), alb[sub].shape), peb[..., None])
    if biome == "jungle":
        for _ in range(2 * k * k):
            ax, ay = rng.uniform(n * 0.7, N - n * 0.7, 2)
            ang = rng.uniform(0, math.tau)
            pts_ = []
            for _s in range(8):
                pts_.append((ax, ay))
                ang += rng.uniform(-0.6, 0.6)
                ax += math.cos(ang) * n * 0.07
                ay += math.sin(ang) * n * 0.07
            xs = [p[0] for p in pts_]
            ys = [p[1] for p in pts_]
            pad = n * 0.03
            x0, x1 = int(max(0, min(xs) - pad)), int(min(N, max(xs) + pad))
            y0, y1 = int(max(0, min(ys) - pad)), int(min(N, max(ys) + pad))
            if x1 <= x0 or y1 <= y0:
                continue
            sub = (slice(y0, y1), slice(x0, x1))
            X, Y = c.X[sub], c.Y[sub]
            dmin = np.full(X.shape, 1e9, F)
            for (p0, p1) in zip(pts_, pts_[1:]):
                vx, vy = p1[0] - p0[0], p1[1] - p0[1]
                ll = vx * vx + vy * vy or 1e-6
                t = np.clip(((X - p0[0]) * vx + (Y - p0[1]) * vy) / ll, 0, 1)
                dmin = np.minimum(dmin, np.hypot(X - p0[0] - vx * t, Y - p0[1] - vy * t) - n * 0.011)
            cv = c.cov(dmin)
            height[sub] = height[sub] + cv * n * 0.01
            alb[sub] = mixc(alb[sub], np.broadcast_to(rgb("#6f4d2e"), alb[sub].shape), cv[..., None] * 0.8)
    if biome == "frost":
        # compacted snow drifts over frozen earth: soft, low-frequency, gently lit
        drift = tile_fbm(N, 2 * k, seed + 4, 5, 0.45)
        ripples = tile_fbm(N, 7 * k, seed + 8, 3)
        snow = smoothstep(0.3, 0.62, drift)
        earth = mixc(dk, lt, base[..., None])
        sc = mixc(rgb("#b7cadf"), rgb("#e3eef8"), ripples[..., None])
        alb = mixc(earth, sc, (0.35 + 0.6 * snow)[..., None])
        height = (drift * 0.7 + ripples * 0.15) * n * 0.09
    col = lit_surface(alb, height, 1.0, 0.5, 0.7, spec=0.2 if biome == "frost" else 0.03, power=16, ao=0.6)
    if biome == "ember":
        sp = (Noise(seed + 11, 256).at(x * 0.6, y * 0.6) > 0.95).astype(F)
        col = col + sp[..., None] * np.array([0.8, 0.3, 0.04], F)
    tops = cut_tiles(opaque(col), n, k)
    out = {}
    for i, t in enumerate(tops):
        out[f"t{i}"] = t
        out[f"f{i}"] = opaque(_front_face(t[..., :3], n, FACE_SEED, dk * 0.7, deep, B, biome, depth=0.24,
                                          kind="dirt"))
    return out


def wall_atlas(B, n, seed, biome, k=K):
    """Seamless rock-mass sheet; returns {"t<i>": top tile, "f<i>": tile with a lit brick face}."""
    N = n * k
    rng = np.random.default_rng(seed)
    a, b, mortar = (rgb(c) for c in B["wall"])
    nz = tile_fbm(N, 4 * k, seed, 5)
    fine = tile_fbm(N, 20 * k, seed + 1, 3)
    f1, f2, idx, pts = voronoi_grid(N, N, 2 * k, 2 * k, rng, 0.9)
    edge = (f2 - f1) * 0.5
    slab = smoothstep(n * 0.01, n * 0.06, edge)
    tint = rng.uniform(-1, 1, len(pts)).astype(F)[idx]
    if biome == "frost":
        facet = rng.uniform(-1, 1, (len(pts), 2)).astype(F)
        x, y = grid(N)
        dx = ((x - pts[idx, 0] + N / 2) % N) - N / 2
        dy = ((y - pts[idx, 1] + N / 2) % N) - N / 2
        height = (facet[idx, 0] * dx + facet[idx, 1] * dy) * 0.08 * slab + nz * n * 0.04
        alb = mixc(a, b, nz[..., None]) * (0.75 + 0.12 * tint[..., None])
        alb = mixc(mortar, alb, slab[..., None])
        col = lit_surface(alb, height, 1.0, 0.42, 0.55, spec=0.35, power=40, wrap=True)
    else:
        height = (slab * 0.7 + nz * 0.35 + fine * 0.1) * n * 0.05
        alb = mixc(a, b, (nz[..., None] - 0.2) * 1.2) * (0.62 + 0.1 * tint[..., None])
        alb = alb * (0.85 + 0.3 * fine[..., None])
        alb = mixc(mortar * 0.8, alb, slab[..., None])
        if biome == "jungle":
            m1, m2 = rgb(B["moss"][0]), rgb(B["moss"][1])
            mn = tile_fbm(N, 5 * k, seed + 5, 5)
            moss = smoothstep(0.5, 0.7, mn + (1 - slab) * 0.15)
            alb = mixc(alb, mixc(m1, m2, fine[..., None]) * 0.75, moss[..., None])
            height = height + moss * n * 0.02
        if biome == "ember":
            soot = smoothstep(0.4, 0.8, tile_fbm(N, 5 * k, seed + 6, 4))
            alb = alb * (1 - 0.35 * soot[..., None])
        col = lit_surface(alb, height, 1.0, 0.42, 0.6, spec=0.06, power=18, ao=0.8)
    tops = cut_tiles(opaque(col), n, k)
    fa, fb = (rgb(c) for c in B["wall_front"])
    out = {}
    for i, t in enumerate(tops):
        out[f"t{i}"] = t
        out[f"f{i}"] = opaque(_front_face(t[..., :3], n, FACE_SEED, fa, fb, B, biome))
    return out


def _crack_dist(n, seed):
    rng = np.random.default_rng(seed + 50)
    c = Canvas(n)
    d_all = np.full((n, n), 1e9, F)
    for k in range(3):
        x, y = n * rng.uniform(0.3, 0.7), n * 0.08
        ang = math.pi / 2 + rng.uniform(-0.5, 0.5)
        w = n * 0.028
        while y < n * 0.92 and 0 < x < n:
            nx = x + math.cos(ang) * n * 0.09
            ny = y + math.sin(ang) * n * 0.09
            d_all = np.minimum(d_all, c.d_capsule(x, y, nx, ny, w, w * 0.8))
            if rng.random() < 0.3:
                bx = nx + math.cos(ang + rng.choice([-1, 1]) * 1.1) * n * 0.12
                by = ny + math.sin(ang + 0.6) * n * 0.08
                d_all = np.minimum(d_all, c.d_capsule(nx, ny, bx, by, w * 0.6, w * 0.2))
            x, y = nx, ny
            ang += rng.uniform(-0.6, 0.6)
            w *= 0.9
        if k == 0 and rng.random() < 0.6:
            break
    return c, d_all


def crack_overlay(n, seed):
    """Dark fissures with a hot core, drawn over a wall tile to mark it breakable."""
    c, d_all = _crack_dist(n, seed)
    dark = c.cov(d_all - n * 0.02) * 0.8
    core = c.cov(d_all)
    col = np.zeros((n, n, 3), F) * dark[..., None]
    a = np.clip(dark + core * 0.2, 0, 1)
    col = col + core[..., None] * np.array([0.55, 0.22, 0.04], F)
    return np.concatenate([col, a[..., None]], -1).astype(F)
