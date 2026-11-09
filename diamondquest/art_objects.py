"""Objects, characters, effects and icons, painted procedurally at any resolution."""
import math

import numpy as np

from .paint import (F, Canvas, Noise, grid, lit_surface, mixc, ramp, rgb, smoothstep,
                    tile_fbm, voronoi, jitter_points)

HUMAN_ACCENT = (59, 232, 255)
AI_ACCENT = (255, 190, 64)


def _c(*v):
    return np.array(v, F) / 255.0


# ===================================================================== rocks
def boulder(n, seed, B, biome):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    cx, cy, r = n * 0.5, n * 0.52, n * 0.4
    ang = np.arctan2(c.Y - cy, c.X - cx)
    nz = Noise(seed, 16)
    wob = nz.fbm(np.cos(ang) * 1.5 + 3, np.sin(ang) * 1.5 + 3, 3)
    dist = np.hypot(c.X - cx, (c.Y - cy) * 1.06)
    d = dist - r * (0.9 + 0.2 * wob)
    k = np.clip(-d / (r * 0.9), 0, 1)
    dome = np.sqrt(np.clip(1 - (1 - k) ** 2, 0, 1)) * r * 0.9
    bump = Noise(seed + 1, 64).fbm(c.X / n * 6, c.Y / n * 6, 5)
    pits = np.zeros_like(bump)
    for _ in range(5):
        px, py = cx + rng.uniform(-0.5, 0.5) * r, cy + rng.uniform(-0.5, 0.5) * r
        rr = rng.uniform(0.06, 0.13) * r
        pd = np.hypot(c.X - px, c.Y - py)
        pits += np.clip(1 - pd / rr, 0, 1) ** 2
    hgt = dome + bump * r * 0.25 - pits * r * 0.12
    a, b = rgb(B["rock"][0]), rgb(B["rock"][1])
    alb = mixc(a, b, bump[..., None] * 1.3 - 0.2)
    alb = alb * (0.9 + 0.2 * Noise(seed + 2, 64).fbm(c.X / n * 18, c.Y / n * 18, 2))[..., None]
    if biome == "jungle":
        moss = smoothstep(0.55, 0.7, Noise(seed + 3, 32).fbm(c.X / n * 5, c.Y / n * 5, 4)
                          + (cy - c.Y) / n * 0.6)
        alb = mixc(alb, rgb("#5f8a30"), moss[..., None])
    elif biome == "frost":
        snow = smoothstep(0.45, 0.6, (cy - c.Y) / r * 0.5 + 0.35 +
                          Noise(seed + 3, 32).fbm(c.X / n * 6, c.Y / n * 6, 3) * 0.3)
        alb = mixc(alb, rgb("#f2f8ff"), snow[..., None])
    col = lit_surface(alb, hgt, 1.0, 0.38, 0.8, spec=0.18, power=20, wrap=False)
    occl = np.clip(0.55 + 0.45 * (k ** 0.35), 0, 1)
    col *= occl[..., None]
    # cracks
    if rng.random() < 0.8:
        x0, y0 = cx + rng.uniform(-0.3, 0.3) * r, cy - r * 0.6
        dd = np.full_like(d, 1e9)
        for _ in range(4):
            x1, y1 = x0 + rng.uniform(-0.25, 0.25) * r, y0 + r * 0.3
            dd = np.minimum(dd, c.d_capsule(x0, y0, x1, y1, n * 0.006))
            x0, y0 = x1, y1
        col *= (1 - 0.6 * c.cov(dd) * (k > 0.1))[..., None]
    if biome == "ember":
        glow = c.cov(d + r * 0.0) * smoothstep(0.72, 0.8, Noise(seed + 9, 32).fbm(c.X / n * 9, c.Y / n * 9, 3))
        col += glow[..., None] * np.array([0.6, 0.18, 0.02], F)
    c.over(np.clip(col, 0, 1), c.cov(d))
    return c.px


# ===================================================================== gems
def gem(n, kind):
    """Side view brilliant-cut diamond with individually lit facets."""
    if kind == "r":
        base, deep, light = _c(255, 40, 80), _c(120, 0, 30), _c(255, 190, 205)
    else:
        base, deep, light = _c(160, 90, 255), _c(60, 20, 140), _c(225, 205, 255)
    c = Canvas(n)
    P = lambda x, y: (x * n, y * n)  # noqa: E731
    gy, ty, cy_ = 0.40, 0.17, 0.90
    G = [P(x, gy) for x in (0.06, 0.2, 0.35, 0.5, 0.65, 0.8, 0.94)]
    T = [P(x, ty) for x in (0.27, 0.385, 0.5, 0.615, 0.73)]
    C = P(0.5, cy_)
    facets = [
        ([G[0], T[0], G[1]], 0.95, "crown"),
        ([T[0], T[1], G[2], G[1]], 0.75, "crown"),
        ([T[1], T[2], G[3], G[2]], 1.1, "crown"),
        ([T[2], T[3], G[4], G[3]], 0.65, "crown"),
        ([T[3], T[4], G[5], G[4]], 0.5, "crown"),
        ([T[4], G[6], G[5]], 0.35, "crown"),
    ]
    mids = [((G[i][0] + G[i + 1][0]) / 2, gy * n) for i in range(6)]
    pav_val = [0.85, 0.45, 0.7, 0.3, 0.55, 0.2]
    for i in range(6):
        facets.append(([G[i], mids[i], C], pav_val[i], "pav"))
        facets.append(([mids[i], G[i + 1], C], pav_val[(i + 3) % 6] * 0.85, "pav"))
    # table (thin bright band on top)
    tbl = [P(0.27, ty), P(0.31, ty - 0.035), P(0.69, ty - 0.035), P(0.73, ty)]
    sil = c.d_poly([G[0], T[0], P(0.31, ty - 0.035), P(0.69, ty - 0.035), T[4], G[6], C])
    # soft glow behind
    glow = np.clip(1 - np.hypot(c.X / n - 0.5, c.Y / n - 0.5) / 0.55, 0, 1) ** 2
    c.over(base, glow * 0.0)
    for pts, v, part in facets:
        d = c.d_poly(pts)
        cxp = sum(p[0] for p in pts) / len(pts)
        cyp = sum(p[1] for p in pts) / len(pts)
        # gradient across the facet (toward the light: top-left)
        g = np.clip(((c.X - cxp) * -0.6 + (c.Y - cyp) * -0.8) / (n * 0.25) * 0.5 + 0.5, 0, 1)
        if part == "crown":
            col = mixc(base, light, (v * 0.55 + g[..., None] * 0.35))
        else:
            col = mixc(deep, base, (v * 0.8 + g[..., None] * 0.3))
            # internal reflection streaks
            stripes = 0.5 + 0.5 * np.sin((c.X - c.Y * 0.6) / n * 40 + v * 9)
            col = col + (stripes[..., None] ** 8) * light * 0.25
        c.over(np.clip(col, 0, 1.2), c.cov(d))
    c.over(mixc(light, np.ones(3, F), 0.6), c.cov(c.d_poly(tbl)))
    # facet edges: thin bright lines
    lines = np.full((n, n), 1e9, F)
    for pts, _, _ in facets:
        for a_, b_ in zip(pts, pts[1:] + pts[:1]):
            lines = np.minimum(lines, c.d_capsule(a_[0], a_[1], b_[0], b_[1], n * 0.004))
    edge_a = c.cov(lines) * c.cov(sil + n * 0.004)
    c.over(mixc(light, np.ones(3, F), 0.5), edge_a * 0.55)
    # outline rim
    rim = c.cov(np.abs(sil) - n * 0.008)
    c.over(deep * 0.6, rim * 0.9)
    # big specular streak on the crown
    hs = c.d_poly([P(0.36, 0.2), P(0.45, 0.2), P(0.31, 0.37), P(0.22, 0.37)])
    c.over(np.ones(3, F), c.cov(hs) * 0.55)
    c.over(np.ones(3, F), c.cov(c.d_circle(0.42 * n, 0.24 * n, 0.025 * n)) * 0.9)
    return c.px


def flare(n):
    """Four-point star glint for gem sparkles (additive)."""
    x, y = grid(n)
    u, v = (x / n - 0.5) * 2, (y / n - 0.5) * 2
    r = np.hypot(u, v)
    rays = np.exp(-np.abs(u) * 40) * np.clip(1 - np.abs(v), 0, 1) ** 3 + \
        np.exp(-np.abs(v) * 40) * np.clip(1 - np.abs(u), 0, 1) ** 3
    diag = (np.exp(-np.abs(u - v) * 30) + np.exp(-np.abs(u + v) * 30)) * np.clip(1 - r, 0, 1) ** 4 * 0.4
    core = np.exp(-(r / 0.12) ** 2)
    a = np.clip(rays + diag + core, 0, 1)
    return np.concatenate([np.ones((n, n, 3), F) * a[..., None], a[..., None]], -1).astype(F)


# ===================================================================== fire
def fire_frames(n, frames=24, seed=3, small=False):
    w, h = n, int(n * 1.5)
    x, y = grid(w, h)
    u = (x / w - 0.5) * 2
    v = 1 - y / h                       # 0 bottom .. 1 top
    nz = Noise(seed, 4)
    nz2 = Noise(seed + 1, 4)
    out = []
    stops = [(0.0, (60, 0, 0)), (0.18, (150, 20, 5)), (0.38, (240, 90, 10)),
             (0.6, (255, 170, 40)), (0.82, (255, 230, 140)), (1.0, (255, 255, 235))]
    for k in range(frames):
        t = k / frames * 4.0
        n1 = nz.fbm(x / w * 3, y / h * 4 + t, 4)
        n2 = nz2.fbm(x / w * 6, y / h * 6 + t * 1.5, 3)
        inten = np.zeros_like(u)
        for (ox, width, height, ph) in ((0.0, 0.62, 0.95, 0), (-0.3, 0.34, 0.62, 1.3), (0.32, 0.36, 0.7, 2.6)):
            vv = v / height
            sway = (n1 - 0.5) * 0.7 * vv + math.sin(k / frames * math.tau + ph) * 0.05 * vv
            ww = width * np.clip(1 - vv, 0, 1) ** 0.65 * (1 + 0.25 * (1 - vv))
            uu = (u - ox - sway) / np.maximum(ww, 1e-3)
            shape = np.clip(1 - np.abs(uu), 0, 1) * (vv < 1) * (v > -0.02)
            f = np.clip(shape * 1.6 - (n2 * 0.9) * vv - vv * 0.55, 0, 1)
            inten = np.maximum(inten, f)
        base_glow = np.exp(-((u / 0.6) ** 2 + ((v - 0.03) / 0.05) ** 2))
        inten = np.clip(inten + base_glow * 0.5, 0, 1)
        col = ramp(inten, stops)
        a = smoothstep(0.04, 0.32, inten)
        out.append(np.concatenate([col * a[..., None], a[..., None]], -1).astype(F))
    return out


def fire_pit(n, seed):
    """Charred ground under a fire tile."""
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    nz = tile_fbm(n, 6, seed, 4)
    x, y = grid(n)
    rad = np.hypot(x / n - 0.5, (y / n - 0.6) * 1.3)
    a = smoothstep(0.5, 0.25, rad + (nz - 0.5) * 0.15)
    col = mixc(_c(30, 18, 14), _c(70, 30, 18), nz[..., None])
    c.over(col, a * 0.85)
    for _ in range(9):
        px, py = n * rng.uniform(0.25, 0.75), n * rng.uniform(0.55, 0.78)
        rr = n * rng.uniform(0.03, 0.06)
        d = c.d_ellipse(px, py, rr * 1.3, rr)
        c.shape(d, _c(40, 32, 30), bevel=rr * 0.8, ambient=0.4, gloss=0.2)
        c.over(_c(255, 110, 30), c.cov(c.d_ellipse(px, py + rr * 0.3, rr * 0.7, rr * 0.35)) * 0.8)
    return c.px


# ===================================================================== snake
def snake_parts(n, B):
    body, pat, belly = rgb(B["snake"][0]), rgb(B["snake"][1]), rgb(B["snake"][2])
    parts = {}
    # body segment: round, scaled, central blotch
    s = int(n * 0.34)
    c = Canvas(s)
    cx = cy = s / 2
    d = c.d_circle(cx, cy, s * 0.46)
    pts = jitter_points(s, s, 6, 6, np.random.default_rng(4), 0.3)
    f1, f2, _ = voronoi(s, s, pts)
    scales = smoothstep(0, s * 0.05, (f2 - f1) * 0.5)
    blot = np.clip(1 - np.hypot(c.X - cx, c.Y - cy) / (s * 0.24), 0, 1)
    blot = smoothstep(0.2, 0.35, blot)
    col = mixc(body, pat, blot[..., None]) * (0.75 + 0.25 * scales[..., None])
    c.shape(d, col, bevel=s * 0.42, gloss=0.55, power=30, ambient=0.45, diffuse=0.75,
            outline=s * 0.03, outline_col=belly * 0.5)
    parts["seg"] = c.px
    # head (pointing right)
    hw, hh = int(n * 0.52), int(n * 0.4)
    c = Canvas(hw, hh)
    cx, cy = hw * 0.45, hh * 0.5
    d = c.d_poly([(hw * 0.06, cy - hh * 0.26), (hw * 0.55, cy - hh * 0.38), (hw * 0.9, cy - hh * 0.16),
                  (hw * 0.96, cy), (hw * 0.9, cy + hh * 0.16), (hw * 0.55, cy + hh * 0.38),
                  (hw * 0.06, cy + hh * 0.26)])
    d = d - hh * 0.06
    pts = jitter_points(hw, hh, 7, 5, np.random.default_rng(6), 0.3)
    f1, f2, _ = voronoi(hw, hh, pts, tile=False)
    scales = smoothstep(0, hh * 0.05, (f2 - f1) * 0.5)
    stripe = smoothstep(0.1, 0.0, np.abs(c.Y - cy) / hh - 0.05) * (c.X < hw * 0.75)
    col = mixc(body, pat, stripe[..., None] * 0.8) * (0.78 + 0.22 * scales[..., None])
    c.shape(d, col, bevel=hh * 0.3, gloss=0.6, power=30, ambient=0.45, outline=hh * 0.04,
            outline_col=belly * 0.5)
    for ey in (cy - hh * 0.2, cy + hh * 0.2):
        ex = hw * 0.62
        c.shape(c.d_ellipse(ex, ey, hh * 0.11, hh * 0.09), _c(250, 220, 60), bevel=hh * 0.06,
                gloss=1.0, ambient=0.8, outline=hh * 0.02, outline_col=(0, 0, 0))
        c.over(_c(10, 10, 10), c.cov(c.d_ellipse(ex + hh * 0.01, ey, hh * 0.025, hh * 0.08)))
        c.over(np.ones(3, F), c.cov(c.d_circle(ex - hh * 0.03, ey - hh * 0.03, hh * 0.022)))
    for ny in (cy - hh * 0.07, cy + hh * 0.07):
        c.over(_c(10, 10, 10), c.cov(c.d_circle(hw * 0.88, ny, hh * 0.02)) * 0.8)
    parts["head"] = c.px
    # tongue
    tw, th = int(n * 0.24), int(n * 0.1)
    c = Canvas(tw, th)
    cy = th / 2
    d = c.d_capsule(0, cy, tw * 0.6, cy, th * 0.08)
    d = np.minimum(d, c.d_capsule(tw * 0.6, cy, tw * 0.95, cy - th * 0.3, th * 0.07, th * 0.03))
    d = np.minimum(d, c.d_capsule(tw * 0.6, cy, tw * 0.95, cy + th * 0.3, th * 0.07, th * 0.03))
    c.shape(d, _c(230, 40, 80), bevel=th * 0.06, gloss=0.6)
    parts["tongue"] = c.px
    return parts


# ===================================================================== portal
def portal_ring(n, seed, B):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    cx = cy = n / 2
    R, r = n * 0.47, n * 0.33
    dist = np.hypot(c.X - cx, c.Y - cy)
    d = np.maximum(dist - R, r - dist)
    ang = np.arctan2(c.Y - cy, c.X - cx)
    seg = np.abs(np.sin(ang * 6))
    blocks = smoothstep(0.02, 0.12, seg)
    nz = Noise(seed, 32).fbm(c.X / n * 8, c.Y / n * 8, 4)
    stone = mixc(rgb(B["wall_front"][0]) * 0.9, rgb(B["wall"][1]) * 1.2, nz[..., None])
    stone = stone * (0.7 + 0.3 * blocks[..., None])
    c.shape(d, stone, bevel=(R - r) * 0.45, gloss=0.15, ambient=0.5, outline=n * 0.006,
            outline_col=(0.05, 0.04, 0.03))
    # runes carved into the ring
    rd = np.full_like(d, 1e9)
    for k in range(12):
        a0 = k / 12 * math.tau + math.pi / 12
        px, py = cx + math.cos(a0) * (R + r) / 2, cy + math.sin(a0) * (R + r) / 2
        s = (R - r) * 0.22
        kind = rng.integers(0, 4)
        if kind == 0:
            rd = np.minimum(rd, np.abs(c.d_circle(px, py, s)) - s * 0.18)
        elif kind == 1:
            rd = np.minimum(rd, c.d_capsule(px - s, py - s, px + s, py + s, s * 0.18))
            rd = np.minimum(rd, c.d_capsule(px - s, py + s, px + s, py - s, s * 0.18))
        elif kind == 2:
            rd = np.minimum(rd, c.d_capsule(px, py - s, px, py + s, s * 0.18))
            rd = np.minimum(rd, c.d_capsule(px - s, py, px + s * 0.6, py - s * 0.6, s * 0.18))
        else:
            rd = np.minimum(rd, np.abs(c.d_poly([(px, py - s), (px + s, py + s * 0.8),
                                                (px - s, py + s * 0.8)])) - s * 0.16)
    c.over(_c(20, 16, 14), c.cov(rd) * 0.85)
    runes = np.concatenate([np.ones((n, n, 3), F) * c.cov(rd)[..., None], c.cov(rd)[..., None]], -1)
    return c.px, runes.astype(F)


def vortex(n, arms=3, tight=7.0):
    x, y = grid(n)
    u, v = x / n - 0.5, y / n - 0.5
    r = np.hypot(u, v) * 2
    a = np.arctan2(v, u)
    spiral = 0.5 + 0.5 * np.sin(a * arms + np.log(r + 1e-3) * tight)
    core = np.exp(-(r / 0.25) ** 2)
    fall = np.clip(1 - r, 0, 1) ** 1.2
    inten = np.clip((spiral ** 2 * 0.8 + core) * fall, 0, 1)
    return np.concatenate([np.ones((n, n, 3), F) * inten[..., None], inten[..., None]], -1).astype(F)


def padlock(n):
    c = Canvas(n)
    cx = n / 2
    sh = np.abs(c.d_capsule(cx - n * 0.18, n * 0.5, cx - n * 0.18, n * 0.3, 0) + 0) * 0
    shack = np.abs(c.d_box(cx, n * 0.32, n * 0.18, n * 0.2, n * 0.18)) - n * 0.05
    shack = np.maximum(shack, c.Y - n * 0.5)
    c.shape(shack, (0.75, 0.75, 0.8), bevel=n * 0.04, gloss=0.9, ambient=0.5)
    body = c.d_box(cx, n * 0.64, n * 0.28, n * 0.22, n * 0.06)
    c.shape(body, (0.95, 0.75, 0.3), bevel=n * 0.06, gloss=0.8, ambient=0.5,
            outline=n * 0.012, outline_col=(0.3, 0.2, 0.05))
    kh = np.minimum(c.d_circle(cx, n * 0.6, n * 0.05), c.d_box(cx, n * 0.7, n * 0.02, n * 0.06))
    c.over(_c(40, 25, 5), c.cov(kh))
    del sh
    return c.px


# ===================================================================== characters
SKIN = _c(240, 196, 160)


def _person(n, pid, pose, frame):
    """Draw one character frame. pose: 'down' | 'up' | 'side' (facing right)."""
    W, H = int(n * 0.92), int(n * 1.28)
    c = Canvas(W, H)
    ph = frame / 4 * math.tau
    walking = frame > 0
    sw = math.sin(ph) if walking else 0.0
    bob = -abs(math.sin(ph)) * H * 0.015 if walking else 0.0
    cx = W * 0.5
    acc = _c(*HUMAN_ACCENT) if pid == 0 else _c(*AI_ACCENT)

    def sh(d, col, bev, **kw):
        c.shape(d, col, bevel=bev, ambient=kw.pop("ambient", 0.5), diffuse=kw.pop("diffuse", 0.75),
                outline=kw.pop("outline", H * 0.008), outline_col=kw.pop("outline_col", (0.06, 0.05, 0.07)),
                **kw)

    if pid == 0:
        jacket = mixc(acc, _c(20, 70, 95), 0.55)
        pants = _c(170, 150, 110)
        boots = _c(95, 60, 35)
        helmet = _c(225, 210, 170)
    else:
        jacket = _c(170, 120, 50)        # brass armour
        pants = _c(70, 74, 86)           # steel legs
        boots = _c(48, 50, 60)
        helmet = _c(80, 86, 100)
    hip_y = H * 0.7 + bob
    foot_y = H * 0.93
    # ---- legs
    if pose == "side":
        for k, s in ((0, sw), (1, -sw)):
            fx = cx + s * W * 0.12
            col = pants * (0.85 if k else 1.0)
            sh(c.d_capsule(cx, hip_y, fx, foot_y - H * 0.05, W * 0.075, W * 0.065), col, W * 0.06)
            sh(c.d_ellipse(fx + W * 0.04, foot_y - H * 0.02, W * 0.1, H * 0.035), boots, W * 0.05,
               gloss=0.3)
    else:
        for side, lift in ((-1, max(0, sw)), (1, max(0, -sw))):
            fx = cx + side * W * 0.11
            fy = foot_y - lift * H * 0.04
            sh(c.d_capsule(fx, hip_y, fx, fy - H * 0.04, W * 0.08, W * 0.07), pants, W * 0.06)
            if pid == 1:
                sh(c.d_circle(fx, (hip_y + fy) / 2, W * 0.05), _c(200, 150, 60), W * 0.04, gloss=0.8)
            sh(c.d_ellipse(fx, fy - H * 0.015, W * 0.09, H * 0.035), boots, W * 0.05, gloss=0.3)
    # ---- backpack (behind body unless facing up)
    body_top = H * 0.42 + bob
    if pose == "side":
        bx = cx - W * 0.2
        sh(c.d_box(bx, H * 0.55 + bob, W * 0.13, H * 0.13, W * 0.05),
           _c(120, 80, 45) if pid == 0 else _c(60, 64, 76), W * 0.06, gloss=0.2)
        if pid == 1:
            for k in range(3):
                c.over(_c(30, 30, 36), c.cov(c.d_box(bx - W * 0.05, H * 0.5 + k * H * 0.04 + bob,
                                                     W * 0.06, H * 0.008, H * 0.004)))
    # ---- torso
    if pose == "side":
        torso = c.d_box(cx, H * 0.56 + bob, W * 0.15, H * 0.15, W * 0.08)
    else:
        torso = c.d_box(cx, H * 0.56 + bob, W * 0.25, H * 0.15, W * 0.1)
    sh(torso, jacket, W * 0.12, gloss=0.5 if pid == 1 else 0.15)
    if pid == 0:
        if pose != "side":
            c.over(_c(90, 60, 30), c.cov(c.d_box(cx, H * 0.66 + bob, W * 0.25, H * 0.022)))
            sh(c.d_box(cx, H * 0.66 + bob, W * 0.04, H * 0.022, H * 0.006), _c(240, 200, 80), W * 0.02,
               gloss=1.0)
        if pose == "down":
            for sx in (-1, 1):
                c.over(_c(80, 52, 30), c.cov(c.d_capsule(cx + sx * W * 0.13, body_top + H * 0.02,
                                                         cx + sx * W * 0.1, H * 0.64 + bob, W * 0.025)))
            # scarf
            sh(c.d_box(cx, body_top + H * 0.03, W * 0.17, H * 0.03, H * 0.02), acc, W * 0.03)
            sh(c.d_capsule(cx + W * 0.08, body_top + H * 0.04, cx + W * 0.12, body_top + H * 0.12,
                           W * 0.03), acc * 0.85, W * 0.02)
    else:
        if pose in ("down", "side"):
            core_x = cx + (W * 0.05 if pose == "side" else 0)
            sh(c.d_circle(core_x, H * 0.55 + bob, W * 0.075), _c(255, 120, 40), W * 0.05, gloss=1.0,
               ambient=1.0)
            c.over(_c(255, 240, 200), c.cov(c.d_circle(core_x, H * 0.55 + bob, W * 0.03)))
        for rx in (-0.18, 0.18) if pose != "side" else (0.1,):
            for ry in (0.47, 0.65):
                sh(c.d_circle(cx + rx * W, H * ry + bob, W * 0.018), _c(230, 200, 120), W * 0.015,
                   gloss=1.0, outline=0)
        if pose == "up":
            for k in range(3):
                sh(c.d_box(cx, H * 0.5 + k * H * 0.05 + bob, W * 0.16, H * 0.012, H * 0.006),
                   _c(50, 52, 60), H * 0.008, outline=0)
    # ---- backpack seen from behind
    if pose == "up":
        pack = _c(120, 80, 45) if pid == 0 else _c(60, 64, 76)
        sh(c.d_box(cx, H * 0.57 + bob, W * 0.2, H * 0.13, W * 0.07), pack, W * 0.08, gloss=0.2)
        if pid == 0:
            sh(c.d_capsule(cx - W * 0.2, H * 0.44 + bob, cx + W * 0.2, H * 0.44 + bob, H * 0.035),
               _c(60, 110, 90), H * 0.03)                                         # bedroll
            c.over(_c(80, 52, 30), c.cov(c.d_box(cx, H * 0.6 + bob, W * 0.2, H * 0.012)))
        else:
            sh(c.d_circle(cx, H * 0.57 + bob, W * 0.08), _c(40, 42, 50), W * 0.05, gloss=0.6)
            for k in range(6):
                a0 = k / 6 * math.tau
                sh(c.d_circle(cx + math.cos(a0) * W * 0.08, H * 0.57 + bob + math.sin(a0) * W * 0.08,
                              W * 0.025), _c(150, 110, 50), W * 0.02, outline=0)
    # ---- arms
    arm_col = jacket * 0.92
    hand = SKIN if pid == 0 else _c(110, 116, 130)
    if pose == "side":
        ax, ay = cx + W * 0.02, H * 0.47 + bob
        ex, ey = ax - sw * W * 0.12, H * 0.66 + bob
        sh(c.d_capsule(ax, ay, ex, ey, W * 0.06, W * 0.05), arm_col, W * 0.05)
        sh(c.d_circle(ex, ey + H * 0.01, W * 0.05), hand, W * 0.04)
    else:
        for side, s in ((-1, sw), (1, -sw)):
            ax, ay = cx + side * W * 0.27, H * 0.47 + bob
            ex, ey = cx + side * W * 0.31, H * 0.66 + bob + s * H * 0.025
            sh(c.d_capsule(ax, ay, ex, ey, W * 0.065, W * 0.055), arm_col, W * 0.05)
            sh(c.d_circle(ex, ey + H * 0.012, W * 0.055), hand, W * 0.04)
            if pid == 1:
                sh(c.d_circle(ax, ay, W * 0.06), _c(200, 150, 60), W * 0.05, gloss=0.9)
    # ---- head
    hy = H * 0.3 + bob
    hr = W * 0.21
    if pid == 0:
        if pose != "up":
            sh(c.d_circle(cx + (W * 0.03 if pose == "side" else 0), hy, hr), SKIN, hr * 0.8, gloss=0.15,
               ambient=0.6)
        if pose == "down":
            for ex in (-0.35, 0.35):
                c.over(_c(30, 25, 30), c.cov(c.d_ellipse(cx + ex * hr, hy + hr * 0.15, hr * 0.12, hr * 0.17)))
                c.over(np.ones(3, F), c.cov(c.d_circle(cx + ex * hr - hr * 0.04, hy + hr * 0.08, hr * 0.05)))
            c.over(_c(230, 140, 130), c.cov(c.d_ellipse(cx - hr * 0.6, hy + hr * 0.42, hr * 0.16, hr * 0.08)) * 0.5)
            c.over(_c(230, 140, 130), c.cov(c.d_ellipse(cx + hr * 0.6, hy + hr * 0.42, hr * 0.16, hr * 0.08)) * 0.5)
            mouth = np.abs(c.d_circle(cx, hy + hr * 0.3, hr * 0.22)) - hr * 0.035
            mouth = np.maximum(mouth, hy + hr * 0.38 - c.Y)
            c.over(_c(120, 50, 50), c.cov(mouth))
        elif pose == "side":
            ex = cx + hr * 0.55
            c.over(_c(30, 25, 30), c.cov(c.d_ellipse(ex, hy + hr * 0.12, hr * 0.1, hr * 0.17)))
            sh(c.d_circle(cx + hr * 1.05, hy + hr * 0.3, hr * 0.14), SKIN, hr * 0.1)
        # pith helmet
        hx = cx + (W * 0.03 if pose == "side" else 0)
        brim = c.d_ellipse(hx + (hr * 0.25 if pose == "side" else 0), hy - hr * 0.35,
                           hr * (1.45 if pose != "side" else 1.5), hr * 0.38)
        sh(brim, helmet * 0.92, hr * 0.25, gloss=0.2)
        dome = c.d_ellipse(hx, hy - hr * 0.62, hr * 1.02, hr * 0.85)
        dome = np.maximum(dome, c.Y - (hy - hr * 0.3))
        sh(dome, helmet, hr * 0.7, gloss=0.3)
        band = np.maximum(c.d_ellipse(hx, hy - hr * 0.62, hr * 1.04, hr * 0.87),
                          np.abs(c.Y - (hy - hr * 0.42)) - hr * 0.1)
        c.over(acc * 0.9, c.cov(band))
        if pose != "up":
            lx = hx + (hr * 0.8 if pose == "side" else 0)
            sh(c.d_circle(lx, hy - hr * 0.5, hr * 0.2), _c(60, 60, 66), hr * 0.12, gloss=0.8)
            c.over(_c(255, 255, 220), c.cov(c.d_circle(lx, hy - hr * 0.5, hr * 0.12)))
    else:
        hx = cx + (W * 0.03 if pose == "side" else 0)
        sh(c.d_box(hx, hy, hr * (1.05 if pose != "side" else 0.85), hr * 0.9, hr * 0.45), helmet, hr * 0.6,
           gloss=0.8, power=40)
        if pose != "up":
            vx = hx + (hr * 0.3 if pose == "side" else 0)
            vw = hr * (0.8 if pose != "side" else 0.5)
            c.over(_c(25, 12, 8), c.cov(c.d_box(vx, hy + hr * 0.05, vw, hr * 0.22, hr * 0.12)))
            c.over(_c(255, 110, 50), c.cov(c.d_box(vx, hy + hr * 0.05, vw * 0.8, hr * 0.08, hr * 0.06)))
            c.over(_c(255, 230, 190), c.cov(c.d_box(vx, hy + hr * 0.05, vw * 0.35, hr * 0.04, hr * 0.03)))
        else:
            for k in range(3):
                c.over(_c(40, 42, 50), c.cov(c.d_box(hx, hy - hr * 0.3 + k * hr * 0.3, hr * 0.7, hr * 0.05)))
        # antenna
        sh(c.d_capsule(hx + hr * 0.4, hy - hr * 0.8, hx + hr * 0.6, hy - hr * 1.5, W * 0.015), _c(60, 62, 70),
           W * 0.01, outline=0)
        sh(c.d_circle(hx + hr * 0.6, hy - hr * 1.55, W * 0.035), _c(255, 90, 50), W * 0.03, gloss=1.0,
           ambient=1.0)
        for ex in (-1, 1):
            if pose == "side" and ex == -1:
                continue
            sh(c.d_circle(hx + ex * hr * (1.0 if pose != "side" else 0.0) + (hr * -0.3 if pose == "side" else 0),
                          hy, hr * 0.22), _c(200, 150, 60), hr * 0.15, gloss=0.9)
    return c.px


def character_set(n, pid, pose):
    return {str(f): _person(n, pid, pose, f) for f in range(4)}


def characters(n):
    """Returns {(pid, pose, frame): image} plus anchor info."""
    out = {}
    for pid in (0, 1):
        for pose in ("down", "up", "side"):
            for f in range(4):
                out[(pid, pose, f)] = _person(n, pid, pose, f)
    return out


def portrait(n, pid):
    img = _person(n, pid, "down", 0)
    H, W = img.shape[:2]
    top = int(H * 0.04)
    side = int(W * 0.08)
    return img[top:top + int(H * 0.52), side:W - side]


# ===================================================================== tools and effects
def hammer(n):
    """Hammer pointing up; handle base at the bottom centre."""
    W, H = int(n * 0.5), n
    c = Canvas(W, H)
    cx = W / 2
    grain = 0.5 + 0.5 * np.sin(c.X / W * 30 + Noise(5, 16).fbm(c.X / W * 2, c.Y / H * 8, 3) * 6)
    wood = mixc(_c(110, 70, 35), _c(165, 115, 60), grain[..., None])
    c.shape(c.d_capsule(cx, H * 0.97, cx, H * 0.25, W * 0.075, W * 0.06), wood, W * 0.07, gloss=0.25,
            outline=W * 0.012, outline_col=(0.15, 0.08, 0.03))
    c.shape(c.d_box(cx, H * 0.85, W * 0.09, H * 0.06, W * 0.03), _c(60, 40, 30), W * 0.05)
    head = c.d_box(cx, H * 0.17, W * 0.46, H * 0.11, W * 0.05)
    c.shape(head, _c(150, 156, 170), W * 0.12, gloss=1.0, power=40, outline=W * 0.012,
            outline_col=(0.1, 0.1, 0.12))
    c.over(np.ones(3, F), c.cov(c.d_box(cx - W * 0.1, H * 0.11, W * 0.25, H * 0.012, H * 0.006)) * 0.7)
    return c.px


def hook_head(n):
    """Grappling hook pointing right: shaft with an eye, three barbed prongs curling back."""
    W = H = n
    c = Canvas(W, H)
    cy = H * 0.5
    metal = _c(175, 182, 196)
    d = c.d_capsule(W * 0.14, cy, W * 0.62, cy, H * 0.045)
    d = np.minimum(d, np.abs(c.d_circle(W * 0.1, cy, H * 0.07)) - H * 0.025)
    tip = (W * 0.62, cy)
    for sgn, spread in ((-1, 1.0), (1, 1.0), (0, 0.0)):
        if sgn == 0:
            d = np.minimum(d, c.d_capsule(tip[0], cy, W * 0.92, cy, H * 0.04, H * 0.018))
            continue
        # curved prong: arc from the tip outwards and back toward the shaft
        pts = []
        for k in range(9):
            a = k / 8
            ang = -math.pi * 0.05 + a * math.pi * 0.85
            px = tip[0] + math.sin(ang) * W * 0.24
            py = cy + sgn * (1 - math.cos(ang)) * H * 0.22 * spread
            pts.append((px, py))
        for (p0, p1) in zip(pts, pts[1:]):
            d = np.minimum(d, c.d_capsule(p0[0], p0[1], p1[0], p1[1], H * 0.04, H * 0.036))
        ex, ey = pts[-1]
        d = np.minimum(d, c.d_poly([(ex - W * 0.06, ey - sgn * H * 0.02), (ex + W * 0.02, ey + sgn * H * 0.06),
                                    (ex + W * 0.03, ey - sgn * H * 0.04)]))
    c.shape(d, metal, H * 0.04, gloss=1.0, power=40, outline=H * 0.014, outline_col=(0.1, 0.1, 0.12))
    return c.px


def chain_link(n):
    W, H = n, int(n * 0.6)
    c = Canvas(W, H)
    d = np.abs(c.d_box(W / 2, H / 2, W * 0.42, H * 0.3, H * 0.3)) - H * 0.09
    c.shape(d, _c(160, 165, 178), H * 0.08, gloss=1.0, power=40, outline=H * 0.03,
            outline_col=(0.1, 0.1, 0.12))
    return c.px


def smoke(n, seed):
    x, y = grid(n)
    r = np.hypot(x / n - 0.5, y / n - 0.5) * 2
    nz = Noise(seed, 8).fbm(x / n * 4, y / n * 4, 4)
    a = np.clip((1 - r) * 1.4 - (1 - nz) * 0.7, 0, 1) ** 1.3
    col = np.ones((n, n, 3), F) * (0.75 + 0.25 * nz[..., None])
    return np.concatenate([col * a[..., None], a[..., None]], -1).astype(F)


def debris(n, seed, B):
    rng = np.random.default_rng(seed)
    c = Canvas(n)
    pts = []
    k = rng.integers(5, 8)
    for i in range(k):
        a = i / k * math.tau + rng.uniform(-0.3, 0.3)
        rr = n * rng.uniform(0.28, 0.45)
        pts.append((n / 2 + math.cos(a) * rr, n / 2 + math.sin(a) * rr))
    col = mixc(rgb(B["rock"][0]), rgb(B["rock"][1]), rng.random())
    c.shape(c.d_poly(pts), col, n * 0.25, gloss=0.2, ambient=0.5)
    return c.px


def leaf(n, seed, color):
    c = Canvas(n)
    d = c.d_ellipse(n / 2, n / 2, n * 0.42, n * 0.2)
    c.shape(d, np.array(color, F) / 255, n * 0.15, gloss=0.3, ambient=0.6)
    c.over(np.array(color, F) / 400, c.cov(c.d_capsule(n * 0.1, n / 2, n * 0.9, n / 2, n * 0.015)))
    return c.px


def snowflake(n):
    c = Canvas(n)
    d = np.full((n, n), 1e9, F)
    for k in range(3):
        a = k / 3 * math.pi
        dx, dy = math.cos(a) * n * 0.45, math.sin(a) * n * 0.45
        d = np.minimum(d, c.d_capsule(n / 2 - dx, n / 2 - dy, n / 2 + dx, n / 2 + dy, n * 0.04))
    c.shape(d, (1, 1, 1), n * 0.03, ambient=0.9)
    return c.px


def slash_arc(n):
    """Motion trail for the hammer swing (additive, white)."""
    x, y = grid(n)
    u, v = x / n - 0.5, y / n - 0.5
    r = np.hypot(u, v) * 2
    a = np.arctan2(v, u)
    band = np.exp(-((r - 0.82) / 0.1) ** 2)
    sweep = np.clip((a + math.pi) / (1.4 * math.pi), 0, 1) * (a < 0.4 * math.pi)
    inten = np.clip(band * sweep ** 2, 0, 1)
    return np.concatenate([np.ones((n, n, 3), F) * inten[..., None], inten[..., None]], -1).astype(F)


# ===================================================================== mine
def mine(n):
    """Spiked land mine (neutral metal; owner colour ring + blinking light added at runtime)."""
    c = Canvas(n)
    cx, cy, r = n * 0.5, n * 0.56, n * 0.25
    c.shape(c.d_ellipse(cx, cy + r * 0.95, r * 1.25, r * 0.32), (0, 0, 0), opacity=0.45, soft=n * 0.05)
    spikes = np.full((n, n), 1e9, F)
    for k in range(8):
        a = k / 8 * math.tau + math.pi / 8
        x0, y0 = cx + math.cos(a) * r * 0.8, cy + math.sin(a) * r * 0.8
        x1, y1 = cx + math.cos(a) * r * 1.45, cy + math.sin(a) * r * 1.45
        spikes = np.minimum(spikes, c.d_capsule(x0, y0, x1, y1, n * 0.035, n * 0.018))
        spikes = np.minimum(spikes, c.d_circle(x1, y1, n * 0.03))
    c.shape(spikes, _c(120, 124, 136), n * 0.025, gloss=0.9, power=40, outline=n * 0.01, outline_col=(0.08, 0.08, 0.1))
    body = c.d_circle(cx, cy, r)
    c.shape(body, _c(70, 74, 84), r * 0.9, gloss=0.8, power=30, outline=n * 0.012, outline_col=(0.06, 0.06, 0.08))
    band = np.maximum(body + n * 0.01, np.abs(c.Y - cy) - n * 0.03)
    c.shape(band, _c(200, 170, 60), n * 0.015, gloss=0.6)
    for k in range(6):
        a = k / 6 * math.tau
        c.shape(c.d_circle(cx + math.cos(a) * r * 0.62, cy + math.sin(a) * r * 0.62 * 0.9, n * 0.022),
                _c(150, 154, 166), n * 0.015, gloss=1.0)
    c.shape(c.d_circle(cx, cy - r * 0.1, n * 0.06), _c(40, 10, 10), n * 0.03, gloss=1.0)
    return c.px


# ===================================================================== UI icons (white, tinted at runtime)
def icon(name, n):
    c = Canvas(n)
    w = (1, 1, 1)
    u = n / 24

    def S(d, bev=u * 1.2):
        c.shape(d, w, bevel=bev, ambient=0.78, diffuse=0.35, gloss=0.25)
    if name == "hammer":
        S(c.d_capsule(6 * u, 20 * u, 15 * u, 9 * u, 1.4 * u))
        pts = [(10 * u, 4 * u), (18 * u, 1.5 * u), (22.5 * u, 9 * u), (14.5 * u, 12 * u)]
        S(c.d_poly(pts) - 0.6 * u)
    elif name == "push":
        S(c.d_circle(15.5 * u, 12 * u, 6.5 * u))
        S(c.d_capsule(2 * u, 12 * u, 6.5 * u, 12 * u, 1.3 * u))
        S(c.d_poly([(5.5 * u, 8 * u), (9.5 * u, 12 * u), (5.5 * u, 16 * u)]))
    elif name == "hook":
        ring = np.abs(c.d_circle(11 * u, 13 * u, 6 * u)) - 1.3 * u
        ring = np.maximum(ring, -(c.Y - 13 * u) * 1.0)
        S(np.minimum(ring, c.d_capsule(5 * u, 13 * u, 5 * u, 3 * u, 1.3 * u)))
        S(c.d_poly([(14 * u, 10 * u), (17 * u, 4 * u), (20 * u, 10 * u)]))
        S(c.d_capsule(17 * u, 13 * u, 17 * u, 8 * u, 1.3 * u))
        S(np.abs(c.d_circle(5 * u, 2.5 * u, 1.6 * u)) - 0.7 * u, u * 0.5)
    elif name == "wait":
        S(c.d_poly([(6 * u, 3 * u), (18 * u, 3 * u), (12 * u, 12 * u)]) - 0.4 * u)
        S(c.d_poly([(6 * u, 21 * u), (18 * u, 21 * u), (12 * u, 12 * u)]) - 0.4 * u)
        S(c.d_box(12 * u, 2.5 * u, 8 * u, 1 * u, 0.6 * u))
        S(c.d_box(12 * u, 21.5 * u, 8 * u, 1 * u, 0.6 * u))
    elif name == "heart":
        d = np.minimum(c.d_circle(8 * u, 9 * u, 5 * u), c.d_circle(16 * u, 9 * u, 5 * u))
        d = np.minimum(d, c.d_poly([(3.3 * u, 10.6 * u), (20.7 * u, 10.6 * u), (12 * u, 21 * u)]))
        S(d, u * 2)
    elif name == "lock":
        sh = np.maximum(np.abs(c.d_box(12 * u, 9 * u, 5 * u, 6 * u, 5 * u)) - 1.3 * u, c.Y - 11 * u)
        S(np.minimum(sh, c.d_box(12 * u, 15 * u, 7.5 * u, 6 * u, 1.5 * u)))
    elif name == "exit":
        S(np.abs(c.d_box(12 * u, 13 * u, 7 * u, 9 * u, 6 * u)) - 1.4 * u)
        S(c.d_circle(12 * u, 13 * u, 3 * u))
    elif name == "music":
        S(c.d_ellipse(8 * u, 18 * u, 3.5 * u, 2.8 * u))
        S(c.d_ellipse(18 * u, 16 * u, 3.5 * u, 2.8 * u))
        S(c.d_capsule(11 * u, 18 * u, 11 * u, 5 * u, 1.1 * u))
        S(c.d_capsule(21 * u, 16 * u, 21 * u, 3 * u, 1.1 * u))
        S(c.d_poly([(10 * u, 4 * u), (22 * u, 2 * u), (22 * u, 6 * u), (10 * u, 8 * u)]))
    elif name == "sound":
        S(c.d_poly([(3 * u, 9 * u), (8 * u, 9 * u), (14 * u, 3 * u), (14 * u, 21 * u), (8 * u, 15 * u),
                    (3 * u, 15 * u)]))
        for rr in (5, 9):
            arc = np.abs(c.d_circle(13 * u, 12 * u, rr * u)) - 1.0 * u
            S(np.maximum(arc, 15 * u - c.X), u * 0.6)
    elif name == "eye":
        S(np.abs(c.d_ellipse(12 * u, 12 * u, 10 * u, 6 * u)) - 1.2 * u)
        S(c.d_circle(12 * u, 12 * u, 3.5 * u))
    elif name == "mine":
        d = c.d_circle(12 * u, 13 * u, 6 * u)
        for k in range(8):
            a = k / 8 * math.tau
            d = np.minimum(d, c.d_capsule(12 * u + math.cos(a) * 5 * u, 13 * u + math.sin(a) * 5 * u,
                                          12 * u + math.cos(a) * 9.5 * u, 13 * u + math.sin(a) * 9.5 * u, 1.2 * u))
        S(d)
        c.over((0.1, 0.1, 0.1), c.cov(c.d_circle(12 * u, 12 * u, 2 * u)))
    elif name == "statue":
        S(c.d_box(10 * u, 11 * u, 7 * u, 8 * u, 2.5 * u))
        S(c.d_box(10 * u, 21 * u, 8 * u, 1.6 * u, 0.6 * u))
        c.over((0.1, 0.1, 0.1), c.cov(c.d_circle(12.5 * u, 10 * u, 2.2 * u)))
        S(c.d_box(19 * u, 10 * u, 5 * u, 0.9 * u, 0.4 * u), u * 0.4)
    elif name == "camera":
        S(c.d_box(12 * u, 13 * u, 10 * u, 7 * u, 2 * u) - 0 * u)
        c.over((0.1, 0.1, 0.1), c.cov(c.d_circle(12 * u, 13 * u, 4.5 * u)))
        S(c.d_circle(12 * u, 13 * u, 3 * u))
        S(c.d_box(8 * u, 5 * u, 3 * u, 2 * u, 0.6 * u))
    return c.px


# ===================================================================== lens statue
def statue(n, B, biome):
    """Two-cell lens statue in profile facing RIGHT: a carved idol head with one big lens eye
    (top cell) on a stone body with folded arms and a pedestal (bottom cell). Width n,
    height 2n. Flipped at runtime to face left; the eye glow and beam are added at runtime."""
    c = Canvas(n, 2 * n)
    u = n / 100.0
    nz = Noise(77, 64)
    grain = nz.fbm(c.X / n * 9, c.Y / n * 9, 4)
    a, b = rgb(B["rock"][0]), rgb(B["rock"][1])
    stone = mixc(a * 0.92, b, grain[..., None] * 1.2 - 0.1)
    stone = stone * (0.88 + 0.24 * Noise(78, 64).fbm(c.X / n * 30, c.Y / n * 30, 2))[..., None]
    dark = stone * 0.55
    OUT = dict(outline=1.3 * u, outline_col=(0.04, 0.04, 0.04))
    # ground shadow and pedestal (bottom of the body cell)
    c.shape(c.d_ellipse(50 * u, 195 * u, 42 * u, 5 * u), (0, 0, 0), opacity=0.5, soft=4 * u)
    c.shape(c.d_box(50 * u, 186 * u, 40 * u, 9 * u, 2 * u), dark, 4 * u, gloss=0.1, **OUT)
    c.shape(c.d_box(50 * u, 175 * u, 35 * u, 3.5 * u, 1.5 * u), stone * 0.8, 2.5 * u, gloss=0.15)
    for k in range(5):
        c.over(dark * 0.6, c.cov(c.d_box(22 * u + k * 14 * u, 186 * u, 1.0 * u, 7 * u)) * 0.8)
    # body: a tapering carved torso
    torso = c.d_poly([(22 * u, 96 * u), (74 * u, 96 * u), (70 * u, 172 * u), (26 * u, 172 * u)]) - 3 * u
    c.shape(torso, stone * 0.95, 8 * u, gloss=0.16, power=18, **OUT)
    # folded arms across the chest
    arm = c.d_capsule(30 * u, 118 * u, 66 * u, 128 * u, 7 * u)
    arm2 = c.d_capsule(66 * u, 118 * u, 32 * u, 130 * u, 6.5 * u)
    c.shape(arm2, stone * 0.9, 5 * u, gloss=0.2, **OUT)
    c.shape(arm, stone * 1.02, 5 * u, gloss=0.2, **OUT)
    # belt and glyph panel
    c.shape(c.d_box(48 * u, 146 * u, 22 * u, 4 * u, 1.5 * u), stone * 1.05, 2.5 * u, gloss=0.2)
    for k in range(4):
        c.over(dark * 0.5, c.cov(c.d_box((35 + k * 9) * u, 146 * u, 0.9 * u, 3 * u)) * 0.8)
    for k in range(2):
        c.over(dark * 0.55, c.cov(np.abs(c.d_circle((40 + k * 16) * u, 160 * u, 3.2 * u)) - 0.8 * u) * 0.75)
    # neck
    c.shape(c.d_box(46 * u, 92 * u, 19 * u, 9 * u, 3 * u), stone * 0.8, 4 * u, **OUT)
    # head block
    head = c.d_box(46 * u, 55 * u, 27 * u, 30 * u, 9 * u)
    head = np.minimum(head, c.d_box(60 * u, 60 * u, 16 * u, 22 * u, 6 * u))
    c.shape(head, stone, 7 * u, gloss=0.18, power=18, **OUT)
    band = c.d_box(45 * u, 27 * u, 31 * u, 6 * u, 3 * u)
    c.shape(band, stone * 1.05, 3 * u, gloss=0.2, **OUT)
    for k in range(4):
        x = 24 * u + k * 13 * u
        tri = c.d_poly([(x - 5 * u, 22 * u), (x, 11 * u), (x + 5 * u, 22 * u)])
        c.shape(tri, stone * 0.95, 2 * u, outline=1.0 * u, outline_col=(0.04, 0.04, 0.04))
    for k in range(7):
        c.over(dark * 0.5, c.cov(c.d_box(18 * u + k * 9 * u, 27 * u, 0.9 * u, 4 * u)) * 0.7)
    c.shape(c.d_ellipse(29 * u, 55 * u, 6 * u, 10 * u), stone * 0.85, 3 * u)
    c.over(dark * 0.5, c.cov(np.abs(c.d_ellipse(29 * u, 55 * u, 3.5 * u, 6 * u)) - 0.8 * u) * 0.8)
    c.shape(c.d_box(63 * u, 42 * u, 14 * u, 3.5 * u, 2 * u), stone * 1.08, 2.5 * u, gloss=0.2)
    nose = c.d_poly([(66 * u, 45 * u), (80 * u, 63 * u), (77 * u, 66 * u), (65 * u, 65 * u)])
    c.shape(nose - 1.0 * u, stone * 1.02, 3 * u, gloss=0.2, outline=1.0 * u, outline_col=(0.05, 0.05, 0.05))
    c.shape(c.d_box(67 * u, 74 * u, 10 * u, 4.5 * u, 2.5 * u), stone * 0.98, 2.5 * u, gloss=0.15)
    c.over(dark * 0.35, c.cov(c.d_box(67 * u, 74 * u, 8 * u, 0.9 * u)))
    for y in (35 * u, 81 * u):
        c.over(dark * 0.5, c.cov(c.d_box(44 * u, y, 24 * u, 0.8 * u)) * 0.85)
    # eye socket and the lens (eye centre at 64% across, 26% down)
    ex, ey = 64 * u, 52 * u
    c.shape(c.d_circle(ex, ey, 9 * u), (0.03, 0.03, 0.05), 2 * u)
    lens_d = c.d_circle(ex, ey, 6.4 * u)
    rr = np.clip(np.hypot(c.X - ex + 2 * u, c.Y - ey + 2 * u) / (7 * u), 0, 1)
    lens = mixc(np.array([1.0, 1.0, 1.0], F), np.array([0.55, 0.85, 1.0], F), rr[..., None])
    c.over(lens, c.cov(lens_d))
    c.over(np.array([0.3, 0.6, 0.9], F), c.cov(np.abs(lens_d) - 0.7 * u) * 0.8)
    c.over(np.ones(3, F), c.cov(c.d_circle(ex - 2.3 * u, ey - 2.4 * u, 1.8 * u)) * 0.95)
    body = c.cov(np.minimum(np.minimum(head, band), torso))
    if biome == "jungle":
        moss = smoothstep(0.58, 0.72, Noise(81, 32).fbm(c.X / n * 5, c.Y / n * 5, 4) +
                          np.abs(c.Y - 100 * u) / n * -0.25 + 0.1)
        c.over(rgb("#5f8a30") * (0.75 + 0.4 * grain[..., None]), moss * body * 0.8)
    elif biome == "frost":
        snow = smoothstep(0.5, 0.62, (29 * u - c.Y) / n * 3.0 + 0.47 +
                          Noise(82, 32).fbm(c.X / n * 7, c.Y / n * 7, 3) * 0.35)
        snow2 = smoothstep(0.55, 0.65, (100 * u - c.Y) / n * 4.0 + 0.5 +
                           Noise(84, 32).fbm(c.X / n * 7, c.Y / n * 7, 3) * 0.3) * (c.Y > 92 * u)
        c.over(rgb("#f2f8ff"), np.maximum(snow * c.cov(np.minimum(head, band)), snow2 * c.cov(torso)) * 0.95)
    else:
        cracks = smoothstep(0.74, 0.8, Noise(83, 32).fbm(c.X / n * 8, c.Y / n * 8, 3))
        c.add(np.array([0.9, 0.3, 0.05], F), cracks * body * 0.9)
    return c.px


def ice_block(n):
    """Translucent ice shell drawn over a frozen explorer."""
    c = Canvas(n)
    u = n / 100.0
    d = c.d_box(50 * u, 54 * u, 40 * u, 44 * u, 12 * u)
    shade = np.clip(0.55 + (c.Y / n - 0.5) * 0.5 + (c.X / n - 0.5) * 0.2, 0, 1)
    col = mixc(np.array([0.85, 0.96, 1.0], F), np.array([0.45, 0.72, 0.95], F), shade[..., None])
    c.shape(d, col, 6 * u, gloss=0.9, power=40, ambient=0.8, opacity=0.5)
    rng = np.random.default_rng(5)
    lines = np.full((n, n), 1e9, F)
    for _ in range(9):
        x0, y0 = rng.uniform(15, 85) * u, rng.uniform(15, 92) * u
        a = rng.uniform(0, math.pi)
        L = rng.uniform(12, 30) * u
        lines = np.minimum(lines, c.d_capsule(x0, y0, x0 + math.cos(a) * L, y0 + math.sin(a) * L, 0.5 * u))
    c.over(np.ones(3, F), c.cov(lines) * c.cov(d) * 0.55)
    c.over(np.ones(3, F), c.cov(np.abs(d) - 1.2 * u) * 0.85)
    hl = c.d_poly([(20 * u, 18 * u), (34 * u, 14 * u), (24 * u, 60 * u), (16 * u, 62 * u)])
    c.over(np.ones(3, F), c.cov(hl) * 0.35)
    return c.px
