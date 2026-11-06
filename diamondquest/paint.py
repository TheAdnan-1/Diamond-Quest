"""Procedural painting toolkit (numpy).

Everything the game shows is painted here at whatever resolution the screen needs:
noise, Voronoi cells, height-map lighting, anti-aliased signed-distance shapes with
automatic bevel shading, blurs, colour ramps, and conversion to pygame surfaces.

Images are float32 arrays shaped (h, w, 4) holding *premultiplied* RGBA in 0..1.
"""
import math

import numpy as np
import pygame

F = np.float32
LIGHT = np.array([-0.45, -0.65, 0.62], F)
LIGHT /= np.linalg.norm(LIGHT)
HALF = LIGHT + np.array([0, 0, 1], F)
HALF /= np.linalg.norm(HALF)


# ------------------------------------------------------------------ colour helpers
def rgb(c):
    """(r, g, b) in 0..255 or hex string -> float32 array 0..1"""
    if isinstance(c, str):
        c = c.lstrip("#")
        c = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    return np.array(c[:3], F) / 255.0


def mixc(a, b, t):
    return a + (b - a) * t


def ramp(t, stops):
    """Map t (array, 0..1) through colour stops [(pos, (r,g,b)), ...] -> (h, w, 3)."""
    t = np.clip(t, 0, 1)
    pos = np.array([p for p, _ in stops], F)
    cols = np.array([rgb(c) for _, c in stops], F)
    out = np.empty(t.shape + (3,), F)
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, cols[:, ch])
    return out


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ noise
class Noise:
    """Lattice value noise with a periodic table, so any period-aligned field tiles."""

    def __init__(self, seed, period=64):
        rng = np.random.default_rng(seed)
        self.p = period
        self.t = rng.random((period, period)).astype(F)

    def at(self, x, y):
        p = self.p
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        tx = (x - x0).astype(F)
        ty = (y - y0).astype(F)
        tx = tx * tx * tx * (tx * (tx * 6 - 15) + 10)
        ty = ty * ty * ty * (ty * (ty * 6 - 15) + 10)
        xa, xb = x0 % p, (x0 + 1) % p
        ya, yb = y0 % p, (y0 + 1) % p
        t = self.t
        a = t[ya, xa]
        b = t[ya, xb]
        c = t[yb, xa]
        d = t[yb, xb]
        return (a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx)) * ty

    def fbm(self, x, y, octaves=4, lac=2.0, gain=0.5):
        tot = np.zeros(np.broadcast(x, y).shape, F)
        amp, norm, f = 1.0, 0.0, 1.0
        for i in range(octaves):
            tot += amp * self.at(x * f + i * 17.3, y * f + i * 31.7)
            norm += amp
            amp *= gain
            f *= lac
        return tot / norm


def grid(w, h=None):
    h = h or w
    y, x = np.mgrid[0:h, 0:w].astype(F)
    return x + 0.5, y + 0.5


def tile_fbm(n, freq, seed, octaves=4, gain=0.5):
    """Seamlessly tiling fBm on an n x n image, freq = lattice cells across (int)."""
    nz = Noise(seed, period=freq)
    x, y = grid(n)
    tot = np.zeros((n, n), F)
    amp, norm, f = 1.0, 0.0, 1
    for i in range(octaves):
        sub = Noise(seed * 7 + i, period=freq * f) if i else nz
        tot += amp * sub.at(x / n * freq * f, y / n * freq * f)
        norm += amp
        amp *= gain
        f *= 2
    return tot / norm


def voronoi(w, h, pts, tile=True):
    """pts: (n, 2) in pixel units. Returns F1, F2 (pixels) and nearest index (row-chunked)."""
    if tile:
        offs = [(dx, dy) for dx in (-w, 0, w) for dy in (-h, 0, h)]
        allp = np.concatenate([pts + np.array(o, F) for o in offs])
        ids = np.tile(np.arange(len(pts)), len(offs))
    else:
        allp, ids = pts, np.arange(len(pts))
    xs = np.arange(w, dtype=F) + 0.5
    F1 = np.empty((h, w), F)
    F2 = np.empty((h, w), F)
    IDX = np.empty((h, w), np.int64)
    rows = max(1, int(6e6 // max(1, w * len(allp))))
    for y0 in range(0, h, rows):
        ys = np.arange(y0, min(h, y0 + rows), dtype=F)[:, None] + 0.5
        d = np.sqrt((xs[None, :, None] - allp[:, 0]) ** 2 + (ys[..., None] - allp[:, 1]) ** 2)
        part = np.argpartition(d, 1, axis=-1)[..., :2]
        d1 = np.take_along_axis(d, part, -1)
        order = np.argsort(d1, axis=-1)
        d1 = np.take_along_axis(d1, order, -1)
        i1 = np.take_along_axis(part, order, -1)[..., 0]
        F1[y0:y0 + len(ys)] = d1[..., 0]
        F2[y0:y0 + len(ys)] = d1[..., 1]
        IDX[y0:y0 + len(ys)] = ids[i1]
    return F1, F2, IDX


def voronoi_grid(w, h, cols, rows, rng, jit=0.8):
    """Fast tileable Voronoi for grid-jittered points: each pixel checks its 5x5 neighbour cells.
    Returns F1, F2, nearest index (j*cols+i) and the points array."""
    cw, ch = w / cols, h / rows
    jx = (rng.random((rows, cols)) - 0.5) * jit
    jy = (rng.random((rows, cols)) - 0.5) * jit
    px = ((np.arange(cols)[None, :] + 0.5 + jx) * cw).astype(F)
    py = ((np.arange(rows)[:, None] + 0.5 + jy) * ch).astype(F)
    x, y = grid(w, h)
    ci = np.floor(x / cw).astype(np.int64)
    cj = np.floor(y / ch).astype(np.int64)
    b1 = np.full((h, w), np.inf, F)
    b2 = np.full((h, w), np.inf, F)
    idx = np.zeros((h, w), np.int64)
    for dj in range(-2, 3):
        for di in range(-2, 3):
            nj, ni = cj + dj, ci + di
            wj, wi = nj % rows, ni % cols
            qx = px[wj, wi] + (ni // cols) * w
            qy = py[wj, wi] + (nj // rows) * h
            d = np.sqrt((x - qx) ** 2 + (y - qy) ** 2)
            closer = d < b1
            b2 = np.where(closer, b1, np.minimum(b2, d))
            idx = np.where(closer, wj * cols + wi, idx)
            b1 = np.where(closer, d, b1)
    pts = np.stack([px.ravel() * 0 + np.broadcast_to(px, (rows, cols)).ravel(),
                    np.broadcast_to(py, (rows, cols)).ravel()], 1).astype(F)
    return b1, b2, idx, pts


def cut_tiles(img, n, k):
    """Split a (k*n)^2 seamless atlas into k*k tiles, row-major."""
    return [img[r * n:(r + 1) * n, c * n:(c + 1) * n].copy() for r in range(k) for c in range(k)]


def jitter_points(w, h, cols, rows, rng, jit=0.8):
    pts = []
    for j in range(rows):
        for i in range(cols):
            pts.append(((i + 0.5 + (rng.random() - 0.5) * jit) * w / cols,
                        (j + 0.5 + (rng.random() - 0.5) * jit) * h / rows))
    return np.array(pts, F)


# ------------------------------------------------------------------ filters / lighting
def blur(a, sigma, wrap=True):
    """Gaussian blur of a 2-D (or 3-D, per channel) array via FFT (wraps around)."""
    if sigma <= 0:
        return a
    if a.ndim == 3:
        return np.stack([blur(a[..., i], sigma, wrap) for i in range(a.shape[2])], -1)
    h, w = a.shape
    pad = 0 if wrap else int(sigma * 3) + 1
    if pad:
        a = np.pad(a, pad, mode="edge")
    H, W = a.shape
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    g = np.exp(-2 * (math.pi ** 2) * sigma * sigma * (fx * fx + fy * fy))
    out = np.fft.irfft2(np.fft.rfft2(a) * g, s=a.shape).astype(F)
    if pad:
        out = out[pad:pad + h, pad:pad + w]
    return out


def normals(height, strength=1.0, wrap=True):
    if wrap:
        dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5
        dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 0.5
    else:
        dy, dx = np.gradient(height)
    n = np.stack([-dx * strength, -dy * strength, np.ones_like(height)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n.astype(F)


def light(n, ambient=0.45, diffuse=0.75, spec=0.0, power=24.0):
    """Returns (shade multiplier, specular add) from a normal map."""
    d = np.clip((n * LIGHT).sum(-1), 0, 1)
    s = np.clip((n * HALF).sum(-1), 0, 1) ** power * spec if spec else 0.0
    return ambient + diffuse * d, s


def lit_surface(albedo, height, strength=1.0, ambient=0.45, diffuse=0.75, spec=0.0,
                power=24.0, wrap=True, ao=0.0):
    n = normals(height, strength, wrap)
    sh, sp = light(n, ambient, diffuse, spec, power)
    col = albedo * sh[..., None] + (sp[..., None] if spec else 0)
    if ao:
        occl = np.clip(1 - ao * np.clip(blur(height, 3, wrap) - height, 0, 1) * 6, 0.35, 1)
        col *= occl[..., None]
    return col


# ------------------------------------------------------------------ signed distances
class Canvas:
    """A premultiplied RGBA float image with SDF painting helpers (pixel units)."""

    def __init__(self, w, h=None):
        self.w = int(w)
        self.h = int(h or w)
        self.px = np.zeros((self.h, self.w, 4), F)
        self.X, self.Y = grid(self.w, self.h)

    # ---- distance functions (negative inside)
    def d_circle(self, cx, cy, r):
        return np.hypot(self.X - cx, self.Y - cy) - r

    def d_ellipse(self, cx, cy, rx, ry):
        k = np.hypot((self.X - cx) / rx, (self.Y - cy) / ry)
        return (k - 1) * min(rx, ry)

    def d_box(self, cx, cy, hw, hh, r=0.0):
        qx = np.abs(self.X - cx) - (hw - r)
        qy = np.abs(self.Y - cy) - (hh - r)
        out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        return out + np.minimum(np.maximum(qx, qy), 0) - r

    def d_capsule(self, ax, ay, bx, by, r, r2=None):
        px, py = self.X - ax, self.Y - ay
        vx, vy = bx - ax, by - ay
        ll = vx * vx + vy * vy or 1e-6
        t = np.clip((px * vx + py * vy) / ll, 0, 1)
        rr = r if r2 is None else r + (r2 - r) * t
        return np.hypot(px - vx * t, py - vy * t) - rr

    def d_poly(self, pts):
        pts = np.asarray(pts, F)
        x, y = self.X, self.Y
        d = (x - pts[0, 0]) ** 2 + (y - pts[0, 1]) ** 2
        s = np.ones_like(x)
        n = len(pts)
        j = n - 1
        for i in range(n):
            ex, ey = pts[j, 0] - pts[i, 0], pts[j, 1] - pts[i, 1]
            wx, wy = x - pts[i, 0], y - pts[i, 1]
            t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey + 1e-9), 0, 1)
            bx, by = wx - ex * t, wy - ey * t
            d = np.minimum(d, bx * bx + by * by)
            c1 = y >= pts[i, 1]
            c2 = y < pts[j, 1]
            c3 = ex * wy > ey * wx
            flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
            s = np.where(flip, -s, s)
            j = i
        return s * np.sqrt(d)

    # ---- painting
    @staticmethod
    def cov(d, soft=1.0):
        return np.clip(0.5 - d / soft, 0, 1).astype(F)

    def over(self, color, alpha):
        """color: (h,w,3) or rgb; alpha: (h,w) coverage*opacity (straight)."""
        a = np.clip(alpha, 0, 1)[..., None]
        c = color if isinstance(color, np.ndarray) and color.ndim == 3 else np.asarray(color, F)
        self.px[..., :3] = c * a + self.px[..., :3] * (1 - a)
        self.px[..., 3:] = a + self.px[..., 3:] * (1 - a)

    def add(self, color, alpha):
        a = np.clip(alpha, 0, 1)[..., None]
        c = color if isinstance(color, np.ndarray) and color.ndim == 3 else np.asarray(color, F)
        self.px[..., :3] = np.clip(self.px[..., :3] + c * a, 0, 1)

    def shape(self, d, color, bevel=0.0, opacity=1.0, outline=0.0, outline_col=(0, 0, 0),
              gloss=0.0, power=30.0, ambient=0.5, diffuse=0.7, tex=None, soft=1.0,
              shade_d=None, rim=0.0):
        """Fill an SDF with optional bevel lighting, texture modulation and outline."""
        col = np.broadcast_to(np.asarray(color, F), d.shape + (3,)).copy() if not \
            (isinstance(color, np.ndarray) and color.ndim == 3) else color.copy()
        if tex is not None:
            col *= tex[..., None] if tex.ndim == 2 else tex
        sd = d if shade_d is None else shade_d
        if bevel > 0:
            k = np.clip(-sd / bevel, 0, 1)
            hgt = np.sqrt(1 - (1 - k) ** 2) * bevel
            n = normals(hgt, 1.0, wrap=False)
            sh, sp = light(n, ambient, diffuse, gloss, power)
            col = col * sh[..., None] + (sp[..., None] if gloss else 0)
            if rim:
                edge = np.clip(1 - np.clip(-sd / (bevel * 0.6), 0, 1), 0, 1) ** 2
                col = col + rim * edge[..., None] * np.clip(-n[..., 1:2] * 0 + 0.6, 0, 1)
        if outline > 0:
            self.over(np.asarray(rgb(outline_col) if not isinstance(outline_col, np.ndarray)
                                 else outline_col, F),
                      self.cov(d - outline, soft) * opacity)
        self.over(np.clip(col, 0, 1.5), self.cov(d, soft) * opacity)

    def surface(self):
        return to_surface(self.px)


# ------------------------------------------------------------------ conversion
def to_surface(px, premult=True):
    """premultiplied float RGBA -> pygame SRCALPHA surface (straight alpha)."""
    a = np.clip(px[..., 3:4], 0, 1)
    if premult:
        c = np.where(a > 1e-4, px[..., :3] / np.maximum(a, 1e-4), 0)
    else:
        c = px[..., :3]
    out = np.concatenate([np.clip(c, 0, 1), a], -1)
    data = (out * 255 + 0.5).astype(np.uint8)
    h, w = data.shape[:2]
    return pygame.image.frombuffer(data.tobytes(), (w, h), "RGBA").copy()


def opaque(rgb_img):
    h, w = rgb_img.shape[:2]
    out = np.concatenate([np.clip(rgb_img, 0, 1), np.ones((h, w, 1), F)], -1)
    return out.astype(F)


def scaled(surface, factor):
    w, h = surface.get_size()
    return pygame.transform.smoothscale(surface, (max(1, int(w * factor)), max(1, int(h * factor))))
