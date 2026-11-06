"""GPU rendering layer (SDL2 renderer via pygame._sdl2).

The game renders at the monitor's real resolution: textures are scaled and blended on the
graphics card, so 1080p, 4K and 8K all get sharp output at full frame rate.
UI code works in *design units* (a 1280x720 reference frame scaled by `u`).
"""
import math
import os
import sys

import numpy as np
import pygame
from pygame._sdl2 import video

from . import paint

BLEND = pygame.BLENDMODE_BLEND
ADD = pygame.BLENDMODE_ADD
MOD = pygame.BLENDMODE_MOD


def setup_platform():
    """Must run before pygame.init(): crisp output on high-DPI screens, smooth texture scaling."""
    os.environ.setdefault("SDL_RENDER_SCALE_QUALITY", "2")          # best supported filtering
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    os.environ.setdefault("SDL_HINT_VIDEO_HIGHDPI_DISABLED", "0")
    if sys.platform == "win32":
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)        # per-monitor aware
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def _soft_disc(n, power=2.0, edge=0.0):
    x, y = paint.grid(n)
    r = np.hypot(x / n - 0.5, y / n - 0.5) * 2
    if edge:
        a = np.clip((1 - r) / edge, 0, 1)
    else:
        a = np.clip(1 - r, 0, 1) ** power
    return np.concatenate([np.ones((n, n, 3), np.float32) * a[..., None], a[..., None]], -1)


def _ring(n, width):
    x, y = paint.grid(n)
    r = np.hypot(x / n - 0.5, y / n - 0.5) * 2
    d = np.abs(r - (1 - width)) / width
    a = np.clip(1 - d, 0, 1) ** 1.5
    return np.concatenate([np.ones((n, n, 3), np.float32) * a[..., None], a[..., None]], -1)


def _rounded(rp, filled=True, stroke=1.0):
    """Rounded-square texture for 9-slicing: corner radius rp pixels (2px stretchable centre)."""
    n = int(rp * 2 + 6)
    c = paint.Canvas(n)
    d = c.d_box(n / 2, n / 2, n / 2 - 0.5, n / 2 - 0.5, rp)
    if filled:
        a = c.cov(d)
    else:
        a = c.cov(np.abs(d + stroke / 2) - stroke / 2)
    return np.concatenate([np.ones((n, n, 3), np.float32) * a[..., None], a[..., None]], -1), n


class GPU:
    def __init__(self, title, fullscreen=False):
        os.environ.setdefault('SDL_RENDER_SCALE_QUALITY','2')
        info = pygame.display.get_desktop_sizes() if hasattr(pygame.display, "get_desktop_sizes") else []
        desk = info[0] if info else (1920, 1080)
        self.desktop = desk
        size = (max(960, int(desk[0] * 0.82)), max(540, int(desk[1] * 0.82)))
        try:
            self.win = video.Window(title, size=size, resizable=True, allow_highdpi=True)
        except TypeError:
            self.win = video.Window(title, size=size, resizable=True)
        try:
            self.r = video.Renderer(self.win, accelerated=1, vsync=True)
            self.accelerated = True
        except Exception:
            self.r = video.Renderer(self.win, accelerated=-1)
            self.accelerated = False
        self.fullscreen = False
        if fullscreen:
            self.toggle_fullscreen()
        self.overbright = self._probe_overbright()
        self.txt = {}
        self.fonts = {}
        self.rounded = {}
        self.prims = {}
        self.resize()
        self._build_prims()

    def _probe_overbright(self):
        """Light maps use dst = 2 * light * scene when the GPU supports custom blending, so
        torches and lamps can brighten the scene beyond its base colours."""
        try:
            mode = self.r.compose_custom_blend_mode((7, 3, 1), (1, 2, 1))
            t = video.Texture(self.r, (4, 4), target=True)
            t.blend_mode = mode
            self.r.target = None
            t.draw(dstrect=(0, 0, 1, 1))
            self.MOD2X = mode
            return True
        except Exception:
            self.MOD2X = MOD
            return False

    # ------------------------------------------------------------ window
    def resize(self):
        vp = self.r.get_viewport()
        self.W, self.H = max(1, vp.w), max(1, vp.h)
        self.u = min(self.W / 1280, self.H / 720)
        self.DW, self.DH = self.W / self.u, self.H / self.u
        # mouse coordinates arrive in window points; on retina/scaled displays pixels differ
        ws = self.win.size
        self.mouse_scale = (self.W / max(1, ws[0]), self.H / max(1, ws[1]))
        self.txt.clear()
        self.rounded.clear()

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.win.set_windowed()
        else:
            self.win.set_fullscreen(desktop=True)
        self.fullscreen = not self.fullscreen

    def mouse(self):
        x, y = pygame.mouse.get_pos()
        return x * self.mouse_scale[0], y * self.mouse_scale[1]

    def mouse_u(self):
        x, y = self.mouse()
        return x / self.u, y / self.u

    # ------------------------------------------------------------ textures
    def texture(self, img):
        """numpy premultiplied image or Surface -> Texture."""
        surf = img if isinstance(img, pygame.Surface) else paint.to_surface(img)
        t = video.Texture.from_surface(self.r, surf)
        t.blend_mode = BLEND
        return t

    def target(self, w, h):
        t = video.Texture(self.r, (max(1, int(w)), max(1, int(h))), target=True)
        return t

    def _build_prims(self):
        P = self.prims
        P["glow"] = self.texture(_soft_disc(256, 2.2))
        P["light"] = self.texture(_soft_disc(256, 1.6))
        P["disc"] = self.texture(_soft_disc(128, edge=0.06))
        P["soft"] = self.texture(_soft_disc(128, 1.0))
        P["ring"] = self.texture(_ring(256, 0.08))
        P["ring_soft"] = self.texture(_ring(256, 0.25))
        sh = _soft_disc(128, 1.4)
        sh[..., :3] = 0
        P["shadow"] = self.texture(sh)
        bar = np.zeros((16, 64, 4), np.float32)
        yy = np.abs(np.arange(16) - 7.5) / 8
        a = np.clip((1 - yy) * 2.2, 0, 1)
        bar[...] = a[:, None, None]
        P["bar"] = self.texture(bar)
        w = np.ones((4, 4, 4), np.float32)
        P["white"] = self.texture(w)
        grad = np.zeros((256, 4, 4), np.float32)
        g = np.linspace(1, 0, 256)[:, None]
        grad[...] = g[..., None]
        P["vgrad"] = self.texture(grad)
        hgrad = np.zeros((4, 256, 4), np.float32)
        hgrad[...] = np.linspace(1, 0, 256)[None, :, None]
        P["hgrad"] = self.texture(hgrad)
        hatch = np.zeros((64, 64, 4), np.float32)
        x, y = paint.grid(64)
        s = (np.sin((x + y) / 64 * math.tau * 4) > 0.3).astype(np.float32)
        hatch[...] = s[..., None]
        P["hatch"] = self.texture(hatch)
        vig = np.zeros((144, 256, 4), np.float32)
        x, y = paint.grid(256, 144)
        rr = np.hypot((x / 256 - 0.5) * 1.0, (y / 144 - 0.5) * 1.15) * 2
        va = np.clip((rr - 0.55) / 0.75, 0, 1) ** 1.6
        vig[..., 3] = va
        P["vignette"] = self.texture(vig)

    # ------------------------------------------------------------ primitive drawing (pixels)
    def clear(self, col=(0, 0, 0)):
        self.r.draw_color = (int(col[0]), int(col[1]), int(col[2]), 255)
        self.r.clear()

    def draw(self, tex, x, y, w, h, color=None, alpha=255, mode=BLEND, angle=0.0, origin=None,
             flip=False, src=None):
        tex.blend_mode = mode
        tex.color = color if color is not None else (255, 255, 255)
        tex.alpha = int(max(0, min(255, alpha)))
        if angle or origin or flip:
            tex.draw(srcrect=src, dstrect=(x, y, w, h), angle=angle, origin=origin, flip_x=flip)
        else:
            tex.draw(srcrect=src, dstrect=(x, y, w, h))

    def draw_c(self, tex, cx, cy, w, h, **kw):
        self.draw(tex, cx - w / 2, cy - h / 2, w, h, **kw)

    def glow(self, cx, cy, r, color, inten=1.0, tex="glow"):
        if inten <= 0.01 or r < 1:
            return
        self.draw(self.prims[tex], cx - r, cy - r, r * 2, r * 2, color=color,
                  alpha=255 * min(1.0, inten), mode=ADD)

    def disc(self, cx, cy, r, color, alpha=255, mode=BLEND):
        self.draw(self.prims["disc"], cx - r, cy - r, r * 2, r * 2, color=color, alpha=alpha, mode=mode)

    def ring(self, cx, cy, r, color, alpha=255, soft=False, mode=BLEND):
        self.draw(self.prims["ring_soft" if soft else "ring"], cx - r, cy - r, r * 2, r * 2,
                  color=color, alpha=alpha, mode=mode)

    def line(self, x1, y1, x2, y2, width, color, alpha=255, mode=BLEND):
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy)
        if ln < 0.5:
            return
        ang = math.degrees(math.atan2(dy, dx))
        w = max(1.0, width) * 1.6
        self.draw(self.prims["bar"], (x1 + x2) / 2 - ln / 2, (y1 + y2) / 2 - w / 2, ln, w,
                  color=color, alpha=alpha, mode=mode, angle=ang)

    def rect(self, x, y, w, h, color, alpha=255, mode=BLEND):
        self.draw(self.prims["white"], x, y, w, h, color=color, alpha=alpha, mode=mode)

    # ------------------------------------------------------------ rounded rects (9-slice)
    def _rtex(self, rp, filled, stroke=1.0):
        key = (int(rp), filled, round(stroke, 1))
        t = self.rounded.get(key)
        if t is None:
            img, n = _rounded(max(1, int(rp)), filled, stroke)
            t = (self.texture(img), n)
            self.rounded[key] = t
        return t

    def rrect(self, x, y, w, h, radius, color, alpha=255, filled=True, stroke=1.0, mode=BLEND):
        rp = max(1, min(int(radius), int(min(w, h) / 2) - 2))
        tex, n = self._rtex(rp, filled, stroke)
        tex.blend_mode = mode
        tex.color = color
        tex.alpha = int(max(0, min(255, alpha)))
        c = rp + 2                      # source corner size
        xs = [x, x + c, x + w - c, x + w]
        ys = [y, y + c, y + h - c, y + h]
        sx = [0, c, n - c, n]
        if w < 2 * c or h < 2 * c:
            tex.draw(dstrect=(x, y, w, h))
            return
        for j in range(3):
            for i in range(3):
                dw, dh = xs[i + 1] - xs[i], ys[j + 1] - ys[j]
                if dw <= 0 or dh <= 0:
                    continue
                if not filled and i == 1 and j == 1:
                    continue
                tex.draw(srcrect=(sx[i], sx[j], sx[i + 1] - sx[i], sx[j + 1] - sx[j]),
                         dstrect=(xs[i], ys[j], dw, dh))

    # ------------------------------------------------------------ text
    def font(self, px, bold=False, display=False):
        key = (px, bold, display)
        f = self.fonts.get(key)
        if f is None:
            from pathlib import Path
            root = Path(__file__).resolve().parent / "fonts"
            name = "Display-Bold.ttf" if display else ("Body-Semibold.ttf" if bold else "Body-Regular.ttf")
            f = pygame.font.Font(str(root / name), max(6, int(px)))
            self.fonts[key] = f
        return f

    def text_tex(self, s, px, color=(255, 255, 255), bold=False, display=False):
        key = (s, int(px), bold, display)
        t = self.txt.get(key)
        if t is None:
            surf = self.font(int(px), bold, display).render(str(s), True, (255, 255, 255))
            t = video.Texture.from_surface(self.r, surf)
            if len(self.txt) > 2500:
                self.txt.clear()
            self.txt[key] = t
        return t

    def text(self, s, x, y, px, color=(255, 255, 255), bold=False, anchor="topleft", alpha=255,
             shadow=False, display=False, mode=BLEND):
        if s == "" or s is None:
            return pygame.Rect(x, y, 0, 0)
        t = self.text_tex(s, px, color, bold, display)
        w, h = t.width, t.height
        rect = pygame.Rect(0, 0, w, h)
        setattr(rect, anchor, (int(x), int(y)))
        if shadow:
            o = max(1, px * 0.06)
            self.draw(t, rect.x, rect.y + o, w, h, color=(0, 0, 0), alpha=alpha * 0.6)
        self.draw(t, rect.x, rect.y, w, h, color=color, alpha=alpha, mode=mode)
        return rect

    def pill(self, w, h, radius, c1, c2):
        """Cached horizontal-gradient rounded rectangle texture (pixel size)."""
        w, h, radius = max(2, int(w)), max(2, int(h)), int(radius)
        key = ("pill", w, h, radius, tuple(c1), tuple(c2))
        t = self.rounded.get(key)
        if t is None:
            c = paint.Canvas(w, h)
            d = c.d_box(w / 2, h / 2, w / 2 - 0.5, h / 2 - 0.5, min(radius, h / 2 - 0.5, w / 2 - 0.5))
            a = c.cov(d)
            k = (c.X / w)[..., None]
            col = (np.array(c1, np.float32) / 255) * (1 - k) + (np.array(c2, np.float32) / 255) * k
            img = np.concatenate([col * a[..., None], a[..., None]], -1).astype(np.float32)
            t = self.texture(img)
            if len(self.rounded) > 400:
                self.rounded.clear()
            self.rounded[key] = t
        return t

    def text_size(self, s, px, bold=False, display=False):
        t = self.text_tex(s, px, (255, 255, 255), bold, display)
        return t.width, t.height

    def present(self):
        self.r.present()

    def screenshot(self):
        return self.r.to_surface()


class UI:
    """Design-unit drawing helpers on top of GPU (1 unit = gpu.u pixels)."""

    def __init__(self, gpu):
        self.g = gpu

    @property
    def u(self):
        return self.g.u

    def P(self, v):
        return v * self.g.u

    def panel(self, x, y, w, h, radius=16, fill=(12, 25, 29), alpha=205, border=(159, 130, 77),
              border_alpha=70, shadow=True, sheen=True):
        u = self.g.u
        X, Y, W, H, R = x * u, y * u, w * u, h * u, radius * u
        if shadow:
            sr = R + 14 * u
            self.g.rrect(X - 10 * u, Y - 4 * u, W + 20 * u, H + 22 * u, sr, (0, 0, 0), 70)
        self.g.rrect(X, Y, W, H, R, fill, alpha)
        if sheen and R < H * 0.4:
            self.g.draw(self.g.prims["vgrad"], X + R * 0.6, Y + 1, W - R * 1.2, H * 0.45,
                        color=(255, 255, 255), alpha=16)
        if border_alpha:
            self.g.rrect(X, Y, W, H, R, border, border_alpha, filled=False, stroke=max(1.0, u * 0.9))

    def rrect(self, x, y, w, h, radius, color, alpha=255, filled=True, stroke=1.0, mode=BLEND):
        u = self.g.u
        self.g.rrect(x * u, y * u, w * u, h * u, radius * u, color, alpha, filled,
                     stroke * u if not filled else 1.0, mode)

    def text(self, s, x, y, size, color=(232, 238, 255), bold=False, anchor="topleft", alpha=255,
             shadow=False, display=False):
        u = self.g.u
        r = self.g.text(s, x * u, y * u, size * u, color, bold, anchor, alpha, shadow, display)
        return pygame.Rect(r.x / u, r.y / u, r.w / u, r.h / u)

    def text_w(self, s, size, bold=False, display=False):
        return self.g.text_size(s, size * self.g.u, bold, display)[0] / self.g.u

    def glow(self, x, y, r, color, inten=1.0):
        u = self.g.u
        self.g.glow(x * u, y * u, r * u, color, inten)

    def tex(self, tex, x, y, w, h, **kw):
        u = self.g.u
        self.g.draw(tex, x * u, y * u, w * u, h * u, **kw)

    def tex_c(self, tex, cx, cy, w, h, **kw):
        u = self.g.u
        self.g.draw(tex, (cx - w / 2) * u, (cy - h / 2) * u, w * u, h * u, **kw)

    def line(self, x1, y1, x2, y2, width, color, alpha=255):
        u = self.g.u
        self.g.line(x1 * u, y1 * u, x2 * u, y2 * u, width * u, color, alpha)

    def rect(self, x, y, w, h, color, alpha=255, mode=BLEND):
        u = self.g.u
        self.g.rect(x * u, y * u, w * u, h * u, color, alpha, mode)

    def disc(self, cx, cy, r, color, alpha=255):
        u = self.g.u
        self.g.disc(cx * u, cy * u, r * u, color, alpha)

    def ring(self, cx, cy, r, color, alpha=255, soft=False):
        u = self.g.u
        self.g.ring(cx * u, cy * u, r * u, color, alpha, soft)

    def hgradient(self, x, y, w, h, c1, c2, radius=0, alpha=255):
        """Left-to-right two-colour gradient fill (used for primary buttons and bars)."""
        u = self.g.u
        X, Y, W, H = x * u, y * u, w * u, h * u
        if radius:
            qw = max(2, int(round(W / 4)) * 4)
            t = self.g.pill(qw, H, radius * u, c1, c2)
            self.g.draw(t, X, Y, W, H, alpha=alpha)
        else:
            self.g.rect(X, Y, W, H, c2, alpha)
            self.g.draw(self.g.prims["hgrad"], X, Y, W, H, color=c1, alpha=alpha)
