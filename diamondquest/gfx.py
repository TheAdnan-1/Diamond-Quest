"""Drawing helpers, palette, fonts and procedurally generated sprites (no image files needed)."""
import math
import random

import pygame

TILE = 48

# ---------------------------------------------------------------- palette
BG = (8, 11, 22)
INK = (241, 234, 217)
INK_DIM = (169, 183, 182)
INK_FAINT = (116, 138, 142)
PANEL = (14, 20, 38)
PANEL_EDGE = (255, 255, 255)

HUMAN = (100, 224, 211)
HUMAN_DARK = (18, 110, 150)
AI = (235, 184, 96)
AI_DARK = (150, 92, 20)

VIOLET = (170, 110, 255)
VIOLET_LIGHT = (222, 196, 255)
VIOLET_DARK = (96, 46, 190)
RED = (255, 62, 98)
RED_LIGHT = (255, 170, 184)
RED_DARK = (160, 18, 52)

FLOOR_A = (15, 19, 33)
FLOOR_B = (17, 22, 37)
DIRT_A = (66, 46, 64)
DIRT_B = (82, 58, 78)
WALL_A = (40, 48, 76)
WALL_B = (58, 68, 104)
WALL_TOP = (96, 110, 156)
CRACK = (255, 157, 60)
FIRE_1 = (255, 92, 40)
FIRE_2 = (255, 176, 66)
FIRE_3 = (255, 240, 170)
SNAKE = (84, 224, 122)
SNAKE_DARK = (28, 120, 70)
ROCK = (118, 126, 158)
ROCK_DARK = (62, 68, 94)
ROCK_LIGHT = (176, 184, 214)
DANGER = (255, 70, 90)
GOOD = (90, 240, 160)


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_col(a, b, t):
    return tuple(int(lerp(a[i], b[i], t)) for i in range(3))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def ease_out(t):
    return 1 - (1 - t) ** 3


def ease_in(t):
    return t * t


def ease_in_out(t):
    return 3 * t * t - 2 * t * t * t


def ease_back(t):
    c = 1.70158
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2


# ---------------------------------------------------------------- fonts
_FONTS = {}
_FONT_NAMES = "inter,segoeui,sfprodisplay,helveticaneue,roboto,ubuntu,dejavusans,arial"


def font(size, bold=False):
    key = (size, bold)
    if key not in _FONTS:
        f = pygame.font.SysFont(_FONT_NAMES, size, bold=bold)
        _FONTS[key] = f
    return _FONTS[key]


_TEXT = {}


def text(s, size, color=INK, bold=False):
    key = (s, size, color, bold)
    surf = _TEXT.get(key)
    if surf is None:
        surf = font(size, bold).render(str(s), True, color)
        if len(_TEXT) > 3000:
            _TEXT.clear()
        _TEXT[key] = surf
    return surf


def blit_text(dst, s, size, pos, color=INK, bold=False, anchor="topleft", alpha=255,
              shadow=False):
    surf = text(s, size, color, bold)
    rect = surf.get_rect(**{anchor: pos})
    if shadow:
        sh = text(s, size, (0, 0, 0), bold)
        sh = sh.copy()
        sh.set_alpha(int(140 * alpha / 255))
        dst.blit(sh, rect.move(0, 2))
    if alpha < 255:
        surf = surf.copy()
        surf.set_alpha(alpha)
    dst.blit(surf, rect)
    return rect


def gradient_text(s, size, c1, c2, bold=True):
    base = font(size, bold).render(s, True, (255, 255, 255))
    w, h = base.get_size()
    grad = pygame.Surface((w, h), pygame.SRCALPHA)
    for x in range(w):
        col = lerp_col(c1, c2, x / max(1, w - 1))
        pygame.draw.line(grad, col + (255,), (x, 0), (x, h))
    grad.blit(base, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return grad


# ---------------------------------------------------------------- shapes
def rrect(dst, color, rect, radius=10, alpha=255, width=0):
    rect = pygame.Rect(rect)
    if alpha >= 255 and len(color) == 3:
        pygame.draw.rect(dst, color, rect, width, border_radius=radius)
        return
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    col = color[:3] + ((color[3] if len(color) > 3 else alpha),)
    pygame.draw.rect(s, col, s.get_rect(), width, border_radius=radius)
    dst.blit(s, rect.topleft)


def glass(dst, rect, radius=16, alpha=190, edge=34, tint=PANEL):
    rect = pygame.Rect(rect)
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, tint + (alpha,), s.get_rect(), border_radius=radius)
    # soft top sheen
    sheen = pygame.Surface((rect.w, rect.h // 2), pygame.SRCALPHA)
    for y in range(sheen.get_height()):
        a = int(18 * (1 - y / max(1, sheen.get_height())))
        pygame.draw.line(sheen, (255, 255, 255, a), (0, y), (rect.w, y))
    mask = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
    sheen_full = pygame.Surface(rect.size, pygame.SRCALPHA)
    sheen_full.blit(sheen, (0, 0))
    sheen_full.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    s.blit(sheen_full, (0, 0))
    pygame.draw.rect(s, PANEL_EDGE + (edge,), s.get_rect(), 1, border_radius=radius)
    dst.blit(s, rect.topleft)


_GLOW = {}


def glow(radius, color, intensity=1.0):
    """Radial glow sprite for additive blending."""
    key = (int(radius), color, round(intensity, 2))
    g = _GLOW.get(key)
    if g is None:
        r = max(2, int(radius))
        g = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        for i in range(r, 0, -1):
            t = i / r
            a = (1 - t) ** 2 * intensity
            col = tuple(int(c * a) for c in color)
            pygame.draw.circle(g, col + (255,), (r, r), i)
        g = g.convert() if pygame.display.get_surface() else g
        g.set_colorkey((0, 0, 0))
        _GLOW[key] = g
    return g


def add_glow(dst, center, radius, color, intensity=1.0):
    g = glow(radius, color, intensity)
    dst.blit(g, (center[0] - g.get_width() // 2, center[1] - g.get_height() // 2),
             special_flags=pygame.BLEND_ADD)


_LIGHT = {}


def light_hole(radius):
    """Alpha mask used to cut light out of the darkness layer."""
    r = int(radius)
    s = _LIGHT.get(r)
    if s is None:
        s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        for i in range(r, 0, -2):
            t = i / r
            a = int(255 * (1 - t) ** 1.4)
            pygame.draw.circle(s, (0, 0, 0, a), (r, r), i)
        _LIGHT[r] = s
    return s


def star(dst, center, size, color, rot=0.0, width=0):
    pts = []
    for i in range(8):
        ang = rot + i * math.pi / 4
        rad = size if i % 2 == 0 else size * 0.32
        pts.append((center[0] + math.cos(ang) * rad, center[1] + math.sin(ang) * rad))
    pygame.draw.polygon(dst, color, pts, width)


def heart(dst, center, size, color):
    x, y = center
    r = size * 0.32
    pygame.draw.circle(dst, color, (int(x - r), int(y - r * 0.3)), int(r))
    pygame.draw.circle(dst, color, (int(x + r), int(y - r * 0.3)), int(r))
    pygame.draw.polygon(dst, color, [(x - 2 * r, y), (x + 2 * r, y), (x, y + size * 0.62)])
    pygame.draw.polygon(dst, color, [(x - 1.95 * r, y - r * 0.15), (x + 1.95 * r, y - r * 0.15),
                                     (x, y + size * 0.6)])


# ---------------------------------------------------------------- sprites
def _hash(r, c, k=0):
    return (r * 73856093 ^ c * 19349663 ^ k * 83492791) & 0xFFFFFFFF


def gem_sprite(kind, size):
    """A faceted diamond. kind 'v' violet or 'r' red."""
    base, light, dark = ((VIOLET, VIOLET_LIGHT, VIOLET_DARK) if kind == "v"
                         else (RED, RED_LIGHT, RED_DARK))
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    W = size

    def P(x, y):
        return (x * W, y * W)
    top_l, top_r = P(0.32, 0.18), P(0.68, 0.18)
    gl, gr = P(0.12, 0.40), P(0.88, 0.40)
    mid = P(0.5, 0.40)
    bot = P(0.5, 0.90)
    # crown facets
    pygame.draw.polygon(s, light, [top_l, top_r, mid])
    pygame.draw.polygon(s, lerp_col(light, base, 0.45), [top_l, mid, gl])
    pygame.draw.polygon(s, lerp_col(light, base, 0.25), [top_r, gr, mid])
    # pavilion facets
    pygame.draw.polygon(s, base, [gl, mid, bot])
    pygame.draw.polygon(s, dark, [mid, gr, bot])
    pygame.draw.polygon(s, lerp_col(base, dark, 0.4), [P(0.31, 0.40), mid, bot])
    # outline + highlights
    pygame.draw.polygon(s, lerp_col(light, (255, 255, 255), 0.6),
                        [top_l, top_r, gr, bot, gl], max(1, W // 24))
    pygame.draw.line(s, (255, 255, 255), P(0.36, 0.22), P(0.46, 0.22), max(1, W // 20))
    pygame.draw.line(s, (255, 255, 255), P(0.2, 0.40), P(0.3, 0.52), max(1, W // 28))
    return s


def rock_sprite(size):
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    cx, cy, r = size / 2, size / 2 + 1, size * 0.42
    pts = []
    rng = random.Random(7)
    for i in range(11):
        a = i / 11 * math.tau
        rr = r * (0.9 + rng.random() * 0.14)
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr * 0.94))
    pygame.draw.polygon(s, ROCK_DARK, pts)
    inner = [(cx + (x - cx) * 0.86 - 2, cy + (y - cy) * 0.82 - 3) for x, y in pts]
    pygame.draw.polygon(s, ROCK, inner)
    hi = [(cx + (x - cx) * 0.45 - 6, cy + (y - cy) * 0.38 - 8) for x, y in pts]
    pygame.draw.polygon(s, ROCK_LIGHT, hi)
    for (dx, dy, rr) in ((0.18, 0.15, 0.08), (-0.12, 0.22, 0.05), (0.2, -0.05, 0.045)):
        pygame.draw.circle(s, ROCK_DARK, (int(cx + dx * size), int(cy + dy * size)),
                           max(1, int(rr * size)))
    pygame.draw.polygon(s, (30, 34, 50), pts, 2)
    return s


def tool_icon(name, size, color):
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    u = size / 24
    if name == "hammer":
        pygame.draw.line(s, color, (6 * u, 20 * u), (15 * u, 9 * u), max(2, int(2.6 * u)))
        head = [(10 * u, 4 * u), (19 * u, 2 * u), (22 * u, 9 * u), (14 * u, 12 * u)]
        pygame.draw.polygon(s, color, head)
    elif name == "push":
        pygame.draw.circle(s, color, (int(15 * u), int(12 * u)), int(6.5 * u), max(2, int(2 * u)))
        pygame.draw.line(s, color, (2 * u, 12 * u), (7 * u, 12 * u), max(2, int(2.4 * u)))
        pygame.draw.polygon(s, color, [(6 * u, 8 * u), (9.5 * u, 12 * u), (6 * u, 16 * u)])
    elif name == "hook":
        pygame.draw.line(s, color, (4 * u, 4 * u), (4 * u, 13 * u), max(2, int(2.2 * u)))
        pygame.draw.arc(s, color, (4 * u - 1, 8 * u, 12 * u, 12 * u), math.pi, math.tau,
                        max(2, int(2.4 * u)))
        pygame.draw.line(s, color, (16 * u, 14 * u), (16 * u, 10 * u), max(2, int(2.4 * u)))
        pygame.draw.polygon(s, color, [(13 * u, 11 * u), (16 * u, 6 * u), (19 * u, 11 * u)])
        for i in range(3):
            pygame.draw.circle(s, color, (int(4 * u), int((2 + i * 0) * u)), max(1, int(1.6 * u)))
    elif name == "life":
        heart(s, (size / 2, size / 2 - u), size * 0.8, color)
    return s


class TileArt:
    """Static terrain art drawn once per cell into the world cache."""

    def __init__(self):
        self.size = TILE

    def draw(self, dst, world, r, c, x, y):
        from .world import DIRT, WALL, CRACKED, FIRE, EXIT
        T = TILE
        t = world.terrain[r][c]
        h = _hash(r, c)
        rect = pygame.Rect(x, y, T, T)
        if t == WALL or t == CRACKED:
            pygame.draw.rect(dst, (12, 14, 26), rect)
            open_below = r + 1 < world.h and world.terrain[r + 1][c] not in (WALL, CRACKED)
            open_above = r - 1 >= 0 and world.terrain[r - 1][c] not in (WALL, CRACKED)
            open_l = c - 1 >= 0 and world.terrain[r][c - 1] not in (WALL, CRACKED)
            open_r = c + 1 < world.w and world.terrain[r][c + 1] not in (WALL, CRACKED)
            # two staggered bricks per tile
            shade = (h % 9) / 30
            face = lerp_col(WALL_A, WALL_B, 0.35 + shade)
            for i, (bx, by, bw, bh) in enumerate(((1, 1, T - 2, T // 2 - 2),
                                                   (1, T // 2 + 1, T // 2 - 2, T // 2 - 2),
                                                   (T // 2 + 1, T // 2 + 1, T // 2 - 2, T // 2 - 2))):
                br = pygame.Rect(x + bx, y + by, bw, bh)
                col = lerp_col(face, WALL_TOP, 0.12 if i == 0 else 0.0)
                pygame.draw.rect(dst, col, br, border_radius=4)
                pygame.draw.line(dst, lerp_col(col, (255, 255, 255), 0.18), (br.x + 3, br.y + 1),
                                 (br.right - 4, br.y + 1), 2)
                pygame.draw.line(dst, lerp_col(col, (0, 0, 0), 0.35), (br.x + 3, br.bottom - 1),
                                 (br.right - 4, br.bottom - 1), 2)
            if open_above:
                pygame.draw.rect(dst, WALL_TOP, (x + 1, y + 1, T - 2, 4), border_radius=2)
            if open_below:
                pygame.draw.rect(dst, (8, 9, 18), (x, y + T - 6, T, 6))
            if open_l:
                pygame.draw.line(dst, (8, 9, 18), (x, y), (x, y + T), 2)
            if open_r:
                pygame.draw.line(dst, (8, 9, 18), (x + T - 1, y), (x + T - 1, y + T), 2)
            if t == CRACKED:
                rng = random.Random(h)
                pts = [(x + T * 0.5 + rng.uniform(-4, 4), y + 4)]
                for i in range(1, 6):
                    pts.append((x + T * 0.5 + rng.uniform(-12, 12), y + 4 + i * (T - 10) / 5))
                pygame.draw.lines(dst, (60, 28, 10), False, pts, 6)
                pygame.draw.lines(dst, CRACK, False, pts, 2)
                br = pts[2]
                pygame.draw.line(dst, CRACK, br, (br[0] + rng.choice((-10, 10)), br[1] + 6), 2)
            return
        # floor-ish base
        base = FLOOR_A if (r + c) % 2 else FLOOR_B
        pygame.draw.rect(dst, base, rect)
        pygame.draw.rect(dst, lerp_col(base, (120, 140, 220), 0.06), rect, 1)
        if h % 4 == 0:
            pygame.draw.circle(dst, lerp_col(base, (255, 255, 255), 0.07),
                               (x + 8 + h % 30, y + 10 + (h >> 4) % 28), 2)
        if h % 7 == 0:
            pygame.draw.line(dst, lerp_col(base, (0, 0, 0), 0.4), (x + 10 + h % 20, y + 30),
                             (x + 18 + h % 20, y + 34), 2)
        if t == DIRT:
            pygame.draw.rect(dst, DIRT_A, rect.inflate(-2, -2), border_radius=8)
            pygame.draw.rect(dst, DIRT_B, (x + 3, y + 3, T - 6, T - 12), border_radius=8)
            rng = random.Random(h)
            for _ in range(7):
                px, py = x + rng.randint(6, T - 6), y + rng.randint(6, T - 8)
                pygame.draw.circle(dst, lerp_col(DIRT_A, (20, 10, 20), 0.4), (px, py),
                                   rng.randint(1, 3))
            for _ in range(3):
                px, py = x + rng.randint(6, T - 6), y + rng.randint(6, T - 14)
                pygame.draw.circle(dst, lerp_col(DIRT_B, (255, 220, 240), 0.18), (px, py), 1)
        elif t == FIRE:
            pygame.draw.rect(dst, (46, 26, 30), rect.inflate(-4, -4), border_radius=10)
            pygame.draw.rect(dst, (90, 40, 30), rect.inflate(-14, -14), border_radius=10)
        elif t == EXIT:
            pygame.draw.rect(dst, (16, 20, 34), rect.inflate(-4, -4), border_radius=12)
