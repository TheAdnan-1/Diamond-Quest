"""AI vs AI extras (used only in that mode; the other modes never touch this file).

* ArenaAgent: the normal AI, plus a decisive exit: as soon as walking out would win
  (after paying the life the route costs, and even if the rival grabs its best diamond
  on the way), it heads straight for the exit.
* ROUND_CAP: an AI vs AI match ends after this many rounds; totals decide.
* Same-brain matches (Minimax vs Minimax, MCTS vs MCTS) are played on a mirrored cave
  (left half reflected onto the right, exit on the centre line). The explorer that moves
  first decides with the real AI; the other answers each turn with the mirror image of
  that turn. Both halves are the same cave, so every round ends level: a draw.
"""
import random
import threading
import time

from . import config as C
from . import mapgen
from .ai import AIAgent
from .fuzzy import RISK_SYSTEM, risk_label
from .logic import danger_analysis
from .search import GridGraph, dijkstra_map
from .world import World, WALL, CRACKED, FIRE, EXIT, FLOOR, DIRT, DIRS

ROUND_CAP = 250            # default; the player can change it in Settings
CAP_OPTIONS = (100, 150, 250, 350, 500)
# Balanced-cave quality (each explorer gets one half): enough diamonds, enough of them
# cheap to reach, and an exit with no fire toll on the way
# tour_*: a greedy walk that picks up TOUR_GEMS diamonds must fit in this many turns / life
STRICT = dict(min_gems=16, min_cheap=12, gem_life=20, exit_life=5, tour_turns=95, tour_life=45)
RELAXED = dict(min_gems=13, min_cheap=10, gem_life=25, exit_life=10, tour_turns=125, tour_life=70)
TOUR_GEMS = 12
START_TRIES = 8
SEARCH_SECONDS = 20            # after this, settle for the best "relaxed" cave found
_CACHE = {}

MIRROR_DIR = {"left": "right", "right": "left", "up": "up", "down": "down", "wait": "wait", None: None}


# ===================================================================== decisive exit
class ArenaAgent(AIAgent):
    """arena=True: playing in a Balanced cave, where each explorer has a sealed half.
    The rival can never be reached there, so mine blockades are pointless, and with the
    scores level the search sees 'leave for a draw' and 'one more diamond' as almost equal,
    which made it flip between them every turn. So in a Balanced cave the AI commits to the
    target it picks and follows it through."""

    STALE_TURNS = 6          # no progress for this long: give the target up
    BAN_TURNS = 30
    EASY_LIFE = 10           # before walking out, still grab diamonds this cheap ...
    EASY_TURNS = 20          # ... and this close

    def __init__(self, pid=1, brain="minimax", seed=None, arena=False):
        super().__init__(pid, brain, seed=seed)
        self.arena = arena
        self.commit = None
        self.commit_best = None
        self.commit_stale = 0
        self.no_go = {}
        self.turn_count = 0

    def _lockdown(self, world, st, skip=None):
        return None if self.arena else super()._lockdown(world, st, skip)

    def _try_mine(self, world, here, nxt):
        return None if self.arena else super()._try_mine(world, here, nxt)

    def decide(self, world):
        t0 = time.time()
        esc = self._escape(world, t0)
        if esc is not None:
            if self.arena:
                self.commit = "EXIT"
            return esc
        tool, tdir, move, info = super().decide(world)
        if not self.arena or getattr(world.players[self.pid], "frozen", 0):
            return tool, tdir, move, info
        return self._steady(world, tool, tdir, move, info, t0)

    # ------------------------------------------------------------ commitment
    def _graph(self, world, to_exit):
        la = danger_analysis(world, self.pid)
        priced = self._rock_holds(world)
        hard = {c for c, k in la["kinds"].items() if ("Crush" in k and c not in priced) or "LateBite" in k}
        return GridGraph(world, self.pid, life_weight=1.0, danger=hard, allow_exit=to_exit), la

    def _steady(self, world, tool, tdir, move, info, t0):
        me = world.players[self.pid]
        here = (me.r, me.c)
        self.turn_count += 1
        self.no_go = {g: t for g, t in self.no_go.items() if t > self.turn_count}
        exit_ok = world.exit_open_for(self.pid) and world.exit_pos is not None

        want = info.get("target")
        if info.get("target_kind") == "exit":
            want = "EXIT"
        elif want not in world.gems:
            want = None
        if want is None and exit_ok:
            want = "EXIT"                     # nothing worth a trip: walk out (a draw if level)

        dm = dijkstra_map(GridGraph(world, self.pid, life_weight=1.0, allow_exit=exit_ok), here)

        def reachable(t):
            cell = world.exit_pos if t == "EXIT" else t
            return cell in dm and me.life - dm[cell][2] > 0

        if want == "EXIT":
            # leaving with an easy diamond still lying next to the way out wastes it
            easy = [g for g in world.gems if g in dm and g not in self.no_go
                    and dm[g][2] <= self.EASY_LIFE and dm[g][1] <= self.EASY_TURNS
                    and me.life - dm[g][2] > 20]
            if easy:
                want = min(easy, key=lambda g: (dm[g][1], dm[g]))

        c = self.commit
        if c == "EXIT" and not exit_ok:
            c = None
        if c not in (None, "EXIT") and c not in world.gems:
            c = None                          # collected
        if c is not None and not reachable(c):
            if c != "EXIT":
                self.no_go[c] = self.turn_count + self.BAN_TURNS
            c = None
        if c is not None:
            d = dm[world.exit_pos if c == "EXIT" else c][1]
            if self.commit_best is None or d < self.commit_best:
                self.commit_best, self.commit_stale = d, 0
            else:
                self.commit_stale += 1
                if self.commit_stale >= self.STALE_TURNS:
                    if c != "EXIT":
                        self.no_go[c] = self.turn_count + self.BAN_TURNS
                    c = None
        if c is None and want is not None and want not in self.no_go and reachable(want):
            c = want
            self.commit_best, self.commit_stale = None, 0
        self.commit = c
        if c is None:
            return tool, tdir, move, info
        cell = world.exit_pos if c == "EXIT" else c
        if c == want and info.get("target") == cell and move not in (None, "wait"):
            return tool, tdir, move, info     # the search already walks there
        graph, la = self._graph(world, c == "EXIT")
        t2, d2, m2, path, why = self._tactics(world, graph, cell, la["danger"], la["kinds"])
        if m2 in (None, "wait") and t2 is None:
            return tool, tdir, move, info
        info = dict(info)
        info.update(target=cell, path=path, think=time.time() - t0,
                    target_kind="exit" if c == "EXIT" else ("red" if world.gems.get(c) == "r" else "violet"),
                    why=why if c != "EXIT" else "heading out")
        self.last = info
        return t2, d2, m2, info

    def _escape(self, world, t0):
        me = world.players[self.pid]
        op = world.players[1 - self.pid]
        if getattr(me, "frozen", 0) or not world.exit_open_for(self.pid) or world.exit_pos is None:
            return None
        here, ex = (me.r, me.c), world.exit_pos
        dm = dijkstra_map(GridGraph(world, self.pid, life_weight=1.0, allow_exit=True), here)
        if ex not in dm:
            return None
        _, turns, lc = dm[ex]
        if me.life - lc <= 0:
            return None
        mine = me.points + me.life - lc
        theirs = op.total
        if mine <= theirs:
            return None
        # what could the rival still add while we walk out?
        threat = 0
        window = turns - 1
        if window > 0 and op.alive and not op.exited:
            om = dijkstra_map(GridGraph(world, op.pid, life_weight=1.0, allow_exit=False), (op.r, op.c))
            for g, kind in world.gems.items():
                if g in om and om[g][1] <= window and op.life - om[g][2] > 0:
                    val = C.RED_POINTS if kind == "r" else C.VIOLET_POINTS
                    threat = max(threat, val - om[g][2])
        if mine <= theirs + threat:
            return None
        la = danger_analysis(world, self.pid)
        priced = self._rock_holds(world)
        hard = {c for c, k in la["kinds"].items() if ("Crush" in k and c not in priced) or "LateBite" in k}
        graph = GridGraph(world, self.pid, life_weight=1.0, danger=hard, allow_exit=True)
        tool, tdir, move, path, why = self._tactics(world, graph, ex, la["danger"], la["kinds"])
        if move in (None, "wait") and tool is None:
            return None
        fz = RISK_SYSTEM.evaluate(life=me.life, lead=me.total - op.total)
        info = {"brain": self.brain, "risk": fz["value"], "mood": risk_label(fz["value"]), "fuzzy": fz,
                "logic": {k: la[k] for k in ("facts", "rules", "derived")},
                "target": ex, "target_kind": "exit", "exit_reason": "leaving now wins",
                "search": {}, "tour": [], "candidates": len(world.gems), "path": path,
                "why": f"escaping: leaving now wins by {mine - theirs - threat}",
                "think": time.time() - t0}
        self.last = info
        return tool, tdir, move, info


# ===================================================================== mirrored cave
def axis(w):
    return (w.w - 2) // 2          # 50 wide: column 24 is the centre line, 49 stays solid


def mcell(w, cell):
    a = axis(w)
    return (cell[0], 2 * a - cell[1])


def mirror_turn(tool, tdir, move):
    return tool, MIRROR_DIR.get(tdir, tdir), MIRROR_DIR.get(move, move)


def mirror_info(w, info):
    out = dict(info or {})
    for k in ("path", "tour"):
        if out.get(k):
            out[k] = [mcell(w, c) if isinstance(c, tuple) else c for c in out[k]]
    if isinstance(out.get("target"), tuple):
        out["target"] = mcell(w, out["target"])
    return out


def _reach(w, src):
    """Cells reachable with every tool (walls and statues block; rocks can be shoved)."""
    seen = {src}
    stack = [src]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS.values():
            n = (r + dr, c + dc)
            if n in seen or not w.inside(*n):
                continue
            if w.terrain[n[0]][n[1]] == WALL or n in w.statue_cells:
                continue
            seen.add(n)
            stack.append(n)
    return seen


def _mirror_of(src, start_hint, q=STRICT):
    """Build the mirrored cave from src's left half, or None if it isn't playable."""
    a = axis(src)
    W, H = src.w, src.h
    w = World(W, H)
    for r in range(H):
        for c in range(W):
            if c < a:
                t = src.terrain[r][c]
                if t == EXIT:
                    t = FLOOR
                w.terrain[r][c] = t
            elif c > a and c <= 2 * a:
                t = src.terrain[r][2 * a - c]
                w.terrain[r][c] = FLOOR if t == EXIT else t
            else:
                w.terrain[r][c] = WALL
    for r in range(H):
        w.terrain[r][0] = w.terrain[r][2 * a] = w.terrain[r][W - 1] = WALL
    for c in range(W):
        w.terrain[0][c] = w.terrain[H - 1][c] = WALL
    left = lambda cell: 0 < cell[1] < a
    for side in (0, 1):
        f = (lambda cell: cell) if side == 0 else (lambda cell: mcell(w, cell))
        for rid, (r, c) in sorted(src.rocks.items()):
            if left((r, c)):
                w.add_rock(*f((r, c)))
        for (r, c), kind in sorted(src.gems.items()):
            if left((r, c)):
                w.gems[f((r, c))] = kind
        for (r, c) in sorted(src.snakes):
            if left((r, c)):
                w.add_snake(*f((r, c)))
        for (r, c), st in sorted(src.statues.items()):
            if left((r, c)):
                d = st.d if side == 0 else -st.d
                w.add_statue(*f((r, c)), d)
    # exit on the centre line, opening onto both halves, away from beams and hookable things
    lit = w.lit()
    rows = []
    for r in range(2, H - 2):
        if w.terrain[r][a - 1] not in (FLOOR, DIRT) or (r, a - 1) in w.statue_cells:
            continue
        if any((r, c) in lit for c in range(a - 6, a)):
            continue
        busy = any((r, c) in w.rock_at or (r, c) in w.snakes or (r, c) in w.statue_cells
                   for c in range(a - 5, a))
        if any(st.r == r or st.r + 1 == r for st in w.statues.values()):
            continue
        if busy:
            continue
        rows.append(r)
    if not rows:
        return None
    # where to start: the original start chamber first, then a few other safe spots
    rng = random.Random(start_hint)
    lit0 = w.lit(edges=True)

    def safe(cell):
        r, c = cell
        return (1 < c < a - 1 and w.terrain[r][c] == FLOOR and cell not in w.rock_at
                and cell not in w.snakes and cell not in w.gems and cell not in w.statue_cells
                and cell not in lit0 and w.terrain[r - 1][c] == WALL
                and not any((r + dr, c + dc) in w.snakes for dr, dc in DIRS.values()))
    starts = [st for st in src.starts if safe(st)]
    others = [(r, c) for r in range(2, H - 2) for c in range(2, a - 1) if safe((r, c))]
    rng.shuffle(others)
    for s0 in (starts + others)[:START_TRIES]:
        got = _settle_half(w.clone(), s0, rows, q, rng)
        if got is not None:
            return got
    return None


def _settle_half(w, s0, rows, q, rng):
    """Given the mirrored cave and a start, prune what can't be reached, place the exit
    and check the cave plays well. Returns the finished cave or None."""
    a = axis(w)
    # price every route the way the AI does (fire, snakes, beams, cracked walls, rocks)
    probe = w.clone()
    probe.setup_players(s0, mcell(w, s0), 0, mines=C.MINES_PER_PLAYER)
    dm = dijkstra_map(GridGraph(probe, 0, life_weight=1.0, allow_exit=False), s0)
    # diamonds the explorer can never get to are taken out (on both sides), so the AI
    # never wanders after something it cannot have
    for g in [g for g in w.gems if g[1] < a and g not in dm]:
        del w.gems[g]
        w.gems.pop(mcell(w, g), None)
    mine = [g for g in w.gems if g[1] < a]
    cheap = [g for g in mine if dm[g][2] <= q["gem_life"]]
    if len(mine) < q["min_gems"] or len(cheap) < q["min_cheap"]:
        return None
    # the exit: easy to walk to (no fire toll), but not right next to the start
    good = [r for r in rows if (r, a - 1) in dm and dm[(r, a - 1)][2] <= q["exit_life"]
            and dm[(r, a - 1)][1] >= 8]
    if not good:
        return None
    good.sort(key=lambda r: -dm[(r, a - 1)][1])
    er = good[min(len(good) - 1, rng.randrange(max(1, (len(good) + 1) // 2)))]
    w.terrain[er][a] = EXIT
    if w.terrain[er][a - 1] == DIRT:
        w.terrain[er][a - 1] = FLOOR
        w.terrain[er][a + 1] = FLOOR
    w.exit_pos = (er, a)
    lit = w.lit(edges=True)
    if s0 in lit or w.terrain[s0[0]][s0[1]] == FIRE:
        return None
    # compact enough to play well: the diamonds mustn't be strung out along long, burning tunnels
    tt, tl = _greedy_tour(w, s0)
    if tt > q["tour_turns"] or tl > q["tour_life"]:
        return None
    w.starts = [s0, mcell(w, s0)]
    _decoys(w, rng)
    w.settle()
    return w


def _greedy_tour(w, start):
    """Turns and life a nearest-diamond walk from the start needs for TOUR_GEMS diamonds."""
    probe = w.clone()
    probe.setup_players(start, mcell(w, start), 0, mines=C.MINES_PER_PLAYER)
    a = axis(w)
    left = {g for g in w.gems if g[1] < a}
    here, turns, life = start, 0, 0
    for _ in range(TOUR_GEMS):
        dm = dijkstra_map(GridGraph(probe, 0, life_weight=1.0, allow_exit=False), here)
        cand = [g for g in left if g in dm]
        if not cand:
            return 999, 999
        g = min(cand, key=lambda g: dm[g][0])
        turns += dm[g][1]
        life += dm[g][2]
        left.discard(g)
        here = g
    return turns, life


def _decoys(w, rng):
    """Sealed chambers and tunnels carved into solid rock, placed at random (so not mirrored).
    None of them touches any cell an explorer can ever reach (statues included, as they
    can be hooked aside), so play stays perfectly mirrored, but the cave no longer looks
    like a mirror image."""
    reach = _reach(w, w.starts[0]) | _reach(w, w.starts[1])
    near = set(reach)
    for (r, c) in list(w.statue_cells):
        for dr in (-2, -1, 0, 1, 2):
            for dc in (-2, -1, 0, 1, 2):
                near.add((r + dr, c + dc))
    for (r, c) in list(reach):
        for dr, dc in DIRS.values():
            near.add((r + dr, c + dc))
    made = set()
    for _ in range(2500):
        if rng.random() < 0.3:                        # a tunnel
            n = rng.randint(4, 10)
            hh, ww = (1, n) if rng.random() < 0.6 else (n, 1)
        else:                                         # a chamber
            hh, ww = rng.randint(2, 6), rng.randint(3, 8)
        r0, c0 = rng.randint(1, w.h - hh - 1), rng.randint(1, w.w - ww - 1)
        cells = [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)]
        if any(cell in near or w.terrain[cell[0]][cell[1]] != WALL and cell not in made
               or cell in w.rock_at or cell in w.statue_cells or cell in w.gems or cell in w.snakes
               for cell in cells):
            continue
        if any(not (1 <= r < w.h - 1 and 1 <= c < w.w - 1) for r, c in cells):
            continue
        if len(made) > 0.17 * w.w * w.h:
            break
        tunnel = hh == 1 or ww == 1
        corners = {(r0, c0), (r0, c0 + ww - 1), (r0 + hh - 1, c0), (r0 + hh - 1, c0 + ww - 1)}
        for r, c in cells:
            if not tunnel and (r, c) in corners and rng.random() < 0.5:
                continue
            roll = rng.random()
            edge = r in (r0, r0 + hh - 1) or c in (c0, c0 + ww - 1)
            w.terrain[r][c] = DIRT if (edge and roll < 0.3) or roll < 0.08 else (FIRE if roll > 0.97 else FLOOR)
            made.add((r, c))
        if not tunnel:
            floor = [(r, c) for r, c in cells if w.terrain[r][c] == FLOOR and (r, c) not in w.rock_at
                     and (r, c) not in w.snakes]
            base = [cell for cell in floor if cell[0] == r0 + hh - 1]
            if base and rng.random() < 0.45:
                w.add_rock(*rng.choice(base))
            floor = [cell for cell in floor if cell not in w.rock_at]
            if floor and rng.random() < 0.25:
                w.add_snake(*rng.choice(floor))


def _hflip(src):
    """The same cave seen in a mirror (used when its start chamber is on the right)."""
    W = src.w
    f = lambda cell: (cell[0], W - 1 - cell[1])
    w = World(W, src.h)
    w.terrain = [row[::-1] for row in src.terrain]
    for rid, (r, c) in sorted(src.rocks.items()):
        w.add_rock(*f((r, c)))
    w.gems = {f(c): k for c, k in src.gems.items()}
    for c in sorted(src.snakes):
        w.add_snake(*f(c))
    for c, st in sorted(src.statues.items()):
        w.add_statue(*f(c), -st.d)
    w.starts = [f(s) for s in src.starts]
    w.exit_pos = f(src.exit_pos) if src.exit_pos else None
    return w


def reachable(w):
    """Cells either explorer could ever get to (the mirrored part of the cave)."""
    return _reach(w, w.starts[0]) | _reach(w, w.starts[1])


_BUILDING = {}
_LOCK = threading.Lock()


def mirrored_map(seed):
    """A same-looking cave for this seed, mirrored so a same-brain match ends level.
    Safe to call from several threads: a cave being built is waited for, not rebuilt."""
    with _LOCK:
        hit = _CACHE.get(seed)
        ev = None
        if hit is None:
            ev = _BUILDING.get(seed)
            owner = ev is None
            if owner:
                ev = _BUILDING[seed] = threading.Event()
    if hit is None and not owner:
        ev.wait()
        with _LOCK:
            hit = _CACHE.get(seed)
    if hit is None:
        try:
            hit = _build_mirrored(seed)
        finally:
            with _LOCK:
                if hit is not None:
                    _CACHE[seed] = hit
                    while len(_CACHE) > 6:
                        _CACHE.pop(next(iter(_CACHE)))
                _BUILDING.pop(seed, None)
            if ev is not None:
                ev.set()
    return hit[0].clone(), dict(hit[1])


def _build_mirrored(seed):
    t0 = time.time()
    meta0 = None
    fallback = None
    for k in range(60):
        s = seed if k == 0 else (seed * 1103515245 + 12345 * k) % (1 << 30)
        src, meta = mapgen.generate(s)
        if meta0 is None:
            meta0 = meta               # the cave's name, biome and story come from the seed
        for cand in (src, _hflip(src)):
            w = _mirror_of(cand, s)
            if w is not None:
                return w, dict(meta0)
            if fallback is None:
                fallback = _mirror_of(cand, s, RELAXED)
        if fallback is not None and time.time() - t0 > SEARCH_SECONDS:
            break
    if fallback is not None:
        return fallback, dict(meta0)
    return mapgen.generate(seed)      # never seen in testing; plays as a normal map


def safe_leader_turn(world, pid, tool, tdir, move):
    """In a mirrored match, keep the first mover from ending the game unevenly on its own
    turn (its twin would answer next turn, but the game would already be over)."""
    def ends_bad(t, td, mv):
        sim = world.clone()
        try:
            sim.take_turn(pid, t, td, mv)
        except ValueError:
            return True
        return sim.over and sim.winner != -1

    if not ends_bad(tool, tdir, move):
        return tool, tdir, move
    if tool and not ends_bad(None, None, move):
        return None, None, move
    for mv in ("wait", "up", "down", "left", "right"):
        if not ends_bad(None, None, mv):
            return None, None, mv
    return tool, tdir, move
