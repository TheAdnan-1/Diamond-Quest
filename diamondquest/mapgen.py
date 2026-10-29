"""Random, fair, tool-driven cave generation.

Layout
  * Rooms (chambers) carved out of solid rock, with soil patches and boulders that
    rest on soil (dig underneath and they fall: "don't rush").
  * Corridors join the rooms (minimum spanning tree + a few loops). Most corridors are
    blocked by an obstacle that needs a tool or costs life:
        cracked wall  -> Hammer
        snake         -> Hammer (or take the bite)
        rock gate     -> Push or Hook the boulder into a pit beside it
        fire          -> walk through and burn
  * Vaults: small sealed chambers off the rooms, behind a cracked wall or a snake.
    Red diamonds live in vaults, so the valuable ones always cost a tool.

The CSP (csp.py) then places the two starts, the exit and the 12 red diamonds
(one per vault) under fairness constraints.
"""
import random
import time
from collections import deque

from . import config as C
from .biomes import ORDER as BIOME_ORDER
from .csp import CSP, Constraint
from .world import World, FLOOR, DIRT, WALL, CRACKED, FIRE, EXIT, DIRS

N4 = list(DIRS.values())


def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def tool_dist(w, src, gates=()):
    """Step distances when every tool is available (walls block; normal rocks block, gate
    rocks don't)."""
    dist = {src: 0}
    q = deque([src])
    gate_cells = set(gates)
    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            n = (r + dr, c + dc)
            if n in dist or not w.inside(*n):
                continue
            if w.terrain[n[0]][n[1]] == WALL or n in w.statue_cells:
                continue
            if n in w.rock_at and n not in gate_cells:
                continue
            dist[n] = dist[(r, c)] + 1
            q.append(n)
    return dist


def free_dist(w, src):
    """Step distances with NO tools: cracked walls, all rocks and snakes block."""
    dist = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            n = (r + dr, c + dc)
            if n in dist or not w.inside(*n):
                continue
            t = w.terrain[n[0]][n[1]]
            if t in (WALL, CRACKED) or n in w.rock_at or n in w.snakes or n in w.statue_cells:
                continue
            dist[n] = dist[(r, c)] + 1
            q.append(n)
    return dist


def _reach(w, src, gates, crack=True, gates_open=True):
    """Cells reachable from src; crack=False treats cracked walls as solid, gates_open=False
    treats gate boulders as solid (other rocks always block)."""
    gate_cells = set(gates) if gates_open else set()
    seen = {src}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            n = (r + dr, c + dc)
            if n in seen or not w.inside(*n):
                continue
            t = w.terrain[n[0]][n[1]]
            if t == WALL or (t == CRACKED and not crack) or n in w.statue_cells:
                continue
            if n in w.rock_at and n not in gate_cells:
                continue
            seen.add(n)
            q.append(n)
    return seen


FAIL = {}   # why candidate maps were rejected (diagnostics)


def _safe_reach(w, src, gates, beams):
    """Cells reachable from src without damage you have no way to avoid. Fire blocks (it
    can't be put out). A lens beam blocks unless you reach it from the statue's body side,
    where you can hook the statue away first. Snakes, cracked walls and gate boulders are fine:
    a tool deals with them, and that is your choice."""
    gate_cells = set(gates)
    by_sid = {st.sid: st for st in w.statues.values()}
    hookable = {sid for sid, st in by_sid.items() if w.statue_hook_spots(st)}
    seen = {src}
    q = deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            n = (r + dr, c + dc)
            if n in seen or not w.inside(*n):
                continue
            t = w.terrain[n[0]][n[1]]
            if t in (WALL, FIRE) or n in w.statue_cells:
                continue
            if n in beams and not ((r, c) == (n[0] + 1, n[1]) and beams[n] in hookable):
                continue
            if n in w.rock_at and n not in gate_cells:
                continue
            seen.add(n)
            q.append(n)
    return seen


class Builder:
    def __init__(self, rng):
        self.rng = rng
        self.w = World()
        H, W = self.w.h, self.w.w
        for r in range(H):
            for c in range(W):
                self.w.terrain[r][c] = WALL
        self.rooms = []           # (r0, c0, r1, c1)
        self.room_of = {}         # cell -> room index
        self.corr = set()         # corridor cells (outside rooms)
        self.reserved = set()     # cells nothing else may touch (pits, vault rings...)
        self.no_items = set()     # cells where nothing may be placed
        self.gates = []           # rock gate cells (the rock positions)
        self.vaults = []          # list of (room index, [cells], door)
        self.obstacles = {"cracked": 0, "snake": 0, "gate": 0, "fire": 0, "open": 0}
        self.lens = []            # (guarded tunnel cell, statue cell)

    # ------------------------------------------------------------------ rooms
    def make_rooms(self, target=20):
        w, rng = self.w, self.rng
        H, W = w.h, w.w
        for _ in range(1500):
            if len(self.rooms) >= target:
                break
            rw, rh = rng.randint(5, 9), rng.randint(4, 7)
            r0, c0 = rng.randint(3, H - rh - 4), rng.randint(3, W - rw - 4)
            r1, c1 = r0 + rh - 1, c0 + rw - 1
            if any(not (r1 + 3 < a or b + 3 < r0 or c1 + 3 < cc or d + 3 < c0)
                   for a, cc, b, d in self.rooms):
                continue
            self.rooms.append((r0, c0, r1, c1))
        for i, (r0, c0, r1, c1) in enumerate(self.rooms):
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    w.terrain[r][c] = FLOOR
                    self.room_of[(r, c)] = i
            # irregular edges
            ring = [(r0 - 1, c) for c in range(c0 + 1, c1)] + [(r1 + 1, c) for c in range(c0 + 1, c1)] + \
                   [(r, c0 - 1) for r in range(r0 + 1, r1)] + [(r, c1 + 1) for r in range(r0 + 1, r1)]
            for cell in ring:
                if rng.random() < 0.22:
                    w.terrain[cell[0]][cell[1]] = FLOOR
                    self.room_of[cell] = i
        return len(self.rooms)

    def center(self, i):
        r0, c0, r1, c1 = self.rooms[i]
        return ((r0 + r1) // 2, (c0 + c1) // 2)

    # ------------------------------------------------------------------ corridors
    def make_corridors(self):
        """MST + a few loops. Returns [(plan, chunks)] where plan is 'gate' or 'wall'."""
        rng, n = self.rng, len(self.rooms)
        cs = [self.center(i) for i in range(n)]
        in_tree, edges = {0}, []
        while len(in_tree) < n:
            best = min(((_manh(cs[a], cs[b]), a, b) for a in in_tree for b in range(n) if b not in in_tree))
            edges.append((best[1], best[2]))
            in_tree.add(best[2])
        have = {tuple(sorted(e)) for e in edges}
        for a in range(n):
            near = sorted((b for b in range(n) if b != a), key=lambda b: _manh(cs[a], cs[b]))[:3]
            for b in near:
                e = tuple(sorted((a, b)))
                if e not in have and _manh(cs[a], cs[b]) < 22 and rng.random() < 0.2:
                    have.add(e)
                    edges.append(e)
        out = []
        for a, b in edges:
            plan = "gate" if rng.random() < 0.6 and abs(cs[a][1] - cs[b][1]) >= 5 else "wall"
            out.append((plan, self._carve(cs[a], cs[b], horizontal_first=(plan == "gate") or None)))
        return out

    def _carve(self, p, q, horizontal_first=None):
        w, rng = self.w, self.rng
        (r, c), (tr, tc) = p, q
        path = []
        hf = rng.random() < 0.5 if horizontal_first is None else horizontal_first
        if hf:
            mids = [(r, x) for x in _range(c, tc)] + [(y, tc) for y in _range(r, tr)]
        else:
            mids = [(y, c) for y in _range(r, tr)] + [(tr, x) for x in _range(c, tc)]
        for cell in mids:
            if cell not in path:
                path.append(cell)
        chunks, cur = [], []
        for cell in path:
            if cell in self.room_of:
                if cur:
                    chunks.append(cur)
                cur = []
                continue
            if w.terrain[cell[0]][cell[1]] == WALL:
                w.terrain[cell[0]][cell[1]] = FLOOR
            self.corr.add(cell)
            cur.append(cell)
        if cur:
            chunks.append(cur)
        return chunks

    def _straight(self, cell, horizontal):
        """True when cell is a plain corridor piece with walls on both perpendicular sides."""
        w = self.w
        r, c = cell
        if cell in self.room_of:
            return False
        sides = [(r - 1, c), (r + 1, c)] if horizontal else [(r, c - 1), (r, c + 1)]
        return all(w.terrain[s[0]][s[1]] == WALL and s not in self.corr for s in sides)

    def place_obstacles(self, plans):
        """Every corridor between chambers gets a barrier: rock gates (Push / Hook) where the
        geometry allows, otherwise cracked walls (Hammer); a few snakes and fires for variety."""
        w, rng = self.w, self.rng
        for plan, chunks in plans:
            chunks = [ch for ch in chunks if len(ch) >= 3]
            if not chunks:
                continue
            ch = max(chunks, key=len)
            if plan == "gate" and self._rock_gate(ch):
                self.obstacles["gate"] += 1
                others = [c for c in chunks if c is not ch]
            else:
                others = chunks
            for ch2 in others:
                mid = [x for x in ch2[1:-1] if (self._straight(x, True) or self._straight(x, False))
                       and x not in self.no_items]
                if not mid:
                    self.obstacles["open"] += 1
                    continue
                cell = rng.choice(mid)
                roll = rng.random()
                if roll < 0.82:
                    w.terrain[cell[0]][cell[1]] = CRACKED
                    self.obstacles["cracked"] += 1
                    # a longer corridor sometimes needs two blows
                    if len(ch2) >= 7 and rng.random() < 0.35:
                        far = [x for x in mid if _manh(x, cell) >= 3]
                        if far:
                            c2 = rng.choice(far)
                            w.terrain[c2[0]][c2[1]] = CRACKED
                            self.no_items.add(c2)
                            self.obstacles["cracked"] += 1
                elif roll < 0.93:
                    w.add_snake(*cell)
                    self.obstacles["snake"] += 1
                else:
                    w.terrain[cell[0]][cell[1]] = FIRE
                    self.obstacles["fire"] += 1
                self.no_items.add(cell)

    def _rock_gate(self, ch):
        """Find P1 M P2 in a horizontal stretch; dig a pit under P1 and P2; rock on M.
        Pushing (or hooking) the rock sideways drops it into a pit and opens the corridor."""
        w = self.w
        cands = []
        for i in range(len(ch) - 2):
            a, m, b = ch[i], ch[i + 1], ch[i + 2]
            if not (a[0] == m[0] == b[0] and m[1] - a[1] == 1 and b[1] - m[1] == 1 or
                    a[0] == m[0] == b[0] and a[1] - m[1] == 1 and m[1] - b[1] == 1):
                continue
            r = m[0]
            cols = sorted((a[1], m[1], b[1]))
            if r + 2 >= w.h - 1:
                continue
            ok = all(self._straight(x, True) for x in (a, m, b))
            block = [(r + 1, x) for x in range(cols[0] - 1, cols[2] + 2)] + \
                    [(r + 2, cols[0]), (r + 2, cols[2])]
            ok = ok and all(w.inside(*x) and w.terrain[x[0]][x[1]] == WALL and x not in self.corr
                            and x not in self.reserved for x in block)
            if ok:
                cands.append((r, cols))
        if not cands:
            return False
        r, cols = self.rng.choice(cands)
        for x in (cols[0], cols[2]):
            w.terrain[r + 1][x] = FLOOR
            self.no_items.add((r + 1, x))
        for x in range(cols[0] - 1, cols[2] + 2):
            for rr in (r - 1, r, r + 1, r + 2):
                self.reserved.add((rr, x))
        for x in cols:
            self.no_items.add((r, x))
        w.add_rock(r, cols[1])
        self.gates.append((r, cols[1]))
        return True

    # ------------------------------------------------------------------ lens statues
    def make_lens_gates(self, target=C.NUM_LENS_GATES):
        """A two-cell lens statue (head on top, body below) set into the side wall of a
        vertical tunnel. Its eye shines across the tunnel. From the tunnel cell beside the
        body you can hook it and turn it to face the rock; from the other side you can't
        reach the body, so you pay, wait for your rival to open it, or go round."""
        w, rng = self.w, self.rng
        cands = []
        for (r, c) in self.corr:
            if (r, c) in self.no_items or (r, c) in self.reserved or w.terrain[r][c] != FLOOR:
                continue
            if not self._straight((r, c), False):
                continue
            up, dn = (r - 1, c), (r + 1, c)
            if not all(x in self.corr and w.terrain[x[0]][x[1]] == FLOOR and x not in self.no_items
                       for x in (up, dn)):
                continue
            for s in (-1, 1):
                cands.append((r, c, s))
        rng.shuffle(cands)
        made = 0
        for r, c, s in cands:
            if made >= target:
                break
            head, body = (r, c + s), (r + 1, c + s)
            if not all(w.inside(*x) and w.terrain[x[0]][x[1]] == WALL and x not in self.corr
                       and x not in self.reserved and x not in self.room_of for x in (head, body)):
                continue
            if w.terrain[r][c - s] != WALL or (r, c) in self.no_items:
                continue
            for x in (head, body):
                w.terrain[x[0]][x[1]] = FLOOR
            w.add_statue(head[0], head[1], -s)
            for x in (head, body, (r, c), (r - 1, c), (r + 1, c)):
                self.no_items.add(x)
            for x in (head, body):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        self.reserved.add((x[0] + dr, x[1] + dc))
            self.lens.append(((r, c), head))
            made += 1
        return made

    def make_room_statues(self, target=3):
        """Statues standing against a chamber wall, their beam sweeping a whole row of the
        room. Hook the body from its row to drag the statue one cell out and turn it back
        toward the wall: the row is clear."""
        w, rng = self.w, self.rng
        doorish = set()
        for cell in list(self.corr) + [v[2] for v in self.vaults]:
            for dr, dc in N4 + [(0, 0)]:
                doorish.add((cell[0] + dr, cell[1] + dc))
        opts = []
        for (r0, c0, r1, c1) in self.rooms:
            if c1 - c0 < 5:
                continue
            for r in range(r0, r1):
                opts.append((r, c0, 1))
                opts.append((r, c1, -1))
        rng.shuffle(opts)
        made = 0
        for r, c, d in opts:
            if made >= target:
                break
            head, body = (r, c), (r + 1, c)
            outside = [(r, c - d), (r + 1, c - d)]
            if not all(w.terrain[x[0]][x[1]] == FLOOR and x not in w.rock_at and x not in w.snakes
                       and x not in doorish and x not in self.no_items for x in (head, body)):
                continue
            if not all(w.terrain[x[0]][x[1]] == WALL and x not in self.corr for x in outside):
                continue
            if (r - 1, c) in w.rock_at or any(abs(h[0] - r) < 3 for h in w.statues
                                               if self.room_of.get(h) == self.room_of[head]):
                continue
            # the beam must actually cover a stretch of the room
            if not all(w.terrain[r][c + d * k] in (FLOOR, DIRT) for k in (1, 2, 3)):
                continue
            w.add_statue(r, c, d)
            self.no_items.update((head, body))
            made += 1
        return made

    # ------------------------------------------------------------------ vaults
    def make_vaults(self, target=20):
        w, rng = self.w, self.rng
        options = []
        for i, (r0, c0, r1, c1) in enumerate(self.rooms):
            for r in range(r0, r1 + 1):
                options.append((i, (r, c0), (0, -1)))
                options.append((i, (r, c1), (0, 1)))
            for c in range(c0, c1 + 1):
                options.append((i, (r0, c), (-1, 0)))
                options.append((i, (r1, c), (1, 0)))
        rng.shuffle(options)
        per_room = {}
        for i, edge, (dr, dc) in options:
            if len(self.vaults) >= target:
                break
            if per_room.get(i, 0) >= 3 or edge not in self.room_of:
                continue
            door = (edge[0] + dr, edge[1] + dc)
            depth = rng.choice((2, 2, 3))
            # vault body: `depth` cells beyond the door, 2 wide perpendicular
            pr, pc = int(dc != 0), int(dr != 0)  # perpendicular unit
            body = [(door[0] + dr * k + pr * s, door[1] + dc * k + pc * s)
                    for k in range(1, depth + 1) for s in (0, 1)]
            cells = [door] + body
            ring = set()
            for (r, c) in body:
                for rr in (-1, 0, 1):
                    for cc in (-1, 0, 1):
                        ring.add((r + rr, c + cc))
            ring |= {(door[0] + pr, door[1] + pc), (door[0] - pr, door[1] - pc)}
            ring -= set(cells)
            if not all(w.inside(*x) and 1 <= x[0] < w.h - 1 and 1 <= x[1] < w.w - 1 for x in cells):
                continue
            if not all(w.inside(*x) and w.terrain[x[0]][x[1]] == WALL and x not in self.corr
                       and x not in self.reserved and x not in self.room_of for x in list(cells) + list(ring)):
                continue
            for x in body:
                w.terrain[x[0]][x[1]] = FLOOR
            if dr == 0 and rng.random() < 0.5:
                # boulder plug: push (or hook) it into the vault to get in
                w.terrain[door[0]][door[1]] = FLOOR
                w.add_rock(*door)
                self.gates.append(door)
                self.no_items.add(body[0])
                self.obstacles["vault_rock"] = self.obstacles.get("vault_rock", 0) + 1
            else:
                w.terrain[door[0]][door[1]] = CRACKED
                self.obstacles["vault_cracked"] = self.obstacles.get("vault_cracked", 0) + 1
            self.reserved |= set(cells) | ring
            self.no_items.add(door)
            self.no_items.add(edge)
            self.vaults.append((i, body, door))
            per_room[i] = per_room.get(i, 0) + 1
        return len(self.vaults)

    # ------------------------------------------------------------------ room interiors
    def furnish_rooms(self):
        w, rng = self.w, self.rng
        near_door = set()
        for cell in list(self.corr) + [v[2] for v in self.vaults]:
            for dr, dc in N4 + [(0, 0)]:
                near_door.add((cell[0] + dr, cell[1] + dc))
        for i, (r0, c0, r1, c1) in enumerate(self.rooms):
            cells = [x for x, k in self.room_of.items() if k == i]
            # soil blobs
            for _ in range(rng.randint(1, 3)):
                cr, cc = rng.randint(r0, r1), rng.randint(c0, c1)
                rad = rng.uniform(1.4, 2.6)
                for (r, c) in cells:
                    if (r - cr) ** 2 + (c - cc) ** 2 <= rad * rad and (r, c) not in near_door:
                        w.terrain[r][c] = DIRT
            # boulders resting on soil or ledges
            for (r, c) in cells:
                if (r, c) in near_door or (r + 1, c) in near_door or (r + 2, c) in near_door:
                    continue
                below = w.terrain[r + 1][c] if w.inside(r + 1, c) else WALL
                if below in (DIRT, WALL) and rng.random() < 0.1:
                    w.terrain[r][c] = FLOOR
                    w.add_rock(r, c)
        w.settle()
        # a rock that rolled into a corridor or door would cut the map; remove it. So would
        # one that can later drop down a dug-out shaft into a corridor or doorway.
        for cell in list(w.rock_at):
            if cell in self.gates:
                continue
            shaft = [cell]
            r, c = cell
            while w.inside(r + 1, c) and w.terrain[r + 1][c] in (FLOOR, DIRT):
                r += 1
                shaft.append((r, c))
            if any(x in self.corr or x in near_door for x in shaft):
                rid = w.rock_at.pop(cell)
                del w.rocks[rid]


def _range(a, b):
    step = 1 if b >= a else -1
    return range(a, b + step, step)


_CACHE = {}


def generate(seed=None, log=None):
    """A fair cave for this seed (the same seed always gives the same cave; results are
    cached, so asking again is instant)."""
    seed = random.randrange(1 << 30) if seed is None else seed
    hit = _CACHE.get(seed)
    if hit is not None:
        return hit[0].clone(), dict(hit[1])
    w, meta = _generate(seed)
    if len(_CACHE) > 8:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[seed] = (w.clone(), dict(meta))
    return w, meta


def _generate(seed):
    rng = random.Random(seed)
    t0 = time.time()
    for attempt in range(1, 80):
        b = Builder(rng)
        if b.make_rooms() < 14:
            continue
        plans = b.make_corridors()
        b.place_obstacles(plans)
        if b.make_vaults() < C.NUM_RED:
            continue
        b.make_lens_gates()
        b.furnish_rooms()
        b.make_room_statues()
        res = _place_with_csp(b, rng)
        if res is None:
            continue
        meta = _populate(b, res, rng)
        if meta is None:
            continue
        meta.update(seed=seed, attempts=attempt, seconds=round(time.time() - t0, 2),
                    biome=BIOME_ORDER[seed % len(BIOME_ORDER)])
        return b.w, meta
    raise RuntimeError("map generation failed")


def _place_with_csp(b, rng):
    w = b.w
    cache = {}

    def dist(a):
        if a not in cache:
            cache[a] = tool_dist(w, a, b.gates)
        return cache[a]

    beams = w.lit(edges=True)
    statue_rooms = {b.room_of[x] for x in w.statue_cells if x in b.room_of}
    statue_rooms |= {b.room_of[x] for x in beams if x in b.room_of}

    def safe(p):
        r, c = p
        return (w.terrain[r][c] in (FLOOR, DIRT) and p not in w.rock_at and p not in w.snakes
                and p not in w.statue_cells and p not in beams and b.room_of.get(p) not in statue_rooms
                and (r - 1, c) not in w.rock_at and (r - 2, c) not in w.rock_at and p not in b.no_items)
    room_cells = [p for p in b.room_of if safe(p)]
    if len(room_cells) < 100:
        return None
    S = rng.sample(room_cells, min(80, len(room_cells)))
    E = [p for p in rng.sample(room_cells, min(160, len(room_cells))) if w.terrain[p[0]][p[1]] == FLOOR]
    slots = []
    for _room, body, door in b.vaults:
        opts = [x for x in body[-2:] if x not in b.no_items] or [body[-1]]
        slots.append(rng.choice(opts))
    reds = [f"R{i}" for i in range(C.NUM_RED)]
    variables = ["S1", "S2", "E"] + reds
    domains = {"S1": S, "S2": S, "E": E}
    for v in reds:
        domains[v] = list(slots)
    BIG = 10 ** 6
    cons = [
        Constraint(("S1", "S2"), lambda a, c: a < c and _manh(a, c) >= 24 and dist(a).get(c, BIG) >= 32
                   and b.room_of[a] != b.room_of[c], "starts far apart"),
        Constraint(("S1", "S2", "E"), lambda a, c, e: (
            dist(a).get(e, BIG) >= 20 and dist(c).get(e, BIG) >= 20
            and abs(dist(a).get(e, BIG) - dist(c).get(e, -BIG)) <= 3), "fair exit"),
    ]
    for v in reds:
        cons.append(Constraint(("S1", "S2", v), lambda a, c, x: (
            dist(a).get(x, BIG) >= 6 and dist(c).get(x, BIG) >= 6
            and abs(dist(a).get(x, BIG) - dist(c).get(x, -BIG)) <= 30), f"{v} fair"))
    for i in range(len(reds)):
        for j in range(i + 1, len(reds)):
            cons.append(Constraint((reds[i], reds[j]), lambda x, y: x != y, "one red per vault"))

    def balanced(a, c, *xs):
        return abs(sum(dist(a).get(x, BIG) - dist(c).get(x, -BIG) for x in xs)) <= 16
    cons.append(Constraint(["S1", "S2"] + reds, balanced, "total red balance"))
    csp = CSP(variables, domains, cons, rng)
    sol = csp.solve(max_nodes=8000)
    if sol is None:
        return None
    sol["_stats"] = {"nodes": csp.nodes, "backtracks": csp.backtracks, "constraints": len(cons),
                     "variables": len(variables)}
    sol["_dist"] = dist
    sol["_slots"] = slots
    return sol


def _populate(b, sol, rng):
    w = b.w
    dist = sol["_dist"]
    s1, s2, ex = sol["S1"], sol["S2"], sol["E"]
    d1, d2 = dist(s1), dist(s2)
    reds = [sol[f"R{i}"] for i in range(C.NUM_RED)]
    w.starts = [s1, s2]
    for s in (s1, s2):
        w.terrain[s[0]][s[1]] = FLOOR
    w.terrain[ex[0]][ex[1]] = EXIT
    w.exit_pos = ex
    used = {s1, s2, ex} | set(b.no_items)
    for p in reds:
        w.terrain[p[0]][p[1]] = FLOOR
        w.gems[p] = "r"
        used.add(p)

    def free(p):
        return (w.inside(*p) and p not in used and p not in w.rock_at and p not in w.snakes
                and p not in w.statue_cells and (p[0] - 1, p[1]) not in w.rock_at
                and w.terrain[p[0]][p[1]] in (DIRT, FLOOR) and p in d1 and p in d2)

    # violets: two in each start room (an easy beginning), some in spare vaults, rest spread
    gems = list(reds)
    vault_cells = [c for _r, body, _d in b.vaults for c in body]
    for s, dd in ((s1, d1), (s2, d2)):
        room = [p for p, k in b.room_of.items() if k == b.room_of[s] and free(p) and 3 <= dd.get(p, 0) <= 8]
        rng.shuffle(room)
        for p in room[:2]:
            w.gems[p] = "v"
            used.add(p)
            gems.append(p)
    spare = [c for c in vault_cells if free(c)]
    rng.shuffle(spare)
    for p in spare[:6]:
        w.gems[p] = "v"
        used.add(p)
        gems.append(p)
    balance = sum(d1[p] - d2[p] for p in gems)
    cands = [p for p in list(b.room_of) + list(b.corr) if free(p) and d1[p] >= 4 and d2[p] >= 4]
    tries = 0
    while sum(1 for g in w.gems.values() if g == "v") < C.NUM_VIOLET and tries < 4000:
        tries += 1
        sample = [p for p in rng.sample(cands, min(24, len(cands))) if free(p) and
                  all(_manh(p, g) >= 3 for g in gems)]
        if not sample:
            continue
        p = min(sample, key=lambda q: abs(balance + d1[q] - d2[q]) + rng.random() * 3)
        balance += d1[p] - d2[p]
        w.terrain[p[0]][p[1]] = FLOOR
        w.gems[p] = "v"
        used.add(p)
        gems.append(p)

    # extra snakes in rooms (guarding diamonds), up to the configured total; never in a
    # start chamber (fair starts: nothing there can hurt you)
    start_rooms = {b.room_of[s1], b.room_of[s2]}
    room_free = [p for p in b.room_of if free(p) and d1[p] > 6 and d2[p] > 6 and _manh(p, ex) > 2
                 and b.room_of[p] not in start_rooms]
    rng.shuffle(room_free)
    for p in room_free:
        if len(w.snakes) >= C.NUM_SNAKES:
            break
        if any(_manh(p, g) == 1 for g in gems) or rng.random() < 0.15:
            if free(p):
                w.terrain[p[0]][p[1]] = FLOOR
                w.add_snake(*p)
                used.add(p)
    # fire: small patches inside rooms, away from the starts
    fire = sum(1 for r in range(w.h) for c in range(w.w) if w.terrain[r][c] == FIRE)
    seeds = [p for p in b.room_of if free(p) and d1[p] > 7 and d2[p] > 7 and b.room_of[p] not in start_rooms]
    rng.shuffle(seeds)
    for sd in seeds:
        if fire >= C.NUM_FIRE:
            break
        cluster = [sd]
        for _ in range(rng.randint(1, 2)):
            base = rng.choice(cluster)
            dr, dc = rng.choice(N4)
            cluster.append((base[0] + dr, base[1] + dc))
        for q in cluster:
            if fire < C.NUM_FIRE and free(q) and q in b.room_of and _manh(q, ex) > 1 \
                    and b.room_of[q] not in start_rooms:
                w.terrain[q[0]][q[1]] = FIRE
                used.add(q)
                fire += 1
    w.settle()

    # ---- validation: everything reachable with tools, most reds need tools
    for s in (s1, s2):
        d = tool_dist(w, s, b.gates)
        if any(g not in d for g in w.gems) or ex not in d:
            return None
    locked = 0
    for g in reds:
        if g not in free_dist(w, s1) and g not in free_dist(w, s2):
            locked += 1
    if locked < C.NUM_RED - 1:
        return None
    # the tools must matter from BOTH starts: count diamonds that need the hammer / a push
    need_hammer, need_push = [], []
    for s in (s1, s2):
        no_hammer = _reach(w, s, b.gates, crack=False, gates_open=True)
        no_push = _reach(w, s, b.gates, crack=True, gates_open=False)
        need_hammer.append(sum(1 for g in w.gems if g not in no_hammer))
        need_push.append(sum(1 for g in w.gems if g not in no_push))
    if min(need_hammer) < len(w.gems) * 0.5 or min(need_push) < 8:
        key = "hammer" if min(need_hammer) < len(w.gems) * 0.5 else "push"
        FAIL[key] = FAIL.get(key, 0) + 1
        return None
    # fair, safe starts: nothing in a start chamber can hurt you, and from each start you can
    # walk out to a good share of the cave without taking damage you can't choose to avoid
    beams = w.lit(edges=True)
    harm = [(fr, fc) for fr in range(w.h) for fc in range(w.w) if w.terrain[fr][fc] == FIRE]
    for s in (s1, s2):
        room = b.room_of[s]
        cells = [p for p, k in b.room_of.items() if k == room]
        if any(p in beams or p in w.snakes or p in w.statue_cells or w.terrain[p[0]][p[1]] == FIRE
               for p in cells):
            FAIL["start_room"] = FAIL.get("start_room", 0) + 1
            return None
    safe1, safe2 = _safe_reach(w, s1, b.gates, beams), _safe_reach(w, s2, b.gates, beams)
    g1 = sum(1 for g in w.gems if g in safe1)
    g2 = sum(1 for g in w.gems if g in safe2)
    if min(g1, g2) < 12 or abs(g1 - g2) > 6:
        FAIL["safe_reach"] = FAIL.get("safe_reach", 0) + 1
        return None
    del harm
    st = sol["_stats"]
    return {"csp_nodes": st["nodes"], "csp_backtracks": st["backtracks"],
            "csp_constraints": st["constraints"], "csp_variables": st["variables"],
            "balance": balance, "snakes": len(w.snakes), "fire": fire,
            "exit_dist": (d1[ex], d2[ex]), "rooms": len(b.rooms), "vaults": len(b.vaults),
            "gates": len(b.gates), "locked_reds": locked, "obstacles": dict(b.obstacles),
            "lens": len(w.statues),
            "need_hammer": min(need_hammer), "need_push": min(need_push),
            "safe_gems": (g1, g2)}
