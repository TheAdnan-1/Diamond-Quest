"""Game rules: the world state and everything that happens in one turn.

Turn structure (per player):  [optional single tool]  ->  [move 1 cell | wait]
Then: snake stun expiry / bites, win/loss checks.

Gravity is instant: after every tool and every move, any boulder that has lost what held it
up drops straight down (or rolls off another boulder) until something stops it. If the only
thing under it is an explorer, it lands on them (CRUSH_DAMAGE) and rests on their head until
they step out from under it. So pushing or hooking a rock over a gap drops it at once, and
digging out the soil (or taking the diamond) a rock sits on brings it down on you.

Lens statues are two cells tall: a head with a lens eye on top of a stone body. The eye
shines a beam straight ahead (left or right) along the head's row, through every cell until
a wall, cracked wall, soil, boulder or another statue stops it. Stepping into the beam, or
having it swing onto you, costs LENS_DAMAGE life and freezes you for FREEZE_TURNS of your
turns. Hook the statue by its BODY, from the body's row (out of the beam): it slides one cell
toward you (or turns in place if it is next to you, or has no room) and faces the other way.
"""
from . import config as C

FLOOR, DIRT, WALL, CRACKED, FIRE, EXIT = range(6)
TERRAIN_NAMES = ["floor", "dirt", "wall", "cracked wall", "fire", "exit"]

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
TOOLS = ("hammer", "push", "hook", "mine")
TOOL_COST = {"hammer": C.HAMMER_COST, "push": C.PUSH_COST, "hook": C.HOOK_COST, "mine": 0}


class Player:
    __slots__ = ("pid", "name", "r", "c", "life", "points", "violets", "reds",
                 "alive", "exited", "mines", "frozen")

    def __init__(self, pid, name, r, c):
        self.pid, self.name, self.r, self.c = pid, name, r, c
        self.life = C.START_LIFE
        self.points = 0
        self.violets = 0
        self.reds = 0
        self.alive = True
        self.exited = False
        self.mines = 0
        self.frozen = 0           # own turns still to skip after a lens hit

    @property
    def gems(self):
        return self.violets + self.reds

    @property
    def total(self):
        return self.points + max(0, self.life)

    def copy(self):
        p = Player(self.pid, self.name, self.r, self.c)
        for s in Player.__slots__:
            setattr(p, s, getattr(self, s))
        return p


class Snake:
    __slots__ = ("sid", "r", "c", "stuns")

    def __init__(self, sid, r, c):
        self.sid, self.r, self.c = sid, r, c
        self.stuns = {}           # pid -> half-turn index at whose end the stun expires

    def copy(self):
        s = Snake(self.sid, self.r, self.c)
        s.stuns = dict(self.stuns)
        return s


class Statue:
    __slots__ = ("sid", "r", "c", "d")

    def __init__(self, sid, r, c, d):
        self.sid, self.r, self.c, self.d = sid, r, c, d   # d: -1 faces left, +1 faces right

    def copy(self):
        return Statue(self.sid, self.r, self.c, self.d)


class World:
    def __init__(self, w=C.GRID_W, h=C.GRID_H):
        self.w, self.h = w, h
        self.terrain = [[DIRT] * w for _ in range(h)]
        self.rocks = {}           # rock id -> [r, c]
        self.rock_at = {}         # (r, c) -> rock id
        self.falling = set()      # rock ids that moved down last tick (map building only)
        self.resting = {}         # rock id -> pids it is pinned on (it fell on their heads)
        self.gems = {}            # (r, c) -> 'v' | 'r'
        self.snakes = {}          # (r, c) -> Snake
        self.players = []
        self.exit_pos = None
        self.starts = []
        self.t = 0                # half-turn counter
        self.current = 0
        self.first = 0
        self.rounds = 0
        self.over = False
        self.winner = None        # pid, or -1 for draw
        self.end_reason = ""
        self._next_rock = 0
        self._next_snake = 0
        self.tool_used = False
        self.mines = {}
        self.mines_enabled = False
        self.statues = {}         # head cell (r, c) -> Statue; its body is at (r + 1, c)
        self.statue_cells = {}    # head AND body cells -> head cell
        self._next_statue = 0

    # ------------------------------------------------------------------ setup
    def add_statue(self, r, c, d):
        s = Statue(self._next_statue, r, c, d)
        self._next_statue += 1
        self.statues[(r, c)] = s
        self.statue_cells[(r, c)] = (r, c)
        self.statue_cells[(r + 1, c)] = (r, c)
        return s

    def _move_statue(self, st, r, c):
        del self.statues[(st.r, st.c)]
        del self.statue_cells[(st.r, st.c)]
        del self.statue_cells[(st.r + 1, st.c)]
        st.r, st.c = r, c
        self.statues[(r, c)] = st
        self.statue_cells[(r, c)] = (r, c)
        self.statue_cells[(r + 1, c)] = (r, c)

    def add_rock(self, r, c):
        rid = self._next_rock
        self._next_rock += 1
        self.rocks[rid] = [r, c]
        self.rock_at[(r, c)] = rid
        return rid

    def add_snake(self, r, c):
        s = Snake(self._next_snake, r, c)
        self._next_snake += 1
        self.snakes[(r, c)] = s
        return s

    def setup_players(self, human_start, ai_start, first, mines=0):
        """pid 0 = human, pid 1 = AI. mines = how many mines each player starts with."""
        self.players = [Player(0, "You", *human_start), Player(1, "AI", *ai_start)]
        for pl in self.players:
            pl.mines = mines
        self.mines = {}
        self.mines_enabled = mines > 0
        self.current = self.first = first
        self.t = 0
        self.rounds = 0
        self.over = False
        self.winner = None
        self.end_reason = ""
        self.tool_used = False

    def clone(self):
        w = World.__new__(World)
        w.w, w.h = self.w, self.h
        w.terrain = [row[:] for row in self.terrain]
        w.rocks = {k: v[:] for k, v in self.rocks.items()}
        w.rock_at = dict(self.rock_at)
        w.falling = set(self.falling)
        w.resting = {k: set(v) for k, v in getattr(self, "resting", {}).items()}
        w.gems = dict(self.gems)
        w.mines = dict(getattr(self, "mines", {}))
        w.mines_enabled = getattr(self, "mines_enabled", False)
        w.snakes = {k: s.copy() for k, s in self.snakes.items()}
        w.statues = {k: s.copy() for k, s in getattr(self, "statues", {}).items()}
        w.statue_cells = dict(getattr(self, "statue_cells", {}))
        w._next_statue = getattr(self, "_next_statue", 0)
        w.players = [p.copy() for p in self.players]
        for a in ("exit_pos", "starts", "t", "current", "first", "rounds", "over",
                  "winner", "end_reason", "_next_rock", "_next_snake", "tool_used"):
            setattr(w, a, getattr(self, a))
        return w

    # ---------------------------------------------------------------- queries
    def inside(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w

    def players_at(self, r, c):
        return [p for p in self.players if p.alive and not p.exited and p.r == r and p.c == c]

    def exit_open_for(self, pid):
        return self.players[pid].gems >= C.EXIT_UNLOCK_GEMS

    def lit(self, edges=False):
        """Cells lit by statue beams: {cell: statue id}. With edges=True also the soil cell
        where a beam ends: dig into it and the beam follows you in."""
        out = {}
        for (r, c), s in getattr(self, "statues", {}).items():
            cc = c + s.d
            while self.inside(r, cc):
                t = self.terrain[r][cc]
                if t == DIRT and edges and (r, cc) not in self.rock_at:
                    out[(r, cc)] = s.sid
                    break
                if t in (WALL, CRACKED, DIRT) or (r, cc) in self.rock_at or (r, cc) in self.statue_cells:
                    break
                out[(r, cc)] = s.sid
                cc += s.d
        return out

    def statue_hook_spots(self, st):
        """Cells in the statue's BODY row from which it can be hooked (either side)."""
        out = []
        r = st.r + 1
        for side in (-1, 1):
            for k in range(1, C.HOOK_RANGE + 1):
                cell = (r, st.c + side * k)
                if not self.inside(*cell):
                    break
                t = self.terrain[cell[0]][cell[1]]
                if t in (WALL, CRACKED, FIRE) or cell in self.rock_at or cell in self.statue_cells \
                        or cell in self.snakes:
                    break
                out.append(cell)
                if not self.is_empty_floor(*cell, ignore_players=True):
                    break                     # the hook line ends at anything else
        return out

    def is_empty_floor(self, r, c, ignore_players=False):
        """Nothing at all on a floor cell (used by rocks, push, hook line)."""
        if not self.inside(r, c) or self.terrain[r][c] != FLOOR:
            return False
        if (r, c) in self.rock_at or (r, c) in self.gems or (r, c) in self.snakes \
                or (r, c) in getattr(self, "statue_cells", {}):
            return False
        if not ignore_players and self.players_at(r, c):
            return False
        return True

    def can_enter(self, pid, r, c):
        if not self.inside(r, c):
            return False
        t = self.terrain[r][c]
        if t in (WALL, CRACKED):
            return False
        if (r, c) in self.rock_at or (r, c) in getattr(self, "statue_cells", {}):
            return False
        if t == EXIT and not self.exit_open_for(pid):
            return False
        return True

    def snake_stunned(self, s):
        return bool(s.stuns)

    # ------------------------------------------------------------ tool checks
    def tool_target(self, pid, tool, d):
        """Return (ok, info) where info describes what the tool would do."""
        p = self.players[pid]
        if d not in DIRS:
            return False, "bad direction"
        if getattr(p, "frozen", 0):
            return False, "you're frozen"
        dr, dc = DIRS[d]
        tr, tc = p.r + dr, p.c + dc
        if tool == "hammer":
            if not self.inside(tr, tc):
                return False, "nothing to hammer"
            if (tr, tc) in self.snakes:
                return True, ("snake", (tr, tc))
            if self.terrain[tr][tc] == CRACKED:
                return True, ("wall", (tr, tc))
            if self.terrain[tr][tc] == FIRE:
                return False, "fire can't be hammered"
            return False, "nothing to hammer there"
        if tool == "push":
            if dr != 0:
                return False, "rocks can only be pushed sideways"
            if (tr, tc) not in self.rock_at:
                return False, "no rock to push"
            br, bc = tr, tc + dc
            if not self.is_empty_floor(br, bc):
                return False, "rock is blocked"
            return True, ("rock", (tr, tc), (br, bc))
        if tool == "hook":
            if dr != 0:
                return False, "the hook only works sideways"
            for k in range(1, C.HOOK_RANGE + 1):
                cr, cc = p.r, p.c + dc * k
                if not self.inside(cr, cc):
                    return False, "nothing in hook range"
                if (cr, cc) in getattr(self, "statue_cells", {}):
                    head = self.statue_cells[(cr, cc)]
                    if head == (cr, cc):
                        return False, "the hook can't grab a statue's head - hook its body"
                    # it slides one cell toward you (or just turns: next to you, or no room)
                    dst = head
                    if k > 1 and self.is_empty_floor(head[0], head[1] - dc):
                        dst = (head[0], head[1] - dc)
                    return True, ("statue", head, dst)
                if (cr, cc) in self.rock_at or (cr, cc) in self.snakes:
                    if k == 1:
                        return False, "already next to you"
                    kind = "rock" if (cr, cc) in self.rock_at else "snake"
                    return True, (kind, (cr, cc), (p.r, p.c + dc))
                if self.terrain[cr][cc] == FIRE:
                    return False, "the hook can't pass or grab fire"
                if not self.is_empty_floor(cr, cc):
                    return False, "hook line is blocked"
            return False, "nothing in hook range"
        if tool == "mine":
            if p.mines <= 0:
                return False, "no mines left"
            if not self.inside(tr, tc) or self.terrain[tr][tc] not in (FLOOR, DIRT):
                return False, "can't place a mine there"
            cell = (tr, tc)
            if cell in self.rock_at or cell in self.gems or cell in self.snakes or \
                    cell in self.mines or self.players_at(tr, tc) or cell in self.statue_cells:
                return False, "that cell is occupied"
            return True, ("mine", cell)
        return False, "unknown tool"

    def valid_tools(self, pid):
        out = []
        for tool in TOOLS:
            for d in DIRS:
                ok, info = self.tool_target(pid, tool, d)
                if ok:
                    out.append((tool, d, info))
        return out

    def valid_moves(self, pid, world=None):
        p = self.players[pid]
        if getattr(p, "frozen", 0):
            return []
        return [d for d, (dr, dc) in DIRS.items() if self.can_enter(pid, p.r + dr, p.c + dc)]

    # ------------------------------------------------------------ the turn
    def _damage(self, p, amount, why, ev):
        p.life -= amount
        ev.append({"type": "damage", "pid": p.pid, "amount": amount, "why": why,
                   "pos": (p.r, p.c)})
        if p.life <= 0 and p.alive:
            p.alive = False
            ev.append({"type": "death", "pid": p.pid, "pos": (p.r, p.c), "why": why})

    def apply_tool(self, pid, tool, d, ev):
        ok, info = self.tool_target(pid, tool, d)
        if not ok:
            raise ValueError(info)
        p = self.players[pid]
        kind = info[0]
        if tool == "hammer":
            pos = info[1]
            if kind == "snake":
                self.snakes[pos].stuns[pid] = self.t + 2
                ev.append({"type": "hammer", "pid": pid, "target": "snake", "pos": pos,
                           "sid": self.snakes[pos].sid})
            else:
                self.terrain[pos[0]][pos[1]] = FLOOR
                ev.append({"type": "hammer", "pid": pid, "target": "wall", "pos": pos})
        elif tool == "push":
            src, dst = info[1], info[2]
            rid = self.rock_at.pop(src)
            self.rocks[rid] = list(dst)
            self.rock_at[dst] = rid
            self.falling.discard(rid)
            ev.append({"type": "push", "pid": pid, "rock": rid, "from": src, "to": dst})
        elif tool == "hook":
            src, dst = info[1], info[2]
            if kind == "statue":
                st = self.statues[src]
                self._move_statue(st, *dst)
                st.d = -st.d
                ev.append({"type": "hook", "pid": pid, "obj": "statue", "sid": st.sid,
                           "from": src, "to": dst, "facing": st.d})
            elif kind == "rock":
                rid = self.rock_at.pop(src)
                self.rocks[rid] = list(dst)
                self.rock_at[dst] = rid
                self.falling.discard(rid)
                ev.append({"type": "hook", "pid": pid, "obj": "rock", "rock": rid,
                           "from": src, "to": dst})
            else:
                s = self.snakes.pop(src)
                s.r, s.c = dst
                self.snakes[dst] = s
                ev.append({"type": "hook", "pid": pid, "obj": "snake", "sid": s.sid,
                           "from": src, "to": dst})
        elif tool == "mine":
            cell = info[1]
            self.mines[cell] = pid
            p.mines -= 1
            ev.append({"type": "mine_place", "pid": pid, "pos": cell})
        if tool in ("push", "hook") and kind == "rock":
            self._rock_onto_mine(info[2], ev)
        if TOOL_COST[tool]:
            self._damage(p, TOOL_COST[tool], tool, ev)

    # ------------------------------------------------------------ lens beams
    def _freeze(self, p, sid, ev):
        if not p.alive or p.exited or getattr(p, "frozen", 0):
            return
        ev.append({"type": "freeze", "pid": p.pid, "pos": (p.r, p.c), "sid": sid})
        p.frozen = C.FREEZE_TURNS
        self._damage(p, C.LENS_DAMAGE, "lens", ev)

    def _beam_sweep(self, before, ev):
        """Anyone the beams have just swung onto is hit."""
        if not getattr(self, "statues", None):
            return
        now = self.lit()
        for cell, sid in now.items():
            if cell in before:
                continue
            for p in self.players_at(*cell):
                self._freeze(p, sid, ev)

    def _rock_onto_mine(self, cell, ev):
        """A boulder landing on a mine sets it off harmlessly."""
        if cell in getattr(self, "mines", {}):
            owner = self.mines.pop(cell)
            ev.append({"type": "mine_boom", "pos": cell, "owner": owner})

    def apply_move(self, pid, d, ev):
        p = self.players[pid]
        if d == "wait" or d is None:
            ev.append({"type": "wait", "pid": pid, "pos": (p.r, p.c)})
            return
        dr, dc = DIRS[d]
        nr, nc = p.r + dr, p.c + dc
        if not self.can_enter(pid, nr, nc):
            raise ValueError("can't move there")
        dug = self.terrain[nr][nc] == DIRT
        if dug:
            self.terrain[nr][nc] = FLOOR
        ev.append({"type": "move", "pid": pid, "from": (p.r, p.c), "to": (nr, nc),
                   "dir": d, "dig": dug})
        p.r, p.c = nr, nc
        cell = (nr, nc)
        if cell in self.gems:
            kind = self.gems.pop(cell)
            pts = C.RED_POINTS if kind == "r" else C.VIOLET_POINTS
            p.points += pts
            if kind == "r":
                p.reds += 1
            else:
                p.violets += 1
            ev.append({"type": "collect", "pid": pid, "pos": cell, "kind": kind,
                       "points": pts})
        if self.terrain[nr][nc] == FIRE:
            self._damage(p, C.FIRE_DAMAGE, "fire", ev)
        if cell in self.snakes and p.alive:
            s = self.snakes[cell]
            if not s.stuns:
                self._bite(p, s, ev)
        if cell in self.mines and self.mines[cell] != pid and p.alive:
            owner = self.mines.pop(cell)
            ev.append({"type": "mine", "pid": pid, "pos": cell, "owner": owner})
            self._damage(p, C.MINE_DAMAGE, "mine", ev)
        if self.terrain[nr][nc] == EXIT and p.alive:
            p.exited = True
            ev.append({"type": "exit", "pid": pid, "pos": cell})
            return
        if p.alive and getattr(self, "statues", None):
            beams = self.lit()
            if cell in beams:
                self._freeze(p, beams[cell], ev)

    def must_leave(self, pid):
        """Why the player should not wait this turn (fire or a waking snake), or None."""
        p = self.players[pid]
        if getattr(p, "frozen", 0):
            return None
        if self.terrain[p.r][p.c] == FIRE:
            return "fire"
        if (p.r, p.c) in self.snakes:
            return "snake"
        return None

    def _bite(self, p, s, ev):
        del self.snakes[(s.r, s.c)]
        ev.append({"type": "bite", "pid": p.pid, "pos": (s.r, s.c), "sid": s.sid})
        self._damage(p, C.SNAKE_BITE, "snake", ev)

    def _stun_expiry(self, ev):
        for pos, s in list(self.snakes.items()):
            if not s.stuns:
                continue
            for pid in [k for k, exp in s.stuns.items() if exp <= self.t]:
                del s.stuns[pid]
            if not s.stuns:
                ev.append({"type": "wake", "sid": s.sid, "pos": pos})
                here = self.players_at(*pos)
                if here:
                    # the snake bites whoever's stun just ran out first, otherwise the other
                    here.sort(key=lambda q: q.pid != self.current)
                    self._bite(here[0], s, ev)

    def gravity_tick(self, ev, settle=False):
        """One gravity step. Returns True if anything moved."""
        moved = False
        new_falling = set()
        for rid in sorted(self.rocks, key=lambda k: -self.rocks[k][0]):
            r, c = self.rocks[rid]
            below = (r + 1, c)
            if self.is_empty_floor(*below):
                self._move_rock(rid, (r, c), below)
                new_falling.add(rid)
                moved = True
                ev.append({"type": "fall", "rock": rid, "from": (r, c), "to": below})
                self._rock_onto_mine(below, ev)
                continue
            if not settle and rid in self.falling:
                victims = self.players_at(*below)
                if victims:
                    for v in victims:
                        ev.append({"type": "crush", "pid": v.pid, "pos": below, "rock": rid})
                        self._damage(v, C.CRUSH_DAMAGE, "boulder", ev)
                    continue
                if below in self.snakes:
                    s = self.snakes.pop(below)
                    ev.append({"type": "snake_crushed", "sid": s.sid, "pos": below, "rock": rid})
                    continue
            if below in self.rock_at:
                for side in (-1, 1):
                    sp, sb = (r, c + side), (r + 1, c + side)
                    if self.is_empty_floor(*sp) and self.is_empty_floor(*sb):
                        self._move_rock(rid, (r, c), sb)
                        new_falling.add(rid)
                        moved = True
                        ev.append({"type": "fall", "rock": rid, "from": (r, c), "to": sb,
                                   "roll": side})
                        self._rock_onto_mine(sb, ev)
                        break
        self.falling = new_falling
        return moved

    def resolve_gravity(self, ev, active=()):
        """Let every unsupported boulder fall now, as far as it goes. `active` holds rocks
        just moved by a tool (they crush a snake they land on)."""
        resting = self.resting = getattr(self, "resting", {})
        active = set(active)
        for _ in range(self.h * 4):
            moved = False
            for rid in sorted(self.rocks, key=lambda k: -self.rocks[k][0]):
                r, c = self.rocks[rid]
                below = (r + 1, c)
                if not self.inside(*below):
                    continue
                if self.is_empty_floor(*below):
                    self._move_rock(rid, (r, c), below)
                    resting.pop(rid, None)
                    active.add(rid)
                    moved = True
                    ev.append({"type": "fall", "rock": rid, "from": (r, c), "to": below})
                    self._rock_onto_mine(below, ev)
                    continue
                victims = self.players_at(*below)
                bare = (self.terrain[below[0]][below[1]] in (FLOOR, FIRE, EXIT) and below not in self.rock_at
                        and below not in self.gems and below not in self.snakes
                        and below not in getattr(self, "statue_cells", {}))
                if victims and bare and self.terrain[below[0]][below[1]] == FLOOR:
                    if not resting.get(rid):
                        # nothing but explorers under it: it comes down on them
                        for v in victims:
                            ev.append({"type": "crush", "pid": v.pid, "pos": below, "rock": rid})
                            self._damage(v, C.CRUSH_DAMAGE, "boulder", ev)
                        resting[rid] = {v.pid for v in victims}
                    continue
                resting.pop(rid, None)
                if below in self.snakes and rid in active and self.terrain[below[0]][below[1]] == FLOOR:
                    s = self.snakes.pop(below)
                    ev.append({"type": "snake_crushed", "sid": s.sid, "pos": below, "rock": rid})
                    moved = True
                    continue
                if below in self.rock_at:
                    for side in (-1, 1):
                        sp, sb = (r, c + side), (r + 1, c + side)
                        if self.is_empty_floor(*sp) and self.is_empty_floor(*sb):
                            self._move_rock(rid, (r, c), sb)
                            active.add(rid)
                            moved = True
                            ev.append({"type": "fall", "rock": rid, "from": (r, c), "to": sb,
                                       "roll": side})
                            self._rock_onto_mine(sb, ev)
                            break
            if not moved:
                break
        self.falling = set()

    def _move_rock(self, rid, src, dst):
        del self.rock_at[src]
        self.rocks[rid] = list(dst)
        self.rock_at[dst] = rid

    def settle(self):
        for _ in range(self.h * 3):
            if not self.gravity_tick([], settle=True):
                break
        self.falling = set()

    def use_tool(self, pid, tool, tool_dir):
        """First half of a turn: one tool (takes no turn). Returns events."""
        if self.over:
            raise ValueError("game over")
        if pid != self.current:
            raise ValueError("not your turn")
        if self.tool_used:
            raise ValueError("only one tool per turn")
        ev = []
        before = self.lit() if getattr(self, "statues", None) else {}
        self.apply_tool(pid, tool, tool_dir, ev)
        moved = [e["rock"] for e in ev if e["type"] in ("push", "hook") and "rock" in e]
        self.resolve_gravity(ev, moved)
        self._beam_sweep(before, ev)
        self.tool_used = True
        if not self.players[pid].alive:
            self._check_end(ev)
        return ev

    def end_turn(self, pid, move="wait"):
        """Second half of a turn: move one cell or wait, then the world reacts."""
        if self.over:
            raise ValueError("game over")
        if pid != self.current:
            raise ValueError("not your turn")
        ev = []
        p = self.players[pid]
        was_frozen = getattr(p, "frozen", 0) > 0
        if was_frozen and move not in (None, "wait"):
            raise ValueError("you're frozen")
        before = self.lit() if getattr(self, "statues", None) else {}
        if p.alive:
            self.apply_move(pid, move, ev)
            if not p.exited:
                self.resolve_gravity(ev)        # soil dug, a diamond taken, a cell left...
            self._beam_sweep(before, ev)        # digging soil can let a beam through
            if was_frozen and p.alive:
                p.frozen -= 1
                if p.frozen == 0:
                    ev.append({"type": "thaw", "pid": pid, "pos": (p.r, p.c)})
            # fire never goes out: standing in it for another turn burns again
            if move in (None, "wait") and self.terrain[p.r][p.c] == FIRE:
                ev.append({"type": "burn", "pid": pid, "pos": (p.r, p.c)})
                self._damage(p, C.FIRE_DAMAGE, "fire", ev)
        if p.alive and not p.exited:
            self._stun_expiry(ev)
        if not self.over and not p.exited:
            before = self.lit() if getattr(self, "statues", None) else {}
            self.resolve_gravity(ev)            # a snake bite can free a rock too
            self._beam_sweep(before, ev)
        self._check_end(ev)
        self.tool_used = False
        if not self.over:
            self.t += 1
            self.current = 1 - self.current
            if self.current == self.first:
                self.rounds += 1
                if self.rounds >= C.MAX_ROUNDS:
                    self._finish(-2, "Round limit reached", ev)
        return ev

    def take_turn(self, pid, tool=None, tool_dir=None, move="wait"):
        """A full turn. Raises ValueError on illegal input (nothing is changed then)."""
        if getattr(self.players[pid], "frozen", 0):
            if tool or move not in (None, "wait"):
                raise ValueError("you're frozen")
        if move not in (None, "wait") and tool is None:
            p = self.players[pid]
            dr, dc = DIRS[move]
            if not self.can_enter(pid, p.r + dr, p.c + dc):
                raise ValueError("can't move there")
        ev = []
        if tool:
            ev += self.use_tool(pid, tool, tool_dir)
            if self.over:
                return ev
            p = self.players[pid]
            if move not in (None, "wait"):
                dr, dc = DIRS[move]
                if not self.can_enter(pid, p.r + dr, p.c + dc):
                    move = "wait"
        ev += self.end_turn(pid, move)
        return ev

    def _check_end(self, ev):
        a, b = self.players
        if not a.alive or not b.alive:
            if not a.alive and not b.alive:
                self._finish(-1, "Both explorers fell", ev)
            else:
                dead = a if not a.alive else b
                self._finish(1 - dead.pid, f"{dead.name} ran out of life", ev)
            return
        for p in self.players:
            if p.exited:
                self._finish(-2, f"{p.name} reached the exit", ev)
                return
        if not self.gems:
            self._finish(-2, "Every diamond has been collected", ev)

    def _finish(self, winner, reason, ev):
        self.over = True
        if winner == -2:  # decide by totals
            a, b = self.players
            winner = 0 if a.total > b.total else 1 if b.total > a.total else -1
        self.winner = winner
        self.end_reason = reason
        ev.append({"type": "game_over", "winner": winner, "reason": reason})
