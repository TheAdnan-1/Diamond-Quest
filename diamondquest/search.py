"""Uninformed and informed search on the game grid.

Every algorithm returns a SearchResult with the path, the order in which nodes
were expanded (used by the in-game Algorithm Lab to animate the search), and cost.

    DFS, DLS, IDDFS, Bidirectional BFS                 -- uninformed
    Best-First (generic, f = g), Greedy Best-First,    -- informed
    A*  (f = g + h)
"""
import heapq
import itertools
from collections import deque

from . import config as C
from .world import WALL, CRACKED, FIRE, EXIT, DIRS, DIRT


class SearchResult:
    def __init__(self, name, path, expanded, cost=None, note=""):
        self.name = name
        self.path = path or []
        self.expanded = expanded
        self.cost = cost
        self.note = note

    @property
    def found(self):
        return bool(self.path)


# ------------------------------------------------------------------ heuristics
def manhattan(a, b):
    """Admissible and consistent on a 4-connected grid with step cost >= 1."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ------------------------------------------------------------- grid adapters
class GridGraph:
    """Neighbour / step-cost view of the world for one player."""

    def __init__(self, world, pid, life_weight=1.0, danger=None, rocks_passable=False,
                 allow_exit=None, lens_cheap=True):
        self.world = world
        self.pid = pid
        self.lw = life_weight
        self.danger = danger or set()
        self.rocks_passable = rocks_passable
        self.allow_exit = world.exit_open_for(pid) if allow_exit is None else allow_exit
        self.forbid = set()
        self.lit = world.lit(edges=True) if getattr(world, "statues", None) else {}
        # beams whose statue can be hooked from behind are "doors": a detour and a hook,
        # not a 40-life freeze (the AI's tactics walk round and turn the statue)
        self.cheap = set()
        if lens_cheap and self.lit:
            by_sid = {s.sid: s for s in world.statues.values()}
            ok = {sid for sid, s in by_sid.items() if any(c not in self.lit for c in world.statue_hook_spots(s))}
            self.cheap = {c for c, sid in self.lit.items() if sid in ok}

    def passable(self, r, c):
        w = self.world
        if not w.inside(r, c):
            return False
        t = w.terrain[r][c]
        if t == WALL:
            return False
        if t == EXIT and not self.allow_exit:
            return False
        if (r, c) in self.forbid or (r, c) in w.statue_cells:
            return False
        if (r, c) in w.rock_at and not self.rocks_passable:
            return False
        return True

    def neighbors(self, cell):
        r, c = cell
        w = self.world
        for dr, dc in DIRS.values():
            n = (r + dr, c + dc)
            if self.passable(*n):
                yield n
            elif dr == 0 and n in w.rock_at and not self.rocks_passable:
                # a boulder can be pushed sideways if the cell beyond it is free
                beyond = (r, c + 2 * dc)
                if w.is_empty_floor(*beyond):
                    yield n

    def _cheap(self, cell, prev):
        """Entering a hookable beam from the statue's body side (the row below the beam):
        stop, hook the body, walk on. From any other side the beam really freezes you."""
        return prev is not None and cell in self.cheap and prev == (cell[0] + 1, cell[1])

    def life_cost(self, cell, prev=None):
        """Life lost by stepping into cell from prev (tools used optimally)."""
        w = self.world
        cost = 0
        if w.terrain[cell[0]][cell[1]] == FIRE:
            cost += 10
        if cell in w.snakes:
            cost += 5           # hammer it rather than take a 15 bite
        if w.terrain[cell[0]][cell[1]] == CRACKED:
            cost += 5
        if cell in w.rock_at:
            cost += 8
        mines = getattr(w, "mines", None)
        if mines and cell in mines and mines[cell] != self.pid:
            cost += 25
        if cell in self.lit:
            cost += C.HOOK_COST if self._cheap(cell, prev) else C.LENS_DAMAGE
        if (cell[0] - 1, cell[1]) in w.rock_at and (
                w.terrain[cell[0]][cell[1]] == DIRT or cell in w.gems or cell in w.snakes):
            cost += C.CRUSH_DAMAGE          # taking away what holds a rock up drops it on you
        return cost

    def turn_cost(self, cell, prev=None):
        """Turns spent entering cell (a lens beam freezes you for a while)."""
        if cell in self.lit:
            return 2 if self._cheap(cell, prev) else 1 + C.FREEZE_TURNS
        return 1

    def step_cost(self, cell, prev=None):
        c = self.turn_cost(cell, prev) + self.lw * self.life_cost(cell, prev)
        if cell in self.danger:
            c += 60
        return c


# ---------------------------------------------------------------- utilities
def _rebuild(parent, node):
    path = [node]
    while parent.get(node) is not None:
        node = parent[node]
        path.append(node)
    return path[::-1]


def path_cost(graph, path):
    return sum(graph.step_cost(n, p) for p, n in zip(path, path[1:]))


# ---------------------------------------------------------------- DFS
def dfs(graph, start, goal, max_expand=20000):
    stack = [start]
    parent = {start: None}
    expanded = []
    while stack and len(expanded) < max_expand:
        node = stack.pop()
        expanded.append(node)
        if node == goal:
            p = _rebuild(parent, node)
            return SearchResult("DFS", p, expanded, path_cost(graph, p))
        for n in graph.neighbors(node):
            if n not in parent:
                parent[n] = node
                stack.append(n)
    return SearchResult("DFS", None, expanded)


# ---------------------------------------------------------------- DLS / IDDFS
def dls(graph, start, goal, limit, max_expand=60000, _expanded=None):
    """Depth-limited DFS. Graph-search variant: a node is re-opened only if reached
    at a shallower depth, which keeps it complete within the limit."""
    expanded = _expanded if _expanded is not None else []
    best_depth = {start: 0}
    stack = [(start, 0, None)]
    parent = {}
    cutoff = False
    while stack and len(expanded) < max_expand:
        node, depth, par = stack.pop()
        if best_depth.get(node, 1 << 30) < depth:
            continue
        parent[node] = par
        expanded.append(node)
        if node == goal:
            p = _rebuild(parent, node)
            return SearchResult(f"DLS (limit {limit})", p, expanded, path_cost(graph, p))
        if depth == limit:
            cutoff = True
            continue
        for n in graph.neighbors(node):
            if depth + 1 < best_depth.get(n, 1 << 30):
                best_depth[n] = depth + 1
                stack.append((n, depth + 1, node))
    note = "cutoff reached" if cutoff else "no path"
    return SearchResult(f"DLS (limit {limit})", None, expanded, note=note)


def iddfs(graph, start, goal, max_depth=400, max_expand=150000):
    expanded = []
    for limit in range(0, max_depth + 1):
        layer = []
        res = dls(graph, start, goal, limit, max_expand=max_expand - len(expanded),
                  _expanded=layer)
        expanded.extend(layer)
        if res.found:
            return SearchResult(f"IDDFS (depth {limit})", res.path, expanded, res.cost)
        if res.note != "cutoff reached" or len(expanded) >= max_expand:
            break
    return SearchResult("IDDFS", None, expanded)


# ---------------------------------------------------------------- Bidirectional
def bidirectional(graph, start, goal, max_expand=20000):
    if start == goal:
        return SearchResult("Bidirectional", [start], [start], 0)
    pf, pb = {start: None}, {goal: None}
    qf, qb = deque([start]), deque([goal])
    expanded = []
    while qf and qb and len(expanded) < max_expand:
        for q, mine, other in ((qf, pf, pb), (qb, pb, pf)):
            for _ in range(len(q)):
                node = q.popleft()
                expanded.append(node)
                for n in graph.neighbors(node):
                    if n in mine:
                        continue
                    mine[n] = node
                    if n in other:
                        a = _rebuild(pf, n)
                        b = _rebuild(pb, n)[::-1]
                        p = a + b[1:]
                        return SearchResult("Bidirectional", p, expanded, path_cost(graph, p))
                    q.append(n)
    return SearchResult("Bidirectional", None, expanded)


# ---------------------------------------------------------------- Best-first family
def best_first(graph, start, goal, f, name="Best-First", max_expand=40000):
    """Generic best-first search: always expand the open node with the lowest f(node, g)."""
    tie = itertools.count()
    g = {start: 0}
    parent = {start: None}
    open_ = [(f(start, 0), next(tie), start)]
    closed = set()
    expanded = []
    while open_ and len(expanded) < max_expand:
        _, _, node = heapq.heappop(open_)
        if node in closed:
            continue
        closed.add(node)
        expanded.append(node)
        if node == goal:
            p = _rebuild(parent, node)
            return SearchResult(name, p, expanded, g[node])
        for n in graph.neighbors(node):
            ng = g[node] + graph.step_cost(n, node)
            if n not in g or ng < g[n]:
                if n in closed and name.startswith("Greedy"):
                    continue
                g[n] = ng
                parent[n] = node
                heapq.heappush(open_, (f(n, ng), next(tie), n))
    return SearchResult(name, None, expanded)


def uniform_best_first(graph, start, goal):
    return best_first(graph, start, goal, lambda n, g: g, "Best-First (f = g)")


def greedy(graph, start, goal, h=manhattan):
    return best_first(graph, start, goal, lambda n, g: h(n, goal), "Greedy Best-First")


def astar(graph, start, goal, h=manhattan):
    return best_first(graph, start, goal, lambda n, g: g + h(n, goal), "A*")


ALGORITHMS = [
    ("DFS", lambda g, s, t: dfs(g, s, t)),
    ("DLS", lambda g, s, t: dls(g, s, t, limit=max(10, int(manhattan(s, t) * 1.6)))),
    ("IDDFS", lambda g, s, t: iddfs(g, s, t)),
    ("Bidirectional", lambda g, s, t: bidirectional(g, s, t)),
    ("Best-First", lambda g, s, t: uniform_best_first(g, s, t)),
    ("Greedy", lambda g, s, t: greedy(g, s, t)),
    ("A*", lambda g, s, t: astar(g, s, t)),
]


# ---------------------------------------------------------------- Dijkstra map
def dijkstra_map(graph, source):
    """Single-source costs. Returns dict cell -> (cost, turns, life)."""
    tie = itertools.count()
    best = {source: (0.0, 0, 0)}
    heap = [(0.0, next(tie), source)]
    done = set()
    while heap:
        cost, _, node = heapq.heappop(heap)
        if node in done:
            continue
        done.add(node)
        _, turns, life = best[node]
        for n in graph.neighbors(node):
            nc = cost + graph.step_cost(n, node)
            if n not in best or nc < best[n][0]:
                best[n] = (nc, turns + graph.turn_cost(n, node), life + graph.life_cost(n, node))
                heapq.heappush(heap, (nc, next(tie), n))
    return best


def bfs_distances(world, source, rocks_block=True, cracked_ok=True):
    """Plain step distances (used by the map generator)."""
    dist = {source: 0}
    q = deque([source])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            n = (r + dr, c + dc)
            if n in dist or not world.inside(*n):
                continue
            t = world.terrain[n[0]][n[1]]
            if t == WALL or (t == CRACKED and not cracked_ok):
                continue
            if rocks_block and n in world.rock_at:
                continue
            dist[n] = dist[(r, c)] + 1
            q.append(n)
    return dist
