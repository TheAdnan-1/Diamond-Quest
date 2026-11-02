"""Adversarial search for the AI's strategy.

A 50x50 game lasts hundreds of single-cell moves, far beyond any look-ahead, so
the AI searches over *macro actions*: "go collect diamond X" or "go to the exit".
Travel costs (turns and life) come from Dijkstra maps on the real grid.

The two players act by clock: whoever has spent fewer turns acts next.
A diamond belongs to whoever *arrives* first: a player may steal a diamond the
rival has set out for if they can get there sooner (the rival's trip is wasted).

  * Minimax with alpha-beta pruning, iterative deepening under a time budget.
    Leaves are scored by playing the game out to the end with a fast greedy
    policy for both sides (who exits when, who runs out of diamonds), so the
    utility is the predicted final result, not a guess at mid-game.
  * Monte Carlo Tree Search (UCT) with epsilon-greedy playouts to the end.
Both use the same utility function: final (points + life) difference, with a
bonus for winning and a large penalty for dying.
"""
import math
import random
import time

from . import config as C

EXIT = "EXIT"
INF = 10 ** 6
TAKEN_STEPS = 24        # how many of the rival's predicted next picks the route planner concedes
LIFE_FLOOR = 10         # never plan a trip that leaves less life than this


class MacroModel:
    """Travel table between the players, the diamonds and the exit."""

    def __init__(self, nodes, values, table, ai_pid, unlock=C.EXIT_UNLOCK_GEMS, horizon=INF):
        self.nodes = nodes          # diamond cells
        self.values = values        # diamond cell -> points
        self.table = table          # (from_key, to_key) -> (turns, life)
        self.ai = ai_pid
        self.unlock = unlock
        self.horizon = horizon      # turns left before the round limit ends the game
        self.tours = {}             # player -> planned order of diamonds (then EXIT)

    def travel(self, a, b):
        return self.table.get((a, b))


class MState:
    __slots__ = ("pos", "clock", "points", "life", "gems", "remaining", "over", "last", "claims",
                 "idle", "end", "walk")

    def __init__(self, pos, clock, points, life, gems, remaining, over=False, last=None,
                 claims=None, idle=None, end=INF, walk=None):
        self.pos, self.clock, self.points, self.life = pos, clock, points, life
        self.gems, self.remaining, self.over, self.last = gems, remaining, over, last
        self.claims = claims or {}          # diamond -> (who, arrival turn), not yet final
        self.idle = idle or [False, False]
        self.end = end                      # turn at which someone walks out (game over)
        self.walk = walk                    # (who, life the walk out costs) while under way

    def child(self):
        return MState(list(self.pos), list(self.clock), list(self.points), list(self.life),
                      list(self.gems), self.remaining, self.over, self.last, dict(self.claims),
                      list(self.idle), self.end, self.walk)

    def to_move(self):
        a, b = self.idle
        if a and b:
            return None
        if a:
            return 1
        if b:
            return 0
        if self.clock[0] == self.clock[1]:
            return 1 - self.last if self.last is not None else 0
        return 0 if self.clock[0] < self.clock[1] else 1


# ----------------------------------------------------------------- moves
def _options(model, s, who):
    """(score, target) for every diamond `who` can sensibly go for (incl. steals)."""
    here = s.pos[who]
    clk = s.clock[who]
    out = []
    T = model.table
    V = model.values
    life = s.life[who]
    limit = min(model.horizon, s.end - 1)
    for g in s.remaining:
        tr = T.get((here, g))
        if tr is None:
            continue
        turns, lc = tr
        if life - lc < LIFE_FLOOR or clk + turns > limit:
            continue
        out.append(((V[g] - lc) / (turns + 2.0), g))
    for g, (owner, arr) in s.claims.items():
        if owner == who:
            continue
        tr = T.get((here, g))
        if tr is None:
            continue
        turns, lc = tr
        if clk + turns < arr and life - lc >= LIFE_FLOOR and clk + turns <= limit:
            out.append(((V[g] - lc) / (turns + 2.0), g))
    out.sort(reverse=True)
    return out


def exit_wins(model, s, who):
    """Leaving now beats the rival even if they grab their best diamond on the way."""
    if s.gems[who] < model.unlock:
        return False
    tr = model.travel(s.pos[who], EXIT)
    if tr is None:
        return False
    turns, lc = tr
    if s.life[who] - lc <= 0 or s.clock[who] + turns >= s.end:
        return False
    op = 1 - who
    mine = s.points[who] + s.life[who] - lc
    theirs = s.points[op] + s.life[op]
    if mine <= theirs:
        return False
    # what can the rival still add while we walk out?
    window = s.clock[who] + turns - s.clock[op]
    threat = 0
    if window > 0 and not s.idle[op]:
        for g in s.remaining:
            t2 = model.travel(s.pos[op], g)
            if t2 and t2[0] <= window and s.life[op] - t2[1] > 0:
                threat = max(threat, model.values[g] - t2[1])
    return mine > theirs + threat


def can_exit(model, s, who):
    if s.gems[who] < model.unlock:
        return False
    tr = model.travel(s.pos[who], EXIT)
    return tr is not None and s.life[who] - tr[1] > 0 and s.clock[who] + tr[0] < s.end


def actions(model, s, who, limit=5):
    """Candidate macro actions for player `who`, best-looking first."""
    acts = [g for _, g in _options(model, s, who)[:limit]]
    if can_exit(model, s, who):
        if exit_wins(model, s, who):
            acts.insert(0, EXIT)
        else:
            acts.append(EXIT)
    return acts


def _available(model, s, who, g):
    """Can `who` still collect g first (and afford the trip)?"""
    tr = model.table.get((s.pos[who], g))
    if tr is None or s.life[who] - tr[1] < LIFE_FLOOR:
        return False
    arr = s.clock[who] + tr[0]
    if arr > min(model.horizon, s.end - 1):
        return False
    if g in s.remaining:
        return True
    c = s.claims.get(g)
    return c is not None and c[0] != who and arr < c[1]


def tour_choice(model, s, who):
    """Next stop on `who`'s planned tour, or None when the tour has nothing left."""
    for g in model.tours.get(who, ()):
        if g == EXIT:
            return None
        if _available(model, s, who, g):
            return g
    return None


def greedy_choice(model, s, who, rng=None, eps=0.0):
    """Fast policy used in playouts: leave when that wins, otherwise the next stop of the
    player's planned tour, otherwise the best value-per-turn diamond (occasionally the
    2nd/3rd best when exploring)."""
    if exit_wins(model, s, who):
        return EXIT
    if who in model.tours and not (rng is not None and eps and rng.random() < eps):
        g = tour_choice(model, s, who)
        if g is not None:
            return g
    opts = _options(model, s, who)
    if not opts:
        return EXIT if can_exit(model, s, who) else None
    if rng is not None and eps and rng.random() < eps and len(opts) > 1:
        return opts[rng.randrange(min(3, len(opts)))][1]
    return opts[0][1]


def apply(model, s, who, act):
    n = s.child()
    turns, life = model.travel(s.pos[who], act)
    n.clock[who] += turns
    n.life[who] -= life
    n.pos[who] = act
    n.last = who
    if act == EXIT:
        # the game ends when we arrive; the rival keeps playing until then
        if n.walk is not None:                # we get out first: the rival stops mid-walk
            n.life[n.walk[0]] += n.walk[1]
        n.end = n.clock[who]
        n.walk = (who, life)
        n.idle[who] = True
        if n.idle[1 - who] or n.clock[1 - who] >= n.end:
            n.over = True
        return n
    v = model.values[act]
    if act in s.claims:                      # stealing a diamond the rival set out for
        owner, _ = s.claims[act]
        n.points[owner] -= v
        n.gems[owner] -= 1
    else:
        n.remaining = s.remaining - {act}
    n.points[who] += v
    n.gems[who] += 1
    n.claims[act] = (who, n.clock[who])
    # claims the other player can no longer beat become final
    n.claims = {g: c for g, c in n.claims.items() if c[1] > n.clock[1 - c[0]]}
    if n.life[who] <= 0:
        n.over = True
    if not n.remaining and not n.claims:
        n.over = True                         # every diamond collected: game ends
    if n.clock[who] >= model.horizon and n.clock[1 - who] >= model.horizon:
        n.over = True
    return n


def utility(model, s):
    """AI's point of view: final (points + life) difference, +40 for a win, -40 for a loss,
    and +-1000 when someone runs out of life (an instant loss)."""
    ai, op = model.ai, 1 - model.ai
    if s.life[ai] <= 0 and s.life[op] <= 0:
        return 0.0
    if s.life[ai] <= 0:
        return -1000.0
    if s.life[op] <= 0:
        return 1000.0
    m = (s.points[ai] + s.life[ai]) - (s.points[op] + s.life[op])
    return m + (40 if m > 0 else -40 if m < 0 else 0)


def playout(model, s, rng=None, eps=0.0, max_steps=60):
    """Play the macro game to the end: each player follows its tour, then greedy."""
    for _ in range(max_steps):
        if s.over:
            break
        who = s.to_move()
        if who is None:
            break
        a = greedy_choice(model, s, who, rng, eps)
        if a is None:
            s = s.child()
            s.idle[who] = True
            continue
        s = apply(model, s, who, a)
    return s


# ------------------------------------------------------------------ tour planning
def _timeline(model, s, who, steps=24):
    """Where the rival's greedy route takes them if left alone: (times, totals, taken)."""
    st = s.child()
    st.idle[1 - who] = True
    times, totals, taken = [st.clock[who]], [st.points[who] + st.life[who]], {}
    saved = model.tours.pop(who, None)
    for _ in range(steps):
        if st.over or st.idle[who]:
            break
        a = greedy_choice(model, st, who)
        if a is None or a == EXIT:
            break
        st = apply(model, st, who, a)
        times.append(st.clock[who])
        totals.append(st.points[who] + st.life[who])
        taken[a] = st.clock[who]
    if saved is not None:
        model.tours[who] = saved
    return times, totals, taken


def plan_tours(model, s, who, width=36, branch=9, keep=10):
    """Beam search for good diamond orders for `who` (an orienteering problem): collect
    enough diamonds to open the exit, as fast and cheaply as possible, skipping the ones
    the rival will reach first, ranked by the lead we would have at each moment."""
    import bisect
    op = 1 - who
    times, totals, taken = _timeline(model, s, op)
    if len(taken) > TAKEN_STEPS:
        taken = dict(sorted(taken.items(), key=lambda kv: kv[1])[:TAKEN_STEPS])

    def op_total(t):
        return totals[max(0, bisect.bisect_right(times, t) - 1)]

    T, V = model.table, model.values
    unlock = model.unlock
    limit = min(model.horizon, s.end - 1)
    start = (s.pos[who], s.clock[who], s.points[who], s.life[who], s.gems[who], ())
    beam = [start]
    done = []
    pool = list(s.remaining)
    max_len = min(len(pool), max(0, unlock - s.gems[who]) + 5)
    for _ in range(max_len):
        cand = {}
        for pos, t, pts, life, gems, seq in beam:
            opts = []
            for g in pool:
                if g in seq:
                    continue
                tr = T.get((pos, g))
                if tr is None:
                    continue
                turns, lc = tr
                arr = t + turns
                if life - lc < LIFE_FLOOR or arr > limit or taken.get(g, INF) <= arr:
                    continue
                opts.append(((V[g] - lc) / (turns + 2.0), g, arr, lc))
            opts.sort(reverse=True)
            for _, g, arr, lc in opts[:branch]:
                np_, nl = pts + V[g], life - lc
                key = (g, frozenset(seq) | {g})
                m = np_ + nl - op_total(arr) - 0.25 * arr
                if key not in cand or cand[key][0] < m:
                    cand[key] = (m, (g, arr, np_, nl, gems + 1, seq + (g,)))
        if not cand:
            break
        ranked = sorted(cand.values(), key=lambda x: -x[0])[:width]
        beam = [x[1] for x in ranked]
        for pos, t, pts, life, gems, seq in beam:
            if gems >= unlock:
                tr = T.get((pos, EXIT))
                if tr and life - tr[1] > 0:
                    fin = pts + life - tr[1] - op_total(t + tr[0])
                    done.append((fin, seq + (EXIT,)))
    if not done:
        done = [(b[2] + b[3] - op_total(b[1]), b[5]) for b in beam]
    done.sort(key=lambda x: -x[0])
    out, seen = [], set()
    for _, seq in done:
        if seq[:3] not in seen:            # keep variety in the opening moves
            seen.add(seq[:3])
            out.append(seq)
        if len(out) >= keep:
            break
    return out


def best_tour(model, s, who, prev=None, budget=0.15):
    """Score candidate tours by playing the game out (rival greedy, reacting to us)."""
    t_end = time.time() + budget
    scale = max(1.0, min(3.0, budget / 0.12))            # more time: a wider route search
    cands = plan_tours(model, s, who, width=int(36 * scale), branch=9 + int(scale), keep=int(10 * scale))
    if prev:
        prev = tuple(g for g in prev if g == EXIT or g in s.remaining)
        if prev and prev not in cands:
            cands.insert(0, prev)
    if not cands:
        return None, None, []
    scored = []
    saved = model.tours.get(who)
    for i, tour in enumerate(cands):
        if i and time.time() > t_end:
            break
        model.tours[who] = tour
        v = utility(model, playout(model, s))
        if prev and tour == prev:
            v += 10                        # stick with the plan unless clearly beaten
        scored.append((v, tour))
    if saved is None:
        model.tours.pop(who, None)
    else:
        model.tours[who] = saved
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], scored[0][0], scored


# ------------------------------------------------------------------ Minimax
def minimax(model, root, depth=6, budget=0.45, limits=(6, 4), must=None):
    """Alpha-beta over macro actions with iterative deepening; leaves are greedy playouts."""
    stats = {"nodes": 0, "pruned": 0, "playouts": 0, "depth": 0}
    t_end = time.time() + budget
    cache = {}

    class Timeout(Exception):
        pass

    def leaf(s):
        stats["playouts"] += 1
        return utility(model, playout(model, s))

    def rec(s, d, alpha, beta):
        stats["nodes"] += 1
        if stats["nodes"] & 63 == 0 and time.time() > t_end:
            raise Timeout
        if s.over:
            return utility(model, s)
        if d == 0:
            return leaf(s)
        who = s.to_move()
        if who is None:
            return utility(model, s)
        acts = actions(model, s, who, limits[0] if who == model.ai else limits[1])
        if not acts:
            n = s.child()
            n.idle[who] = True
            return rec(n, d, alpha, beta)
        if who == model.ai:
            v = -math.inf
            for a in acts:
                v = max(v, rec(apply(model, s, who, a), d - 1, alpha, beta))
                alpha = max(alpha, v)
                if alpha >= beta:
                    stats["pruned"] += 1
                    break
        else:
            v = math.inf
            for a in acts:
                v = min(v, rec(apply(model, s, who, a), d - 1, alpha, beta))
                beta = min(beta, v)
                if alpha >= beta:
                    stats["pruned"] += 1
                    break
        return v

    root_acts = actions(model, root, model.ai, limits[0] + 1)
    for m in (must or ()):
        if m not in root_acts and model.travel(root.pos[model.ai], m):
            root_acts.append(m)               # always re-check the plan we are following
    if not root_acts:
        return None, dict(stats, value=0, scored=[])
    best, best_v, scored = root_acts[0], -math.inf, []
    order = list(root_acts)
    for d in range(1, depth + 1):
        try:
            res = []
            alpha = -math.inf
            for a in order:
                v = rec(apply(model, root, model.ai, a), d - 1, alpha, math.inf)
                res.append((a, v))
                alpha = max(alpha, v)
        except Timeout:
            break
        res.sort(key=lambda x: -x[1])
        best, best_v = res[0]
        scored = res
        order = [a for a, _ in res]           # best-first ordering for the next depth
        stats["depth"] = d
        if time.time() > t_end:
            break
    cache.clear()
    stats["value"] = best_v
    stats["scored"] = scored
    return best, stats


# ------------------------------------------------------------------ MCTS
class _Node:
    __slots__ = ("state", "parent", "act", "who", "children", "untried", "n", "w")

    def __init__(self, model, state, parent=None, act=None, who=None, limit=6):
        self.state, self.parent, self.act, self.who = state, parent, act, who
        self.children = []
        self.n = 0
        self.w = 0.0
        if state.over:
            self.untried = []
        else:
            mover = state.to_move() if parent is not None else model.ai
            self.untried = [] if mover is None else [
                (mover, a) for a in actions(model, state, mover, limit)]
            self.untried.reverse()                 # pop() takes the most promising first


def mcts(model, root_state, budget=0.5, max_iter=20000, c=0.9, rng=None, must=None):
    rng = rng or random.Random()
    root = _Node(model, root_state, limit=7)
    for m in (must or ()):
        if model.travel(root_state.pos[model.ai], m) and all(a != m for _, a in root.untried):
            root.untried.append((model.ai, m))
    t0 = time.time()
    it = 0
    while it < max_iter and time.time() - t0 < budget:
        it += 1
        node = root
        # 1. selection
        while not node.untried and node.children:
            mover_is_ai = node.children[0].who == model.ai
            logn = math.log(node.n)

            def ucb(ch):
                mean = ch.w / ch.n
                if not mover_is_ai:
                    mean = 1 - mean
                return mean + c * math.sqrt(logn / ch.n)
            node = max(node.children, key=ucb)
        # 2. expansion
        if node.untried:
            who, a = node.untried.pop()
            child = _Node(model, apply(model, node.state, who, a), node, a, who)
            node.children.append(child)
            node = child
        # 3. simulation: epsilon-greedy playout to the end of the game
        s = playout(model, node.state, rng, eps=0.15)
        reward = 1 / (1 + math.exp(-utility(model, s) / 40))   # squash to (0, 1), AI view
        # 4. backpropagation
        while node is not None:
            node.n += 1
            node.w += reward
            node = node.parent
    if not root.children:
        return None, {"iterations": it}
    best = max(root.children, key=lambda ch: (ch.n, ch.w / ch.n))
    return best.act, {"iterations": it,
                      "visits": [(ch.act, ch.n, ch.w / ch.n) for ch in root.children],
                      "scored": [(ch.act, ch.w / ch.n) for ch in root.children],
                      "value": best.w / best.n}
