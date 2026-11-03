"""PDDL support: parser, problem generator, grounder and a forward-search planner.

  * parse_domain / parse_problem   -- read STRIPS PDDL (s-expressions)
  * make_problem(world, pid)        -- write a PDDL problem for the area around a player
  * ground()                        -- instantiate actions, pruned by static predicates
  * plan()                          -- breadth-first forward search (optimal plan length)

Note on the domain: `dig` leaves (dirt ?c) unchanged. Re-entering a dug cell costs the
same one step either way, and keeping dirt static keeps the state space tiny.
"""
import os
from collections import deque
from itertools import product

from .world import WALL, CRACKED, DIRT, EXIT, DIRS

DOMAIN_PATH = os.path.join(os.path.dirname(__file__), "pddl", "domain.pddl")


# ------------------------------------------------------------------ parsing
def _tokens(text):
    out = []
    for line in text.splitlines():
        line = line.split(";", 1)[0]
        out.extend(line.replace("(", " ( ").replace(")", " ) ").split())
    return out


def parse_sexpr(text):
    toks = _tokens(text)
    pos = 0

    def read():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "(":
            lst = []
            while toks[pos] != ")":
                lst.append(read())
            pos += 1
            return lst
        return t.lower()
    return read()


def _conj(expr):
    """Turn (and a b (not c)) into ([pos...], [neg...]) of literal tuples."""
    if not expr:
        return [], []
    items = expr[1:] if expr[0] == "and" else [expr]
    pos, neg = [], []
    for it in items:
        if it[0] == "not":
            neg.append(tuple(it[1]))
        else:
            pos.append(tuple(it))
    return pos, neg


class Action:
    def __init__(self, name, params, pre, add, dele):
        self.name, self.params, self.pre, self.add, self.dele = name, params, pre, add, dele


class Domain:
    def __init__(self, name, predicates, actions):
        self.name, self.predicates, self.actions = name, predicates, actions


def parse_domain(text):
    sx = parse_sexpr(text)
    name = sx[1][1]
    predicates, actions = [], []
    for part in sx[2:]:
        if part[0] == ":predicates":
            predicates = [p[0] for p in part[1:]]
        elif part[0] == ":action":
            aname = part[1]
            fields = dict(zip(part[2::2], part[3::2]))
            params = fields.get(":parameters", [])
            pre, _ = _conj(fields.get(":precondition", []))
            add, dele = _conj(fields.get(":effect", []))
            actions.append(Action(aname, params, pre, add, dele))
    return Domain(name, predicates, actions)


def parse_problem(text):
    sx = parse_sexpr(text)
    objs, init, goal = [], [], []
    for part in sx[2:]:
        if part[0] == ":objects":
            objs = part[1:]
        elif part[0] == ":init":
            init = [tuple(f) for f in part[1:]]
        elif part[0] == ":goal":
            goal, _ = _conj(part[1])
    return objs, set(init), goal


# ------------------------------------------------------------- problem maker
def cell_name(c):
    return f"c{c[0]}_{c[1]}"


def make_problem(world, pid, radius=6):
    """Write a PDDL problem for the window around player pid. Returns (text, cells)."""
    p = world.players[pid]
    cells = []
    for r in range(p.r - radius, p.r + radius + 1):
        for c in range(p.c - radius, p.c + radius + 1):
            if not world.inside(r, c):
                continue
            t = world.terrain[r][c]
            if (r, c) != (p.r, p.c) and (t == WALL or (r, c) in world.rock_at
                                         or (r, c) in world.snakes or t == EXIT
                                         or (r, c) in getattr(world, "statue_cells", {})):
                continue
            cells.append((r, c))
    cs = set(cells)
    init = [f"(at {cell_name((p.r, p.c))})"]
    for (r, c) in cells:
        t = world.terrain[r][c]
        n = cell_name((r, c))
        if t == DIRT:
            init.append(f"(dirt {n})")
        elif t == CRACKED:
            init.append(f"(cracked {n})")
        else:
            init.append(f"(open {n})")
        if (r, c) in world.gems:
            init.append(f"(gem {n})")
        for dr, dc in DIRS.values():
            if (r + dr, c + dc) in cs:
                init.append(f"(adj {n} {cell_name((r + dr, c + dc))})")
    objs = " ".join(cell_name(c) for c in cells)
    text = (f"(define (problem diamond-quest-local)\n  (:domain diamond-quest)\n"
            f"  (:objects {objs})\n  (:init\n    " + "\n    ".join(init) + ")\n"
            "  (:goal (has-gem)))\n")
    return text, cells


# ----------------------------------------------------------------- grounding
def ground(domain, objects, init):
    fluent_preds = {lit[0] for a in domain.actions for lit in a.add + a.dele}
    static = {f for f in init if f[0] not in fluent_preds}
    by_pred = {}
    for f in static:
        by_pred.setdefault(f[0], []).append(f)
    grounded = []
    for a in domain.actions:
        static_pre = [l for l in a.pre if l[0] not in fluent_preds]
        bindings = [{}]
        for lit in static_pre:                  # join on static facts
            nb = []
            for b in bindings:
                for f in by_pred.get(lit[0], []):
                    ok, b2 = True, dict(b)
                    for var, val in zip(lit[1:], f[1:]):
                        if var.startswith("?"):
                            if b2.setdefault(var, val) != val:
                                ok = False
                                break
                        elif var != val:
                            ok = False
                            break
                    if ok:
                        nb.append(b2)
            bindings = nb
        for b in bindings:
            unbound = [v for v in a.params if v not in b]
            for vals in product(objects, repeat=len(unbound)) if unbound else [()]:
                full = dict(b, **dict(zip(unbound, vals)))

                def inst(lit):
                    return (lit[0],) + tuple(full.get(x, x) for x in lit[1:])
                pre = frozenset(inst(l) for l in a.pre if l[0] in fluent_preds)
                grounded.append((f"({a.name} {' '.join(full[v] for v in a.params)})", pre,
                                 frozenset(inst(l) for l in a.add),
                                 frozenset(inst(l) for l in a.dele)))
    return grounded, frozenset(f for f in init if f[0] in fluent_preds)


# ------------------------------------------------------------------- planner
def plan(domain, objects, init, goal, max_states=200000):
    actions, state0 = ground(domain, objects, init)
    goal = frozenset(goal)
    index = {}
    for i, (_, pre, _, _) in enumerate(actions):
        for lit in pre:
            index.setdefault(lit, []).append(i)
    parent = {state0: None}
    q = deque([state0])
    expanded = 0
    while q and expanded < max_states:
        s = q.popleft()
        expanded += 1
        if goal <= s:
            steps = []
            while parent[s] is not None:
                s, name = parent[s]
                steps.append(name)
            return steps[::-1], {"grounded": len(actions), "expanded": expanded}
        seen_act = set()
        for lit in s:
            for i in index.get(lit, ()):
                if i in seen_act:
                    continue
                seen_act.add(i)
                name, pre, add, dele = actions[i]
                if pre <= s:
                    ns = (s - dele) | add
                    if ns not in parent:
                        parent[ns] = (s, name)
                        q.append(ns)
    return None, {"grounded": len(actions), "expanded": expanded}


def plan_for_player(world, pid, radius=6, save_dir=None):
    with open(DOMAIN_PATH) as f:
        dtext = f.read()
    domain = parse_domain(dtext)
    for rad in (radius, radius + 3, radius + 6):
        ptext, cells = make_problem(world, pid, rad)
        if any(c in world.gems for c in cells):
            break
    objs, init, goal = parse_problem(ptext)
    steps, stats = plan(domain, objs, init, goal)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "problem.pddl"), "w") as f:
            f.write(ptext)
        with open(os.path.join(save_dir, "plan.txt"), "w") as f:
            f.write("\n".join(steps or ["; no plan found"]) + "\n")
    path = []
    if steps:
        p = world.players[pid]
        path = [(p.r, p.c)]
        for st in steps:
            parts = st.strip("()").split()
            if parts[0] in ("move", "dig"):
                r, c = parts[2][1:].split("_")
                path.append((int(r), int(c)))
    stats["objects"] = len(objs)
    stats["facts"] = len(init)
    return steps, path, stats
