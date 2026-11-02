"""The AI rival: how all the pieces fit together each turn.

  1. Fuzzy (Mamdani) risk appetite  <- AI life + score lead
  2. Logic engine danger map        <- cells around the AI
  3. Dijkstra maps -> travel table between both players, every diamond and the exit
  4. Route planning: beam search over diamond orders (orienteering), each candidate tour
     scored by playing the game out against a reacting rival
  5. Strategy: Minimax (alpha-beta, iterative deepening) or MCTS over macro actions, with
     full-game playouts at the leaves; commitment to the current goal stops dithering
  6. Tactics: A* path to the target with life-weighted step costs; hammer, push,
     blocking, mines; a gravity simulation stops it from sealing itself in
"""
import random
import time

from . import config as C
from .adversarial import (MacroModel, MState, EXIT, minimax, mcts, exit_wins,
                          best_tour, tour_choice, _timeline as adv_timeline)
from .fuzzy import RISK_SYSTEM, risk_label
from .logic import danger_analysis
from .search import GridGraph, astar, dijkstra_map, manhattan, bfs_distances, path_cost
from .world import DIRS, CRACKED, FIRE, WALL, FLOOR, DIRT

DIR_OF = {v: k for k, v in DIRS.items()}


class AIAgent:
    def __init__(self, pid=1, brain="minimax", seed=None):
        self.pid = pid
        self.brain = brain
        self.rng = random.Random(seed)
        self.last = {}
        self.budget = 1.0                     # seconds of search per turn

    # ------------------------------------------------------------------ main
    def decide(self, world):
        t0 = time.time()
        me = world.players[self.pid]
        op = world.players[1 - self.pid]
        info = {"brain": self.brain}
        if getattr(me, "frozen", 0):
            info.update(why=f"frozen solid ({me.frozen} more turn{'s' if me.frozen > 1 else ''})",
                        think=time.time() - t0, path=[], target=None, target_kind=None, search={})
            self.last = info
            return None, None, "wait", info

        # 1. fuzzy risk: how many turns a point of life is worth on the way
        lead = me.total - op.total
        fz = RISK_SYSTEM.evaluate(life=me.life, lead=lead)
        risk = fz["value"]
        lw = 1.1 - 0.3 * risk
        info.update(risk=risk, mood=risk_label(risk), fuzzy=fz, life_weight=lw)

        # 2. logic danger map
        la = danger_analysis(world, self.pid)
        info["logic"] = {k: la[k] for k in ("facts", "rules", "derived")}
        danger = la["danger"]

        # 3. macro model (every diamond, both players, the exit)
        # fire and snakes are already priced in life; only surprises (falling rocks, a snake
        # about to wake) get the extra danger penalty when routing
        priced = self._rock_holds(world)
        hard = {c for c, k in la["kinds"].items() if ("Crush" in k and c not in priced) or "LateBite" in k}
        graph = GridGraph(world, self.pid, life_weight=lw, danger=hard, allow_exit=False)
        model = self._build_model(world, graph)
        info["candidates"] = len(model.values)

        # anti-stall: if we haven't moved for a while (and aren't frozen), drop the goal
        # we keep failing to reach and take a little more risk
        self.turn_no = getattr(self, "turn_no", 0) + 1
        here0 = (me.r, me.c)
        if here0 == getattr(self, "last_pos", None):
            self.still = getattr(self, "still", 0) + 1
        else:
            self.still = 0
        self.last_pos = here0
        banned = getattr(self, "banned", {})
        self.banned = banned = {g: t for g, t in banned.items() if t > self.turn_no}
        if self.still >= 4:
            for g in (getattr(self, "goal", None), getattr(self, "lock_target", None)):
                if g is not None:
                    banned[g] = self.turn_no + 25
            self.goal = None
            self.tour = None
            self.lock_off_until = self.turn_no + 30
            self.still = 0
        for g in banned:
            model.values.pop(g, None)

        # 4. strategy
        P = world.players
        root = MState(["P0", "P1"], [0, 0], [P[0].points, P[1].points], [P[0].life, P[1].life],
                      [P[0].gems, P[1].gems], frozenset(model.values), last=1 - self.pid)
        # 4a. route planning: beam search over diamond orders, each candidate tour
        #     scored by playing the game out against a reacting rival
        tour, tour_v, _ = best_tour(model, root, self.pid, prev=getattr(self, "tour", None),
                                    budget=self.budget * 0.35)
        self.tour = tour
        if tour:
            model.tours[self.pid] = tour
        plan_first = tour_choice(model, root, self.pid) if tour else None
        info["tour"] = [g for g in (tour or ()) if g != EXIT][:12]
        # 4b. adversarial check of the next step (the rival may contest or steal)
        goal = getattr(self, "goal", None)
        if goal not in world.gems or goal == (me.r, me.c):
            goal = None
        must = [g for g in (plan_first, goal) if g is not None]
        if self.brain == "mcts":
            target, st = mcts(model, root, budget=self.budget * 0.65, rng=self.rng, must=must)
            tol = 0.08
        else:
            target, st = minimax(model, root, depth=8, budget=self.budget * 0.65, must=must)
            tol = 10.0
        info["search"] = st
        scored = dict(st.get("scored", []))
        best_v = scored.get(target)
        # commitment: keep walking to the current goal unless something is clearly better,
        # then prefer the planned tour, then the raw search result
        if best_v is not None and target != EXIT:
            if goal is not None and goal in scored and scored[goal] >= best_v - 1.5 * tol:
                target = goal
            elif plan_first is not None and plan_first in scored and \
                    scored[plan_first] >= best_v - tol:
                target = plan_first
        self.goal = target if target not in (None, EXIT) else None
        # finish it: if leaving now wins, walk out (even mid-chase)
        if exit_wins(model, root, self.pid):
            target = EXIT
            info["exit_reason"] = "leaving now wins"
        if target is None or (getattr(self, "hold_count", 0) >= 3 and target != EXIT):
            target = self._fallback(world, model) or target
        target_cell = world.exit_pos if target == EXIT else target
        # losing the race? lock the exit down with mines before the rival can walk out
        lock_tool = None
        self.lock_target = None
        if target != EXIT and self.turn_no >= getattr(self, "lock_off_until", 0):
            lock = self._lockdown(world, st)
            if lock:
                kind, data, why_lock = lock
                info["lockdown"] = why_lock
                if kind == "mine":
                    lock_tool = data
                    nxt = self._lockdown(world, st, skip=data[1])
                    if nxt and nxt[0] == "goto":
                        target_cell = nxt[1]
                else:
                    target_cell = data
                    self.lock_target = data
        info["target"] = target_cell
        info["target_kind"] = ("exit" if target == EXIT else None if target is None else
                               ("red" if world.gems.get(target) == "r" else "violet"))
        self.model, self.root = model, root

        # 5 + 6. tactics (the exit is only walkable when it is the goal)
        if target == EXIT:
            graph = GridGraph(world, self.pid, life_weight=lw, danger=hard, allow_exit=True)
        tool, tdir, move, path, why = self._tactics(world, graph, target_cell, danger,
                                                     la["kinds"])
        # never dig out a shaft that drops a boulder into our only way out
        for _ in range(3):
            if move in (None, "wait") or not self._self_trap(world, tool, tdir, move):
                break
            dr, dc = DIRS[move]
            graph.forbid.add((me.r + dr, me.c + dc))
            self.route = None
            tool, tdir, move, path, why = self._tactics(world, graph, target_cell, danger,
                                                         la["kinds"])
            why = why + " (avoiding a rock trap)"
        else:
            if move not in (None, "wait") and self._self_trap(world, tool, tdir, move):
                tool, tdir, move, why = None, None, "wait", "a rock would trap me"
        if lock_tool:
            d_mine, cell_mine = lock_tool
            if tool is not None and move not in (None, "wait") and \
                    not world.can_enter(self.pid, me.r + DIRS[move][0], me.c + DIRS[move][1]):
                move = "wait"                 # that step needed a tool; the mine comes first
            tool, tdir = "mine", d_mine
            why = "mining the way to the exit"
        elif info.get("lockdown"):
            why = "racing to mine the exit"
        if move == "wait" and la["must_leave"]:
            exits = self._safe_moves(world, danger) or [
                d for d, (dr, dc) in DIRS.items() if world.can_enter(self.pid, me.r + dr, me.c + dc)
                and (me.r + dr, me.c + dc) != world.exit_pos]
            if exits:
                move, why = exits[0], "stepping out of the fire"
        self.hold_count = getattr(self, "hold_count", 0) + 1 if move in (None, "wait") and \
            why.startswith(("holding", "racing")) else 0
        info.update(path=path, why=why, think=time.time() - t0)
        self.last = info
        return tool, tdir, move, info

    def _self_trap(self, world, tool, tdir, move):
        """Would this turn let a boulder fall and wall us off from most of the cave?"""
        me = world.players[self.pid]
        dr, dc = DIRS[move]
        cells = [(me.r, me.c), (me.r + dr, me.c + dc)]
        if tool in ("push", "hook") and tdir:
            tr, tc = DIRS[tdir]
            cells += [(me.r + tr * k, me.c + tc * k) for k in range(1, 5)]
        if not any((r - k, c) in world.rock_at for r, c in cells for k in range(1, 12)):
            return False
        def reach(w, src):
            g = GridGraph(w, self.pid, life_weight=1.0, allow_exit=True)
            dm = dijkstra_map(g, src)
            return sum(1 for x in w.gems if x in dm) + (3 if w.exit_pos in dm else 0)
        before = reach(world, (me.r, me.c))
        sim = world.clone()
        try:
            sim.take_turn(self.pid, tool, tdir, move)
        except ValueError:
            return False
        p = sim.players[self.pid]
        pos = (p.r, p.c)
        p.r, p.c = -9, -9                 # we will walk on: let what is above us fall too
        sim.resolve_gravity([])
        p.r, p.c = pos
        after = reach(sim, pos)
        return before >= 4 and after < before * 0.6

    def _lockdown(self, world, st, skip=None):
        """When our search says we are losing and the rival is close to opening the exit,
        spend mines where the rival must step to get out: the exit's open neighbours (or the
        chokepoints just before it). Our own mines never hurt us.
        Returns ("mine", (dir, cell), why) when we can place one now, ("goto", cell, why)
        to walk to a placing spot, or None."""
        me = world.players[self.pid]
        op = world.players[1 - self.pid]
        ex = world.exit_pos
        if not getattr(world, "mines_enabled", False) or me.mines <= 0 or ex is None:
            return None
        if op.gems < C.EXIT_UNLOCK_GEMS - 3 or op.exited or not op.alive:
            return None
        v = st.get("value", 0)
        losing = (v < 0.45) if self.brain == "mcts" else (v < 0)
        lethal = op.life <= C.MINE_DAMAGE
        if not losing and op.total <= me.total and not lethal:
            return None
        op_d = bfs_distances(world, (op.r, op.c))
        if ex not in op_d:
            return None
        # cells the rival must cross: exit neighbours, else chokepoints on its route
        nbrs = [(ex[0] + dr, ex[1] + dc) for dr, dc in DIRS.values()]
        open_n = [n for n in nbrs if world.inside(*n) and world.terrain[n[0]][n[1]] in (FLOOR, DIRT)
                  and n not in world.rock_at and n not in world.statue_cells]
        mined = set(world.mines)
        todo = [n for n in open_n if n not in mined]
        if len(todo) > me.mines:
            # too many ways in: mine the rival's route where a block costs it most
            ex_d = bfs_distances(world, ex)
            route = [c for c in op_d if c in ex_d and op_d[c] + ex_d[c] == op_d[ex] and 0 < ex_d[c] <= 7
                     and world.terrain[c[0]][c[1]] in (FLOOR, DIRT) and c not in mined
                     and c not in world.rock_at and c not in world.statue_cells]
            scored = []
            for c in route:
                sim = world.clone()
                sim.terrain[c[0]][c[1]] = WALL
                after = bfs_distances(sim, (op.r, op.c)).get(ex, 999)
                scored.append((min(after - op_d[ex], C.MINE_DAMAGE), -ex_d[c], c))
            scored.sort(reverse=True)
            todo = [c for gain, _, c in scored if gain >= (2 if lethal else 4)][:me.mines]
        todo = [c for c in todo if c != skip and not world.players_at(*c) and c not in world.gems
                and c not in world.snakes and world.terrain[c[0]][c[1]] in (FLOOR, DIRT)]
        if not todo:
            return None
        here = (me.r, me.c)
        my_d = bfs_distances(world, here)
        best = None
        for n in todo:
            for d, (dr, dc) in DIRS.items():
                stand = (n[0] - dr, n[1] - dc)          # from here a mine goes `d` onto n
                if stand == ex or stand not in my_d or stand in mined or stand in world.statue_cells:
                    continue
                if stand == here:
                    if world.tool_target(self.pid, "mine", d)[0]:
                        return "mine", (d, n), "mining the exit"
                    continue                  # can't place it from here: never wait on it
                cost = my_d[stand]
                if best is None or cost < best[0]:
                    best = (cost, stand, n)
        if best is None:
            return None
        # only worth it if we get there before the rival can walk out
        if best[0] + 1 >= op_d[ex] + (0 if op.gems >= C.EXIT_UNLOCK_GEMS else 6):
            return None
        return "goto", best[1], "racing to mine the exit"

    def _fallback(self, world, model):
        """Nothing worth a trip under the safety margin: leave if we can, otherwise the
        cheapest diamond that is still worth more than it costs (or nothing)."""
        me = world.players[self.pid]
        key = "P%d" % self.pid
        tr = model.travel(key, EXIT)
        if world.exit_open_for(self.pid) and tr and me.life - tr[1] > 0:
            return EXIT
        best = None
        for g, v in model.values.items():
            t = model.travel(key, g)
            if t and me.life - t[1] > 0 and v - t[1] > 0:
                sc = (v - t[1]) / (t[0] + 2)
                if best is None or sc > best[0]:
                    best = (sc, g)
        return best[1] if best else None

    def _try_mine(self, world, here, nxt):
        """Strategic mines. The rival's next diamonds (as our model predicts its route) and,
        once it is getting close, the exit are its goals. A mine on a cell next to us is
        worth the detour it forces on the way to each goal (a true chokepoint costs a 25-life
        toll instead, or a knockout when the rival is low). Place the best one if it matters."""
        me = world.players[self.pid]
        op = world.players[1 - self.pid]
        if me.mines <= 0 or world.tool_used or not op.alive or op.exited:
            return None
        model, root = getattr(self, "model", None), getattr(self, "root", None)
        if model is None:
            return None
        goals = []
        _times, _tot, taken = adv_timeline(model, root, op.pid)
        for i, g in enumerate(sorted(taken, key=taken.get)[:3]):
            goals.append((g, (1.0, 0.6, 0.4)[i]))
        ex = world.exit_pos
        if ex is not None and op.gems >= C.EXIT_UNLOCK_GEMS - 6:
            goals.append((ex, 1.2 if op.gems >= C.EXIT_UNLOCK_GEMS - 3 else 0.5))
        if not goals:
            return None
        # the rival walks round mines it can see: treat ours as walls
        base_w = world.clone()
        for cell, owner in world.mines.items():
            if owner == self.pid:
                base_w.terrain[cell[0]][cell[1]] = WALL
        src = (op.r, op.c)
        base = bfs_distances(base_w, src)
        goals = [(g, wt) for g, wt in goals if g in base and g != src]
        if not goals:
            return None
        lethal = op.life <= C.MINE_DAMAGE
        best = None
        for d, (dr, dc) in DIRS.items():
            cell = (here[0] + dr, here[1] + dc)
            if cell == src or cell not in base:
                continue
            ok, _ = world.tool_target(self.pid, "mine", d)
            if not ok:
                continue
            sim = base_w.clone()
            sim.terrain[cell[0]][cell[1]] = WALL
            after = bfs_distances(sim, src)
            value = 0.0
            for g, wt in goals:
                if g == cell:
                    continue
                if g not in after:          # no way round: walk over it or give up
                    value += wt * (100 if lethal else C.MINE_DAMAGE)
                else:
                    value += wt * min(after[g] - base[g], C.MINE_DAMAGE)
            if value >= 6 and (best is None or value > best[0]):
                best = (value, d)
        if best:
            return "mine", best[1], f"mining your route (worth {best[0]:.0f})"
        return None

    # ----------------------------------------------------------- macro model
    def _signature(self, world):
        return hash((tuple(map(tuple, world.terrain)), frozenset(world.rock_at),
                     frozenset(world.snakes), frozenset(getattr(world, "mines", {}).items()),
                     frozenset((k, s.d) for k, s in getattr(world, "statues", {}).items())))

    def _build_model(self, world, graph):
        """Travel table. Diamond-to-diamond and diamond-to-exit legs are cached until the
        cave itself changes (a wall breaks, a rock moves, a snake dies, a mine appears)."""
        me = world.players[self.pid]
        op = world.players[1 - self.pid]
        gems = list(world.gems)
        values = {g: (C.RED_POINTS if world.gems[g] == "r" else C.VIOLET_POINTS) for g in gems}
        sig = self._signature(world)
        cache = getattr(self, "_legs", None)
        if cache is None or cache[0] != sig:
            cache = (sig, {}, None)
            self._legs = cache
        legs = cache[1]
        base = GridGraph(world, self.pid, life_weight=1.0, allow_exit=False)
        ex = world.exit_pos
        if cache[2] is None:
            em = dijkstra_map(GridGraph(world, self.pid, life_weight=1.0, allow_exit=True), ex)
            self._legs = cache = (sig, legs, em)
        em = cache[2]
        table = {}
        for g in gems:
            if g not in legs:
                dm = dijkstra_map(base, g)
                legs[g] = {h: (dm[h][1], dm[h][2]) for h in world.gems if h in dm and h != g}
                if g in em:
                    legs[g][EXIT] = (em[g][1], max(0, em[g][2] - base.life_cost(g)))
            for h, tr in legs[g].items():
                if h == EXIT or h in values:
                    table[(g, h)] = tr
        # (local dangers only steer the next steps, not the long-range plan)
        my = dijkstra_map(GridGraph(world, self.pid, life_weight=graph.lw, allow_exit=False),
                          (me.r, me.c))
        og = GridGraph(world, op.pid, life_weight=1.0, allow_exit=False)
        om = dijkstra_map(og, (op.r, op.c))
        for key, dm, p in (("P%d" % self.pid, my, me), ("P%d" % op.pid, om, op)):
            for g in gems:
                if g in dm:
                    table[(key, g)] = (dm[g][1], dm[g][2])
            here = (p.r, p.c)
            if here in em:
                table[(key, EXIT)] = (em[here][1], max(0, em[here][2] - base.life_cost(here)))
        horizon = max(1, C.MAX_ROUNDS - world.rounds)
        return MacroModel(gems, values, table, self.pid, horizon=horizon)

    # --------------------------------------------------------------- tactics
    def _tactics(self, world, graph, target, danger, kinds):
        me = world.players[self.pid]
        here = (me.r, me.c)
        crush = {c for c, k in kinds.items() if "Crush" in k} - self._rock_holds(world)

        path = []
        old = getattr(self, "route", None)
        if old and old[0] == target and here in old[1]:
            rest = old[1][old[1].index(here):]
            nxt0 = rest[1] if len(rest) > 1 else None
            if nxt0 is not None and nxt0 not in crush and nxt0 not in graph.lit and (
                    world.can_enter(self.pid, *nxt0) or nxt0 in world.rock_at or
                    world.terrain[nxt0[0]][nxt0[1]] == CRACKED):
                path = rest
        if not path and target is not None:
            res = astar(graph, here, target)
            if not res.found:
                g2 = GridGraph(world, self.pid, graph.lw, graph.danger, rocks_passable=True,
                               allow_exit=graph.allow_exit)
                res = astar(g2, here, target)
            path = res.path
        # walking out ends the game, so any life left is enough; a stuck AI takes more risk
        margin = 1 if target == world.exit_pos or getattr(self, "hold_count", 0) >= 3 else 8
        path = self._survivable(world, graph, here, target, path, margin)
        self.route = (target, path)

        # a blow we can land right now: drop a rock on the rival, swing a beam onto them
        hit = self._try_attack(world, path)
        if hit:
            return hit
        # a lens statue in hook range: turning it may clear our way (or catch the rival)
        lens = self._try_statue(world, graph, target, path)
        if lens:
            return lens
        if len(path) > 1 and path[1] in graph.lit:
            # about to walk into a beam: go round to the statue's body row and turn it, if we
            # can get there without crossing a beam; otherwise plan with the real price
            sub = self._statue_approach(world, graph, path[1])
            if sub:
                self.route = (target, sub)
                nxt = sub[1]
                d = DIR_OF[(nxt[0] - here[0], nxt[1] - here[1])]
                return None, None, d, sub, "going round to turn a lens statue"
            true = GridGraph(world, self.pid, graph.lw, graph.danger, allow_exit=graph.allow_exit,
                             lens_cheap=False)
            res = astar(true, here, target)
            if res.found:
                path = res.path
                self.route = (target, path)

        # standing in a snake cell: our stun (or the rival's) runs out soon, so move on
        in_snake = here in world.snakes
        block = None if in_snake else self._try_block(world)

        if len(path) < 2:
            mine = self._try_mine(world, here, None)
            if mine:
                return mine[0], mine[1], "wait", path, mine[2]
            return None, None, "wait", path, "holding position"

        nxt = path[1]
        d = DIR_OF[(nxt[0] - here[0], nxt[1] - here[1])]
        if nxt in crush:
            # gravity is instant: nothing is "about to fall". Stepping there brings a rock
            # down on us (e.g. stepping straight down from under one). Go round if we can,
            # take it if we can afford it, otherwise pick the least harmful step.
            g2 = GridGraph(world, self.pid, graph.lw, graph.danger, allow_exit=graph.allow_exit)
            g2.forbid = set(graph.forbid) | crush
            res = astar(g2, here, target) if target is not None else None
            if res is not None and res.found and len(res.path) > 1 and \
                    path_cost(g2, res.path) <= path_cost(graph, path) + C.CRUSH_DAMAGE * graph.lw:
                path = res.path
                self.route = (target, path)
                nxt = path[1]
                d = DIR_OF[(nxt[0] - here[0], nxt[1] - here[1])]
            elif me.life <= C.CRUSH_DAMAGE + 8:
                return self._least_bad(world, kinds, path)
        self.waits = 0
        if nxt in world.snakes or world.terrain[nxt[0]][nxt[1]] == CRACKED:
            return "hammer", d, d, path, f"hammering {d}"
        if nxt in world.rock_at:
            ok, _ = world.tool_target(self.pid, "push", d)
            if ok:
                return "push", d, d, path, f"pushing a rock {d}"
            moves = self._safe_moves(world, danger)
            return None, None, (moves[0] if moves else "wait"), path, "rock in the way"

        if block:
            tool, tdir, why = block
            sim = world.clone()
            sim.apply_tool(self.pid, tool, tdir, [])
            if sim.can_enter(self.pid, *nxt):
                return tool, tdir, d, path, why
            return tool, tdir, "wait", path, why
        mine = self._try_mine(world, here, nxt)
        if mine:
            return mine[0], mine[1], d, path, mine[2]
        return None, None, d, path, ("heading to the exit" if target == world.exit_pos
                                     else "heading to a diamond")

    def _try_statue(self, world, graph, target, path):
        """Hook a lens statue when the turned beam saves more than the hook costs: a
        shorter or safer route for us, and a bonus if the beam swings onto the rival."""
        if target is None or world.tool_used:
            return None
        opts = [(d, info) for tool, d, info in world.valid_tools(self.pid)
                if tool == "hook" and info[0] == "statue"]
        if not opts:
            return None
        me = world.players[self.pid]
        here = (me.r, me.c)
        true = GridGraph(world, self.pid, graph.lw, graph.danger, allow_exit=graph.allow_exit,
                         lens_cheap=False)
        cur = path_cost(true, path) if len(path) > 1 else 10 ** 6
        best = None
        for d, _info in opts:
            sim = world.clone()
            try:
                ev = sim.use_tool(self.pid, "hook", d)
            except ValueError:
                continue
            if not sim.players[self.pid].alive:
                continue
            g2 = GridGraph(sim, self.pid, graph.lw, graph.danger, allow_exit=graph.allow_exit,
                           lens_cheap=False)
            res = astar(g2, here, target)
            if not res.found or len(res.path) < 2:
                continue
            new = path_cost(g2, res.path) + graph.lw * C.HOOK_COST
            bonus = sum(C.LENS_DAMAGE + 15 for e in ev if e["type"] == "freeze" and e["pid"] != self.pid)
            gain = cur - new + bonus
            if gain > 4 and (best is None or gain > best[0]):
                best = (gain, d, res.path, bonus)
        if not best:
            return None
        gain, d, p2, bonus = best
        self.route = (target, p2)
        nxt = p2[1]
        mv = DIR_OF[(nxt[0] - here[0], nxt[1] - here[1])]
        why = "turning a lens statue on you" if bonus else "turning a lens statue out of my way"
        return "hook", d, mv, p2, why

    def _try_attack(self, world, path):
        """Try every push / hook we can make now; take the one that hurts the rival most
        for what it costs us (a knockout wins the game outright)."""
        if world.tool_used:
            return None
        me = world.players[self.pid]
        op = world.players[1 - self.pid]
        here = (me.r, me.c)
        best = None
        for tool, d, info in world.valid_tools(self.pid):
            if tool not in ("push", "hook"):
                continue
            sim = world.clone()
            try:
                ev = sim.use_tool(self.pid, tool, d)
            except ValueError:
                continue
            if not sim.players[self.pid].alive:
                continue
            op_loss = sum(e["amount"] for e in ev if e["type"] == "damage" and e["pid"] == op.pid)
            my_loss = sum(e["amount"] for e in ev if e["type"] == "damage" and e["pid"] == self.pid)
            gain = op_loss - my_loss + (1000 if not sim.players[op.pid].alive else 0)
            if op_loss and gain >= 20 and (best is None or gain > best[0]):
                mv = "wait"
                if len(path) > 1 and sim.can_enter(self.pid, *path[1]) and path[1] not in sim.lit():
                    mv = DIR_OF[(path[1][0] - here[0], path[1][1] - here[1])]
                best = (gain, tool, d, mv, not sim.players[op.pid].alive)
        if not best:
            return None
        gain, tool, d, mv, ko = best
        why = "finishing you off" if ko else ("dropping a rock on you" if tool == "push"
                                              else "hooking something onto you")
        return tool, d, mv, path, why

    @staticmethod
    def _rock_holds(world):
        """Cells holding a rock up (soil, a diamond, a snake): entering one brings the rock
        down on you. Their price is in the route costs, so tactics don't dodge them."""
        return {(r + 1, c) for (r, c) in world.rock_at
                if world.inside(r + 1, c) and (world.terrain[r + 1][c] == DIRT or (r + 1, c) in world.gems
                                               or (r + 1, c) in world.snakes)}

    def _survivable(self, world, graph, here, target, path, margin=8):
        """Never walk a route whose damage adds up to (nearly) all our life: mines, fire,
        beams and tool costs along the way. Re-route round the harmful cells, or stay put."""
        if len(path) < 2:
            return path
        me = world.players[self.pid]
        true = GridGraph(world, self.pid, graph.lw, graph.danger, allow_exit=graph.allow_exit,
                         lens_cheap=False)
        steps = list(zip(path, path[1:]))
        cum = sum(true.life_cost(n, p) for p, n in steps)
        if me.life - cum >= margin:
            return path
        harmful = {n for p, n in steps if true.life_cost(n, p) >= 10}
        g2 = GridGraph(world, self.pid, graph.lw, graph.danger, allow_exit=graph.allow_exit)
        g2.forbid = set(graph.forbid) | harmful
        res = astar(g2, here, target)
        if res.found:
            cum2 = sum(true.life_cost(n, p) for p, n in zip(res.path, res.path[1:]))
            if me.life - cum2 >= margin:
                return res.path
        # no safe way there: only take the first step if it costs nothing
        if true.life_cost(path[1], here) == 0:
            return path[:2]
        return [here]

    def _statue_approach(self, world, graph, beam_cell):
        """Shortest beam-free walk to a cell in the body row of the statue lighting beam_cell."""
        sid = graph.lit.get(beam_cell)
        st = next((s for s in world.statues.values() if s.sid == sid), None)
        if st is None:
            return None
        me = world.players[self.pid]
        here = (me.r, me.c)
        spots = [c for c in world.statue_hook_spots(st) if c not in graph.lit]
        if not spots or here in spots:
            return None
        g = GridGraph(world, self.pid, graph.lw, graph.danger, allow_exit=False, lens_cheap=False)
        g.forbid = set(graph.lit)
        best = None
        for sp in spots:
            res = astar(g, here, sp)
            if res.found and len(res.path) <= 14 and (best is None or len(res.path) < len(best)):
                best = res.path
        return best if best and len(best) > 1 else None

    def _least_bad(self, world, kinds, path):
        """Boxed in: pick the neighbouring cell that costs the least life."""
        me = world.players[self.pid]
        here = (me.r, me.c)
        cost_of = {"Crush": C.CRUSH_DAMAGE, "Burn": C.FIRE_DAMAGE, "Bite": C.HAMMER_COST,
                   "LateBite": C.SNAKE_BITE, "Freeze": C.LENS_DAMAGE}
        best = (C.CRUSH_DAMAGE if here in kinds and "Crush" in kinds[here] else 0, "wait")
        if world.terrain[me.r][me.c] == FIRE:
            best = (best[0] + C.FIRE_DAMAGE, "wait")
        if here in world.snakes:
            best = (C.SNAKE_BITE, "wait")
        for d, (dr, dc) in DIRS.items():
            n = (me.r + dr, me.c + dc)
            if not world.can_enter(self.pid, *n) or n == world.exit_pos:
                continue
            cost = sum(cost_of[k] for k in kinds.get(n, []))
            if cost < best[0]:
                best = (cost, d)
        d = best[1]
        if d != "wait":
            n = (me.r + DIRS[d][0], me.c + DIRS[d][1])
            if n in world.snakes:
                return "hammer", d, d, path, "squeezing past a snake"
        return None, None, d, path, "boxed in, taking the safest step"

    def _safe_moves(self, world, danger):
        me = world.players[self.pid]
        out = []
        for d, (dr, dc) in DIRS.items():
            n = (me.r + dr, me.c + dc)
            if world.can_enter(self.pid, *n) and n not in danger and n not in world.snakes \
                    and n != world.exit_pos:
                out.append(d)
        self.rng.shuffle(out)
        return out

    def _try_block(self, world):
        """If the human is close to a red diamond, see whether pushing or hooking a rock
        lengthens their route a lot."""
        me = world.players[self.pid]
        op = world.players[1 - self.pid]
        if me.life < 45:
            return None
        reds = [g for g, k in world.gems.items() if k == "r" and manhattan(g, (op.r, op.c)) <= 7]
        if not reds:
            return None
        before = bfs_distances(world, (op.r, op.c))
        reds = [g for g in reds if g in before]
        if not reds:
            return None
        goal = min(reds, key=lambda g: before[g])
        if before[goal] > 8:
            return None
        mine = bfs_distances(world, (me.r, me.c)).get(goal, 99)
        if mine <= before[goal]:
            return None          # we can win the race anyway
        best = None
        for tool, d, info in world.valid_tools(self.pid):
            if tool == "hammer" or info[0] != "rock":
                continue
            sim = world.clone()
            sim.apply_tool(self.pid, tool, d, [])
            sim.resolve_gravity([])
            after = bfs_distances(sim, (op.r, op.c)).get(goal, 999)
            gain = after - before[goal]
            if gain >= 5 and (best is None or gain > best[0]):
                best = (gain, tool, d)
        if best:
            return best[1], best[2], f"blocking you with a {best[1]} (+{min(best[0], 99)} steps)"
        return None
