"""Inference engine for propositional logic (definite / Horn clauses).

  * tell(fact) / rule(premises, conclusion)
  * forward_chain()  -- data-driven, fires rules until nothing new is derived
  * ask(query)       -- goal-driven backward chaining with a proof trace

Used for the "Danger Sense": the world near a player is translated into
propositional facts such as Rock_12_5, Falling_12_5, Fire_13_5, AwakeSnake_9_9,
ground rules are generated per cell, and the engine derives Danger_r_c / Safe_r_c.
"""
from collections import defaultdict, deque

from .world import WALL, CRACKED, FIRE, DIRT


class KB:
    def __init__(self):
        self.facts = set()
        self.rules = []                     # (frozenset premises, conclusion)
        self._by_premise = defaultdict(list)

    def tell(self, sym):
        self.facts.add(sym)

    def rule(self, premises, conclusion):
        idx = len(self.rules)
        self.rules.append((frozenset(premises), conclusion))
        for p in premises:
            self._by_premise[p].append(idx)

    def forward_chain(self):
        """Classic count-based forward chaining (linear in KB size)."""
        count = [len(p) for p, _ in self.rules]
        inferred = set()
        agenda = deque(self.facts)
        derived = set(self.facts)
        while agenda:
            p = agenda.popleft()
            if p in inferred:
                continue
            inferred.add(p)
            for i in self._by_premise.get(p, ()):
                count[i] -= 1
                if count[i] == 0:
                    concl = self.rules[i][1]
                    if concl not in derived:
                        derived.add(concl)
                        agenda.append(concl)
        return derived

    def ask(self, query, _seen=None, _trace=None):
        """Backward chaining. Returns (proved, trace)."""
        seen = set() if _seen is None else _seen
        trace = [] if _trace is None else _trace
        if query in self.facts:
            trace.append(f"{query}  [fact]")
            return True, trace
        if query in seen:
            return False, trace
        seen.add(query)
        for prem, concl in self.rules:
            if concl != query:
                continue
            sub = []
            if all(self.ask(p, seen, sub)[0] for p in prem):
                trace.extend(sub)
                trace.append(f"{' & '.join(sorted(prem))}  =>  {query}")
                return True, trace
        return False, trace


def sym(name, cell):
    return f"{name}_{cell[0]}_{cell[1]}"


def danger_analysis(world, pid, radius=7):
    """Build a KB for the area around player pid and derive dangerous / safe cells."""
    p = world.players[pid]
    kb = KB()
    cells = []
    for r in range(p.r - radius, p.r + radius + 1):
        for c in range(p.c - radius, p.c + radius + 1):
            if world.inside(r, c):
                cells.append((r, c))
    kb.tell(sym("Me", (p.r, p.c)))
    beams = world.lit(edges=True) if getattr(world, "statues", None) else {}
    for cell in cells:
        r, c = cell
        t = world.terrain[r][c]
        if cell in world.rock_at:
            kb.tell(sym("Rock", cell))
            if world.rock_at[cell] in world.falling:
                kb.tell(sym("Falling", cell))
        if t == FIRE:
            kb.tell(sym("Fire", cell))
        if t == DIRT or cell in world.gems or cell in world.snakes:
            kb.tell(sym("Holds", cell))        # entering it removes what holds a rock up
        if cell in beams:
            kb.tell(sym("Beam", cell))
        if t not in (WALL, CRACKED) and cell not in world.rock_at:
            kb.tell(sym("Open", cell))
        if world.is_empty_floor(r, c):
            kb.tell(sym("Vacant", cell))
        if cell in world.snakes:
            s = world.snakes[cell]
            if pid in s.stuns:
                kb.tell(sym("StunnedByMe", cell))
            elif s.stuns:
                kb.tell(sym("StunnedByOther", cell))
            else:
                kb.tell(sym("AwakeSnake", cell))
        # ---- ground rules for this cell
        up, up2 = (r - 1, c), (r - 2, c)
        kb.rule([sym("Falling", up)], sym("Crush", cell))                      # R1
        kb.rule([sym("Falling", up2), sym("Vacant", up)], sym("Crush", cell))  # R2
        kb.rule([sym("Rock", up), sym("Vacant", cell)], sym("Crush", cell))    # R3
        kb.rule([sym("Rock", up), sym("Holds", cell)], sym("Crush", cell))     # R3a: dig under a rock
        kb.rule([sym("Rock", up2), sym("Me", up)], sym("Crush", cell))         # R3b: dig-under trap
        for side in (-1, 1):                                                   # R3c: a rock that
            kb.rule([sym("Me", up), sym("Rock", (r - 2, c + side)),            # rolls into the cell
                     sym("Rock", (r - 1, c + side)), sym("Vacant", up2)],      # I am leaving
                    sym("Crush", cell))
        kb.rule([sym("Fire", cell)], sym("Burn", cell))                        # R4
        kb.rule([sym("Fire", cell), sym("Me", cell)], sym("MustLeave", cell))  # R4b: burns again
        kb.rule([sym("AwakeSnake", cell)], sym("Bite", cell))                  # R5
        kb.rule([sym("StunnedByOther", cell)], sym("LateBite", cell))          # R6
        kb.rule([sym("Beam", cell)], sym("Freeze", cell))                      # R6b: lens beam
        for hz in ("Crush", "Burn", "Bite", "LateBite", "Freeze"):
            kb.rule([sym(hz, cell)], sym("Danger", cell))                      # R7
        kb.rule([sym("Rock", up), sym("Open", cell)], sym("Unstable", cell))   # R8
    derived = kb.forward_chain()
    must_leave = sym("MustLeave", (p.r, p.c)) in derived
    danger, kinds, safe, unstable = set(), {}, set(), set()
    for cell in cells:
        if sym("Danger", cell) in derived:
            danger.add(cell)
            kinds[cell] = [h for h in ("Crush", "Burn", "Bite", "LateBite", "Freeze")
                           if sym(h, cell) in derived]
        elif sym("Open", cell) in derived:
            safe.add(cell)                       # negation as failure: not provably dangerous
        if sym("Unstable", cell) in derived:
            unstable.add(cell)
    return {"danger": danger, "kinds": kinds, "safe": safe, "unstable": unstable,
            "must_leave": must_leave,
            "facts": len(kb.facts), "rules": len(kb.rules), "derived": len(derived), "kb": kb}
