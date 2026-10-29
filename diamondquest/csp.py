"""A small, general Constraint Satisfaction Problem solver.

Backtracking search with
  * MRV  (minimum remaining values) variable ordering,
  * degree heuristic as a tie-breaker,
  * forward checking for n-ary constraints (when a constraint has exactly one
    unassigned variable left, that variable's domain is filtered).
"""
import random


class Constraint:
    def __init__(self, scope, fn, name=""):
        self.scope = tuple(scope)
        self.fn = fn
        self.name = name

    def ok(self, assignment):
        return self.fn(*(assignment[v] for v in self.scope))


class CSP:
    def __init__(self, variables, domains, constraints, rng=None):
        self.variables = list(variables)
        self.domains = {v: list(domains[v]) for v in self.variables}
        self.constraints = constraints
        self.rng = rng or random.Random()
        self.by_var = {v: [] for v in self.variables}
        for con in constraints:
            for v in con.scope:
                self.by_var[v].append(con)
        self.nodes = 0
        self.backtracks = 0

    def _consistent(self, var, assignment):
        for con in self.by_var[var]:
            if all(v in assignment for v in con.scope) and not con.ok(assignment):
                return False
        return True

    def _select(self, assignment, domains):
        unassigned = [v for v in self.variables if v not in assignment]
        return min(unassigned, key=lambda v: (
            len(domains[v]),
            -sum(1 for con in self.by_var[v]
                 if any(u not in assignment and u != v for u in con.scope)),
            self.rng.random()))

    def _forward_check(self, var, assignment, domains):
        new = dict(domains)
        for con in self.by_var[var]:
            free = [v for v in con.scope if v not in assignment]
            if len(free) != 1:
                continue
            u = free[0]
            kept = []
            for val in new[u]:
                assignment[u] = val
                if con.ok(assignment):
                    kept.append(val)
                del assignment[u]
            if not kept:
                return None
            new[u] = kept
        return new

    def solve(self, max_nodes=50000):
        domains = {v: self.rng.sample(d, len(d)) for v, d in self.domains.items()}
        return self._backtrack({}, domains, max_nodes)

    def _backtrack(self, assignment, domains, max_nodes):
        if len(assignment) == len(self.variables):
            return dict(assignment)
        if self.nodes >= max_nodes:
            return None
        var = self._select(assignment, domains)
        for val in domains[var]:
            self.nodes += 1
            assignment[var] = val
            if self._consistent(var, assignment):
                pruned = self._forward_check(var, assignment, domains)
                if pruned is not None:
                    pruned[var] = [val]
                    res = self._backtrack(assignment, pruned, max_nodes)
                    if res is not None:
                        return res
            del assignment[var]
            self.backtracks += 1
            if self.nodes >= max_nodes:
                return None
        return None
