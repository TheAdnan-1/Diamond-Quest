"""Mamdani fuzzy inference system.

Pipeline:  crisp inputs -> fuzzification (membership functions)
           -> rule evaluation (AND = min, OR = max)
           -> implication (min, clips the output set)
           -> aggregation (max)
           -> defuzzification (centroid)

Used by the AI to decide its *risk appetite* from its life and its score lead.
"""


def trimf(a, b, c):
    def f(x):
        if x <= a or x >= c:
            return 1.0 if x == b else 0.0
        return (x - a) / (b - a) if x < b else (c - x) / (c - b)
    return f


def trapmf(a, b, c, d):
    def f(x):
        if x < a or x > d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if x < b:
            return (x - a) / (b - a) if b > a else 1.0
        return (d - x) / (d - c) if d > c else 1.0
    return f


class FuzzyVar:
    def __init__(self, name, lo, hi, terms, samples=201):
        self.name, self.lo, self.hi = name, lo, hi
        self.terms = terms                      # term name -> membership function
        step = (hi - lo) / (samples - 1)
        self.universe = [lo + i * step for i in range(samples)]

    def fuzzify(self, x):
        x = max(self.lo, min(self.hi, x))
        return {t: round(f(x), 4) for t, f in self.terms.items()}


class Mamdani:
    def __init__(self, inputs, output, rules):
        self.inputs = {v.name: v for v in inputs}
        self.output = output
        self.rules = rules          # list of ({var: term}, out_term, "and"|"or")

    def evaluate(self, **crisp):
        fz = {name: var.fuzzify(crisp[name]) for name, var in self.inputs.items()}
        strength = {t: 0.0 for t in self.output.terms}
        fired = []
        for ante, out_term, op in self.rules:
            degs = [fz[v][t] for v, t in ante.items()]
            s = min(degs) if op == "and" else max(degs)
            if s > 0:
                fired.append((ante, out_term, s))
            strength[out_term] = max(strength[out_term], s)          # aggregation
        agg = []
        for x in self.output.universe:
            mu = 0.0
            for t, f in self.output.terms.items():
                mu = max(mu, min(strength[t], f(x)))                   # implication
            agg.append(mu)
        area = sum(agg)
        value = sum(x * m for x, m in zip(self.output.universe, agg)) / area if area else 0.5
        return {"value": value, "fuzzified": fz, "strength": strength, "fired": fired,
                "aggregate": agg}


def build_risk_system():
    life = FuzzyVar("life", 0, 100, {
        "low": trapmf(0, 0, 25, 45),
        "medium": trimf(30, 55, 80),
        "high": trapmf(65, 85, 100, 100),
    })
    lead = FuzzyVar("lead", -150, 150, {
        "behind": trapmf(-150, -150, -60, -10),
        "even": trimf(-40, 0, 40),
        "ahead": trapmf(10, 60, 150, 150),
    })
    risk = FuzzyVar("risk", 0, 1, {
        "cautious": trapmf(0, 0, 0.2, 0.4),
        "balanced": trimf(0.3, 0.5, 0.7),
        "bold": trapmf(0.6, 0.8, 1, 1),
    })
    R = [
        ({"life": "low", "lead": "behind"}, "balanced", "and"),
        ({"life": "low", "lead": "even"}, "cautious", "and"),
        ({"life": "low", "lead": "ahead"}, "cautious", "and"),
        ({"life": "medium", "lead": "behind"}, "bold", "and"),
        ({"life": "medium", "lead": "even"}, "balanced", "and"),
        ({"life": "medium", "lead": "ahead"}, "cautious", "and"),
        ({"life": "high", "lead": "behind"}, "bold", "and"),
        ({"life": "high", "lead": "even"}, "bold", "and"),
        ({"life": "high", "lead": "ahead"}, "balanced", "and"),
    ]
    return Mamdani([life, lead], risk, R)


RISK_SYSTEM = build_risk_system()


def risk_label(v):
    return "Cautious" if v < 0.38 else "Bold" if v > 0.62 else "Balanced"
