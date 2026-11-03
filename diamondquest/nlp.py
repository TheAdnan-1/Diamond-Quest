"""Natural-language commands: tokenizer + context-free grammar + recursive-descent parser.

Grammar (terminals are word classes from the lexicon):

    S        -> CMD (CONJ* CMD)*              (commands may also simply follow each other)
    CMD      -> TOOLCMD | MOVECMD | WAITCMD
    TOOLCMD  -> [USE [DET]] TOOL [PREP] [DET] [OBJ] [PREP] [DET] DIR
    MOVECMD  -> [MOVEV] DIR [COUNT]          (MOVEV is required unless DIR starts the command)
    WAITCMD  -> WAITV [COUNT]
    COUNT    -> NUM [UNIT] | ADVCOUNT
    CONJ     -> 'then' | 'and' | 'and then' | ',' | 'after that'

Examples:  "hammer left then move left"   "use the hook on the rock to the right"
           "go down 3 times, wait"        "push rock right and walk right twice"
"""
import re

LEXICON = {
    "MOVEV": {"move", "go", "walk", "step", "run", "head", "dig"},
    "WAITV": {"wait", "stay", "skip", "pass", "rest"},
    "USE": {"use", "swing", "throw", "apply"},
    "TOOL": {"hammer": "hammer", "smash": "hammer", "stun": "hammer", "hit": "hammer",
             "break": "hammer", "push": "push", "shove": "push", "hook": "hook",
             "pull": "hook", "grab": "hook", "drag": "hook"},
    "DIR": {"up": "up", "north": "up", "down": "down", "south": "down", "left": "left",
            "west": "left", "right": "right", "east": "right"},
    "OBJ": {"rock", "boulder", "snake", "wall", "stone"},
    "DET": {"the", "a", "that", "this"},
    "PREP": {"on", "to", "at", "towards", "toward", "into"},
    "UNIT": {"times", "time", "steps", "step", "cells", "cell", "tiles", "turns", "turn"},
    "ADVCOUNT": {"once": 1, "twice": 2, "thrice": 3},
    "NUMWORD": {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10},
    "FILLER": {"please", "now", "then?"},
}
CONJ_WORDS = {"then", "and", ",", "after", "that"}


class ParseError(Exception):
    pass


def tokenize(text):
    text = text.lower().replace(",", " , ").replace(".", " ")
    toks = [t for t in re.split(r"\s+", text) if t and t not in LEXICON["FILLER"]]
    out = []
    i = 0
    while i < len(toks):
        if toks[i] == "after" and i + 1 < len(toks) and toks[i + 1] == "that":
            out.append(("CONJ", "after that"))
            i += 2
            continue
        if toks[i] == "and" and i + 1 < len(toks) and toks[i + 1] == "then":
            out.append(("CONJ", "and then"))
            i += 2
            continue
        out.append((classify(toks[i]), toks[i]))
        i += 1
    return out


def classify(w):
    if w in (",", "then", "and"):
        return "CONJ"
    if w.isdigit():
        return "NUM"
    if w in LEXICON["NUMWORD"]:
        return "NUM"
    for cls in ("DIR", "TOOL", "WAITV", "MOVEV", "USE", "OBJ", "DET", "PREP", "UNIT",
                "ADVCOUNT"):
        if w in LEXICON[cls]:
            return cls
    return "UNK"


class Parser:
    """Recursive descent: one method per non-terminal, builds a parse tree."""

    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else ("EOF", "")

    def eat(self, cls):
        t = self.peek()
        if t[0] != cls:
            raise ParseError(f"expected {cls} but found '{t[1] or 'end of input'}'")
        self.i += 1
        return (cls, t[1])

    def opt(self, cls):
        return self.eat(cls) if self.peek()[0] == cls else None

    def parse(self):
        kids = [self.cmd()]
        while self.peek()[0] in ("CONJ", "USE", "TOOL", "WAITV", "MOVEV", "DIR"):
            while self.peek()[0] == "CONJ":
                kids.append(self.eat("CONJ"))
            if self.peek()[0] == "EOF":
                break
            kids.append(self.cmd())
        if self.peek()[0] != "EOF":
            raise ParseError(f"unexpected word '{self.peek()[1]}'")
        return ("S", kids)

    def cmd(self):
        k = self.peek()[0]
        if k in ("USE", "TOOL"):
            return ("CMD", [self.toolcmd()])
        if k == "WAITV":
            return ("CMD", [self.waitcmd()])
        if k in ("MOVEV", "DIR"):
            return ("CMD", [self.movecmd()])
        raise ParseError(f"I don't know how to start a command with '{self.peek()[1]}'")

    def toolcmd(self):
        kids = []
        if self.peek()[0] == "USE":
            kids.append(self.eat("USE"))
            d = self.opt("DET")
            if d:
                kids.append(d)
        kids.append(self.eat("TOOL"))
        for cls in ("PREP", "DET", "OBJ", "PREP", "DET"):
            x = self.opt(cls)
            if x:
                kids.append(x)
        kids.append(self.eat("DIR"))
        return ("TOOLCMD", kids)

    def movecmd(self):
        kids = []
        if self.peek()[0] == "MOVEV":
            kids.append(self.eat("MOVEV"))
            x = self.opt("PREP")
            if x:
                kids.append(x)
        kids.append(self.eat("DIR"))
        c = self.count()
        if c:
            kids.append(c)
        return ("MOVECMD", kids)

    def waitcmd(self):
        kids = [self.eat("WAITV")]
        c = self.count()
        if c:
            kids.append(c)
        return ("WAITCMD", kids)

    def count(self):
        if self.peek()[0] == "NUM":
            kids = [self.eat("NUM")]
            u = self.opt("UNIT")
            if u:
                kids.append(u)
            return ("COUNT", kids)
        if self.peek()[0] == "ADVCOUNT":
            return ("COUNT", [self.eat("ADVCOUNT")])
        return None


def tree_str(node):
    label, kids = node
    if isinstance(kids, str):
        return f"{label}:{kids}"
    return f"[{label} " + " ".join(tree_str(k) for k in kids) + "]"


def _count_value(node):
    if node is None:
        return 1
    tok = node[1][0]
    if tok[0] == "ADVCOUNT":
        return LEXICON["ADVCOUNT"][tok[1]]
    v = tok[1]
    return int(v) if v.isdigit() else LEXICON["NUMWORD"][v]


def interpret(tree):
    """Parse tree -> list of primitive actions ('tool', name, dir) / ('move', dir) / ('wait',)."""
    acts = []
    for cmd in tree[1]:
        if cmd[0] != "CMD":
            continue
        inner = cmd[1][0]
        kids = inner[1]
        get = lambda cls: next((k for k in kids if k[0] == cls), None)
        if inner[0] == "TOOLCMD":
            tool = LEXICON["TOOL"][get("TOOL")[1]]
            acts.append(("tool", tool, LEXICON["DIR"][get("DIR")[1]]))
        elif inner[0] == "MOVECMD":
            n = min(_count_value(get("COUNT")), 20)
            acts.extend([("move", LEXICON["DIR"][get("DIR")[1]])] * n)
        elif inner[0] == "WAITCMD":
            acts.extend([("wait",)] * min(_count_value(get("COUNT")), 20))
    return acts


def parse_command(text):
    """Returns (actions, tree_string) or raises ParseError."""
    toks = tokenize(text)
    if not toks:
        raise ParseError("type a command, e.g. 'hammer left then move left'")
    unk = [t[1] for t in toks if t[0] == "UNK"]
    if unk:
        raise ParseError(f"unknown word '{unk[0]}'")
    tree = Parser(toks).parse()
    return interpret(tree), tree_str(tree)


def to_turns(actions):
    """Group primitive actions into turns: at most one tool, then one move/wait."""
    turns = []
    i = 0
    while i < len(actions):
        a = actions[i]
        if a[0] == "tool":
            nxt = actions[i + 1] if i + 1 < len(actions) else None
            if nxt and nxt[0] in ("move", "wait"):
                turns.append((a[1], a[2], nxt[1] if nxt[0] == "move" else "wait"))
                i += 2
            else:
                turns.append((a[1], a[2], "wait"))
                i += 1
        elif a[0] == "move":
            turns.append((None, None, a[1]))
            i += 1
        else:
            turns.append((None, None, "wait"))
            i += 1
    return turns
