"""Match modes: who controls each explorer, and what to call them on screen.

  hva  Human vs AI     (the original game: pid 0 = you, pid 1 = AI rival)
  hvh  Human vs Human  (hot-seat: two players share the keyboard and mouse)
  ava  AI vs AI        (watch two AI brains race each other)

Only labels and control change; the rules, maps and AI are untouched.
"""

MODES = ("hva", "hvh", "ava")
MODE_NAMES = {"hva": "Human vs AI", "hvh": "Human vs Human", "ava": "AI vs AI"}

_TAG = {"hva": ("YOU", "AI"), "hvh": ("P1", "P2"), "ava": ("AI-1", "AI-2")}
_NAME = {"hva": ("You", "AI"), "hvh": ("Player 1", "Player 2"), "ava": ("AI-1", "AI-2")}
_CARD = {"hva": ("YOU", "AI RIVAL"), "hvh": ("PLAYER 1", "PLAYER 2"), "ava": ("AI-1", "AI-2")}

_state = {"mode": "hva"}


def set_mode(m):
    _state["mode"] = m if m in MODES else "hva"


def mode():
    return _state["mode"]


def is_human(pid):
    m = _state["mode"]
    return m == "hvh" or (m == "hva" and pid == 0)


def human_pids():
    return [p for p in (0, 1) if is_human(p)]


def ai_pids():
    return [p for p in (0, 1) if not is_human(p)]


def tag(pid):
    """Short label over a character / on the minimap: YOU, AI, P1, AI-2 ..."""
    return _TAG[_state["mode"]][pid]


def name(pid):
    """Name used in sentences: You, AI, Player 1, AI-1 ..."""
    return _NAME[_state["mode"]][pid]


def card(pid):
    return _CARD[_state["mode"]][pid]


def is_you(pid):
    """True only for the single human of the original mode (grammar: 'You were' vs 'AI was')."""
    return _state["mode"] == "hva" and pid == 0


def turn_label(pid, thinking):
    m = _state["mode"]
    if m == "hva":
        return "YOUR TURN" if pid == 0 else ("AI IS THINKING" if thinking else "AI MOVES")
    if m == "hvh":
        return f"{card(pid)}'S TURN"
    return f"{tag(pid)} IS THINKING" if thinking else f"{tag(pid)} MOVES"
