"""All game-balance numbers live here so they are easy to tune."""

# ---- Map -------------------------------------------------------------
GRID_W = 50
GRID_H = 50

# ---- Life & points ---------------------------------------------------
START_LIFE = 100
VIOLET_POINTS = 10
RED_POINTS = 25

# ---- Damage (life) ---------------------------------------------------
SNAKE_BITE = 15
FIRE_DAMAGE = 10
CRUSH_DAMAGE = 40

# ---- Tool costs (life) -----------------------------------------------
HAMMER_COST = 5
HOOK_COST = 6
PUSH_COST = 8
HOOK_RANGE = 3          # cells, horizontal, needs a clear line

# ---- Mines (optional rule) -------------------------------------------
MINES_PER_PLAYER = 3    # placed on one of the 4 cells next to you; uses your tool for the turn
MINE_DAMAGE = 25        # life lost by the opponent who steps on it (your own mines are safe)

# ---- Lens statues ---------------------------------------------------
LENS_DAMAGE = 40        # life lost when the statue's beam touches you
FREEZE_TURNS = 2        # ...and you are frozen (skip) for this many of your turns
NUM_LENS_GATES = 5      # statues guarding tunnels, each with a hook alcove behind it

# ---- Exit ------------------------------------------------------------
EXIT_UNLOCK_GEMS = 10   # diamonds a player must hold before the exit opens for them

# ---- Map contents ----------------------------------------------------
NUM_VIOLET = 30
NUM_RED = 12
NUM_SNAKES = 22
NUM_FIRE = 30

# ---- Safety valve ----------------------------------------------------
MAX_ROUNDS = 500        # if nobody exits by then, the game ends and totals are compared
