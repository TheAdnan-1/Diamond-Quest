"""Diamond Quest — application: screens, input, turn flow, AI threading (GPU-rendered)."""
import json
import math
import os
import random
import sys
import threading
import time

import numpy as np
import pygame

from . import config as C
from . import gfx
from . import mapgen
from . import roles
from .ai import AIAgent
from . import arena
from .assets import Assets, tile_size_for
from .audio import Audio
from .biomes import BIOMES, ORDER
from .fuzzy import risk_label
from .gpu import GPU, UI, setup_platform
from .hud import HUD
from .logic import danger_analysis
from .nlp import parse_command, to_turns, ParseError
from .pddl_planner import plan_for_player
from .search import ALGORITHMS, GridGraph, bfs_distances
from .view import WorldView, hexrgb
from .world import DIRS, FLOOR, DIRT, WALL, CRACKED, FIRE, EXIT

KEY_DIR = {pygame.K_UP: "up", pygame.K_w: "up", pygame.K_DOWN: "down", pygame.K_s: "down",
           pygame.K_LEFT: "left", pygame.K_a: "left", pygame.K_RIGHT: "right", pygame.K_d: "right"}
OUT_DIR = os.path.join(os.getcwd(), "diamond_quest_output")
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".diamond_quest", "settings.json")
TITLE_SEED = 12345
TIPS = [
    "Boulders fall when you dig out what holds them up. Don't rush!",
    "Hammer a snake to stun it — only you may stand in its cell next turn.",
    "Push or hook a rock above your rival's path to drop it on them (−40 life).",
    "Fire never goes out: step in, and every turn you stay costs another 10 life.",
    "Reaching the exit ends the game — only leave when your points + life are higher.",
    "Press B in game to watch the AI think: minimax / MCTS search and its fuzzy mood.",
    "Press V for the Algorithm Lab: watch DFS, IDDFS, A* and friends explore the cave.",
]


class Button:
    audio = None
    _last_hover = None

    def __init__(self, rect, label, action, kind="ghost", icon=None):
        self.rect = rect                      # design units (x, y, w, h)
        self.label, self.action, self.kind, self.icon = label, action, kind, icon
        self.hover_t = 0.0
        self._hover = False

    def hit(self, pos):
        x, y, w, h = self.rect
        return x <= pos[0] <= x + w and y <= pos[1] <= y + h

    def press(self):
        if Button.audio:
            Button.audio.play("click", 0.5, 0.8)
        self.action()

    def draw(self, app, mouse, now, dt, selected=False, focus=False):
        ui = app.ui
        hover = self.hit(mouse) or focus
        hover_key=(app.state,self.rect,self.label)
        if hover and Button._last_hover!=hover_key and Button.audio:
            Button.audio.play("hover",0.5,.45)
            Button._last_hover=hover_key
        self._hover = hover
        self.hover_t += ((1 if hover else 0) - self.hover_t) * min(1, dt * 12)
        x, y, w, h = self.rect
        g = 3 * self.hover_t
        x, y, w, h = x - g, y - g * 0.6, w + 2 * g, h + 1.2 * g
        if self.kind == "primary":
            ui.glow(x+w/2,y+h/2,w*.5,(213,170,86),.10+.12*self.hover_t)
            ui.hgradient(x,y,w,h,(222,190,119),(165,117,53),radius=7)
            ui.rrect(x+2,y+2,w-4,h-4,5,(255,235,177),100,filled=False)
            ui.line(x+12,y+4,x+w-12,y+4,1,(255,243,200),130)
            ui.text(self.label,x+w/2,y+h/2,17,(29,29,23),True,'center')
        elif self.kind == 'seg':
            ui.panel(x,y,w,h,7,fill=(29,65,64) if selected else (12,25,29),
                     alpha=245,border=gfx.HUMAN if selected else (169,140,87),
                     border_alpha=190 if selected else int(70+70*self.hover_t),shadow=False)
            ui.text(self.label,x+w/2,y+h/2,15,gfx.HUMAN if selected else gfx.INK,True,'center')
        else:
            ui.panel(x,y,w,h,7,alpha=235,border=(180,150,94),
                     border_alpha=int(75+100*self.hover_t),shadow=False)
            ui.text(self.label,x+w/2,y+h/2,15,gfx.INK,True,'center')


class Slider:
    def __init__(self, rect, getter, setter):
        self.rect, self.get, self.set = rect, getter, setter
        self.drag = False

    def handle(self, e, mouse):
        x, y, w, h = self.rect
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and x - 10 <= mouse[0] <= x + w + 10 \
                and y - 10 <= mouse[1] <= y + h + 10:
            self.drag = True
        if e.type == pygame.MOUSEBUTTONUP:
            self.drag = False
        if self.drag and e.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            self.set(gfx.clamp((mouse[0] - x) / w))

    def draw(self, ui):
        x, y, w, h = self.rect
        v = self.get()
        ui.rrect(x, y + h / 2 - 4, w, 8, 4, (255, 255, 255), 30)
        ui.hgradient(x, y + h / 2 - 4, max(8, w * v), 8, (98, 204, 188), (198, 163, 91), radius=4)
        ui.glow(x + w * v, y + h / 2, 22, gfx.HUMAN, 0.5)
        ui.disc(x + w * v, y + h / 2, 10, (255, 255, 255))
        ui.text(f"{int(round(v * 100))}%", x + w + 18, y + h / 2, 14, gfx.INK, True, "midleft")


class App:
    def __init__(self, seed=None):
        setup_platform()
        pygame.init()
        self.settings = self._load_settings()
        self.g = GPU("Diamond Quest", fullscreen=self.settings.get("fullscreen", False))
        self.ui = UI(self.g)
        self.audio = Audio()
        self.audio.set_volumes(self.settings["music"], self.settings["sfx"])
        Button.audio = self.audio
        self.A = Assets(self.g)
        self.hud = HUD(self.ui, self.A)
        self.clock = pygame.time.Clock()
        self.now = 0.0
        self.seed = seed if seed is not None else random.randrange(1 << 30)
        self.match = {"start": 0, "first": 0, "brain": "minimax", "mines": True, "mode": "hva", "brain2": "mcts"}
        self.toast_msg = None
        self.focus = 0
        self.preview = None
        self.preview_tex = None
        self.world = None
        self.title_view = None
        self.state = "loading"
        self.after_load = self._enter_title
        self.load_t0 = 0.0
        self.tip = random.choice(TIPS)
        self.title_biome = ORDER[TITLE_SEED % len(ORDER)]
        self.A.start(self.g.u, (self.title_biome,))
        self.frame_ms = []
        self.fps_show = False
        self.grad_cache = {}

    # ================================================================ settings
    def _load_settings(self):
        s = {"music": 0.7, "sfx": 0.85, "fullscreen": False, "camera": "me", "minimap": True,
             "ava_cap": arena.ROUND_CAP}
        try:
            with open(SETTINGS_PATH) as f:
                s.update(json.load(f))
        except Exception:
            pass
        return s

    def save_settings(self):
        try:
            os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
            with open(SETTINGS_PATH, "w") as f:
                json.dump(self.settings, f, indent=1)
        except Exception:
            pass

    # ================================================================ helpers
    @property
    def DW(self):
        return self.g.DW

    @property
    def DH(self):
        return self.g.DH

    def gradient_text(self, s, size, c1, c2):
        px = int(size * self.g.u)
        key = (s, px, c1, c2)
        t = self.grad_cache.get(key)
        if t is None:
            surf = self.g.font(px, True, True).render(s, True, (255, 255, 255))
            w, h = surf.get_size()
            a = pygame.surfarray.array_alpha(surf).T.astype(np.float32) / 255 if surf.get_flags() & \
                pygame.SRCALPHA else pygame.surfarray.array3d(surf)[..., 0].T.astype(np.float32) / 255
            xs = np.linspace(0, 1, w)[None, :, None]
            col = np.array(c1, np.float32) / 255 * (1 - xs) + np.array(c2, np.float32) / 255 * xs
            col = np.broadcast_to(col, (h, w, 3)) * (0.85 + 0.15 * np.linspace(1, 0, h)[:, None, None])
            img = np.concatenate([col * a[..., None], a[..., None]], -1)
            t = self.g.texture(img.astype(np.float32))
            self.grad_cache[key] = t
        return t

    def add_log(self, msg, col=gfx.INK_DIM):
        self.log.append((msg, col, self.now))
        self.log = self.log[-40:]

    def toast(self, msg, col=(255, 120, 130)):
        self.toast_msg = (msg, col, self.now)
        if col == (255, 120, 130):
            self.audio.play("error", 0.5, 0.6)

    def busy(self):
        return self.now < self.view.timeline_end

    def human_turn(self):
        """True when the explorer about to move is controlled by a person."""
        return roles.is_human(self.world.current)

    @property
    def me(self):
        """The explorer the screen speaks for: the human whose turn it is (hot-seat),
        else the (only) human, else AI-1 when two AIs play."""
        w = self.world
        if w is not None and roles.mode() == "hvh":
            return w.current
        hp = roles.human_pids()
        return hp[0] if hp else 0

    def _set(self, st):
        self.state = st
        self.focus = 0
        self.audio.set_music_level("game" if st == "game" else "menu")
        if st!="game":self.audio.set_scene(0,0)

    def quit(self):
        self.save_settings()
        try:
            from . import assets as _assets
            if _assets._POOL:
                _assets._POOL.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        pygame.quit()
        os._exit(0)

    # ================================================================ loading
    def load_then(self, biomes, after):
        self.A.start(self.g.u, biomes)
        if self.A.busy():
            self.state = "loading"
            self.after_load = after
            self.tip = random.choice(TIPS)
        else:
            after()

    def _enter_title(self):
        w, meta = mapgen.generate(TITLE_SEED)
        w.setup_players(w.starts[0], w.starts[1], 0)
        self.title_world = w
        self.title_view = WorldView(self.g, self.A, w, meta["biome"], TITLE_SEED)
        self.title_view.free_cam = True
        self.title_view.show_grid = False
        self.audio.set_ambience(meta["biome"])
        self._set("title")
        # warm up the other cave themes in the background
        others = tuple(b for b in ORDER if b not in self.A.loaded_biomes)
        if others:
            self.A.start(self.g.u, others)

    # ================================================================ screens: setup
    def goto_setup(self, new_seed=False):
        if new_seed or self.preview is None:
            if new_seed:
                pf = getattr(self, "prefetch", None)
                if pf and not pf[1].is_alive():
                    self.seed = pf[0]                 # already built in the background
                else:
                    self.seed = random.randrange(1 << 30)
                self.prefetch = None
            self.preview, self.preview_meta = self._make_map(self.seed)
            self.preview_key = (self.seed, self._mirrored())
            self.preview_tex = self._preview_texture(self.preview, self.preview_meta["biome"])
        self.audio.set_ambience(self.preview_meta["biome"])
        self._set("setup")
        self._prefetch_next()
        seed = self.seed           # warm the mirrored cave too, so switching brains is instant
        threading.Thread(target=lambda: arena.mirrored_map(seed), daemon=True).start()

    def _prefetch_next(self):
        """Build the next random cave in the background so New Map is instant."""
        if getattr(self, "prefetch", None):
            return
        nxt = random.randrange(1 << 30)
        balanced = self._mirrored()

        def build():
            mapgen.generate(nxt)
            if balanced:
                arena.mirrored_map(nxt)       # AI vs AI Balanced: have its cave ready too
        th = threading.Thread(target=build, daemon=True)
        th.start()
        self.prefetch = (nxt, th)

    def _mirrored(self):
        """AI vs AI is played in an arena cave (whatever the brains), unless the
        Asymmetric cave type is chosen: then it is a normal cave and a real match."""
        m = self.match
        return m.get("mode") == "ava" and m.get("arena", "sym") != "asym"

    def _twins(self):
        m = self.match
        return self._mirrored() and m.get("brain") == m.get("brain2")

    def _make_map(self, seed):
        return arena.mirrored_map(seed) if self._mirrored() else mapgen.generate(seed)

    def _preview_texture(self, w, biome):
        from .cartography import map_surface
        return self.g.texture(map_surface(w,biome,16,objects=True))

    def start_game(self, same_map=False):
        if not same_map or self.preview is None or getattr(self, "preview_key", None) != (self.seed, self._mirrored()):
            self.preview, self.preview_meta = self._make_map(self.seed)
            self.preview_key = (self.seed, self._mirrored())
            self.preview_tex = None
        biome = self.preview_meta["biome"]
        self.load_then((biome,), self._begin_game)

    def _begin_game(self):
        w, self.map_meta = self._make_map(self.seed)
        self.biome = self.map_meta["biome"]
        st = self.match
        w.setup_players(w.starts[st["start"]], w.starts[1 - st["start"]], st["first"],
                        mines=C.MINES_PER_PLAYER if st.get("mines") else 0)
        if st.get("mode", "hva") != "hva":
            for p in w.players:
                p.name = roles.name(p.pid)
        self.world = w
        self.view = WorldView(self.g, self.A, w, self.biome, self.seed)
        self.view.cam_mode = self.settings.get("camera", "me")
        self.view.audio = self.audio
        self.view.hud_targets = {0: self.hud.target_px(0), 1: self.hud.target_px(1)}
        self.view.on_gem_arrive = lambda pid: self.hud.pulse.__setitem__(pid, self.now)
        self.audio.set_ambience(self.biome)
        self.audio.set_music_level("game")
        self.jingle_done = False
        roles.set_mode(st.get("mode", "hva"))
        brains = {1: st["brain"]} if roles.mode() == "hva" else {0: st["brain"], 1: st.get("brain2", "mcts")}
        Agent = arena.ArenaAgent if roles.mode() == "ava" else AIAgent
        if roles.mode() == "ava":
            balanced = self._mirrored()
            self.ais = {pid: Agent(pid, brains[pid], seed=self.seed + pid - 1, arena=balanced)
                        for pid in roles.ai_pids()}
        else:
            self.ais = {pid: Agent(pid, brains[pid], seed=self.seed + pid - 1) for pid in roles.ai_pids()}
        self.mirror = {"leader": st["first"], "last": None} if self._twins() and w.starts and \
            w.exit_pos and w.exit_pos[1] == arena.axis(w) else None
        self.ai = self.ais.get(1)
        self.ai_infos = {}
        if roles.mode() == "ava":
            self.view.cam_mode = "both"
        self.view.focus_pid = self.me
        self.ai_thread = None
        self.ai_result = None
        self.ai_started = 0.0
        self.ai_info = {}
        self.armed = None
        self.queue = []
        self.queue_pid = None
        self.log = []
        self.panel = None
        self.logic_on = False
        self.logic_cache = None
        self.lab = None
        self.plan = None
        self.console = None
        self.console_hist = []
        self.big_map = False
        self.gameover_t = None
        self.hud.disp.clear()
        self._set("game")
        m = self.map_meta
        self.add_log(f"{BIOMES[self.biome]['name']}: a fair map ({m['csp_constraints']} constraints)",
                     gfx.INK_DIM)
        if roles.mode() != "hva":
            self.add_log(f"Mode: {roles.MODE_NAMES[roles.mode()]}", gfx.INK)
        self.add_log(f"Grab {C.EXIT_UNLOCK_GEMS} diamonds to open the exit.", gfx.HUMAN)
        self.add_log("Red diamonds sit in sealed vaults: Hammer, Push or", gfx.RED_LIGHT)
        self.add_log("Hook your way in. Don't rush!", gfx.RED_LIGHT)

    def _gameover_buttons(self):
        cx, y = self.DW / 2, 586
        return [Button((cx - 300, y, 180, 54), "REMATCH", lambda: self.start_game(True), "primary"),
                Button((cx - 90, y, 180, 54), "NEW MAP", lambda: self.goto_setup(True)),
                Button((cx + 120, y, 180, 54), "MENU", self.to_title)]

    def _pause_buttons(self):
        cx = self.DW / 2
        return [Button((cx - 150, 270, 300, 54), "RESUME", lambda: self._set("game"), "primary"),
                Button((cx - 150, 338, 300, 50), "RESTART MAP", lambda: self.start_game(True)),
                Button((cx - 150, 400, 300, 50), "NEW GAME", lambda: self.goto_setup(True)),
                Button((cx - 150, 462, 300, 50), "SETTINGS", lambda: self.open_settings("pause")),
                Button((cx - 150, 524, 300, 50), "QUIT TO TITLE", self.to_title)]

    def to_title(self):
        if self.title_view:
            self.audio.set_ambience(self.title_view.biome)
        self._set("title")

    # ================================================================ events -> log
    def _apply_events(self, ev):
        self.view.schedule(ev, self.now)
        self.logic_cache = None
        self.plan = None if ev else self.plan
        names = {0: roles.name(0), 1: roles.name(1)}
        for e in ev:
            t, pid = e["type"], e.get("pid")
            who = names.get(pid, "")
            col = gfx.HUMAN if pid == 0 else gfx.AI
            if t == "collect":
                self.add_log(f"{who} grabbed a {'RED' if e['kind'] == 'r' else 'violet'} diamond "
                             f"(+{e['points']})", gfx.RED_LIGHT if e["kind"] == "r" else gfx.VIOLET_LIGHT)
            elif t == "hammer":
                self.add_log(f"{who} hammered a {e['target']}", col)
            elif t == "push":
                self.add_log(f"{who} pushed a rock", col)
            elif t == "hook" and e["obj"] == "statue":
                self.add_log(f"{who} turned a lens statue {'right' if e['facing'] > 0 else 'left'}", col)
            elif t == "hook":
                self.add_log(f"{who} hooked a {e['obj']}", col)
            elif t == "freeze":
                self.add_log(f"{who} {'were' if roles.is_you(pid) else 'was'} frozen by a lens beam! (-{C.LENS_DAMAGE})",
                             (170, 225, 255))
                if roles.is_you(pid):
                    self.toast(f"Frozen! You skip your next {C.FREEZE_TURNS} turns", (170, 225, 255))
                elif roles.mode() == "hvh":
                    self.toast(f"{who} is frozen and skips the next {C.FREEZE_TURNS} turns", (170, 225, 255))
            elif t == "thaw":
                self.add_log(f"{who} thawed out", (170, 225, 255))
            elif t == "bite":
                self.add_log(f"{who} got bitten by a snake!", (255, 110, 120))
            elif t == "crush":
                self.add_log(f"{who} {'were' if roles.is_you(pid) else 'was'} hit by a boulder! (-{C.CRUSH_DAMAGE})",
                             (255, 110, 120))
            elif t == "burn":
                self.add_log(f"{who} stayed in the fire and burned again (-{C.FIRE_DAMAGE})", gfx.FIRE_2)
            elif t == "damage" and e["why"] == "fire" and not any(x["type"] == "burn" for x in ev):
                self.add_log(f"{who} walked into fire (-{C.FIRE_DAMAGE})", gfx.FIRE_2)
            elif t == "mine_place":
                self.add_log(f"{who} planted a mine", col)
            elif t == "mine":
                self.add_log(f"{who} stepped on a mine! (-{C.MINE_DAMAGE})", (255, 150, 90))
            elif t == "mine_boom":
                self.add_log("A boulder set off a mine", (255, 150, 90))
            elif t == "snake_crushed":
                self.add_log("A boulder crushed a snake", gfx.SNAKE)
            elif t == "exit":
                self.add_log(f"{who} reached the exit!", col)
            elif t == "death":
                self.add_log(f"{who} ran out of life", (255, 110, 120))
            elif t == "game_over":
                self.gameover_t = self.view.timeline_end + 0.3
        for p in self.world.players:
            if p.gems == C.EXIT_UNLOCK_GEMS and any(e["type"] == "collect" and e["pid"] == p.pid for e in ev):
                self.add_log(f"The exit is now OPEN for {names[p.pid]}!", gfx.HUMAN if p.pid == 0 else gfx.AI)
                if roles.is_human(p.pid):
                    self.toast("The exit is open for you!" if roles.is_you(p.pid)
                               else f"The exit is open for {names[p.pid]}!", gfx.HUMAN if p.pid == 0 else gfx.AI)
                    self.view.at(self.now + 0.6, lambda: self.audio.play("unlock"))
        self._music_intensity()

    def _music_intensity(self):
        """Dynamic music: tension rises with low life, a close score and an open exit."""
        w = self.world
        a, b = w.players
        k = 0.0
        k += gfx.clamp((60 - a.life) / 50) * 0.45
        k += gfx.clamp(1 - abs(a.total - b.total) / 80) * 0.25
        if w.exit_open_for(0) or w.exit_open_for(1):
            k += 0.35
        k += gfx.clamp(w.rounds / 300) * 0.15
        self.audio.set_intensity(gfx.clamp(k))

    # ================================================================ human turn
    def human_tool(self, tool, d):
        pid = self.world.current
        ok, info = self.world.tool_target(pid, tool, d)
        if not ok:
            self.toast(f"{tool.capitalize()}: {info}")
            return False
        self._apply_events(self.world.use_tool(pid, tool, d))
        self.armed = None
        return True

    def human_move(self, d):
        w = self.world
        pid = w.current
        if getattr(w.players[pid], "frozen", 0) and d != "wait":
            self.toast("You're frozen — wait it out")
            return False
        if d != "wait":
            p = w.players[pid]
            dr, dc = DIRS[d]
            nr, nc = p.r + dr, p.c + dc
            if not w.can_enter(pid, nr, nc):
                if (nr, nc) == w.exit_pos:
                    self.toast(f"Exit locked: collect {C.EXIT_UNLOCK_GEMS - p.gems} more diamonds")
                elif (nr, nc) in w.rock_at:
                    self.toast("A rock blocks the way — try Push (P) or Hook (K)")
                elif w.inside(nr, nc) and w.terrain[nr][nc] == CRACKED:
                    self.toast("Cracked wall — break it with the Hammer (H)")
                else:
                    self.toast("You can't go there")
                return False
            if (nr, nc) == w.exit_pos and w.players[pid].total <= w.players[1 - pid].total:
                self.toast("Careful: you are behind — leaving now loses!", gfx.AI)
        self._apply_events(w.end_turn(pid, d))
        return True

    def tick_frozen(self):
        """A frozen explorer can only wait: skip the human's turn automatically."""
        w = self.world
        if not self.human_turn() or self.busy() or not getattr(w.players[w.current], "frozen", 0):
            self.frozen_since = None
            return
        if getattr(self, "frozen_since", None) is None:
            self.frozen_since = self.now
            return
        if self.now - self.frozen_since > 0.7:
            self.frozen_since = None
            self.queue.clear()
            self.armed = None
            self._apply_events(w.end_turn(w.current, "wait"))

    def run_queue(self):
        if not self.queue or not self.human_turn() or self.busy():
            return
        if self.queue_pid is not None and self.world.current != self.queue_pid:
            return
        tool, td, mv = self.queue.pop(0)
        if tool and not self.world.tool_used:
            if not self.human_tool(tool, td):
                self.queue.clear()
                return
        if mv and not self.human_move(mv):
            self.queue.clear()

    # ================================================================ AI turn
    def _ai_job(self, snapshot, agent=None):
        try:
            self.ai_result = (agent or self.ai).decide(snapshot)
        except Exception as ex:
            self.ai_result = (None, None, "wait", {"why": f"error: {ex}", "think": 0})

    def tick_ai(self):
        w = self.world
        if w.over or w.current not in self.ais or self.busy():
            return
        if self.ai_thread is None:
            self.ai_result = None
            self.ai_started = self.now
            mir = getattr(self, "mirror", None)
            if mir and w.current != mir["leader"] and mir["last"] is not None:
                self.ai_thread = threading.Thread(target=self._twin_job, args=(w, mir["last"]), daemon=True)
                self.ai_thread.start()
                return
            self.ai_thread = threading.Thread(target=self._ai_job, args=(w.clone(), self.ais[w.current]), daemon=True)
            self.ai_thread.start()
            return
        if self.ai_result is None or self.now - self.ai_started < 0.35:
            return
        tool, td, mv, info = self.ai_result
        self.ai_thread = None
        self.ai_info = info
        self.ai_infos[w.current] = info
        pid = w.current
        mir = getattr(self, "mirror", None)
        if mir and pid == mir["leader"]:
            tool, td, mv = arena.safe_leader_turn(w, pid, tool, td, mv)
        try:
            ev = w.take_turn(pid, tool, td, mv)
        except ValueError:
            tool, td, mv = None, None, "wait"
            ev = w.take_turn(pid, None, None, "wait")
        if mir and pid == mir["leader"]:
            mir["last"] = (tool, td, mv, info)
        cap = self.ava_cap()
        if roles.mode() == "ava" and not w.over and w.rounds >= cap:
            w._finish(-2, f"Round limit reached ({cap} rounds)", ev)
        self._apply_events(ev)

    def ava_cap(self):
        try:
            return max(1, int(self.settings.get("ava_cap", arena.ROUND_CAP)))
        except (TypeError, ValueError):
            return arena.ROUND_CAP

    def _twin_job(self, w, last):
        tool, td, mv, info = last
        time.sleep(max(0.0, info.get("think", 0.4)) * random.uniform(0.85, 1.15))
        t2, td2, mv2 = arena.mirror_turn(tool, td, mv)
        self.ai_result = (t2, td2, mv2, arena.mirror_info(w, info))

    # ================================================================ panels
    def _lab_goal(self, dist):
        w = self.world
        for pool in ([g for g, k in w.gems.items() if k == "r" and dist.get(g, 0) >= 10],
                     [g for g, k in w.gems.items() if k == "r" and g in dist],
                     [g for g in w.gems if g in dist]):
            if pool:
                return min(pool, key=lambda g: dist[g])
        return None

    def run_lab(self, idx):
        w = self.world
        p = w.players[self.me]
        goal = self._lab_goal(bfs_distances(w, (p.r, p.c), cracked_ok=False))
        if goal is None:
            self.toast("No reachable diamond to search for")
            return
        name, fn = ALGORITHMS[idx]
        t0 = time.perf_counter()
        res = fn(GridGraph(w, self.me, life_weight=1.0), (p.r, p.c), goal)
        self.lab = {"res": res, "ms": (time.perf_counter() - t0) * 1000, "goal": goal, "t0": self.now,
                    "name": name, "table": None}
        self.panel = "lab"
        self.audio.play("arm", 0.5, 0.5)

    def run_lab_all(self):
        w = self.world
        p = w.players[self.me]
        goal = self._lab_goal(bfs_distances(w, (p.r, p.c), cracked_ok=False))
        if goal is None:
            self.toast("No reachable diamond to search for")
            return
        graph = GridGraph(w, self.me, life_weight=1.0)
        table = []
        for name, fn in ALGORITHMS:
            t0 = time.perf_counter()
            res = fn(graph, (p.r, p.c), goal)
            table.append((name, len(res.expanded), len(res.path) - 1 if res.found else None, res.cost,
                          (time.perf_counter() - t0) * 1000))
        self.lab = {"res": None, "table": table, "goal": goal, "t0": self.now, "name": "Compare"}
        self.panel = "lab"

    def run_plan(self):
        steps, path, stats = plan_for_player(self.world, self.me, save_dir=OUT_DIR)
        self.plan = {"steps": steps, "path": path, "stats": stats, "t0": self.now}
        self.panel = "plan"
        if steps:
            self.add_log(f"PDDL plan: {len(steps)} steps (saved to diamond_quest_output/)", (200, 170, 255))
        else:
            self.toast("The planner found no diamond nearby")

    def logic(self):
        if self.logic_cache is None or self.logic_cache[0] != self.me:
            self.logic_cache = (self.me, danger_analysis(self.world, self.me))
        return self.logic_cache[1]

    # ================================================================ settings screen
    def open_settings(self, back):
        self.settings_back = back
        self._set("settings")

    def _settings_widgets(self):
        cx = self.DW / 2
        x0 = cx - 260
        s = self.settings

        def set_music(v):
            s["music"] = v
            self.audio.set_volumes(s["music"], s["sfx"])

        def set_sfx(v):
            s["sfx"] = v
            self.audio.set_volumes(s["music"], s["sfx"])
            if random.random() < 0.15:
                self.audio.play("gem_v", 0.5, 0.6)
        if not hasattr(self, "_sliders") or self._sliders_dw != self.DW:
            self._sliders = [Slider((x0 + 200, 296, 270, 30), lambda: s["music"], set_music),
                             Slider((x0 + 200, 356, 270, 30), lambda: s["sfx"], set_sfx)]
            self._sliders_dw = self.DW

        def set_fs(v):
            if self.g.fullscreen != v:
                self.g.toggle_fullscreen()
                s["fullscreen"] = v
                self.save_settings()

        def set_cam(v):
            s["camera"] = v
            if self.world is not None:
                self.view.cam_mode = v
        btns = [
            (Button((x0 + 200, 196, 150, 46), "Window", lambda: set_fs(False), "seg"), not self.g.fullscreen),
            (Button((x0 + 360, 196, 150, 46), "Fullscreen", lambda: set_fs(True), "seg"), self.g.fullscreen),
            (Button((x0 + 200, 418, 150, 46), "Both of you", lambda: set_cam("both"), "seg"), s["camera"] == "both"),
            (Button((x0 + 360, 418, 150, 46), "Follow you", lambda: set_cam("me"), "seg"), s["camera"] == "me"),
            (Button((cx - 130, 552, 260, 54), "BACK", self._settings_back, "primary"), False),
        ]

        def set_cap(v):
            s["ava_cap"] = v
            self.save_settings()
        for i, v in enumerate(arena.CAP_OPTIONS):
            btns.append((Button((x0 + 200 + i * 64, 478, 58, 44), str(v), lambda v=v: set_cap(v), "seg"),
                         self.ava_cap() == v))
        return btns

    def _settings_back(self):
        self.save_settings()
        self._set(self.settings_back)

    # ================================================================ input
    def handle(self, e):
        if e.type == pygame.QUIT:
            self.quit()
        if e.type in (pygame.WINDOWSIZECHANGED, pygame.WINDOWRESIZED) if hasattr(pygame, "WINDOWSIZECHANGED") \
                else e.type == pygame.VIDEORESIZE:
            self._on_resize()
            return
        if e.type == pygame.KEYDOWN and (e.key == pygame.K_F11 or
                                         (e.key == pygame.K_RETURN and e.mod & pygame.KMOD_ALT)):
            self.g.toggle_fullscreen()
            self.settings["fullscreen"] = self.g.fullscreen
            return
        if e.type == pygame.KEYDOWN and e.key == pygame.K_F3:
            self.fps_show = not self.fps_show
            return
        mouse = self.g.mouse_u()
        st = self.state
        if st == "title":
            btns = self.title_buttons()
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for b in btns:
                    if b.hit(mouse):
                        b.press()
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_UP, pygame.K_w):
                    self.focus = (self.focus - 1) % len(btns)
                elif e.key in (pygame.K_DOWN, pygame.K_s):
                    self.focus = (self.focus + 1) % len(btns)
                elif e.key in (pygame.K_RETURN, pygame.K_SPACE):
                    btns[self.focus].press()
                elif e.key == pygame.K_ESCAPE:
                    self.quit()
        elif st == "howto":
            if e.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                self._set("title")
        elif st == "settings":
            for sl in self._sliders:
                sl.handle(e, mouse)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for b, _ in self._settings_widgets():
                    if b.hit(mouse):
                        b.press()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self._settings_back()
        elif st == "setup":
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self._set("title")
                elif e.key == pygame.K_RETURN:
                    self.start_game(True)
                elif e.key == pygame.K_n:
                    self.goto_setup(True)
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for b, _ in self.setup_buttons():
                    if b.hit(mouse):
                        b.press()
        elif st == "pause":
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                self._set("game")
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for b in self._pause_buttons():
                    if b.hit(mouse):
                        b.press()
        elif st == "game":
            self._game_event(e, mouse)

    def _on_resize(self):
        old_ts = self.A.ts
        self.g.resize()
        self.grad_cache.clear()
        if self.world is not None:
            self.view.hud_targets = {0: self.hud.target_px(0), 1: self.hud.target_px(1)}
        if tile_size_for(self.g.u) != old_ts and self.state != "loading":
            prev = self.state
            biomes = tuple(self.A.loaded_biomes) or (self.title_biome,)
            self.A.loaded_biomes = set()
            self.load_then(biomes, lambda: self._after_reload(prev))

    def _after_reload(self, prev):
        if self.title_view:
            self.title_view.rebuild()
        if self.world is not None:
            self.view.rebuild()
        self.preview_tex = None
        self._set(prev)

    def _game_event(self, e, mouse):
        w = self.world
        if self.console is not None:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    self.console = None
                    pygame.key.stop_text_input()
                elif e.key == pygame.K_RETURN:
                    self.submit_console()
                elif e.key == pygame.K_BACKSPACE:
                    self.console = self.console[:-1]
            elif e.type == pygame.TEXTINPUT and len(self.console) < 70:
                self.console += e.text
            return
        if e.type == pygame.KEYDOWN:
            k = e.key
            if w.over:
                if self.gameover_t and self.now > self.gameover_t:
                    if k == pygame.K_RETURN:
                        self.start_game(True)
                    elif k == pygame.K_ESCAPE:
                        self.to_title()
                return
            if k == pygame.K_ESCAPE:
                if self.armed:
                    self.armed = None
                elif self.panel or self.big_map:
                    self.panel = None
                    self.big_map = False
                else:
                    self._set("pause")
                    self.audio.set_music_level("menu")
                return
            if k == pygame.K_n:
                self.toast(f"Music {'on' if self.audio.toggle_music() else 'off'}", gfx.INK)
                return
            if k == pygame.K_x:
                self.toast(f"Sound effects {'on' if self.audio.toggle_sfx() else 'off'}", gfx.INK)
                return
            if k in (pygame.K_t, pygame.K_RETURN):
                self.console = ""
                self.panel = "console"
                pygame.key.start_text_input()
                return
            if k == pygame.K_l:
                self.logic_on = not self.logic_on
                self.add_log(f"Danger sense {'ON' if self.logic_on else 'OFF'} (logic engine)", gfx.GOOD)
                return
            if k == pygame.K_v:
                self.panel = None if self.panel == "lab" else "lab"
                return
            if k == pygame.K_b:
                self.panel = None if self.panel == "brain" else "brain"
                return
            if k == pygame.K_o:
                if self.panel == "plan":
                    self.panel, self.plan = None, None
                else:
                    self.run_plan()
                return
            if k == pygame.K_g:
                self.view.show_grid=not self.view.show_grid
                self.toast('Cell grid '+('shown' if self.view.show_grid else 'hidden'),gfx.INK)
                return
            if k == pygame.K_m:
                self.big_map = not self.big_map
                return
            if k == pygame.K_TAB:
                self.toggle_minimap()
                return
            if k == pygame.K_c:
                modes = ["both", "me", "ai"]
                v = self.view
                v.cam_mode = modes[(modes.index(v.cam_mode) + 1) % 3]
                names = {"both": "shows you AND the AI", "me": "follows you", "ai": "follows the AI"}
                if roles.mode() == "hvh":
                    names = {"both": "shows both players", "me": "follows whoever's turn it is",
                             "ai": "follows the other player"}
                elif roles.mode() == "ava":
                    names = {"both": "shows AI-1 AND AI-2", "me": "follows AI-1", "ai": "follows AI-2"}
                self.toast(f"Camera: {names[v.cam_mode]}", gfx.INK)
                return
            if self.panel == "lab" and pygame.K_0 <= k <= pygame.K_7:
                if k == pygame.K_0:
                    self.run_lab_all()
                else:
                    self.run_lab(k - pygame.K_1)
                return
            if not self.human_turn() or self.busy():
                return
            self.queue.clear()
            tool_keys = {pygame.K_h: "hammer", pygame.K_p: "push", pygame.K_k: "hook", pygame.K_e: "mine"}
            if k in tool_keys:
                self.arm(tool_keys[k])
            elif k == pygame.K_SPACE:
                self.armed = None
                self.human_move("wait")
            elif k in KEY_DIR:
                d = KEY_DIR[k]
                if self.armed:
                    self.human_tool(self.armed, d)
                else:
                    self.human_move(d)
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if w.over:
                if self.gameover_t and self.now > self.gameover_t:
                    for b in self._gameover_buttons():
                        if b.hit(mouse):
                            b.press()
                return
            if not self.panel and not self.big_map:
                if not self.settings.get("minimap", True):
                    x, y, cw, ch = self.minimap_chip_rect()
                    if x <= mouse[0] <= x + cw and y <= mouse[1] <= y + ch:
                        self.toggle_minimap()
                        return
                else:
                    hb = getattr(self.hud, "map_hide_rect", None)
                    if hb and hb[0] <= mouse[0] <= hb[0] + hb[2] and hb[1] <= mouse[1] <= hb[1] + hb[3]:
                        self.toggle_minimap()
                        return
            for name, (x, y, ww, hh) in self.hud.dock_rects.items():
                if x <= mouse[0] <= x + ww and y <= mouse[1] <= y + hh:
                    if self.human_turn() and not self.busy():
                        if name == "wait":
                            self.human_move("wait")
                        else:
                            self.arm(name)
                    return
            if self.human_turn() and not self.busy():
                r, c = self.view.screen_to_cell(*self.g.mouse())
                p = w.players[w.current]
                delta = (r - p.r, c - p.c)
                for d, v in DIRS.items():
                    if v == delta:
                        if self.armed:
                            self.human_tool(self.armed, d)
                        else:
                            self.human_move(d)

    def arm(self, tool):
        if self.world.tool_used:
            self.toast("Only one tool per turn")
            return
        if self.armed == tool:
            self.armed = None
            return
        if tool == "mine" and not getattr(self.world, "mines_enabled", False):
            self.toast("Mines are switched off for this match")
            return
        opts = [t for t in self.world.valid_tools(self.world.current) if t[0] == tool]
        if not opts:
            hints = {"hammer": "Stand next to a snake or a cracked wall",
                     "push": "Stand beside a rock with free space behind it",
                     "hook": f"Line up a rock or snake within {C.HOOK_RANGE} cells sideways",
                     "mine": "You need a mine left and a free cell beside you"}
            self.toast(f"No {tool} target. {hints[tool]}")
            return
        self.audio.play("arm", 0.5, 0.7)
        self.armed = tool

    def submit_console(self):
        text = self.console.strip()
        self.console = None
        pygame.key.stop_text_input()
        if not text:
            return
        try:
            acts, tree = parse_command(text)
            turns = to_turns(acts)
            self.console_hist.append((text, tree, None))
            self.queue = turns
            self.queue_pid = self.world.current if self.human_turn() else self.me
            self.add_log(f"Parsed '{text}' -> {len(turns)} turn(s)", (200, 170, 255))
        except ParseError as ex:
            self.console_hist.append((text, None, str(ex)))
            self.toast(f"Parser: {ex}")
        self.console_hist = self.console_hist[-6:]
        self.panel = "console"

    # ================================================================ draw: loading
    def draw_loading(self, dt):
        g, ui = self.g, self.ui
        g.clear((6, 8, 18))
        t = self.now
        for i in range(5):
            a = t * 0.2 + i * 1.3
            ui.glow(self.DW / 2 + math.cos(a) * 380, self.DH / 2 + math.sin(a * 0.8) * 200, 300,
                    [(59, 120, 255), (170, 80, 255), (255, 60, 120), (40, 200, 220), (120, 60, 200)][i], 0.12)
        img = self.gradient_text("DIAMOND QUEST", 66, (250, 232, 185), (177, 132, 64))
        w, h = img.width / g.u, img.height / g.u
        ui.tex(img, self.DW / 2 - w / 2, self.DH / 2 - 120 - h / 2, w, h)
        p = self.A.progress()
        bw = 520
        x, y = self.DW / 2 - bw / 2, self.DH / 2 + 10
        ui.rrect(x, y, bw, 12, 6, (255, 255, 255), 26)
        ui.hgradient(x, y, max(12, bw * p), 12, (59, 232, 255), (170, 110, 255), radius=6)
        ui.glow(x + bw * p, y + 6, 26, gfx.HUMAN, 0.7)
        ts = self.A.ts or 0
        ui.text(f"Preparing your expedition  ·  {int(p * 100)}%", self.DW / 2, y + 34, 15,
                gfx.INK_DIM, True, "center")
        ui.text(f"Carved ruins. Buried riches. A rival with a plan.",
                self.DW / 2, y + 58, 12, gfx.INK_FAINT, False, "center")
        ui.text(self.tip, self.DW / 2, self.DH - 70, 15, gfx.INK, False, "center")
        ui.text("TIP", self.DW / 2, self.DH - 96, 11, gfx.HUMAN, True, "center")

    # ================================================================ draw: title
    def title_buttons(self):
        x=88
        return [Button((x,388,294,52),"BEGIN EXPEDITION",lambda:self.goto_setup(),"primary"),
                Button((x,452,294,46),"HOW TO PLAY",lambda:self._set("howto")),
                Button((x,510,294,46),"SETTINGS",lambda:self.open_settings("title")),
                Button((x,568,294,46),"QUIT",self.quit)]

    def draw_world_backdrop(self, dt, veil=200):
        v = self.title_view
        t = self.now
        ww, hh = v.world.w, v.world.h
        v.cam[0] = ww / 2 + math.sin(t * 0.05) * ww * 0.3
        v.cam[1] = hh / 2 + math.cos(t * 0.037) * hh * 0.28
        v.zoom = 0.85 + 0.1 * math.sin(t * 0.07)
        v.update(dt, t)
        v.draw(t)
        self.g.rect(0, 0, self.g.W, self.g.H, (6, 8, 18), veil)

    def draw_title(self, dt):
        ui,g=self.ui,self.g
        self.draw_world_backdrop(dt,105)
        # A readable title side and a softly lit cast side, with the real cave
        # still moving underneath the composition.
        g.draw(g.prims['hgrad'],0,0,g.W*.82,g.H,color=(3,15,19),alpha=245)
        g.draw(g.prims['vgrad'],0,0,g.W,g.H*.44,color=(3,12,17),alpha=170)
        t=self.now;gold=(211,177,111)
        ui.line(90,78,379,78,1,gold,130)
        ui.text('T H E   E X P E D I T I O N',90,99,12,gold,True)
        for word,sz,yy in [('DIAMOND',65,139),('QUEST',100,207)]:
            tex=self.gradient_text(word,sz,(250,232,185),(177,132,64))
            ui.tex(tex,84,yy,tex.width/g.u,tex.height/g.u)
        ui.text('Fortune favors the clever.',91,331,17,gfx.INK)
        ui.text('Descend. Discover. Outwit your rival.',91,357,12,gfx.INK_DIM)
        cx=self.DW*.74;fy=min(self.DH-92,638)
        ui.glow(cx,fy-225,340,(61,159,154),.19)
        ui.glow(cx+110,fy-210,230,(215,153,62),.16)
        for pid,dx,hh in [(1,118,365),(0,-88,410)]:
            tex=self.A.get(f'keyart_{pid}',400*g.u)
            if tex:
                ww=hh*.8
                self.g.draw(self.g.prims['shadow'],(cx+dx-ww*.45)*g.u,(fy-20)*g.u,
                            ww*.9*g.u,36*g.u,alpha=205)
                ui.tex(tex,cx+dx-ww/2,fy-hh,ww,hh)
        ui.line(cx-212,fy+15,cx+234,fy+15,1,gold,95)
        ui.text('HUMAN INSTINCT',cx-98,fy+33,10,gfx.HUMAN,True,'center')
        ui.text('FIERCE COMPETITION',cx+126,fy+33,10,gfx.AI,True,'center')
        mouse=g.mouse_u()
        for i,b in enumerate(self.title_buttons()):
            if b.hit(mouse):self.focus=i
            b.draw(self,mouse,t,dt,focus=i==self.focus)
        ui.text('DIAMOND QUEST  /  EXPEDITION EDITION',90,self.DH-28,9.5,gfx.INK_FAINT,True)
        ui.text('THREE WORLDS. ONE WAY OUT.',self.DW-50,self.DH-28,9.5,gfx.INK_FAINT,True,'topright')

    def draw_howto(self, dt):
        ui, g = self.ui, self.g
        self.draw_world_backdrop(dt, 215)
        x, y, w, h = 70, 50, self.DW - 140, self.DH - 100
        ui.panel(x, y, w, h, 24, alpha=230, border_alpha=40)
        ui.text("HOW TO PLAY", x + 40, y + 30, 30, gfx.INK, True, display=True)
        left = [
            ("The race", gfx.HUMAN, True),
            "You and the AI start on opposite sides of a fair, random cave",
            "and take turns. Collect diamonds, then reach the exit.",
            ("gems", None, None),
            f"The exit opens for a player holding {C.EXIT_UNLOCK_GEMS} diamonds.",
            "Whoever exits ends the game: highest points + life wins.",
            "The game also ends when every diamond is collected.",
            f"Life at 0 = instant loss. Everyone starts with {C.START_LIFE}.",
            "Modes: Human vs AI · Human vs Human (share one keyboard) · AI vs AI.",
            "",
            ("Each turn", gfx.HUMAN, True),
            "Optionally use ONE tool (it costs life, not a turn),",
            "then move one cell (no diagonals) or wait. You may share a cell.",
            "",
            ("Dangers", (255, 120, 130), True),
            f"Snake bite -{C.SNAKE_BITE}   Fire -{C.FIRE_DAMAGE}   Boulder -{C.CRUSH_DAMAGE}",
            "Red diamonds sit in vaults sealed by cracked walls,",
            "  snakes or boulders - you need your tools to get in.",
            "Boulders drop at once when you dig out what holds them up",
            "  (right onto you!). Push or hook one onto your rival!",
            "Fire never goes out and can't be hammered or hooked —",
            "  it burns you on entry and again every turn you stay.",
        ]
        right = [
            ("Tools", gfx.AI, True),
            ("tool:hammer", f"Hammer (H) -{C.HAMMER_COST}: stun a snake or break a cracked wall.", None),
            "   A stunned snake bites no one. Only the hammerer may stay one",
            "   turn in its cell; anyone else is bitten when it wakes.",
            ("tool:push", f"Push (P) -{C.PUSH_COST}: shove a rock sideways into a free cell.", None),
            ("tool:hook", f"Hook (K) -{C.HOOK_COST}: pull a rock or snake ≤{C.HOOK_RANGE} cells away (sideways,",
             None),
            "   clear line) next to you. Great for blocking your rival!",
            ("tool:mine", f"Mine (E): plant one of your {C.MINES_PER_PLAYER} on a cell beside you (uses your",
             None),
            f"   tool for the turn). Your rival loses {C.MINE_DAMAGE} life stepping on it.",
            ("tool:statue", f"Lens statue: its beam freezes you (-{C.LENS_DAMAGE}, skip {C.FREEZE_TURNS} turns).",
             None),
            "   Hook its body (lower cell) from the body's row, safe from the",
            "   beam: it slides 1 cell to you and turns to face the other way.",
            "",
            ("Controls", gfx.AI, True),
            "Arrows / WASD or click a neighbouring cell to move",
            "H / P / K / E arm a tool, then press a direction · Space wait",
            "T type a command: \"hammer left then go left\"",
            "L danger sense · V algorithm lab · B AI brain · O PDDL plan",
            "C camera · G cell grid · M big map · Tab hide map · N music · X sound · Esc pause",
            "",
            ("Press any key to go back", gfx.INK_FAINT, True),
        ]
        for col, lines in ((x + 44, left), (x + w / 2 + 10, right)):
            yy = y + 96
            for ln in lines:
                if isinstance(ln, tuple) and ln[0] == "gems":
                    for i, (k, lab) in enumerate((("gem_v", f"Violet = {C.VIOLET_POINTS} pts"),
                                                  ("gem_r", f"Red = {C.RED_POINTS} pts"))):
                        tex = self.A.get(k, 30 * g.u)
                        if tex:
                            ui.tex(tex, col + i * 190, yy - 4, 28, 28)
                        ui.text(lab, col + 34 + i * 190, yy + 2, 15, gfx.VIOLET_LIGHT if i == 0 else gfx.RED_LIGHT,
                                True)
                    yy += 28
                elif isinstance(ln, tuple) and str(ln[0]).startswith("tool:"):
                    self.hud.icon(ln[0][5:], col, yy - 2, 22, gfx.AI)
                    ui.text(ln[1], col + 30, yy, 14.5, gfx.INK_DIM)
                    yy += 22
                elif isinstance(ln, tuple):
                    ui.text(ln[0], col, yy, 17, ln[1], True)
                    yy += 27
                else:
                    ui.text(ln, col, yy, 14.5, gfx.INK_DIM)
                    yy += 22

    def draw_settings(self, dt):
        ui, g = self.ui, self.g
        if self.settings_back == "pause" and self.world is not None:
            self.view.draw(self.now, None)
            g.rect(0, 0, g.W, g.H, (4, 6, 14), 215)
        else:
            self.draw_world_backdrop(dt, 215)
        cx = self.DW / 2
        x0 = cx - 260
        ui.panel(x0 - 40, 100, 600, 540, 24, alpha=230, border_alpha=40)
        ui.text("SETTINGS", cx, 140, 30, gfx.INK, True, "center", display=True)
        mouse = g.mouse_u()
        rows = [("Display", 219), ("Music volume", 311), ("Sound effects", 371), ("Camera", 441),
                ("AI vs AI round limit", 500)]
        for lab, yy in rows:
            ui.text(lab, x0, yy, 16, gfx.INK, True, "midleft")
        for b, sel in self._settings_widgets():
            b.draw(self, mouse, self.now, dt, selected=sel)
        for sl in self._sliders:
            sl.draw(ui)
        ren = "GPU accelerated" if g.accelerated else "software rendering"
        light = "overbright lighting" if g.overbright else "standard lighting"
        ui.text(f"{g.W} x {g.H}  ·  {ren}  ·  {light}  ·  tiles painted at {self.A.ts}px", cx, 618, 12,
                gfx.INK_FAINT, False, "center")

    SETUP_ROWS = (134, 214, 294, 374, 454)        # game mode, start, first, brain, mines

    def setup_buttons(self):
        ox = 690
        m = self.match
        m.setdefault("mode", "hva")
        m.setdefault("brain2", "mcts")
        m.setdefault("arena", "sym")
        roles.set_mode(m["mode"])
        out = []
        y_mode, y_start, y_first, y_brain, y_mines = self.SETUP_ROWS

        def seg(y, opts, key, x=ox, bw=180, step=190, h=44):
            for i, (lab, val) in enumerate(opts):
                out.append((Button((x + i * step, y, bw, h), lab,
                                   lambda k=key, v=val: self.match.__setitem__(k, v), "seg"), m[key] == val))
        seg(y_mode, [("Human vs AI", "hva"), ("Human vs Human", "hvh"), ("AI vs AI", "ava")], "mode")
        if m["mode"] == "ava":
            seg(y_start, [("Start A", 0), ("Start B", 1)], "start", bw=128, step=136)
            seg(y_start, [("Balanced", "sym"), ("Asymmetric", "asym")], "arena", x=ox + 290, bw=128, step=136)
        else:
            seg(y_start, [("Start A", 0), ("Start B", 1)], "start")
        seg(y_first, [(roles.name(0), 0), (roles.name(1), 1)], "first")
        brains = [("Minimax", "minimax"), ("MCTS", "mcts")]
        if m["mode"] == "hva":
            seg(y_brain, brains, "brain")
        elif m["mode"] == "ava":
            seg(y_brain, brains, "brain", bw=128, step=136)
            seg(y_brain, brains, "brain2", x=ox + 290, bw=128, step=136)
        seg(y_mines, [(f"Mines: {C.MINES_PER_PLAYER} each", True), ("No mines", False)], "mines")
        out.append((Button((ox, 612, 250, 52), "NEW MAP", lambda: self.goto_setup(True)), False))
        out.append((Button((ox + 270, 612, 270, 52), "START MATCH", lambda: self.start_game(True), "primary"),
                    False))
        return out

    def draw_setup(self, dt):
        ui, g = self.ui, self.g
        self.draw_world_backdrop(dt, 205)
        if getattr(self, "preview_key", None) != (self.seed, self._mirrored()):
            self.preview, self.preview_meta = self._make_map(self.seed)
            self.preview_key = (self.seed, self._mirrored())
            self.preview_tex = None
        B = BIOMES[self.preview_meta["biome"]]
        ui.text("MATCH SETUP", 70, 34, 30, gfx.INK, True, display=True)
        ui.text("Every match is a new cave, made fair by a constraint solver.", 70, 76, 15, gfx.INK_DIM)
        px, py, ps = 70, 112, 560
        ui.panel(px, py, ps, ps, 20, alpha=230, border_alpha=40)
        if self.preview_tex is None:
            self.preview_tex = self._preview_texture(self.preview, self.preview_meta["biome"])
        inner = ps - 32
        ui.tex(self.preview_tex, px + 16, py + 16, inner, inner)
        sc = inner / self.preview.w
        for i, s in enumerate(self.preview.starts):
            human = i == self.match["start"]
            col = gfx.HUMAN if human else gfx.AI
            who = roles.tag(0 if human else 1)
            cx, cy = px + 16 + (s[1] + 0.5) * sc, py + 16 + (s[0] + 0.5) * sc
            k = 0.5 + 0.5 * math.sin(self.now * 4)
            ui.glow(cx, cy, 40, col, 0.6 + 0.3 * k)
            ui.disc(cx, cy, 14, (10, 14, 26))
            ui.ring(cx, cy, 15, col)
            ui.text("AB"[i], cx, cy, 15, col, True, "center")
            ui.text(who, cx, cy + 26, 12, col, True, "center", shadow=True)
        ex = self.preview.exit_pos
        cx, cy = px + 16 + (ex[1] + 0.5) * sc, py + 16 + (ex[0] + 0.5) * sc
        ui.glow(cx, cy, 36, gfx.HUMAN, 0.7)
        ui.ring(cx, cy, 11, (255, 255, 255))
        ui.text("EXIT", cx, cy + 22, 12, (255, 255, 255), True, "center", shadow=True)
        ox = 690
        mouse = g.mouse_u()
        btns = self.setup_buttons()
        mode = self.match["mode"]
        y_mode, y_start, y_first, y_brain, y_mines = self.SETUP_ROWS
        start_lab = {"hva": "YOUR STARTING POINT", "hvh": "PLAYER 1 STARTING POINT",
                     "ava": "AI-1 STARTING POINT"}[mode]
        brain_lab = {"hva": "AI BRAIN", "hvh": "AI BRAIN", "ava": "AI-1 BRAIN"}[mode]
        for lab, yy in (("GAME MODE", y_mode), (start_lab, y_start), ("WHO MOVES FIRST", y_first),
                        (brain_lab, y_brain), ("MINES", y_mines)):
            ui.text(lab, ox, yy - 22, 13, gfx.INK_FAINT, True)
        if mode == "ava":
            ui.text("AI-2 BRAIN", ox + 290, y_brain - 22, 13, gfx.INK_FAINT, True)
            ui.text("CAVE TYPE", ox + 290, y_start - 22, 13, gfx.INK_FAINT, True)
        elif mode == "hvh":
            ui.text("No AI in this mode: two players take turns on one keyboard.", ox, y_brain + 13, 13,
                    gfx.INK_DIM)
        for b, sel in btns:
            b.draw(self, mouse, self.now, dt, selected=sel)
        m = self.preview_meta
        ix, iy, iw, ih = ox, 514, 550, 84
        ui.panel(ix, iy, iw, ih, 18, alpha=200, border_alpha=30)
        ui.rrect(ix + 16, iy + 18, 4, 18, 2, B["accent"], 255)
        ui.text(B["name"].upper(), ix + 28, iy + 16, 17, B["accent"], True)
        ui.text(B["blurb"], ix + 28, iy + 40, 13, gfx.INK_DIM)
        ui.text(f"{m.get('need_hammer', 0)} diamonds need the hammer · {m.get('need_push', 0)} need a push · "
                f"{m.get('lens', 0)} lens statues · {m['csp_constraints']} constraints", ix + 28, iy + 62 - 4, 12,
                gfx.GOOD)
        ui.text("Enter: start · N: new map · Esc: back", ox, 676, 12, gfx.INK_FAINT)

    # ================================================================ draw: game
    def overlays(self):
        fns = []
        w = self.world
        if self.armed and self.human_turn():
            opts = [t for t in w.valid_tools(w.current) if t[0] == self.armed]

            def fn(v, now, opts=opts):
                g = self.g
                ts = v.ts
                for tool, d, info in opts:
                    x, y = v.S(info[1])
                    k = 0.5 + 0.5 * math.sin(now * 6)
                    g.rrect(x - ts * 0.46, y - ts * 0.46, ts * 0.92, ts * 0.92, ts * 0.16, gfx.HUMAN,
                            120 + 100 * k, filled=False, stroke=max(2, ts * 0.05))
                    g.glow(x, y, ts * 0.6, gfx.HUMAN, 0.25 * k)
                    if len(info) > 2:
                        x2, y2 = v.S(info[2])
                        g.line(x, y, x2, y2, max(2, ts * 0.04), gfx.HUMAN)
                        g.disc(x2, y2, ts * 0.08, gfx.HUMAN)
                    g.text(d.upper(), x, y - ts * 0.62, max(10, ts * 0.2), gfx.HUMAN, True, "center", shadow=True)
            fns.append(fn)
        if self.logic_on:
            la = self.logic()

            def lf(v, now, la=la):
                g = self.g
                ts = v.ts
                labels = {"Crush": "ROCK", "Burn": "FIRE", "Bite": "BITE", "LateBite": "WAKES", "Freeze": "BEAM"}
                for cell in la["danger"]:
                    x, y = v.S(cell)
                    k = 0.5 + 0.5 * math.sin(now * 4 + cell[0])
                    g.draw(g.prims["hatch"], x - ts * 0.46, y - ts * 0.46, ts * 0.92, ts * 0.92, color=gfx.DANGER,
                           alpha=60 + 50 * k)
                    g.rrect(x - ts * 0.46, y - ts * 0.46, ts * 0.92, ts * 0.92, ts * 0.12, gfx.DANGER, 200,
                            filled=False, stroke=max(2, ts * 0.04))
                    kinds = la["kinds"].get(cell, [])
                    if kinds:
                        g.text(labels[kinds[0]], x, y + ts * 0.28, max(9, ts * 0.18), (255, 220, 225), True,
                               "center", shadow=True)
                for cell in la["unstable"]:
                    x, y = v.S(cell)
                    g.text("LOOSE", x, y - ts * 0.38, max(8, ts * 0.16), gfx.AI, True, "center", shadow=True)
            fns.append(lf)
        if self.panel == "lab" and self.lab and self.lab.get("res"):
            lab = self.lab

            def lab_fn(v, now, lab=lab):
                g = self.g
                ts = v.ts
                res = lab["res"]
                k = gfx.clamp((now - lab["t0"]) / 1.6)
                n = int(len(res.expanded) * k)
                tot = max(1, len(res.expanded))
                for i, cell in enumerate(res.expanded[:n]):
                    x, y = v.S(cell)
                    if -ts < x < g.W + ts and -ts < y < g.H + ts:
                        c = gfx.lerp_col((80, 140, 255), (190, 120, 255), i / tot)
                        g.rrect(x - ts * 0.4, y - ts * 0.4, ts * 0.8, ts * 0.8, ts * 0.12, c, 70)
                if k >= 1 and res.found:
                    pts = [v.S(c) for c in res.path]
                    for a, b in zip(pts, pts[1:]):
                        g.line(a[0], a[1], b[0], b[1], max(3, ts * 0.09), (255, 255, 255))
                        g.line(a[0], a[1], b[0], b[1], max(2, ts * 0.05), gfx.HUMAN)
                x, y = v.S(lab["goal"])
                g.ring(x, y, ts * 0.5, gfx.RED_LIGHT, 255)
            fns.append(lab_fn)
        if self.panel == "plan" and self.plan and self.plan["path"]:
            plan = self.plan

            def pf(v, now, plan=plan):
                g = self.g
                ts = v.ts
                pts = [v.S(c) for c in plan["path"]]
                for i, (a, b) in enumerate(zip(pts, pts[1:])):
                    g.line(a[0], a[1], b[0], b[1], max(2, ts * 0.06), (200, 170, 255))
                    k = (now * 2 - i * 0.15) % 1.0
                    g.disc(a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, max(2, ts * 0.06),
                           (255, 255, 255))
                for i, c in enumerate(plan["path"][1:], 1):
                    x, y = v.S(c)
                    g.text(str(i), x, y, max(9, ts * 0.2), (220, 200, 255), True, "center", shadow=True)
            fns.append(pf)
        if self.panel == "brain" and self.ai_info.get("path"):
            tcol = gfx.HUMAN if roles.mode() == "ava" and self.ai_info is self.ai_infos.get(0) else gfx.AI
            tname = "AI TARGET" if roles.mode() != "ava" else \
                f"{roles.tag(0 if self.ai_info is self.ai_infos.get(0) else 1)} TARGET"
            info = self.ai_info

            def bf(v, now, info=info, tcol=tcol, tname=tname):
                g = self.g
                ts = v.ts
                pts = [v.S(c) for c in info["path"][:60]]
                for i in range(0, len(pts) - 1, 2):
                    g.line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], max(2, ts * 0.06), tcol)
                if info.get("target"):
                    x, y = v.S(info["target"])
                    k = 0.5 + 0.5 * math.sin(now * 5)
                    g.ring(x, y, ts * (0.45 + 0.08 * k), tcol, 255)
                    g.text(tname, x, y - ts * 0.65, max(9, ts * 0.18), tcol, True, "center", shadow=True)
            fns.append(bf)
        return fns

    def draw_panel(self):
        ui, hud = self.ui, self.hud
        if self.panel == "lab":
            lines = [("Search from you to a red diamond (ringed)" if roles.mode() == "hva" else
                      f"Search from {roles.name(self.me)} to a red diamond (ringed)", gfx.INK_DIM)]
            for i, (name, _) in enumerate(ALGORITHMS):
                lines.append((f"  {i + 1}   {name}", gfx.INK))
            lines.append(("  0   Compare all seven", gfx.INK))
            lines.append(("", gfx.INK))
            lab = self.lab
            if lab and lab.get("res"):
                res = lab["res"]
                lines += [(res.name, gfx.HUMAN, True),
                          (f"nodes expanded  {len(res.expanded)}", gfx.INK_DIM),
                          (f"path length     {len(res.path) - 1 if res.found else '—'} steps", gfx.INK_DIM),
                          (f"path cost       {res.cost:.0f}" if res.cost is not None else "path cost  —", gfx.INK_DIM),
                          (f"time            {lab['ms']:.1f} ms", gfx.INK_DIM)]
                if res.note:
                    lines.append((res.note, gfx.AI))
            elif lab and lab.get("table"):
                lines.append(("algorithm      nodes   steps   cost", gfx.HUMAN, True))
                for name, n, steps, cost, ms in lab["table"]:
                    lines.append((f"{name[:13]:<14}{n:>6}  {str(steps) if steps is not None else '—':>6}"
                                  f"  {cost if cost is not None else 0:>5.0f}", gfx.INK_DIM))
            hud.drawer("ALGORITHM LAB", lines, gfx.HUMAN, 440)
        elif self.panel == "brain":
            info = self.ai_info
            if roles.mode() == "hvh":
                x, y, w, h = hud.drawer("AI BRAIN", [], gfx.AI, 456)
                ui.text("No AI in Human vs Human.", x + 18, y + 46, 12.5, gfx.INK_DIM)
                return
            title = "AI BRAIN"
            if roles.mode() == "ava" and info:
                title = f"{roles.tag(0 if info is self.ai_infos.get(0) else 1)} BRAIN"
            x, y, w, h = hud.drawer(title, [], gfx.AI, 456)
            x, y = x + 18, y + 46
            if not info:
                ui.text("The AI hasn't moved yet.", x, y, 12.5, gfx.INK_DIM)
                return
            s = info.get("search", {})
            kind = info.get("target_kind", "?")
            goal = ("Goal: the EXIT" if kind == "exit" else "Goal: none" if kind is None else f"Goal: {kind} diamond")
            ui.text(goal, x, y, 14, gfx.AI, True)
            y += 20
            ui.text(f"Now: {info.get('why', '')}", x, y, 12.5, gfx.INK_DIM)
            y += 19
            if info.get("brain") == "mcts":
                lines = [f"MCTS: {s.get('iterations', 0)} full-game playouts",
                         f"position score {s.get('value', 0):.2f}  (0 = losing, 1 = winning)"]
            else:
                lines = [f"Minimax alpha-beta: depth {s.get('depth', 0)} · {s.get('nodes', 0)} nodes · "
                         f"{s.get('pruned', 0)} cut-offs",
                         f"{s.get('playouts', 0)} playouts · predicted margin {s.get('value', 0):+.0f}"]
            tour = info.get("tour") or []
            if tour:
                lines.append(f"Route plan: {len(tour)} diamonds queued (beam search)")
            for txt in lines:
                ui.text(txt, x, y, 12.5, gfx.INK_DIM)
                y += 19
            ui.text(f"Think time {info.get('think', 0) * 1000:.0f} ms · {info.get('candidates', 0)} candidate diamonds",
                    x, y, 12.5, gfx.INK_DIM)
            y += 28
            fz = info.get("fuzzy")
            if fz:
                ui.text("FUZZY RISK (Mamdani)", x, y, 11.5, gfx.AI, True)
                y += 19
                y = hud.bars(x, y, "life", fz["fuzzified"]["life"], gfx.GOOD)
                y = hud.bars(x, y + 4, "score lead", fz["fuzzified"]["lead"], gfx.HUMAN)
                y += 8
                risk = fz["value"]
                ui.text(f"risk = {risk:.2f}  ·  {risk_label(risk)}", x, y, 14, gfx.AI, True)
                y += 22
                ui.hgradient(x, y, 300, 10, (90, 240, 160), (255, 70, 90), radius=5)
                ui.disc(x + 300 * risk, y + 5, 8, (255, 255, 255))
                y += 24
            lg = info.get("logic")
            if lg:
                ui.text(f"Logic KB: {lg['facts']} facts · {lg['rules']} rules · {lg['derived']} derived", x, y,
                        12.5, gfx.INK_DIM)
        elif self.panel == "plan" and self.plan:
            pl = self.plan
            st = pl["stats"]
            lines = [(f"{st.get('objects', 0)} objects · {st.get('grounded', 0)} ground actions", gfx.INK_DIM),
                     (f"{st.get('expanded', 0)} states expanded (BFS)", gfx.INK_DIM),
                     ("Files: diamond_quest_output/problem.pddl, plan.txt", gfx.INK_FAINT), ("", gfx.INK)]
            if pl["steps"]:
                lines.append((f"Plan to the nearest diamond: {len(pl['steps'])} steps", (200, 170, 255), True))
                for i, s in enumerate(pl["steps"][:14], 1):
                    lines.append((f"{i:>2}. {s}", gfx.INK_DIM))
                if len(pl["steps"]) > 14:
                    lines.append((f"    … {len(pl['steps']) - 14} more", gfx.INK_FAINT))
            else:
                lines.append(("No plan found nearby.", gfx.AI))
            hud.drawer("PDDL PLANNER", lines, (200, 170, 255), 456)
        elif self.panel == "console":
            lines = [("Type commands in plain English, e.g.", gfx.INK_DIM),
                     ("  hammer left then move left", gfx.INK),
                     ("  use the hook on the rock to the right", gfx.INK),
                     ("  go down 3 times, wait", gfx.INK), ("", gfx.INK)]
            for text, tree, err in self.console_hist[-3:][::-1]:
                lines.append((f"> {text}", gfx.HUMAN, True))
                if err:
                    lines.append((f"  x {err}", (255, 120, 130)))
                else:
                    t = tree
                    while t:
                        lines.append(("  " + t[:46], gfx.INK_DIM))
                        t = t[46:]
            hud.drawer("COMMAND PARSER (NLP)", lines, (200, 170, 255), 456)

    def draw_console(self):
        if self.console is None:
            return
        ui = self.ui
        x, y, w, h = self.DW / 2 - 310, self.DH - 200, 620, 52
        ui.panel(x, y, w, h, 18, alpha=240, border=(200, 170, 255), border_alpha=220)
        caret = "|" if int(self.now * 2) % 2 == 0 else " "
        ui.text(">", x + 18, y + h / 2, 22, (200, 170, 255), True, "midleft")
        ui.text(self.console + caret, x + 40, y + h / 2, 17, gfx.INK, False, "midleft")
        ui.text("Enter to run · Esc to close", x + w - 12, y + h + 6, 11, gfx.INK_FAINT, False, "topright")

    def toggle_minimap(self):
        self.settings["minimap"] = not self.settings.get("minimap", True)
        self.save_settings()
        self.toast(f"Minimap {'shown' if self.settings['minimap'] else 'hidden'} (Tab)", gfx.INK)

    def minimap_chip_rect(self):
        return (self.DW - 16 - 118, self.DH - 16 - 34, 118, 34)

    def draw_game(self, dt, update=True):
        w = self.world
        # keep the explorers out from under the HUD: wider right margin while the map shows
        show_map = self.settings.get("minimap", True)
        self.view.insets = (60, 170, 310 if show_map and not self.panel else 60, 170)
        if update:
            self.view.update(dt, self.now)
        self.view.scene_audio_enabled=self.state=="game" and not w.over
        self.view.draw(self.now, self.overlays())
        ui = self.ui
        mouse = self.g.mouse_u()
        thinking = w.current in self.ais and self.ai_thread is not None
        self.view.focus_pid = self.me
        self.hud.draw_cards(w, self.now, dt, self.ai_infos,
                            {pid: a.brain for pid, a in self.ais.items()})
        self.hud.draw_turn(w, self.now, thinking, w.rounds + 1,
                           self.ava_cap() if roles.mode() == "ava" else None)
        if roles.human_pids():
            self.hud.draw_dock(w, self.now, self.armed, w.tool_used, self.human_turn() and not w.over, mouse,
                               pid=self.me)
        else:
            self.hud.dock_rects = {}
            self.hud.draw_spectator_hint()
        self.hud.draw_log(self.log, self.now)
        if self.big_map:
            self.g.rect(0, 0, self.g.W, self.g.H, (4, 6, 14), 170)
        if self.big_map or (not self.panel and show_map):
            self.hud.draw_minimap(self.view, w, self.now, self.big_map)
        elif not self.panel:
            x, y, cw, ch = self.minimap_chip_rect()
            hover = x <= mouse[0] <= x + cw and y <= mouse[1] <= y + ch
            ui.panel(x, y, cw, ch, 17, alpha=215, border_alpha=90 if hover else 40)
            self.hud.icon("exit", x + 12, y + 8, 18, gfx.HUMAN)
            ui.text("MAP  · Tab", x + 36, y + ch / 2, 12.5, gfx.INK, True, "midleft")
        if self.logic_on:
            la = self.logic()
            ui.text(f"DANGER SENSE  ·  {la['facts']} facts  {la['rules']} rules  → {len(la['danger'])} hazards",
                    self.DW / 2, 82, 11.5, gfx.GOOD, True, "center", shadow=True)
        self.draw_panel()
        self.draw_console()
        p0 = w.players[w.current]
        here = (p0.r, p0.c)
        if not w.over and self.human_turn() and not self.busy():
            warn = None
            if here in w.snakes:
                warn = "Leave the snake's cell this turn - it wakes up!"
            elif w.terrain[p0.r][p0.c] == FIRE:
                warn = "You're standing in fire - move now or burn again!"
            if warn:
                k = 0.5 + 0.5 * math.sin(self.now * 8)
                ui.text(warn, self.DW / 2, 104, 17, gfx.lerp_col((255, 90, 110), (255, 220, 120), k), True,
                        "center", shadow=True)
        if self.queue:
            ui.text(f"Running command… {len(self.queue)} turn(s) queued", self.DW / 2, self.DH - 150, 13,
                    (200, 170, 255), True, "center", shadow=True)
        self.hud.toast(self.toast_msg, self.now)
        if w.over and self.gameover_t and self.now > self.gameover_t:
            self.draw_gameover(dt)

    def draw_gameover(self, dt):
        ui, g = self.ui, self.g
        w = self.world
        if not self.jingle_done:
            self.jingle_done = True
            if roles.mode() == "hva":
                self.audio.play("victory" if w.winner == 0 else "defeat" if w.winner == 1 else "unlock")
            else:
                self.audio.play("victory" if w.winner in (0, 1) else "unlock")
            self.audio.set_intensity(0.0)
        k = gfx.clamp((self.now - self.gameover_t) / 0.6)
        g.rect(0, 0, g.W, g.H, (4, 6, 14), 205 * k)
        if w.winner == 0:
            title, c1, c2 = "VICTORY", (59, 232, 255), (140, 255, 200)
        elif w.winner == 1:
            title, c1, c2 = "DEFEAT", (255, 190, 64), (255, 92, 120)
        else:
            title, c1, c2 = "DRAW", (200, 200, 255), (170, 110, 255)
        if roles.mode() != "hva" and w.winner in (0, 1):
            title = f"{roles.card(w.winner)} WINS"
            c1, c2 = ((59, 232, 255), (140, 255, 200)) if w.winner == 0 else ((255, 190, 64), (255, 92, 120))
        img = self.gradient_text(title, 110 if roles.mode() == "hva" else 84, c1, c2)
        sc = 0.8 + 0.2 * gfx.ease_back(k)
        iw, ih = img.width / g.u * sc, img.height / g.u * sc
        ui.glow(self.DW / 2, 160, 320, c1, 0.3 * k)
        ui.tex(img, self.DW / 2 - iw / 2, 160 - ih / 2, iw, ih, alpha=255 * k)
        ui.text(w.end_reason, self.DW / 2, 236, 18, gfx.INK_DIM, False, "center", alpha=255 * k)
        cx = self.DW / 2
        x, y, cw, ch = cx - 300, 278, 600, 280
        ui.panel(x, y, cw, ch, 24, alpha=int(232 * k), border_alpha=int(40 * k))
        a, b = w.players
        cols = [x + 300, x + 470]
        for i, pid in enumerate((0, 1)):
            tex = self.A.get(f"portrait_{pid}", 60 * g.u)
            if tex:
                ui.tex(tex, cols[i] - 27, y + 14, 54, 60)
            ui.text(roles.tag(pid), cols[i], y + 80, 15, gfx.HUMAN if pid == 0 else gfx.AI, True,
                    "center")
        rows = [("Violet diamonds", a.violets, b.violets), ("Red diamonds", a.reds, b.reds),
                ("Points", a.points, b.points), ("Life left", max(0, a.life), max(0, b.life))]
        yy = y + 108
        for name, va, vb in rows:
            ui.text(name, x + 36, yy, 15, gfx.INK_DIM)
            ui.text(str(va), cols[0], yy + 9, 17, gfx.INK, True, "center")
            ui.text(str(vb), cols[1], yy + 9, 17, gfx.INK, True, "center")
            yy += 30
        ui.rect(x + 28, yy + 4, cw - 56, 1, (255, 255, 255), 60)
        ui.text("TOTAL", x + 36, yy + 18, 17, gfx.INK, True)
        ui.text(str(a.total), cols[0], yy + 28, 28, gfx.HUMAN, True, "center", display=True)
        ui.text(str(b.total), cols[1], yy + 28, 28, gfx.AI, True, "center", display=True)
        mouse = g.mouse_u()
        for btn in self._gameover_buttons():
            btn.draw(self, mouse, self.now, dt)

    def draw_pause(self, dt):
        ui, g = self.ui, self.g
        self.draw_game(dt, update=False)
        g.rect(0, 0, g.W, g.H, (4, 6, 14), 200)
        img = self.gradient_text("PAUSED", 70, (250, 232, 185), (177, 132, 64))
        iw, ih = img.width / g.u, img.height / g.u
        ui.tex(img, self.DW / 2 - iw / 2, 190 - ih / 2, iw, ih)
        mouse = g.mouse_u()
        for b in self._pause_buttons():
            b.draw(self, mouse, self.now, dt)

    def draw_fps(self):
        if not self.fps_show or not self.frame_ms:
            return
        ms = sorted(self.frame_ms[-120:])
        med = ms[len(ms) // 2]
        self.ui.text(f"{1000 / max(1e-3, med):.0f} fps  ·  {med:.1f} ms  ·  {self.g.W}x{self.g.H}", 12, 8, 12,
                     gfx.GOOD, True, shadow=True)

    # ================================================================ main loop
    def frame(self, dt):
        self.now += dt
        self.A.pump(8 if self.state != "loading" else 30)
        st = self.state
        if st == "loading":
            self.draw_loading(dt)
            if not self.A.busy():
                fn, self.after_load = self.after_load, None
                if fn:
                    fn()
        elif st == "title":
            self.draw_title(dt)
        elif st == "howto":
            self.draw_howto(dt)
        elif st == "settings":
            self.draw_settings(dt)
        elif st == "setup":
            self.draw_setup(dt)
        elif st == "game":
            if not self.world.over:
                self.tick_ai()
                self.tick_frozen()
                self.run_queue()
            self.draw_game(dt)
        elif st == "pause":
            self.draw_pause(dt)
        self.draw_fps()
        self.g.present()

    def run(self):
        last = time.perf_counter()
        while True:
            t = time.perf_counter()
            dt = min(0.05, t - last)
            last = t
            for e in pygame.event.get():
                self.handle(e)
            self.frame(dt)
            self.frame_ms.append((time.perf_counter() - t) * 1000)
            if len(self.frame_ms) > 600:
                del self.frame_ms[:300]
            if not self.g.accelerated:
                self.clock.tick(120)


def main():
    seed = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        seed = int(sys.argv[1])
    App(seed).run()
