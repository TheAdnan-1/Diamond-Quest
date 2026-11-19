"""In-game HUD drawn on the GPU in design units (crisp at any resolution)."""
import math

from . import config as C
from . import gfx
from . import roles

TOOLS = [("hammer", "H", C.HAMMER_COST), ("push", "P", C.PUSH_COST), ("hook", "K", C.HOOK_COST),
         ("wait", "SPACE", 0)]


class HUD:
    def __init__(self, ui, assets):
        self.ui = ui
        self.A = assets
        self.disp = {}
        self.pulse = {0: -9.0, 1: -9.0}
        self.dock_rects = {}

    # ------------------------------------------------------------ layout helpers
    @property
    def DW(self):
        return self.ui.g.DW

    @property
    def DH(self):
        return self.ui.g.DH

    def card_rect(self, pid):
        return (16, 16, 330, 118) if pid == 0 else (self.DW - 346, 16, 330, 118)

    def target_px(self, pid):
        x, y, w, h = self.card_rect(pid)
        u = self.ui.u
        return ((x + 214) * u, (y + 92) * u) if pid == 0 else ((x + 116) * u, (y + 92) * u)

    def _num(self, key, value, dt, speed=8.0):
        v = self.disp.get(key, value)
        v += (value - v) * min(1.0, dt * speed)
        if abs(value - v) < 0.05:
            v = value
        self.disp[key] = v
        return v

    def icon(self, name, x, y, size, color=(255, 255, 255), alpha=255):
        t = self.A.get(f"icon_{name}", size * self.ui.u)
        if t is not None:
            self.ui.tex(t, x, y, size, size, color=color, alpha=alpha)

    def sprite(self, name, x, y, w, h, **kw):
        t = self.A.get(name, w * self.ui.u)
        if t is not None:
            self.ui.tex(t, x, y, w, h, **kw)

    # ------------------------------------------------------------ player cards
    def draw_cards(self, world, now, dt, ai_infos, brains):
        """ai_infos / brains: {pid: ...} for every explorer an AI controls."""
        for p in world.players:
            self._card(p, world, now, dt, ai_infos.get(p.pid), brains.get(p.pid))

    def _card(self, p, world, now, dt, ai_info, brain):
        ui = self.ui
        x,y,w,h = self.card_rect(p.pid)
        col = gfx.HUMAN if p.pid==0 else gfx.AI
        active = world.current==p.pid and not world.over
        pulse = max(0, 1-(now-self.pulse[p.pid])/.5)
        ui.panel(x,y,w,h,9,fill=(10,22,25),alpha=244,
                 border=col if active else (163,134,82),border_alpha=150 if active else 86)
        ui.line(x+16,y+1,x+w-16,y+1,1,col,180 if active else 55)
        if pulse:ui.glow(x+w-38,y+38,55,col,pulse*.35)
        ax,ay=x+40,y+37
        ui.disc(ax,ay,25,(21,36,38))
        self.sprite(f"portrait_{p.pid}",ax-26,ay-27,52,54)
        ui.ring(ax,ay,27,(178,146,88),185)
        ui.text("THE ADVENTURER" if p.pid==0 else "THE CHALLENGER",x+79,y+10,9.5,gfx.INK_DIM,True)
        ui.text(roles.card(p.pid),x+79,y+24,18,gfx.INK,True)
        sub=f"{p.mines} mines" if getattr(world,'mines_enabled',False) else "Treasure hunter"
        if brain:sub=brain.upper() + ("  /  "+ai_info['mood'] if ai_info and 'mood' in ai_info else "")
        ui.text(sub,x+80,y+49,9.5,gfx.INK_FAINT)
        pts=self._num(("pts",p.pid),p.points,dt)
        ui.text(str(int(round(pts))),x+w-16,y+5,29,col,True,'topright',display=True)
        ui.text("POINTS",x+w-17,y+43,8.5,gfx.INK_FAINT,True,'topright')
        life=self._num(("life",p.pid),max(0,p.life),dt,6)
        frac=gfx.clamp(life/C.START_LIFE)
        lc=gfx.lerp_col((224,78,76),(102,201,164),gfx.clamp((frac-.15)/.7))
        bx,by,bw=x+16,y+72,w-32
        ui.rrect(bx,by,bw,5,2,(41,55,55))
        if frac:ui.rrect(bx,by,max(4,bw*frac),5,2,lc)
        for i in range(1,10):ui.rect(bx+bw*i/10,by,1,5,(7,18,22),130)
        self.icon('heart',bx,by+14,13,lc)
        ui.text(str(int(round(life))),bx+18,by+12,11,gfx.INK,True)
        self.sprite('gem_v',bx+75,by+11,18,18)
        ui.text(str(p.violets),bx+96,by+12,11,gfx.VIOLET_LIGHT,True)
        self.sprite('gem_r',bx+121,by+11,18,18)
        ui.text(str(p.reds),bx+142,by+12,11,gfx.RED_LIGHT,True)
        need=C.EXIT_UNLOCK_GEMS
        ex=bx+bw-85
        for i in range(need):
            ui.rrect(ex+i*5.5,by+17,3.3,7,1,col if i<p.gems else (51,66,67))
        ui.text('EXIT',bx+bw,by+13,8.5,col if p.gems>=need else gfx.INK_FAINT,True,'topright')

    # ------------------------------------------------------------ turn pill
    def draw_turn(self, world, now, thinking, round_no, cap=None):
        if world.over:
            return
        ui = self.ui
        col = gfx.HUMAN if world.current == 0 else gfx.AI
        label = roles.turn_label(world.current, thinking)
        tw = ui.text_w(label, 18, True)
        w = tw + 78
        x = self.DW / 2 - w / 2
        ui.panel(x, 16, w, 42, 7, alpha=242, border=col, border_alpha=110)
        a = 0.5 + 0.5 * math.sin(now * 5)
        ui.glow(x + 24, 37, 14, col, 0.25 + 0.15 * a)
        ui.disc(x + 24, 37, 5.5, col)
        ui.text(label, x + 40, 37, 18, col, True, "midleft")
        if thinking:
            for i in range(3):
                k = (now * 2.5 - i * 0.2) % 1.0
                ui.disc(x + w - 28 + i * 8, 37, 2 + 2 * math.sin(k * math.pi), col)
        ui.text(f"ROUND {round_no}" + (f" / {cap}" if cap else ""), self.DW / 2, 64, 11, gfx.INK_FAINT, True, "midtop")

    # ------------------------------------------------------------ tool dock
    def draw_dock(self, world, now, armed, tool_used, my_turn, mouse, pid=0):
        ui = self.ui
        me = world.players[pid]
        items = list(TOOLS[:3])
        if getattr(world, "mines_enabled", False):
            items.append(("mine", "E", -1))
        items.append(TOOLS[3])
        cw, ch, gap = 112, 70, 10
        total = len(items) * cw + (len(items) - 1) * gap
        x0 = self.DW / 2 - total / 2
        y0 = self.DH - 100
        self.dock_rects = {}
        for i, (name, key, cost) in enumerate(items):
            x = x0 + i * (cw + gap)
            self.dock_rects[name] = (x, y0, cw, ch)
            hover = x <= mouse[0] <= x + cw and y0 <= mouse[1] <= y0 + ch
            on = armed == name
            disabled = not my_turn or (tool_used and name != "wait") or (name == "mine" and me.mines <= 0)
            ui.panel(x, y0, cw, ch, 8, alpha=245 if on else 237,
                     border=gfx.HUMAN if on else (173, 141, 83), border_alpha=230 if on else (150 if hover else 82))
            if on:
                ui.glow(x + cw / 2, y0 + ch / 2, 70, gfx.HUMAN, 0.3 + 0.15 * math.sin(now * 6))
            ic = gfx.INK_FAINT if disabled else ((255, 255, 255) if not on else gfx.HUMAN)
            self.icon(name, x + 7, y0 + 9, 42, (135, 145, 146) if disabled else (255, 255, 255))
            ui.text(name.capitalize(), x + 52, y0 + 13, 15, gfx.INK_FAINT if disabled else gfx.INK, True)
            if cost == -1:
                sub, scol = f"{me.mines} left", (255, 200, 120) if me.mines else gfx.INK_FAINT
            elif cost:
                sub, scol = f"-{cost} life", (255, 140, 150) if not disabled else gfx.INK_FAINT
            else:
                sub, scol = "skip turn", gfx.INK_FAINT
            ui.text(sub, x + 52, y0 + 34, 11, scol, True)
            kw = ui.text_w(key, 10, True) + 12
            ui.rrect(x + cw - kw - 7, y0 + ch - 23, kw, 17, 5, (255, 255, 255), 26)
            ui.text(key, x + cw - kw / 2 - 7, y0 + ch - 14.5, 10, gfx.INK_DIM, True, "center")
        if tool_used and my_turn:
            ui.text("Tool used — now move or wait", self.DW / 2, y0 - 16, 13, gfx.HUMAN, True, "center",
                    shadow=True)
        elif armed and my_turn:
            ui.text(f"{armed.capitalize()} armed — press a direction (Esc cancels)", self.DW / 2, y0 - 16, 13,
                    gfx.HUMAN, True, "center", shadow=True)
        if roles.mode() == "hvh" and my_turn:
            ui.text(f"{roles.card(pid)}: your move", self.DW / 2, y0 - 36, 13,
                    gfx.HUMAN if pid == 0 else gfx.AI, True, "center", shadow=True)
        hint = ("WASD / Arrows: move   ·   H: hammer   P: push   K: hook   E: mine   ·   Space: wait   ·   C: camera   G: grid   M: map   ·   Esc: menu")
        ui.text(hint, self.DW / 2, self.DH - 14, 11, gfx.INK_FAINT, False, "center")

    def draw_spectator_hint(self):
        """AI vs AI: no tool dock, just the viewing keys."""
        hint = ("Spectating AI-1 vs AI-2   ·   B: AI brain   ·   C: camera   G: grid   M: map   "
                "·   N: music   X: sound   ·   Esc: menu")
        self.ui.text(hint, self.DW / 2, self.DH - 14, 11, gfx.INK_FAINT, False, "center")

    # ------------------------------------------------------------ log
    def draw_log(self, log, now):
        ui = self.ui
        x, y, w, h = 16, self.DH - 166, 310, 132
        ui.panel(x, y, w, h, 8, alpha=219, border_alpha=62)
        ui.text("EVENTS", x + 14, y + 10, 10, gfx.INK_FAINT, True)
        yy = y + 28
        for msg, col, t in log[-6:]:
            age = now - t
            a = 255 if age < 6 else max(110, int(255 - (age - 6) * 30))
            ui.text(msg[:52], x + 14, yy, 12.5, col, False, alpha=a)
            yy += 16.5

    # ------------------------------------------------------------ minimap
    def draw_minimap(self, view, world, now, big=False):
        """Map with clear icons: diamonds, snakes, exit portal, both explorers and a legend."""
        ui = self.ui
        tex = view.minimap()
        if big:
            size = min(self.DH - 190, self.DW - 420)
            pad = 14
            x = self.DW / 2 - (size + pad * 2) / 2
            y = 70
        else:
            size = 236
            pad = 10
            x, y = self.DW - 16 - size - pad * 2, self.DH - 16 - size - pad * 2 - 26
        W = size + pad * 2
        H = size + pad * 2 + (26 if not big else 0)
        ui.panel(x, y, W, H, 8, alpha=244, border_alpha=110)
        mx, my = x + pad, y + pad
        ui.tex(tex, mx, my, size, size)
        sc = size / world.w

        def P(cell):
            return mx + (cell[1] + 0.5) * sc, my + (cell[0] + 0.5) * sc
        # snakes: small green dots
        for cell, sn in world.snakes.items():
            cx, cy = P(cell)
            ui.disc(cx, cy, max(1.6, sc * 0.38), (90, 220, 110) if not sn.stuns else (150, 170, 220))
        # lens statues and their beams
        segs = view.beam_segs if view.beam_segs is not None else view._beam_segments()
        for sid, r, cells, d in segs:
            if cells:
                (x1, y1), (x2, y2) = P(cells[0]), P(cells[-1])
                ui.line(x1 - d * sc * 0.5, y1, x2 + d * sc * 0.5, y2, max(1.4, sc * 0.35), (150, 220, 255), 220)
        for cell, st in getattr(world, "statues", {}).items():
            cx, cy = P(cell)
            hs = max(2.4, sc * 0.55)
            ui.rect(cx - hs, cy - hs, hs * 2, hs * 2 + sc, (200, 200, 190), 255)
            ui.disc(cx + st.d * hs * 0.45, cy, max(1.0, hs * 0.45), (150, 225, 255))
        for cell, owner in getattr(world, "mines", {}).items():
            cx, cy = P(cell)
            ui.disc(cx, cy, max(2.2, sc * 0.5), gfx.HUMAN if owner == 0 else gfx.AI)
            ui.disc(cx, cy, max(1.0, sc * 0.22), (255, 60, 50))
        # diamonds: real gem icons with a glow (reds bigger and pulsing)
        gs_v = max(7.0, sc * 1.9)
        gs_r = max(9.0, sc * 2.4)
        tv = self.A.get("gem_v", gs_v * ui.u)
        tr = self.A.get("gem_r", gs_r * ui.u)
        for cell, kind in world.gems.items():
            cx, cy = P(cell)
            if kind == "r":
                k = 0.5 + 0.5 * math.sin(now * 3 + cell[0] + cell[1])
                ui.glow(cx, cy, gs_r * 1.3, gfx.RED, 0.45 + 0.3 * k)
                if tr:
                    ui.tex(tr, cx - gs_r / 2, cy - gs_r / 2, gs_r, gs_r)
            else:
                ui.glow(cx, cy, gs_v * 1.1, gfx.VIOLET, 0.35)
                if tv:
                    ui.tex(tv, cx - gs_v / 2, cy - gs_v / 2, gs_v, gs_v)
        # exit portal
        ex = world.exit_pos
        ecx, ecy = P(ex)
        open_ = world.exit_open_for(getattr(view, "focus_pid", 0))
        ecol = gfx.HUMAN if open_ else (180, 150, 255)
        k = (now * 0.9) % 1.0
        ui.ring(ecx, ecy, 6 + 14 * k, ecol, 220 * (1 - k))
        ui.glow(ecx, ecy, 22, ecol, 0.8)
        es = max(13.0, sc * 3.0)
        self.icon("exit" if open_ else "lock", ecx - es / 2, ecy - es / 2, es, ecol)
        lab = "EXIT" if open_ else "EXIT (locked)"
        ui.text(lab, ecx, ecy - es / 2 - 2, 10 if not big else 13, ecol, True, "midbottom", shadow=True)
        # players
        for p in world.players:
            if not p.alive or p.exited:
                continue
            pr = view.player_cell_f(p.pid, now)
            cx, cy = mx + (pr[1] + 0.5) * sc, my + (pr[0] + 0.5) * sc
            col = gfx.HUMAN if p.pid == 0 else gfx.AI
            ui.glow(cx, cy, 16, col, 0.9)
            ui.disc(cx, cy, 5 if not big else 7, (10, 14, 26))
            ui.disc(cx, cy, 3.6 if not big else 5.5, col)
            ui.text(roles.tag(p.pid), cx, cy - 7, 9.5 if not big else 12, col, True, "midbottom",
                    shadow=True)
        # current view rectangle
        vw, vh = ui.g.W / view.ts, ui.g.H / view.ts
        vx, vy = mx + (view.cam[0] - vw / 2) * sc, my + (view.cam[1] - vh / 2) * sc
        vx2, vy2 = vx + vw * sc, vy + vh * sc
        vx, vy = max(mx, vx), max(my, vy)
        vx2, vy2 = min(mx + size, vx2), min(my + size, vy2)
        if vx2 > vx and vy2 > vy:
            ui.rrect(vx, vy, vx2 - vx, vy2 - vy, 3, (255, 255, 255), 130, filled=False, stroke=1.2)
        # legend
        nv = sum(1 for g in world.gems.values() if g == "v")
        nr = len(world.gems) - nv
        items = [("gem_v", f"{nv} violet", gfx.VIOLET_LIGHT), ("gem_r", f"{nr} red", gfx.RED_LIGHT)]
        if big:
            lx, ly = x + W + 18, y + 20
            ui.panel(lx - 10, ly - 12, 210, 350, 16, alpha=225, border_alpha=36)
            ui.text("MAP LEGEND", lx + 4, ly, 12, gfx.INK_FAINT, True)
            rows = [("gem_v", f"Violet diamond (+{10})  ·  {nv} left", gfx.VIOLET_LIGHT),
                    ("gem_r", f"Red diamond (+25)  ·  {nr} left", gfx.RED_LIGHT)]
            yy = ly + 26
            for key,terrain_label in ((f'{view.biome}/floor/0','Open path'),(f'{view.biome}/dirt/t0','Diggable ground'),(f'{view.biome}/wall/f0','Solid rock · blocked')):
                texture=self.A.get(key,20*ui.u)
                if texture:ui.tex(texture,lx,yy-3,20,20)
                ui.text(terrain_label,lx+28,yy,12,gfx.INK_DIM,True)
                yy+=26
            for name, text_, col in rows:
                t = self.A.get(name, 22 * ui.u)
                if t:
                    ui.tex(t, lx, yy - 3, 20, 20)
                ui.text(text_, lx + 28, yy, 12, col, True)
                yy += 26
            for col, text_ in ((ecol, "Exit portal"), ((90, 220, 110), "Snake"), ((150, 82, 40), "Cracked wall"),
                               ((235, 105, 30), "Fire"), ((140, 142, 150), "Boulder"),
                               ((150, 220, 255), "Lens statue + beam"), (gfx.HUMAN, roles.name(0) if roles.mode() != "hva" else "You"),
                               (gfx.AI, roles.name(1) if roles.mode() != "hva" else "AI rival")):
                ui.disc(lx + 10, yy + 7, 6, col)
                ui.text(text_, lx + 28, yy, 12, gfx.INK_DIM, True)
                yy += 22
            ui.text("M to close", x + W / 2, y + H + 10, 12, gfx.INK_DIM, True, "midtop")
        else:
            lx, ly = x + pad, y + size + pad * 2 + 2
            for i, (name, text_, col) in enumerate(items):
                t = self.A.get(name, 18 * ui.u)
                if t:
                    ui.tex(t, lx + i * 92, ly - 1, 16, 16)
                ui.text(text_, lx + 20 + i * 92, ly + 1, 11.5, col, True)
            ui.text("M", x + W - pad, ly + 1, 10.5, gfx.INK_FAINT, True, "topright")
            # a small hide button in the map's top-right corner
            bx, by = x + W - 30, y + 6
            self.map_hide_rect = (bx, by, 24, 24)
            ui.disc(bx + 12, by + 12, 11, (10, 14, 26), 220)
            ui.line(bx + 7, by + 12, bx + 17, by + 12, 2.2, gfx.INK_DIM)

    # ------------------------------------------------------------ drawers
    def drawer(self, title, lines, accent=gfx.HUMAN, height=None):
        ui = self.ui
        h = height or min(self.DH - 340, 30 + 18 * len(lines) + 24)
        x, y, w = self.DW - 366, 146, 350
        ui.panel(x, y, w, h, 18, alpha=225, border_alpha=40)
        ui.rrect(x + 14, y + 17, 4, 16, 2, accent, 255)
        ui.text(title, x + 26, y + 15, 15, gfx.INK, True)
        yy = y + 46
        for ln in lines:
            if yy > y + h - 16:
                break
            if isinstance(ln, tuple):
                s, col = ln[0], ln[1]
                bold = len(ln) > 2 and ln[2]
            else:
                s, col, bold = ln, gfx.INK_DIM, False
            ui.text(s, x + 18, yy, 12.5, col, bold)
            yy += 18.5
        return (x, y, w, h)

    def bars(self, x, y, label, memberships, color):
        ui = self.ui
        ui.text(label, x, y, 11, gfx.INK_FAINT, True)
        y += 17
        for term, deg in memberships.items():
            ui.text(term, x, y, 11.5, gfx.INK_DIM)
            ui.rrect(x + 72, y + 3, 156, 9, 4.5, (255, 255, 255), 24)
            if deg > 0:
                ui.rrect(x + 72, y + 3, max(9, 156 * deg), 9, 4.5, color, 255)
            ui.text(f"{deg:.2f}", x + 236, y, 11.5, gfx.INK_DIM)
            y += 17
        return y

    def toast(self, msg, now):
        if not msg:
            return
        ui = self.ui
        s, col, t0 = msg
        k = now - t0
        if k > 2.2:
            return
        a = int(255 * min(1, (2.2 - k) / 0.4))
        tw = ui.text_w(s, 15, True)
        w = tw + 40
        x, y = self.DW / 2 - w / 2, self.DH - 172
        ui.panel(x, y, w, 38, 19, alpha=min(225, a), border=col, border_alpha=int(a * 0.5), shadow=False)
        ui.text(s, self.DW / 2, y + 19, 15, col, True, "center", alpha=a)
