"""
Simulador FEM de Treliças 2D — v3.1 (canvas ao vivo, modo desenho)
Requisitos: Python 3.8+, matplotlib, numpy
Rodar: python fem_treli_v3.py

MODO MANUAL — controles do canvas:
  N / B / A / F     → trocar modo (teclado)
  Clique esquerdo   → ação do modo atual
  Botão direito     → cancelar seleção de barra
  Scroll            → zoom centrado no cursor
  Arrasto botão do meio → pan
  Z                 → desfazer
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
import math
import os
import sys

# ─── PALETA ───────────────────────────────────────────────────────────────────
C = dict(
    bg="#0f1117", bg2="#181b26", bg3="#1e2130",
    fg="#e0e0e0", acc="#2a6fdb", brd="#2a2d3a",
    red="#d05538", grn="#3fcf90", ylw="#f0c060",
    ten="#3fcf90", com="#f07060",
)

# ─── MATERIAIS ────────────────────────────────────────────────────────────────
MATERIAIS = {
    "Aço estrutural":      {"E": 200e9, "A": 10e-4, "cor": "#2a6fdb"},
    "Alumínio 6061":       {"E":  69e9, "A": 15e-4, "cor": "#b0b0b0"},
    "Concreto armado":     {"E":  30e9, "A": 50e-4, "cor": "#8c7b6b"},
    "Madeira (Eucalipto)": {"E":  12e9, "A": 80e-4, "cor": "#c8893a"},
}

# ─── TOPOLOGIAS PRÉ-DEFINIDAS ─────────────────────────────────────────────────
# forcas: {no_idx: (Fx_N, Fy_N)}  — sempre em Newtons
TOPOLOGIAS = {
    # Warren 3 nós: apoios nas extremidades inferiores, carga no vértice
    # Verificado: N_diag=-14.14kN (compressão), N_base=+10kN (tração)
    "Warren simples": {
        "nos":    [(0,0),(4,0),(2,2)],
        "barras": [(0,1),(0,2),(1,2)],
        "fixos":  {0:(True,True), 1:(False,True)},
        "forcas": {2:(0.0,-20e3)},
    },
    # Ponte Pratt 7 nós: cordão inferior em tração, diagonais em tração, montantes em compressão
    "Ponte Pratt": {
        "nos":    [(0,0),(2,0),(4,0),(6,0),(1,2),(3,2),(5,2)],
        "barras": [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(6,3),
                   (1,4),(1,5),(2,5),(2,6),(3,6)],
        "fixos":  {0:(True,True), 3:(False,True)},
        "forcas": {1:(0.0,-20e3), 2:(0.0,-20e3)},
    },
    # Telhado 4 nós: montante central necessário para estabilidade
    # Verificado: diagonais em compressão (-8kN), cordão inferior inativo
    "Telhado": {
        "nos":    [(0,0),(3,0),(6,0),(3,2.4)],
        "barras": [(0,1),(1,2),(0,3),(1,3),(2,3)],
        "fixos":  {0:(True,True), 2:(False,True)},
        "forcas": {3:(0.0,-20e3)},
    },
    # Cantilever: dois nós fixos à esquerda, carga na extremidade livre
    # Verificado: diagonal em compressão (-28.28kN)
    "Cantilever": {
        "nos":    [(0,0),(0,2),(2,2),(2,0)],
        "barras": [(0,1),(1,2),(2,3),(0,2),(1,3)],
        "fixos":  {0:(True,True), 1:(True,True)},
        "forcas": {3:(0.0,-20e3)},
    },
}

# ─── SOLVER FEM ───────────────────────────────────────────────────────────────
def resolver_fem(nos, barras, fixos_dof, forcas_dof):
    """
    nos        : lista de (x, y) em metros
    barras     : lista de (i, j, E_Pa, A_m2)
    fixos_dof  : set de índices DOF travados
    forcas_dof : dict {dof_idx: valor_N}
    Retorna    : (u_m, stresses_Pa)
    """
    n   = len(nos)
    dof = 2 * n
    K   = np.zeros((dof, dof))

    for (i, j, E, A) in barras:
        xi, yi = nos[i]; xj, yj = nos[j]
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        c, s = dx / L, dy / L
        k    = E * A / L
        ke   = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s],
        ])
        dd = [2*i, 2*i+1, 2*j, 2*j+1]
        for r in range(4):
            for col in range(4):
                K[dd[r], dd[col]] += ke[r, col]

    F = np.zeros(dof)
    for d, v in forcas_dof.items():
        F[d] = v

    livres = [d for d in range(dof) if d not in fixos_dof]
    if not livres:
        raise RuntimeError("Nenhum DOF livre — todos os nós estão fixados.")

    Kff = K[np.ix_(livres, livres)]
    det = np.linalg.det(Kff)
    if abs(det) < 1e-12:
        raise RuntimeError(
            "Sistema singular.\n"
            "Causas comuns:\n"
            "• Apoios insuficientes (adicione PIN ou ROLETE)\n"
            "• Barra isolada (nó sem conexão)\n"
            "• Mecanismo (retângulo sem diagonal)\n"
            "• Dois nós sobrepostos"
        )

    u_livres = np.linalg.solve(Kff, F[livres])
    u = np.zeros(dof)
    for idx, d in enumerate(livres):
        u[d] = u_livres[idx]

    stresses = []
    for (i, j, E, A) in barras:
        xi, yi = nos[i]; xj, yj = nos[j]
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        if L < 1e-9:
            stresses.append(0.0); continue
        c, s = dx / L, dy / L
        delta = (u[2*j] - u[2*i]) * c + (u[2*j+1] - u[2*i+1]) * s
        stresses.append(E * delta / L)

    return u, stresses


# ─── CANVAS INTERATIVO ────────────────────────────────────────────────────────
class CanvasManual:
    """Canvas de desenho ao vivo — clique para construir a treliça."""
    SNAP = 0.5    # metros (grid snap)
    PICK = 0.45   # metros (raio de seleção de nó)

    def __init__(self, frame, app):
        self.app        = app
        self.nos        = []   # [(x, y)]  metros
        self.barras     = []   # [(i, j, E, A)]
        self.fixos      = {}   # {ni: (fix_x, fix_y)}
        # BUG FIX: forcas armazena sempre em Newtons
        self.forcas     = {}   # {ni: (Fx_N, Fy_N)}
        self.modo       = 'no'
        self._bar_start = None
        self._mouse_xy  = None
        self._pan_st    = None
        self._zoom      = 70.0
        self._off       = [60.0, 50.0]

        self.fig, self.ax = plt.subplots(figsize=(6.5, 5), facecolor=C["bg"])
        self.fig.subplots_adjust(left=0.03, right=0.98, top=0.97, bottom=0.04)
        self.wgt = FigureCanvasTkAgg(self.fig, master=frame)
        self.wgt.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.wgt.mpl_connect('button_press_event',   self._click)
        self.wgt.mpl_connect('motion_notify_event',  self._move)
        self.wgt.mpl_connect('button_release_event', self._release)
        self.wgt.mpl_connect('scroll_event',         self._scroll)
        self._redraw()

    # ── coordenadas ──────────────────────────────────────────────────────────
    def _snap(self, x, y):
        s = self.SNAP
        return round(x / s) * s, round(y / s) * s

    def _nearest(self, wx, wy):
        best, bd = -1, self.PICK
        for i, (nx, ny) in enumerate(self.nos):
            d = math.hypot(nx - wx, ny - wy)
            if d < bd:
                best, bd = i, d
        return best

    # ── eventos ──────────────────────────────────────────────────────────────
    def _click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        wx, wy = ev.xdata, ev.ydata

        if ev.button == 2:
            self._pan_st = (ev.x, ev.y, self._off[0], self._off[1])
            return

        if ev.button == 3:
            if self.modo == 'barra':
                self._bar_start = None
                self._redraw()
            return

        if ev.button != 1:
            return

        # ── Modo: adicionar nó ──
        if self.modo == 'no':
            sx, sy = self._snap(wx, wy)
            # evita duplicata
            if not any(math.hypot(sx - nx, sy - ny) < 0.05 for nx, ny in self.nos):
                self.nos.append((sx, sy))
                self.app.sync_info()
            self._redraw()

        # ── Modo: adicionar barra ──
        elif self.modo == 'barra':
            ni = self._nearest(wx, wy)
            if ni == -1:
                return
            if self._bar_start is None:
                self._bar_start = ni
            else:
                if self._bar_start != ni:
                    a, b = min(self._bar_start, ni), max(self._bar_start, ni)
                    dup = any(
                        min(bi, bj) == a and max(bi, bj) == b
                        for bi, bj, *_ in self.barras
                    )
                    if not dup:
                        E = self.app.mat_E()
                        A = self.app.mat_A()
                        self.barras.append((self._bar_start, ni, E, A))
                        self.app.sync_info()
                self._bar_start = None
            self._redraw()

        # ── Modo: apoio ──
        elif self.modo == 'apoio':
            ni = self._nearest(wx, wy)
            if ni == -1:
                return
            cur = self.fixos.get(ni, (False, False))
            if not cur[0] and not cur[1]:
                self.fixos[ni] = (True, True)    # livre → PIN
            elif cur[0] and cur[1]:
                self.fixos[ni] = (False, True)   # PIN → ROLETE
            else:
                self.fixos.pop(ni, None)          # ROLETE → livre
            self.app.sync_info()
            self._redraw()

        # ── Modo: força ──
        elif self.modo == 'forca':
            ni = self._nearest(wx, wy)
            if ni == -1:
                return
            # BUG FIX: converte kN → N apenas aqui, armazena em N
            fx_n = self.app.forca_fx() * 1e3
            fy_n = self.app.forca_fy() * 1e3
            if fx_n == 0.0 and fy_n == 0.0:
                self.forcas.pop(ni, None)
            else:
                self.forcas[ni] = (fx_n, fy_n)
            self.app.sync_info()
            self._redraw()

    def _move(self, ev):
        if ev.button == 2 and self._pan_st:
            dx = ev.x - self._pan_st[0]
            dy = ev.y - self._pan_st[1]
            self._off[0] = self._pan_st[2] + dx
            self._off[1] = self._pan_st[3] + dy
            self._redraw()
            return
        if ev.inaxes == self.ax and ev.xdata is not None:
            self._mouse_xy = (ev.xdata, ev.ydata)
        else:
            self._mouse_xy = None
        if self._bar_start is not None:
            self._redraw()

    def _release(self, ev):
        if ev.button == 2:
            self._pan_st = None

    def _scroll(self, ev):
        f = 1.12 if ev.step > 0 else 0.88
        if ev.xdata is not None:
            self._off[0] = ev.xdata * self._zoom * (1 - f) + self._off[0] * f
            self._off[1] = ev.ydata * self._zoom * (1 - f) + self._off[1] * f
        self._zoom = max(12, min(700, self._zoom * f))
        self._redraw()

    # ── redesenho ao vivo ─────────────────────────────────────────────────────
    def _redraw(self):
        ax = self.ax
        ax.cla()
        ax.set_facecolor(C["bg"])
        for sp in ax.spines.values():
            sp.set_color(C["brd"])
        ax.tick_params(colors='#444', labelsize=6)

        W_px, H_px = [s * self.fig.dpi for s in self.fig.get_size_inches()]
        xmin = -self._off[0] / self._zoom
        xmax = (W_px - self._off[0]) / self._zoom
        ymin = -self._off[1] / self._zoom
        ymax = (H_px - self._off[1]) / self._zoom
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', adjustable='datalim')

        # grid
        step = self.SNAP
        for x in np.arange(math.floor(xmin / step) * step, xmax + step, step):
            ax.axvline(x, color='#1c2030', lw=0.5, zorder=0)
        for y in np.arange(math.floor(ymin / step) * step, ymax + step, step):
            ax.axhline(y, color='#1c2030', lw=0.5, zorder=0)

        mat_cor = MATERIAIS[self.app.mat_nome()]["cor"]
        sz = max((xmax - xmin) * 0.035, 0.1)

        # barras
        for (i, j, E, A) in self.barras:
            if i >= len(self.nos) or j >= len(self.nos):
                continue
            x0, y0 = self.nos[i]; x1, y1 = self.nos[j]
            ax.plot([x0, x1], [y0, y1], color="#4a72a8", lw=2.2,
                    solid_capstyle='round', zorder=2)
            L = math.hypot(x1 - x0, y1 - y0)
            ax.text((x0+x1)/2, (y0+y1)/2, f"{L:.2f}m",
                    color="#5a7aaa", fontsize=6, ha='center', va='bottom', zorder=4)

        # linha fantasma (barra em construção)
        if self._bar_start is not None and self._mouse_xy is not None:
            bx, by = self.nos[self._bar_start]
            mx, my = self._mouse_xy
            snap_ni = self._nearest(mx, my)
            if snap_ni != -1 and snap_ni != self._bar_start:
                mx, my = self.nos[snap_ni]
            ax.plot([bx, mx], [by, my], color=C["ylw"],
                    lw=1.5, ls='--', alpha=0.75, zorder=3)

        # apoios
        for ni, (fxb, fyb) in self.fixos.items():
            if ni >= len(self.nos):
                continue
            px, py = self.nos[ni]
            tx = [px, px - sz, px + sz, px]
            ty = [py, py - sz*1.4, py - sz*1.4, py]
            ax.fill(tx, ty, color=C["ylw"], alpha=0.3, zorder=3)
            ax.plot(tx, ty, color=C["ylw"], lw=1, zorder=3)
            ax.plot([px - sz*1.3, px + sz*1.3],
                    [py - sz*1.6, py - sz*1.6],
                    color=C["ylw"], lw=1.2, zorder=3)
            ax.text(px, py - sz*2.1,
                    "PIN" if (fxb and fyb) else "ROL",
                    color=C["ylw"], fontsize=6, ha='center', va='top', zorder=4)

        # forças — forcas já está em N
        fa = max((xmax - xmin) * 0.14, 0.3)
        for ni, (ffx, ffy) in self.forcas.items():
            if ni >= len(self.nos):
                continue
            px, py = self.nos[ni]
            mag = math.hypot(ffx, ffy)
            if mag < 1e-9:
                continue
            ddx, ddy = ffx / mag * fa, ffy / mag * fa
            ax.annotate("", xy=(px + ddx, py + ddy), xytext=(px, py),
                        arrowprops=dict(arrowstyle="->", color=C["acc"], lw=2),
                        zorder=5)
            ax.text(px + ddx*1.12, py + ddy*1.12,
                    f"({ffx/1e3:.1f},{ffy/1e3:.1f})kN",
                    color=C["acc"], fontsize=6, ha='center', va='center', zorder=6)

        # nós
        for i, (x, y) in enumerate(self.nos):
            is_sel  = (self._bar_start == i)
            is_near = (self._mouse_xy is not None and
                       math.hypot(x - self._mouse_xy[0],
                                  y - self._mouse_xy[1]) < self.PICK)
            fc = C["ylw"] if is_sel else (C["acc"] if is_near else mat_cor)
            r  = 7 if (is_sel or is_near) else 5
            ax.scatter(x, y, s=r*r*3, color=fc,
                       edgecolors='white', linewidths=1, zorder=8)
            ax.text(x, y + sz*0.7, str(i),
                    color='white', fontsize=7, ha='center', va='bottom', zorder=9)
            ax.text(x, y - sz*0.9, f"({x:.1f},{y:.1f})",
                    color='#556', fontsize=5.5, ha='center', va='top', zorder=9)

        # dica de modo
        hints = {
            'no':    "🖱 NÓ — clique no grid",
            'barra': ("🖱 BARRA — clique 2º nó | dir: cancelar"
                      if self._bar_start is not None
                      else "🖱 BARRA — clique 1º nó"),
            'apoio': "🖱 APOIO — clique nó → PIN → ROLETE → livre",
            'forca': "🖱 FORÇA — clique nó para aplicar Fx/Fy",
        }
        ax.text(0.01, 0.99, hints.get(self.modo, ""),
                transform=ax.transAxes, color='#889', fontsize=7.5,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc=C["bg2"],
                          ec='none', alpha=0.85))
        ax.text(0.99, 0.99,
                f"Nós: {len(self.nos)}  Barras: {len(self.barras)}",
                transform=ax.transAxes, color='#445',
                fontsize=7, va='top', ha='right')

        self.wgt.draw_idle()

    # ── API pública ───────────────────────────────────────────────────────────
    def set_modo(self, m):
        self.modo = m
        self._bar_start = None
        self._redraw()

    def desfazer(self):
        if self.barras:
            self.barras.pop()
        elif self.nos:
            idx = len(self.nos) - 1
            self.nos.pop()
            self.fixos.pop(idx, None)
            self.forcas.pop(idx, None)
        self.app.sync_info()
        self._redraw()

    def limpar(self):
        self.nos.clear(); self.barras.clear()
        self.fixos.clear(); self.forcas.clear()
        self._bar_start = None
        self.app.sync_info()
        self._redraw()

    def fit(self):
        if not self.nos:
            return
        W_px, H_px = [s * self.fig.dpi for s in self.fig.get_size_inches()]
        xs = [n[0] for n in self.nos]; ys = [n[1] for n in self.nos]
        rng_x = max(max(xs) - min(xs), 1)
        rng_y = max(max(ys) - min(ys), 1)
        self._zoom = min(W_px / rng_x, H_px / rng_y) * 0.60
        self._off[0] = W_px/2 - (min(xs) + max(xs))/2 * self._zoom
        self._off[1] = H_px/2 - (min(ys) + max(ys))/2 * self._zoom
        self._redraw()

    def carregar(self, nos, barras, fixos, forcas):
        """
        Carrega dados de um modelo pré-definido.
        forcas deve estar em Newtons: {ni: (Fx_N, Fy_N)}
        """
        self.nos    = list(nos)
        self.barras = list(barras)
        self.fixos  = dict(fixos)
        self.forcas = dict(forcas)   # já em N — sem conversão
        self._bar_start = None
        self.app.sync_info()
        self.fit()


# ─── PLOT DE RESULTADO ────────────────────────────────────────────────────────
def plotar_resultado(ax_a, ax_d, nos, barras, fixos, forcas_N,
                     u, stresses, mat_cor, mat_nome, escala=200):
    """
    forcas_N : {ni: (Fx_N, Fy_N)} — sempre em Newtons
    """
    nos_arr = np.array(nos, dtype=float)
    # BUG FIX: usa np.ptp() compatível com todas as versões
    span = max(float(np.ptp(nos_arr[:, 0])),
               float(np.ptp(nos_arr[:, 1])), 1.0)
    sz = span * 0.05

    def _base(ax):
        ax.set_facecolor(C["bg"])
        ax.set_aspect('equal')
        for sp in ax.spines.values():
            sp.set_color(C["brd"])
        ax.tick_params(colors='#555', labelsize=6)

    def _draw(ax, nos_p, colored=False):
        _base(ax)
        max_s = max((abs(s) for s in stresses), default=1) or 1
        for idx, (i, j, E, A) in enumerate(barras):
            x0, y0 = nos_p[i]; x1, y1 = nos_p[j]
            if colored:
                s   = stresses[idx]
                nr  = abs(s) / max_s
                col = C["ten"] if s >= 0 else C["com"]
                lw  = 1.5 + 2.5 * nr
            else:
                col = "#3d5a8a"; lw = 1.8
            ax.plot([x0, x1], [y0, y1], color=col, lw=lw,
                    solid_capstyle='round', zorder=2)

        # apoios
        for ni, (fxb, fyb) in fixos.items():
            if ni >= len(nos_p): continue
            px, py = nos_p[ni]
            tx = [px, px-sz, px+sz, px]
            ty = [py, py-sz*1.4, py-sz*1.4, py]
            ax.fill(tx, ty, color=C["ylw"], alpha=0.3, zorder=3)
            ax.plot(tx, ty, color=C["ylw"], lw=1, zorder=3)

        # forças — forcas_N em Newtons
        fa = sz * 2.5
        for ni, (ffx, ffy) in forcas_N.items():
            if ni >= len(nos_p): continue
            px, py = nos_p[ni]
            mag = math.hypot(ffx, ffy)
            if mag < 1e-9: continue
            ddx, ddy = ffx/mag*fa, ffy/mag*fa
            ax.annotate("", xy=(px+ddx, py+ddy), xytext=(px, py),
                        arrowprops=dict(arrowstyle="->", color=C["acc"], lw=1.5),
                        zorder=5)
            ax.text(px+ddx*1.1, py+ddy*1.1,
                    f"{mag/1e3:.1f}kN",
                    color=C["acc"], fontsize=6, ha='center', va='center', zorder=6)

        # nós
        for i, (x, y) in enumerate(nos_p):
            fc = C["ylw"] if i in fixos else mat_cor
            ax.scatter(x, y, s=50, color=fc,
                       edgecolors='white', linewidths=0.8, zorder=6)
            ax.text(x, y + sz*0.6, str(i),
                    color='white', fontsize=7, ha='center', va='bottom', zorder=7)

    ax_a.cla(); ax_d.cla()
    _draw(ax_a, nos_arr, colored=False)
    ax_a.set_title("Estrutura original", color='white', fontsize=8, pad=12)

    # deformada
    nos_def = nos_arr.copy()
    for i in range(len(nos)):
        nos_def[i, 0] += u[2*i]   * escala
        nos_def[i, 1] += u[2*i+1] * escala

    # overlay original em cinza
    for (i, j, E, A) in barras:
        x0, y0 = nos_arr[i]; x1, y1 = nos_arr[j]
        ax_d.plot([x0, x1], [y0, y1], color='#2a3050',
                  lw=1, ls='--', alpha=0.4, zorder=1)

    _draw(ax_d, nos_def, colored=True)
    ax_d.set_title(f"Deformada (×{escala})  [{mat_nome}]",
                   color='white', fontsize=8, pad=12)

    leg = [Line2D([0],[0], color=C["ten"], lw=2, label='Tração'),
           Line2D([0],[0], color=C["com"], lw=2, label='Compressão')]
    ax_d.legend(handles=leg, fontsize=6, facecolor='#1a1d27',
                edgecolor='#333', labelcolor='white', loc='lower right')
    
def resource_path(relative_path):
    """
    Retorna o caminho correto do arquivo tanto no Python normal
    quanto no .exe gerado pelo PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# ─── APP PRINCIPAL ────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador FEM — Treliças 2D  v3.1")

        try:
            self.iconbitmap(resource_path("icone.ico"))
        except Exception as e:
            print(f"Não foi possível carregar o ícone da janela: {e}")

        self.configure(bg=C["bg"])
        self.geometry("1340x800")
        self._build()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _lbl(self, p, t, fg="#666"):
        tk.Label(p, text=t, bg=C["bg2"], fg=fg,
                 font=("Consolas", 8)).pack(anchor="w", padx=12, pady=(7,1))

    def _sep(self, p):
        tk.Frame(p, bg=C["brd"], height=1).pack(fill=tk.X, padx=10, pady=3)

    def _btn(self, p, t, cmd, fg=None, bg=None, **kw):
        return tk.Button(p, text=t, command=cmd,
                         bg=bg or C["bg3"], fg=fg or C["fg"],
                         activebackground=C["acc"], relief=tk.FLAT,
                         font=("Consolas", 9), cursor="hand2", **kw)

    def _ent(self, p, var, w=7):
        return tk.Entry(p, textvariable=var, width=w,
                        bg=C["bg"], fg=C["fg"], insertbackground=C["fg"],
                        font=("Consolas", 9), relief=tk.FLAT)

    # ── layout ────────────────────────────────────────────────────────────────
    def _build(self):
        style = ttk.Style(self); style.theme_use("clam")
        style.configure("TNotebook",     background=C["bg2"], borderwidth=0)
        style.configure("TNotebook.Tab", background=C["bg3"], foreground=C["fg"],
                        padding=[10,4], font=("Consolas",9))
        style.map("TNotebook.Tab",       background=[("selected", C["acc"])])
        # Combobox em tema escuro
        style.configure(
        "Dark.TCombobox",
        fieldbackground=C["bg3"],
        background=C["bg3"],
        foreground=C["fg"],
        arrowcolor=C["fg"],
        bordercolor=C["brd"],
        lightcolor=C["brd"],
        darkcolor=C["brd"],
        selectbackground=C["acc"],
        selectforeground="white"
        )

        style.map(
        "Dark.TCombobox",
        fieldbackground=[
            ("readonly", C["bg3"]),
            ("focus", C["bg3"]),
            ("!disabled", C["bg3"])
        ],
        foreground=[
            ("readonly", C["fg"]),
            ("focus", C["fg"]),
            ("!disabled", C["fg"])
        ],
        background=[
            ("readonly", C["bg3"]),
            ("focus", C["bg3"]),
            ("!disabled", C["bg3"])
        ],
        selectbackground=[
            ("readonly", C["acc"]),
            ("focus", C["acc"])
        ],
        selectforeground=[
            ("readonly", "white"),
            ("focus", "white")
        ]
        )

        # Cores da lista aberta do Combobox
        self.option_add("*TCombobox*Listbox*Background", C["bg3"])
        self.option_add("*TCombobox*Listbox*Foreground", C["fg"])
        self.option_add("*TCombobox*Listbox*selectBackground", C["acc"])
        self.option_add("*TCombobox*Listbox*selectForeground", "white")

        # painel esquerdo
        left = tk.Frame(self, bg=C["bg2"], width=300)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        tk.Label(left, text="FEM  TRELIÇAS  2D", bg=C["bg2"], fg=C["acc"],
                 font=("Consolas",12,"bold")).pack(pady=(14,0))
        tk.Label(left, text="Pontes & Passarelas", bg=C["bg2"], fg="#444",
                 font=("Consolas",7)).pack(pady=(0,8))
        tk.Frame(left, bg=C["brd"], height=1).pack(fill=tk.X, padx=10)

        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        tab_m = tk.Frame(nb, bg=C["bg2"]); nb.add(tab_m, text="  Modelos  ")
        tab_e = tk.Frame(nb, bg=C["bg2"]); nb.add(tab_e, text="  Manual  ")

        # IMPORTANTE: constrói as abas ANTES do canvas (tab_e referencia self._cm via lambda)
        self._build_modelos(tab_m)
        self._build_manual(tab_e)

        # resultados
        tk.Frame(left, bg=C["brd"], height=1).pack(fill=tk.X, padx=10)
        tk.Label(left, text="RESULTADOS", bg=C["bg2"], fg="#444",
                 font=("Consolas",7)).pack(anchor="w", padx=14, pady=(6,1))
        self._txt = tk.Text(left, bg=C["bg"], fg=C["fg"], font=("Consolas",8),
                            height=12, bd=0, state=tk.DISABLED, wrap=tk.NONE)
        self._txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))
        self._txt.tag_config("h", foreground=C["acc"])
        self._txt.tag_config("t", foreground=C["ten"])
        self._txt.tag_config("c", foreground=C["com"])

        # painel direito
        right = tk.Frame(self, bg=C["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        nb2 = ttk.Notebook(right)
        nb2.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._nb2 = nb2
        self._tab_res  = tk.Frame(nb2, bg=C["bg"])
        self._tab_draw = tk.Frame(nb2, bg=C["bg"])
        nb2.add(self._tab_res,  text="  Resultado FEM  ")
        nb2.add(self._tab_draw, text="  Desenho ao vivo  ")

        # canvas resultado (matplotlib estático)
        self._fig, (self._ax_a, self._ax_d) = plt.subplots(
            1, 2, figsize=(9.5, 5.5), facecolor=C["bg"])
        self._fig.subplots_adjust(
            left=0.03, right=0.98, top=0.94, bottom=0.05, wspace=0.18)
        self._pc = FigureCanvasTkAgg(self._fig, master=self._tab_res)
        self._pc.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._placeholder()

        # canvas interativo — criado POR ÚLTIMO (lambdas acima resolvem self._cm no clique)
        self._cm = CanvasManual(self._tab_draw, self)

        # atalhos de teclado (após self._cm existir é seguro, mas lambdas protegem mesmo assim)
        self.bind("n", lambda e: self._set_modo('no'))
        self.bind("b", lambda e: self._set_modo('barra'))
        self.bind("a", lambda e: self._set_modo('apoio'))
        self.bind("f", lambda e: self._set_modo('forca'))
        self.bind("z", lambda e: self._cm.desfazer())

    # ── aba Modelos ───────────────────────────────────────────────────────────
    def _build_modelos(self, p):
        self._lbl(p, "Topologia pré-definida", "#888")
        self._var_topo = tk.StringVar(value=list(TOPOLOGIAS.keys())[0])
        ttk.Combobox(
            p,
            textvariable=self._var_topo,
            values=list(TOPOLOGIAS.keys()),
            state="readonly",
            font=("Consolas", 9),
            style="Dark.TCombobox"
        ).pack(fill=tk.X, padx=12)

        self._sep(p)
        self._lbl(p, "Material", "#888")
        self._var_mat = tk.StringVar(value=list(MATERIAIS.keys())[0])
        self._lbl_mat = tk.Label(p, text="", bg=C["bg2"], fg=C["acc"],
                                  font=("Consolas",7))
        self._lbl_mat.pack(anchor="w", padx=12)
        cb = ttk.Combobox(
            p,
            textvariable=self._var_mat,
            values=list(MATERIAIS.keys()),
            state="readonly",
            font=("Consolas", 9),
            style="Dark.TCombobox"
        )
        cb.pack(fill=tk.X, padx=12)
        cb.bind("<<ComboboxSelected>>", lambda e: self._upd_mat())
        self._upd_mat()
    
        self._sep(p)
        self._lbl(p, "Força no nó de carga (kN)  [0 = usa padrão do modelo]", "#888")
        frm = tk.Frame(p, bg=C["bg2"]); frm.pack(fill=tk.X, padx=12)
        self._rfx = tk.DoubleVar(value=0.0)
        self._rfy = tk.DoubleVar(value=0.0)
        for lbl_t, var in [("Fx:", self._rfx), ("Fy:", self._rfy)]:
            tk.Label(frm, text=lbl_t, bg=C["bg2"], fg="#888",
                     font=("Consolas",8)).pack(side=tk.LEFT)
            self._ent(frm, var, 6).pack(side=tk.LEFT, padx=4)

        self._sep(p)
        self._lbl(p, "Escala deformação (×)", "#888")
        self._escala = tk.IntVar(value=200)
        tk.Scale(p, from_=1, to=1000, orient=tk.HORIZONTAL,
                 variable=self._escala, bg=C["bg2"], fg=C["fg"],
                 troughcolor=C["bg"], highlightthickness=0,
                 activebackground=C["acc"],
                 font=("Consolas",7)).pack(fill=tk.X, padx=12)

        self._sep(p)
        self._btn(p, "▶  GERAR E RESOLVER", self._gerar,
                  fg="white", bg=C["acc"]).pack(fill=tk.X, padx=12, pady=8)
        self._btn(p, "→ Enviar para canvas Manual", self._enviar_canvas,
                  fg=C["acc"]).pack(fill=tk.X, padx=12, pady=(0,6))

    # ── aba Manual ────────────────────────────────────────────────────────────
    def _build_manual(self, p):
        self._lbl(p, "Material das barras", "#888")
        self._var_mat_m = tk.StringVar(value=list(MATERIAIS.keys())[0])
        ttk.Combobox(
            p,
            textvariable=self._var_mat_m,
            values=list(MATERIAIS.keys()),
            state="readonly",
            font=("Consolas", 9),
            style="Dark.TCombobox"
        ).pack(fill=tk.X, padx=12)

        self._sep(p)
        self._lbl(p, "MODO DE DESENHO  (atalhos: N B A F)", "#888")
        modos = [
            ("🔵  Adicionar Nó     (N)", 'no'),
            ("📏  Adicionar Barra  (B)", 'barra'),
            ("🔶  Definir Apoio    (A)", 'apoio'),
            ("↗   Aplicar Força   (F)", 'forca'),
        ]
        self._var_modo = tk.StringVar(value='no')
        for txt, val in modos:
            tk.Radiobutton(p, text=txt, variable=self._var_modo, value=val,
                           bg=C["bg2"], fg=C["fg"], selectcolor=C["bg3"],
                           activebackground=C["bg2"], font=("Consolas",9),
                           command=lambda v=val: self._cm.set_modo(v)
                           ).pack(anchor="w", padx=16, pady=1)

        self._sep(p)
        self._lbl(p, "Força a aplicar (kN)", "#888")
        frm = tk.Frame(p, bg=C["bg2"]); frm.pack(fill=tk.X, padx=12)
        self._mfx = tk.DoubleVar(value=0.0)
        self._mfy = tk.DoubleVar(value=-10.0)
        for lbl_t, var in [("Fx:", self._mfx), ("Fy:", self._mfy)]:
            tk.Label(frm, text=lbl_t, bg=C["bg2"], fg="#888",
                     font=("Consolas",8)).pack(side=tk.LEFT)
            self._ent(frm, var, 6).pack(side=tk.LEFT, padx=3)
        
        self._sep(p)
        self._lbl(p, "Escala deformação (×)", "#888")
        self._escala_manual = tk.IntVar(value=200)
        tk.Scale(p, from_=1, to=1000, orient=tk.HORIZONTAL,
                 variable=self._escala_manual, bg=C["bg2"], fg=C["fg"],
                 troughcolor=C["bg"], highlightthickness=0,
                 activebackground=C["acc"],
                 font=("Consolas",7)).pack(fill=tk.X, padx=12)

        self._sep(p)
        self._lbl(p, "Estrutura atual", "#888")
        self._info = tk.Text(p, bg=C["bg"], fg="#558",
                              font=("Consolas",7), height=5,
                              bd=0, state=tk.DISABLED)
        self._info.pack(fill=tk.X, padx=12)

        self._sep(p)
        frm2 = tk.Frame(p, bg=C["bg2"]); frm2.pack(fill=tk.X, padx=12)
        # BUG FIX: lambdas adiam avaliação de self._cm para o momento do clique
        self._btn(frm2, "↩ (Z)",    lambda: self._cm.desfazer()).pack(side=tk.LEFT, padx=(0,3))
        self._btn(frm2, "🗑 Limpar", self._limpar_m, fg=C["red"]).pack(side=tk.LEFT, padx=3)
        self._btn(frm2, "⊡ Fit",    lambda: self._cm.fit()).pack(side=tk.LEFT, padx=3)

        self._sep(p)
        self._btn(p, "▶  RESOLVER FEM", self._resolver_m,
                  fg="white", bg=C["acc"]).pack(fill=tk.X, padx=12, pady=8)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _set_modo(self, m):
        self._var_modo.set(m)
        self._cm.set_modo(m)

    def sync_info(self):
        cm = self._cm; t = self._info
        t.config(state=tk.NORMAL); t.delete("1.0", tk.END)
        t.insert(tk.END, f"Nós ({len(cm.nos)}): ")
        for i, (x, y) in enumerate(cm.nos):
            t.insert(tk.END, f" {i}({x:.1f},{y:.1f})")
        t.insert(tk.END, f"\nBarras ({len(cm.barras)}): ")
        for bi, bj, *_ in cm.barras:
            t.insert(tk.END, f" {bi}→{bj}")
        t.insert(tk.END, "\nApoios: ")
        for ni, (fxb, fyb) in cm.fixos.items():
            t.insert(tk.END, f" {ni}({'PIN' if fxb and fyb else 'ROL'})")
        t.insert(tk.END, "\nForças: ")
        for ni, (ffx, ffy) in cm.forcas.items():
            t.insert(tk.END, f" {ni}({ffx/1e3:.1f},{ffy/1e3:.1f}kN)")
        t.config(state=tk.DISABLED)

    def mat_nome(self): return self._var_mat_m.get()
    def mat_E(self):    return MATERIAIS[self.mat_nome()]["E"]
    def mat_A(self):    return MATERIAIS[self.mat_nome()]["A"]
    def forca_fx(self): return self._mfx.get()   # retorna kN (canvas converte para N)
    def forca_fy(self): return self._mfy.get()   # retorna kN

    def _upd_mat(self):
        m = MATERIAIS[self._var_mat.get()]
        self._lbl_mat.config(
            text=f"E={m['E']/1e9:.0f} GPa  A={m['A']*1e4:.0f} cm²")

    # ── gerar modelo pré-definido ─────────────────────────────────────────────
    def _gerar(self):
        topo = TOPOLOGIAS[self._var_topo.get()]
        mat  = MATERIAIS[self._var_mat.get()]
        nos  = list(topo["nos"])
        barras = [(i, j, mat["E"], mat["A"]) for i, j in topo["barras"]]

        # condições de contorno
        fixos_dof = set()
        for ni, (fxb, fyb) in topo["fixos"].items():
            if fxb: fixos_dof.add(2*ni)
            if fyb: fixos_dof.add(2*ni+1)

        # BUG FIX: só sobrescreve força do template se o usuário digitou algo ≠ 0
        # Caso contrário usa o padrão do template (já em N)
        forcas_dof = {}
        rfx_n = self._rfx.get() * 1e3   # kN → N
        rfy_n = self._rfy.get() * 1e3
        for ni, (dfx, dfy) in topo["forcas"].items():
            fx = rfx_n if rfx_n != 0.0 else dfx
            fy = rfy_n if rfy_n != 0.0 else dfy
            if fx != 0.0: forcas_dof[2*ni]   = fx
            if fy != 0.0: forcas_dof[2*ni+1] = fy

        self._executar(nos, barras, fixos_dof, forcas_dof,
                       topo["fixos"], topo["forcas"],
                       mat["cor"], self._var_mat.get(), self._escala.get())

    # ── enviar modelo para o canvas manual ────────────────────────────────────
    def _enviar_canvas(self):
        topo = TOPOLOGIAS[self._var_topo.get()]
        mat  = MATERIAIS[self._var_mat.get()]
        self._cm.carregar(
            nos    = topo["nos"],
            barras = [(i, j, mat["E"], mat["A"]) for i, j in topo["barras"]],
            fixos  = topo["fixos"],
            forcas = topo["forcas"],   # já em N no template
        )
        self._nb2.select(1)

    # ── resolver estrutura manual ─────────────────────────────────────────────
    def _resolver_m(self):
        cm = self._cm
        if len(cm.nos) < 2 or len(cm.barras) < 1:
            messagebox.showerror("Erro", "Adicione pelo menos 2 nós e 1 barra.")
            return
        if not cm.fixos:
            messagebox.showerror("Erro",
                "Defina ao menos um apoio.\n"
                "Modo Apoio (A) → clique no nó → PIN ou ROLETE.")
            return

        # monta fixos_dof
        fixos_dof = set()
        for ni, (fxb, fyb) in cm.fixos.items():
            if fxb: fixos_dof.add(2*ni)
            if fyb: fixos_dof.add(2*ni+1)

        # BUG FIX: forcas do canvas já estão em N — monta forcas_dof direto
        forcas_dof = {}
        for ni, (ffx, ffy) in cm.forcas.items():
            if ffx != 0.0: forcas_dof[2*ni]   = ffx
            if ffy != 0.0: forcas_dof[2*ni+1] = ffy

        mat_cor = MATERIAIS[self.mat_nome()]["cor"]
        self._executar(
            cm.nos, cm.barras, fixos_dof, forcas_dof,
            cm.fixos, cm.forcas,        # forcas_N em N — correto para o plot
            mat_cor, self.mat_nome(), self._escala_manual.get()
        )

    def _limpar_m(self):
        self._cm.limpar()
        self.sync_info()

    # ── núcleo: chama solver e plota ──────────────────────────────────────────
    def _executar(self, nos, barras, fixos_dof, forcas_dof,
                  fixos_dict, forcas_N, mat_cor, mat_nome, escala):
        """
        fixos_dof  : set de DOFs travados
        forcas_dof : {dof_idx: valor_N}  → para o solver
        fixos_dict : {ni:(fx,fy)}         → para o plot
        forcas_N   : {ni:(Fx_N,Fy_N)}    → para o plot (em N)
        """
        try:
            u, st = resolver_fem(nos, barras, fixos_dof, forcas_dof)
        except RuntimeError as e:
            messagebox.showerror("Erro FEM", str(e))
            return

        plotar_resultado(self._ax_a, self._ax_d,
                         nos, barras, fixos_dict, forcas_N,
                         u, st, mat_cor, mat_nome, escala)
        self._pc.draw()
        self._mostrar_res(nos, barras, u, st, mat_nome)
        self._nb2.select(0)   # mostra a aba de resultado

    def _mostrar_res(self, nos, barras, u, st, mat_nome):
        t = self._txt
        t.config(state=tk.NORMAL); t.delete("1.0", tk.END)
        t.insert(tk.END, f"Material: {mat_nome}\n", "h")
        t.insert(tk.END, "─"*34 + "\n")
        t.insert(tk.END, "DESLOCAMENTOS NODAIS\n", "h")
        for i in range(len(nos)):
            t.insert(tk.END,
                f"  Nó {i:2d}  ux={u[2*i]*1e3:+.4f}mm"
                f"  uy={u[2*i+1]*1e3:+.4f}mm\n")
        t.insert(tk.END, "\nESFORÇOS POR BARRA\n", "h")
        for k, (i, j, E, A) in enumerate(barras):
            s  = st[k]; N = s * A
            tag = "t" if s >= 0 else "c"
            tp  = "tração" if s >= 0 else "compressão"
            t.insert(tk.END,
                f"  B{k}({i}→{j})  σ={s/1e6:+.2f}MPa"
                f"  N={N/1e3:+.2f}kN  [{tp}]\n", tag)
        t.config(state=tk.DISABLED)

    def _placeholder(self):
        for ax, msg in [(self._ax_a, "Estrutura original"),
                        (self._ax_d, "Deformada")]:
            ax.cla(); ax.set_facecolor(C["bg"])
            for sp in ax.spines.values():
                sp.set_color(C["brd"])
            ax.tick_params(colors='#2a2d3a')
            ax.text(0.5, 0.5, msg, color="#2a2d3a",
                    ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
        self._pc.draw()


if __name__ == "__main__":
    App().mainloop()
