# -*- coding: utf-8 -*-
"""Fig.1 英文重制：扩散先验引导补全流程图（矢量，全英文）。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

ROOT = Path(r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.linewidth"] = 0

BLUE = "#2C5F8A"
ORANGE = "#E07B39"
GREEN = "#4C9A6C"
PURPLE = "#7A5C9E"
GREY = "#5A5A5A"
FILL_B = "#E6EFF6"
FILL_G = "#E7F2EB"
FILL_O = "#FBEADD"
FILL_Y = "#FDF6E4"


def rbox(ax, x, y, w, h, text, fc, ec, fs=7.2, bold=False, rad=0.25, lw=0.8, tc="#1A1A1A"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3, linespacing=1.35)
    return p


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.1, style="-|>", ls="-", rad=0.0):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9,
                        lw=lw, color=color, linestyle=ls, zorder=4,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def cloud(ax, cx, cy, s, rng, color_by_depth=True, alpha=0.9, hole=False, pts=420):
    """示意点云：花瓶轮廓 + 深度着色。"""
    n = pts
    t = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 1, n)
    r = 0.55 + 0.30 * np.sin(v * 2.6) - 0.10 * v
    x = cx + s * r * np.cos(t) * 0.9
    y = cy + s * (v * 2.0 - 1.0)
    if hole:
        keep = ~((np.abs(x - cx) < 0.42 * s) & (y > cy - 0.15 * s))
        x, y = x[keep], y[keep]
    if color_by_depth:
        c = y
    else:
        c = np.zeros_like(y)
    ax.scatter(x, y, c=c, cmap="viridis", s=0.9, lw=0, alpha=alpha, zorder=1)


fig, ax = plt.subplots(figsize=(7.05, 2.72))
ax.set_xlim(0, 100)
ax.set_ylim(0, 40.5)
ax.axis("off")
rng = np.random.default_rng(7)

# --- 输入 ---
cloud(ax, 6.2, 27.5, 4.4, rng, alpha=0.55, hole=True)
ax.text(6.2, 35.6, "Partial cloud $\\mathbf{P}_{obs}$", ha="center", fontsize=7.0)
ax.text(6.2, 34.2, "(depth back-projection)", ha="center", fontsize=6.2, color="#555555")

# 深度图
ax.add_patch(Rectangle((2.2, 5.0), 8.0, 7.4, fc="#F2F2F2", ec="#9A9A9A", lw=0.6))
xs = np.linspace(0, 1, 60)
yz = np.linspace(0, 1, 40)
X, Y = np.meshgrid(xs, yz)
Z = 0.5 + 0.25 * np.sin(4 * X) * np.cos(3 * Y) + 0.15 * Y
ax.pcolormesh(2.2 + X * 8.0, 5.0 + Y * 7.4, Z, cmap="viridis", shading="auto", zorder=1)
ax.text(6.2, 13.4, "Depth map $D$", ha="center", fontsize=7.0)

arrow(ax, 11.2, 27.5, 15.4, 27.5)
arrow(ax, 11.2, 8.7, 15.4, 8.7)

# --- 残损区域检测 ---
ax.add_patch(FancyBboxPatch((15.8, 2.6), 17.0, 32.4,
                            boxstyle="round,pad=0.02,rounding_size=0.5",
                            fc="none", ec="#8C8C8C", lw=0.7, ls=(0, (4, 3)), zorder=2))
ax.text(24.3, 37.7, "Damaged-region detection", ha="center", fontsize=7.0, color="#333333")
rbox(ax, 18.0, 27.0, 12.6, 4.4, "Visibility test", FILL_B, BLUE)
rbox(ax, 18.0, 14.6, 12.6, 4.4, "Boundary extraction\n($k$-NN, $k$=16)", FILL_G, GREEN, fs=6.6)
rbox(ax, 18.0, 3.6, 12.6, 4.4, "Missing-region\ninitialization", FILL_O, ORANGE, fs=6.6)
arrow(ax, 24.3, 26.8, 24.3, 19.2, lw=0.9)
arrow(ax, 24.3, 14.4, 24.3, 8.2, lw=0.9)

arrow(ax, 33.2, 20.0, 36.6, 20.0)

# --- 能量最小化 ---
ax.add_patch(FancyBboxPatch((35.6, 9.0), 36.4, 23.0,
                            boxstyle="round,pad=0.02,rounding_size=0.5",
                            fc=FILL_Y, ec=ORANGE, lw=0.9, zorder=2))
ax.text(54.2, 30.2, "Energy minimization on missing points only", ha="center",
        fontsize=7.0, fontweight="bold", color="#333333")
ax.text(54.2, 22.8,
        "$E=\\lambda_{obs}L_{obs}+\\lambda_{bnd}L_{bnd}"
        "+\\lambda_{smooth}L_{smooth}+\\lambda_{diff}L_{diff}$",
        ha="center", fontsize=6.35)
ax.text(53.8, 17.0,
        "$L_{obs}$: observed points frozen (hard constraint)\n"
        "$L_{diff}=-\\log p_\\theta(\\mathbf{x})$: score energy of\n"
        "missing points under the learned shape prior",
        ha="center", va="center", fontsize=6.3, color="#444444", linespacing=1.5)
ax.text(53.8, 11.4, "optimize only $\\mathbf{P}_{miss}$  $\\rightarrow$  $\\mathbf{P}_{obs}$ unchanged",
        ha="center", fontsize=6.3, color=BLUE)

# --- 预训练扩散先验 ---
rbox(ax, 39.5, 0.8, 29.0, 6.2,
     "Pretrained point-cloud diffusion model\n"
     "score function $s_\\theta(\\mathbf{x}_t,t)$", "#F0EDF6", PURPLE, fs=6.6)
arrow(ax, 54.0, 7.2, 54.0, 8.9, color=PURPLE, lw=1.2)
ax.text(55.2, 8.0, "shape prior", fontsize=5.8, color=PURPLE,
        va="center", ha="left")

arrow(ax, 72.2, 20.0, 74.6, 20.0)

# --- 输出 ---
cloud(ax, 83.6, 25.6, 4.4, rng, alpha=0.85)
ax.text(83.6, 35.4, "Completed cloud $\\mathbf{P}$", ha="center", fontsize=7.0)
ax.text(83.6, 34.0, "16384 points", ha="center", fontsize=6.2, color="#555555")

# 不确定性条带
strip = np.linspace(0, 1, 64)[None, :]
ax.imshow(strip, extent=(77.0, 90.0, 13.0, 15.8), aspect="auto", cmap="YlOrRd", zorder=1)
ax.add_patch(Rectangle((77.0, 13.0), 13.0, 2.8, fc="none", ec="#999999", lw=0.6))
ax.text(83.5, 10.6, "Per-point SEU  $u_{geo}\\in[0,1]$", ha="center", fontsize=6.6)
ax.text(83.5, 7.6, "low $\\rightarrow$ high uncertainty", ha="center",
        fontsize=5.9, color="#666666")

save = ROOT / "fig_en_flow.png"
fig.savefig(save, dpi=1000, bbox_inches="tight", facecolor="white", pad_inches=0.03)
plt.close(fig)
print("saved", save)
