# -*- coding: utf-8 -*-
"""Fig.4 定性对比：GT / Damaged / Restored / 3D points。
灰度点云 + 暖灰底，最后一列按来源着色，800 dpi。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from pathlib import Path

ROOT = Path(r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01")
plt.rcParams["font.family"] = "Arial"

# (行标签, 数据文件, 方位角, 俯仰角)
CASES = [
    ("(a) Grotto statue", "raw/grotto_af4a61846cda41a08186012cdba38a10_40.npz", 140.0, 35.0),
    ("(b) Ceramic ware", "raw/ceramic_b4e5f0e6367e442c995eb1e241e61f74_40.npz", 275.0, 20.0),
    ("(c) Architectural comp.", "raw/structure_c33bb1328788413b85b164875747d9a2_40.npz", 270.0, 15.0),
]
HEAD = ["GT", "Damaged", "Restored", "3D points"]
BG = "#F4F3F0"
PT = "#8C8981"
OBS = "#2E8B57"
FIN = "#D14B47"


def principal(P):
    C = P - P.mean(axis=0)
    w, v = np.linalg.eigh(C.T @ C)
    return v[:, np.argmax(w)]


def basis(d, axis=None):
    """以 d 为视线，axis（默认世界 z）作为画面竖直方向。"""
    d = d / np.linalg.norm(d)
    a = np.array([0.0, 0.0, 1.0]) if axis is None else axis
    up = a - np.dot(a, d) * d
    if np.linalg.norm(up) < 1e-6:
        a = np.array([0.0, 1.0, 0.0])
        up = a - np.dot(a, d) * d
    up /= np.linalg.norm(up)
    e1 = np.cross(up, d)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, up, d])


def best_view(vis, miss, axis, n=36):
    """缺口居中、所见面积大、且缺失点不被观测点遮挡的视角。"""
    cands = []
    for pitch in (12.0, 25.0, 40.0):
        for k in range(n):
            yaw = 2 * np.pi * k / n
            R = basis(np.array([np.cos(yaw) * np.cos(np.deg2rad(pitch)),
                                np.sin(yaw) * np.cos(np.deg2rad(pitch)),
                                np.sin(np.deg2rad(pitch))]), axis)
            xo, yo, zo = proj(vis, R)
            xm, ym, zm = proj(miss, R)
            x0, x1 = min(xo.min(), xm.min()), max(xo.max(), xm.max())
            y0, y1 = min(yo.min(), ym.min()), max(yo.max(), ym.max())
            g = 120
            zi = np.full((g, g), -np.inf)
            ix = np.clip(((xo - x0) / (x1 - x0) * (g - 1)).astype(int), 0, g - 1)
            iy = np.clip(((yo - y0) / (y1 - y0) * (g - 1)).astype(int), 0, g - 1)
            np.maximum.at(zi, (iy, ix), zo)
            occ = zi > -np.inf
            mx = np.clip(((xm - x0) / (x1 - x0) * (g - 1)).astype(int), 0, g - 1)
            my = np.clip(((ym - y0) / (y1 - y0) * (g - 1)).astype(int), 0, g - 1)
            blocked = occ[my, mx] & (zi[my, mx] > zm + 1e-6)
            cx = (xm.mean() - x0) / (x1 - x0) - 0.5
            cy = (ym.mean() - y0) / (y1 - y0) - 0.5
            cands.append((1.0 - blocked.mean(), (x1 - x0) * (y1 - y0), yaw, pitch,
                          max(0.0, 1.0 - 2.0 * np.hypot(cx, cy))))
    amax = max(c[1] for c in cands)
    best = max(cands, key=lambda c: 0.30 * (c[1] / amax) + 0.20 * c[0] + 0.50 * c[4])
    return best[2], best[3]


def proj(P, R):
    Q = P @ R.T
    return Q[:, 0], Q[:, 1], Q[:, 2]


fig, axes = plt.subplots(3, 4, figsize=(7.05, 5.28))
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0.072, right=0.996, top=0.912, bottom=0.008,
                    wspace=0.035, hspace=0.075)

for r, (label, rel, yaw, pitch) in enumerate(CASES):
    d = np.load(ROOT / rel, allow_pickle=True)
    gt, part = d["gt"], d["partial"]
    comp, vis = d["complete"], d["visible"]
    axis = principal(gt)
    if yaw is None:
        yaw, pitch = best_view(vis, comp[len(vis):], axis)
        yaw = float(np.rad2deg(yaw))
        print(f"{label}: auto yaw={yaw:.0f} pitch={pitch:.0f}")
    R = basis(np.array([np.cos(np.deg2rad(yaw)) * np.cos(np.deg2rad(pitch)),
                        np.sin(np.deg2rad(yaw)) * np.cos(np.deg2rad(pitch)),
                        np.sin(np.deg2rad(pitch))]), axis)
    # 同行统一取 GT 的范围，保证四列同尺度
    xg, yg, _ = proj(gt, R)
    padx = 0.06 * (xg.max() - xg.min())
    pady = 0.06 * (yg.max() - yg.min())
    xlim = (xg.min() - padx, xg.max() + padx)
    ylim = (yg.min() - pady, yg.max() + pady)

    for c in range(4):
        ax = axes[r, c]
        ax.set_facecolor(BG)
        if c < 3:
            P = [gt, part, comp][c]
            x, y, z = proj(P, R)
            o = np.argsort(z)
            ax.scatter(x[o], y[o], c=PT, s=0.30, lw=0, alpha=1.0, rasterized=True)
        else:
            xv, yv, _ = proj(vis, R)
            xc, yc, _ = proj(comp[len(vis):], R)
            ax.scatter(xv, yv, c=OBS, s=0.30, lw=0, alpha=0.95, rasterized=True)
            ax.scatter(xc, yc, c=FIN, s=0.30, lw=0, alpha=0.95, rasterized=True)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#DCD9D2")
            sp.set_linewidth(0.5)
        if r == 0:
            ax.set_title(HEAD[c], fontsize=9.2, fontweight="bold", pad=4.0)

    pos = axes[r, 0].get_position()
    fig.text(0.008, (pos.y0 + pos.y1) / 2, label, rotation=90,
             va="center", ha="left", fontsize=7.0, color="#2E2E2E")

out = ROOT / "fig_en_qualitative.png"
fig.savefig(out, dpi=800, facecolor="white", bbox_inches="tight", pad_inches=0.015)
plt.close(fig)
print("saved", out)
