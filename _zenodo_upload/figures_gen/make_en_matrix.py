# -*- coding: utf-8 -*-
"""Fig.5 六类别 × 四损伤档：直接从 raw/*.npz 渲染 Damaged / Restored，灰度点云。
不裁切旧图，全部几何自绘，保证同一行四档同视角、同尺度。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01")
plt.rcParams["font.family"] = "Arial"

# 行顺序与 Table 5 一致
CASES = [
    ("Grotto statue", "grotto_0c90ba77c3684886b859329b1ca2e6d5"),
    ("Architectural comp.", "structure_40fda2162d514873b6a2224da3d2de81"),
    ("Bronze vessel", "bronze_aef09725cfc5489d85f0b3cc74b4bcd5"),
    ("Ceramic ware", "ceramic_0aaf8989743c4cf587710648ef6ed05f"),
    ("Jade", "jade_810cab0dbe70487e8d94791c7f4ff0e6"),
    ("Mural", "mural_f2643c116c15498e881d4dc97394b9e6"),
]
LEVELS = [10, 20, 40, 50]
BG = "#F4F3F0"
PT = "#8C8981"


def principal(P):
    C = P - P.mean(axis=0)
    w, v = np.linalg.eigh(C.T @ C)
    return v[:, np.argmax(w)]


def basis(d, axis):
    d = d / np.linalg.norm(d)
    up = axis - np.dot(axis, d) * d
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0.0, 1.0, 0.0]) - d[1] * d
    up /= np.linalg.norm(up)
    e1 = np.cross(up, d)
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, up, d])


def proj(P, R):
    Q = P @ R.T
    return Q[:, 0], Q[:, 1], Q[:, 2]


def best_view(vis, miss, axis, n=36):
    """缺口可见率为主、所见面积占比为辅，避免把宽扁器物看成刀刃。"""
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
            # 缺失区质心越靠近投影中心，看到的越是「洞」而不是被削掉半边
            cx = (xm.mean() - x0) / (x1 - x0) - 0.5
            cy = (ym.mean() - y0) / (y1 - y0) - 0.5
            off = min(1.0, 2.0 * np.hypot(cx, cy))
            cands.append((1.0 - blocked.mean(), (x1 - x0) * (y1 - y0), yaw, pitch, 1.0 - off))
    amax = max(c[1] for c in cands)
    best = max(cands, key=lambda c: 0.30 * (c[1] / amax) + 0.20 * c[0] + 0.50 * c[4])
    return best[2], best[3], best[0]


fig, axes = plt.subplots(len(CASES), 8, figsize=(7.05, 5.62))
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0.086, right=0.998, top=0.868, bottom=0.004,
                    wspace=0.035, hspace=0.055)

for r, (label, uid) in enumerate(CASES):
    data = {lv: np.load(ROOT / "raw" / f"{uid}_{lv}.npz", allow_pickle=True) for lv in LEVELS}
    gt = data[40]["gt"]
    axis = principal(gt)
    d50 = data[50]
    vis50 = d50["visible"]
    yaw, pitch, cov = best_view(vis50, d50["complete"][len(vis50):], axis)
    print(f"{label}: yaw={np.rad2deg(yaw):.0f} pitch={pitch:.0f} vis={cov:.2f}")
    R = basis(np.array([np.cos(yaw) * np.cos(np.deg2rad(pitch)),
                        np.sin(yaw) * np.cos(np.deg2rad(pitch)),
                        np.sin(np.deg2rad(pitch))]), axis)

    xg, yg, _ = proj(gt, R)
    pad = 0.05 * max(xg.max() - xg.min(), yg.max() - yg.min())
    xlim = (xg.min() - pad, xg.max() + pad)
    ylim = (yg.min() - pad, yg.max() + pad)

    for li, lv in enumerate(LEVELS):
        for ci, key in enumerate(("partial", "complete")):
            ax = axes[r, 2 * li + ci]
            ax.set_facecolor(BG)
            P = data[lv][key]
            x, y, z = proj(P, R)
            o = np.argsort(z)
            ax.scatter(x[o], y[o], c=PT, s=0.22, lw=0, alpha=1.0, rasterized=True)
            if ci == 1:
                # 补全点单独着色，Damaged 与 Restored 的差异才看得出来
                nv = data[lv]["visible"].shape[0]
                xc, yc, _ = proj(P[nv:], R)
                ax.scatter(xc, yc, c="#D14B47", s=0.22, lw=0, rasterized=True)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#DCD9D2")
                sp.set_linewidth(0.45)
            if r == 0:
                ax.set_title("Damaged" if ci == 0 else "Restored",
                             fontsize=6.2, pad=2.0)

    pos = axes[r, 0].get_position()
    fig.text(0.010, (pos.y0 + pos.y1) / 2, label, rotation=90,
             va="center", ha="left", fontsize=6.8, color="#2E2E2E")

for li, lv in enumerate(LEVELS):
    x0 = axes[0, 2 * li].get_position().x0
    x1 = axes[0, 2 * li + 1].get_position().x1
    fig.text((x0 + x1) / 2, 0.892, f"{lv}% damage",
             ha="center", va="center", fontsize=7.6, fontweight="bold")

out = ROOT / "fig_en_matrix_6cat_4lv.png"
fig.savefig(out, dpi=800, facecolor="white", bbox_inches="tight", pad_inches=0.015)
plt.close(fig)
print("saved", out)
