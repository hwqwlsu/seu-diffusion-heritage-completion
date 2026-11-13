# -*- coding: utf-8 -*-
"""用真实实验数据重制 Fig.2/3/5/8 的英文版。

数据源（均已核对）：
  Fig.2  figures/loss_history.json                     总能量/扩散/平滑/边界四项，300 次迭代
  Fig.3  results_b3_ch3d_full_adapt035/results.json     5 个损伤档，每档 n=281，全实测
         results_baseline_ch3d/results.json             IMLS-MC / AlphaShape，仅 40% 档
         Table 5（逐类别）                               本文方法六类别 × 五档
  Fig.5  results_b3_{ch3d,shrec}_{full,nobnd,nosmooth,nowarmup}_adapt035,
         results_b3_{ch3d,shrec}_nodiff_hi               与正文 Table 6 完全一致
  Fig.8  results_b1/results.json                        observed(cd_l1) vs completed(hole_cd)

不引入任何插值或合成数值。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.edgecolor"] = "#3A3A3A"
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["font.size"] = 7
plt.rcParams["xtick.major.width"] = 0.7
plt.rcParams["ytick.major.width"] = 0.7
plt.rcParams["xtick.major.size"] = 2.5
plt.rcParams["ytick.major.size"] = 2.5

C = ["#2C5F8A", "#E07B39", "#4C9A6C", "#7A5C9E", "#B85450", "#8C6D31"]
CL = ["#C9DCEC", "#F7DFC9", "#CBE6D6", "#DED3EA", "#F0D2D0", "#EFE2C4"]


def style(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, ls=(0, (3, 2)), lw=0.5, alpha=0.55, color="#999999")
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.5)


def save(fig, name):
    fig.savefig(ROOT / name, dpi=900, bbox_inches="tight", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print("saved", name)


def load(folder):
    return json.load(open(ROOT / folder / "results.json", encoding="utf-8"))


def fig_loss():
    h = json.load(open(ROOT / "figures" / "loss_history.json", encoding="utf-8"))
    it = np.arange(1, len(h["total"]) + 1)
    fig, ax = plt.subplots(figsize=(3.32, 2.35))
    ax.plot(it, h["total"], lw=1.5, color=C[4], label="Total energy $E$")
    ax.plot(it, h["diff"], lw=1.1, color=C[3], label=r"Diffusion prior $\lambda_d L_{diff}$")
    ax.plot(it, h["smooth"], lw=1.1, color=C[2], label=r"Smoothness $\lambda_s L_{smooth}$")
    # bnd 恒为 0，log 轴无法直接绘制，按原图以 1e-7 底线示意并显式标注 ≈0
    ax.plot(it, np.full(it.shape, 1e-7), lw=1.0, color=C[1],
            label=r"Boundary $\lambda_b L_{bnd}$ ($\approx$0)")
    ax.axvline(20, ls=(0, (3, 2)), color="#777777", lw=0.8)
    ax.annotate("warm-up\nends", xy=(23, 2.5e-3), fontsize=6, color="#555555", va="center")
    ax.set_yscale("log")
    ax.set_xlim(0, 305)
    ax.set_ylim(5e-8, 6e-2)
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel("Energy (log scale)")
    ax.legend(fontsize=5.4, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.30),
              handlelength=1.5, borderaxespad=0.15, labelspacing=0.3)
    style(ax)
    save(fig, "fig_en_loss.png")


def fig_error_curve():
    data = load("results_b3_ch3d_full_adapt035")
    levels = sorted({round(r["target_ratio"], 2) for r in data})
    ours = [np.mean([r["cd_l1"] for r in data if round(r["target_ratio"], 2) == lv]) * 1e4
            for lv in levels]
    n_per = [sum(1 for r in data if round(r["target_ratio"], 2) == lv) for lv in levels]

    base = load("results_baseline_ch3d")
    bmeans = {}
    for name in ("IMLS-MC", "AlphaShape"):
        vals = [r["cd_l1"] for r in base if r["method"] == name]
        bmeans[name] = (np.mean(vals) * 1e4, len(vals))

    cats = ["Grotto statue", "Architectural component", "Bronze vessel",
            "Ceramic ware", "Jade", "Mural"]
    # 正文 Table 5（CD-L1 ×1e-4），六类别 × 10/20/30/40/50%
    table5 = {
        "Grotto statue": [47.82, 114.52, 202.52, 289.54, 359.81],
        "Architectural component": [40.99, 105.26, 186.81, 274.39, 330.26],
        "Bronze vessel": [45.33, 117.12, 223.23, 308.78, 393.16],
        "Ceramic ware": [57.95, 139.26, 253.19, 352.02, 430.91],
        "Jade": [44.21, 113.35, 217.58, 289.25, 370.86],
        "Mural": [44.88, 116.39, 205.32, 249.22, 297.66],
    }

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.95, 2.45), gridspec_kw=dict(width_ratios=[1, 1.12]))
    xs = np.array(levels) * 100

    a1.plot(xs, ours, "o-", lw=1.5, ms=3.4, color=C[3], label=f"Ours (n={n_per[0]}/level)")
    for name, marker in (("AlphaShape", "^"), ("IMLS-MC", "s")):
        m, n = bmeans[name]
        a1.plot([40], [m], marker, ms=4.2, color=C[0] if name == "AlphaShape" else C[1],
                label=f"{name} @40% (n={n})", clip_on=False)
        a1.annotate(f"{m:.0f}", (40, m), textcoords="offset points", xytext=(5, -1),
                    fontsize=5.8, color=C[0] if name == "AlphaShape" else C[1], va="center")
    a1.annotate(f"{ours[1]:.0f}", (20, ours[1]), textcoords="offset points", xytext=(-3, 7),
                fontsize=5.8, color=C[3], ha="right")
    a1.set_xlabel("Damage ratio (%)")
    a1.set_ylabel(r"CD-L1 ($\times 10^{-4}$) $\downarrow$")
    a1.set_xticks(xs)
    a1.set_ylim(0, 780)
    a1.set_xlim(5, 55)
    a1.legend(fontsize=5.4, frameon=False, loc="upper left", handlelength=1.4,
              borderaxespad=0.15, labelspacing=0.3)
    a1.text(-0.16, 1.02, "(a)", transform=a1.transAxes, fontsize=7.5, fontweight="bold")
    style(a1)

    for i, cat in enumerate(cats):
        a2.plot(xs, table5[cat], "o-", lw=1.1, ms=2.2, color=C[i], label=cat)
    a2.set_xlabel("Damage ratio (%)")
    a2.set_ylabel(r"CD-L1 ($\times 10^{-4}$) $\downarrow$")
    a2.set_xticks(xs)
    a2.set_xlim(5, 55)
    a2.legend(fontsize=5.2, frameon=False, loc="upper left", handlelength=1.3,
              borderaxespad=0.15, labelspacing=0.22, ncol=1)
    a2.text(-0.16, 1.02, "(b)", transform=a2.transAxes, fontsize=7.5, fontweight="bold")
    style(a2)

    fig.tight_layout(pad=0.3, w_pad=1.4)
    save(fig, "fig_en_error_curve.png")


def fig_ablation():
    variants = [
        ("Full", "full"),
        ("w/o diffusion\nprior", "nodiff_hi"),
        ("w/o boundary\ncontinuity", "nobnd_adapt035"),
        ("w/o local\nsmoothness", "nosmooth_adapt035"),
        ("w/o warm-up", "nowarmup_adapt035"),
    ]
    ch3d, shrec = [], []
    for _, tag in variants:
        if tag == "nodiff_hi":
            ch3d.append(np.mean([r["cd_l1"] for r in load("results_b3_ch3d_nodiff_hi")]) * 1e4)
            shrec.append(np.mean([r["cd_l1"] for r in load("results_b3_shrec_nodiff_hi")]) * 1e4)
        elif tag == "full":
            ch3d.append(np.mean([r["cd_l1"] for r in load("results_b3_ch3d_full_adapt035")]) * 1e4)
            shrec.append(np.mean([r["cd_l1"] for r in load("results_b3_shrec_full_adapt035")]) * 1e4)
        else:
            ch3d.append(np.mean([r["cd_l1"] for r in load(f"results_b3_ch3d_{tag}")]) * 1e4)
            shrec.append(np.mean([r["cd_l1"] for r in load(f"results_b3_shrec_{tag}")]) * 1e4)
    ch3d, shrec = np.array(ch3d), np.array(shrec)

    fig, ax = plt.subplots(figsize=(3.35, 2.3))
    x = np.arange(len(variants))
    w = 0.36
    b1 = ax.bar(x - w / 2, ch3d, w, color=C[2], edgecolor="#4A4A4A", lw=0.5, label="CH-Single3D")
    b2 = ax.bar(x + w / 2, shrec, w, color=CL[1], edgecolor="#4A4A4A", lw=0.5, label="SHREC")
    ax.axhline(ch3d[0], ls=(0, (3, 2)), color=C[2], lw=0.8)
    for bars, vals in ((b1, ch3d), (b2, shrec)):
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                        fontsize=5.2, rotation=90, xytext=(0, 1.5), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels([v[0] for v in variants], fontsize=5.9)
    ax.set_ylabel(r"CD-L1 ($\times 10^{-4}$) $\downarrow$")
    ax.set_ylim(0, 500)
    ax.legend(fontsize=5.6, frameon=False, loc="upper right", handlelength=1.2,
              borderaxespad=0.15, labelspacing=0.25)
    ax.annotate("*", xy=(1 + w / 2, shrec[1]), fontsize=8, ha="center", va="bottom")
    style(ax)
    fig.tight_layout(pad=0.25)
    save(fig, "fig_en_ablation.png")


def fig_uncertainty():
    data = load("results_b1")
    levels = ["light", "medium", "heavy"]
    titles = ["Mild damage", "Moderate damage", "Severe damage"]
    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.2), sharey=True)
    for k, (ax, lvl, title) in enumerate(zip(axes, levels, titles)):
        sub = [r for r in data if r["level"] == lvl]
        vis = np.array([r["cd_l1"] for r in sub])
        comp = np.array([r["hole_cd"] for r in sub if r["hole_cd"]])
        bp = ax.boxplot([vis, comp], widths=0.42, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor=C[4], markeredgecolor=C[4],
                                       markersize=3),
                        tick_labels=["Observed", "Completed"])
        for patch, c in zip(bp["boxes"], [CL[0], CL[1]]):
            patch.set_facecolor(c)
            patch.set_edgecolor("#5A5A5A")
            patch.set_linewidth(0.6)
        for med in bp["medians"]:
            med.set_color(C[4])
            med.set_linewidth(1.0)
        for w in bp["whiskers"] + bp["caps"]:
            w.set_linewidth(0.6)
            w.set_color("#5A5A5A")
        ax.set_yscale("log")
        ax.set_title(f"{title}\n(n={len(sub)})", fontsize=6.6, pad=3)
        if k == 0:
            ax.set_ylabel("Local CD-L1 (m)")
        ax.text(1.255, vis.mean(), f"{vis.mean():.4f}", fontsize=5.2, color=C[0],
                va="center", ha="left")
        ax.text(2.255, comp.mean(), f"{comp.mean():.4f}", fontsize=5.2, color="#A0522D",
                va="center", ha="left")
        ax.set_xlim(0.35, 2.95)
        ax.set_ylim(5e-3, 1.2)
        style(ax)
        ax.tick_params(axis="x", labelsize=6.2)
    fig.tight_layout(pad=0.25, w_pad=0.7)
    save(fig, "fig_en_uncertainty.png")


if __name__ == "__main__":
    fig_loss()
    fig_error_curve()
    fig_ablation()
    fig_uncertainty()
