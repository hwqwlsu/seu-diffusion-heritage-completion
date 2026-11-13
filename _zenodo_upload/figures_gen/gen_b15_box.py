# -*- coding: utf-8 -*-
"""
重画 fig_B15_2：CH-Single3D 各类别构件点归属率分布
保留箱线图形式，改进可读性：
1. 叠加抖动散点（n=1/n=2 的类别由箱体退化的直线变为可见的真实样本点）
2. 纵轴收紧到 0.58~1.04，去掉 40% 空白
3. 总体均值标注右移，避免压住左侧箱体；均值写入刻度标签
4. 去掉标题中的内部实验代号 B2
5. 加图例说明 菱形=均值、圆点=单样本、虚线=总体均值
数据：results_b2_sam/results.json（30 个样本，真实数据，未改动）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os, json
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.9

ROOT = r'C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01'
OUT = os.path.join(ROOT, 'figures')
DPI = 800

C = ['#2C5F8A', '#E07B39', '#4C9A6C', '#7A5C9E', '#B85450', '#C9922B']

# ---- 读取真实数据 ----
b2 = json.load(open(os.path.join(ROOT, 'results_b2_sam', 'results.json'), encoding='utf-8'))
cats = defaultdict(list)
for r in b2:
    cats[r['category']].append(r['assigned_ratio'])
cat_order = [c for c in ['陶瓷器', '青铜器', '造像雕塑', '建筑构件', '其他'] if c in cats]
data = [np.array(cats[c]) for c in cat_order]
overall = float(np.mean([r['assigned_ratio'] for r in b2]))

# 刻度标签带上样本数与均值，省去图内文字标注
labels = [f'{c}\n(n={len(cats[c])}, 均值 {np.mean(cats[c]):.3f})' for c in cat_order]

fig, ax = plt.subplots(figsize=(10, 5.8))
rng = np.random.default_rng(7)

bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True,
                widths=0.5, showfliers=False,
                meanprops=dict(marker='D', markerfacecolor=C[4], markeredgecolor='white',
                               markersize=8, markeredgewidth=1.2),
                medianprops=dict(color=C[4], lw=2.2),
                whiskerprops=dict(lw=1.6, color='#555555'),
                capprops=dict(lw=1.6, color='#555555'),
                boxprops=dict(lw=1.6, edgecolor='#555555'))

palette = [C[0], C[1], C[2], C[3], C[5]]
# ---- 箱体填充 + 叠加抖动散点 ----
for i, (vals, patch) in enumerate(zip(data, bp['boxes']), start=1):
    patch.set_facecolor(palette[i-1])
    patch.set_alpha(0.72)
    # n=1 无抖动；n=2 左右分开；n>2 随机抖动
    if len(vals) == 1:
        xs = np.full(1, i)
    elif len(vals) == 2:
        xs = np.array([i - 0.1, i + 0.1])
    else:
        xs = i + rng.uniform(-0.16, 0.16, len(vals))
    ax.scatter(xs, vals, s=30, color=palette[i-1], edgecolor='white',
               linewidth=0.7, alpha=0.95, zorder=4)

# ---- 总体均值参考线 + 标注（左端下方，白底避免压住箱体）----
ax.axhline(overall, ls='--', color=C[4], lw=1.8, zorder=2)
ax.text(0.62, overall - 0.014, f'总体均值 {overall:.3f}',
        ha='left', va='top', fontsize=10.5, color=C[4], fontweight='bold', zorder=6,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.88, pad=1.6))

# 轴标题保持原位置（左侧中部竖排）；旋转中文在本机渲染会叠字，改为逐字竖排
ax.set_ylabel('点\n归\n属\n率', rotation=0, fontsize=12, labelpad=16,
              linespacing=1.15, va='center', color='#333333')
ax.set_title('CH-Single3D 各类别构件归属率分布（真实实验数据）',
             fontsize=13.5, fontweight='bold', pad=12)
ax.set_ylim(0.58, 1.05)
ax.set_xlim(0.45, len(cat_order) + 0.55)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', ls='--', lw=0.6, alpha=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=10.5)
ax.tick_params(axis='y', labelsize=11)

# ---- 图例（放在左下空白区）----
from matplotlib.lines import Line2D
handles = [
    Line2D([], [], marker='D', ls='none', markerfacecolor=C[4], markeredgecolor='white',
           markersize=8, label='类别均值'),
    Line2D([], [], marker='o', ls='none', color='#777777', markersize=7, label='单个样本'),
    Line2D([], [], color=C[4], ls='--', lw=1.8, label=f'总体均值 ({overall:.3f})'),
]
ax.legend(handles=handles, loc='lower left', fontsize=10, frameon=True,
          edgecolor='#BBBBBB', fancybox=False, ncol=2)

out = os.path.join(OUT, 'fig_B15_2.png')
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white', pad_inches=0.12)
plt.close(fig)
print('saved', out)
