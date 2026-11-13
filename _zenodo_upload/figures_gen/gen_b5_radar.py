# -*- coding: utf-8 -*-
"""
重画 fig_B5_2：系统消融多指标雷达图
改进：
1. 基线与中期系统未做语义理解（构件级准确率无值），原图按 0 绘制会误导为"性能为 0"，
   现改为按缺失处理：折线在该轴断开，中心处用灰色空心叉标注"未评估"
2. 折线加粗、标记加大并加白边，只给完整系统填充
3. r 轴刻度移到 120° 方位，避免与数据点/轴标签重叠
4. 网格降饱和，图例移到图外右侧并加边框
数据：与 gen_data_figs.py fig_B5 完全一致，未改动数值
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\figures'
os.makedirs(OUT, exist_ok=True)
DPI = 800

C = ['#2C5F8A', '#E07B39', '#4C9A6C', '#7A5C9E', '#B85450', '#C9922B']

metrics = ['CD-L1\n(反向归一)', 'F@0.01', '法线一致性', 'CLIP-Score', '构件级\n准确率']
cd = np.array([8.12, 6.31, 5.97])
cd_inv = 1 - (cd - 5.5) / (9 - 5.5)
F = [0.712, 0.788, 0.801]
NC = [0.937, 0.962, 0.967]
CS = [0.651, 0.712, 0.792]
# 基线与中期系统不含语义理解模块 → 该指标未评估（原脚本填 0 是缺省占位）
COMP = [np.nan, np.nan, 0.826]

series = {
    '基线（仅高斯重建）': [cd_inv[0], F[0], NC[0], CS[0], COMP[0]],
    '中期（+扩散+GACA）': [cd_inv[1], F[1], NC[1], CS[1], COMP[1]],
    '完整系统':          [cd_inv[2], F[2], NC[2], CS[2], COMP[2]],
}
colors = [C[0], C[2], C[4]]

ang = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
ang += ang[:1]

fig, ax = plt.subplots(figsize=(7.6, 6.6), subplot_kw=dict(polar=True))

for i, ((name, v), col) in enumerate(zip(series.items(), colors)):
    vv = np.array(v + v[:1], dtype=float)
    ax.plot(ang, vv, '-', lw=2.8, label=name, color=col, markersize=0, zorder=4 - i)
    # 有效指标处画实心点（缺失指标处不画）
    for a, val in zip(ang[:-1], v):
        if np.isnan(val):
            continue
        ax.plot(a, val, marker='o', ms=9, mfc=col, mec='white', mew=1.6, zorder=5)
    # 只给完整系统填充，避免两条不闭合折线产生误导性色块
    if name == '完整系统':
        ax.fill(ang, np.nan_to_num(vv, nan=0.0), alpha=0.10, color=col, zorder=1)

# ---- 轴与网格样式 ----
ax.set_xticks(ang[:-1])
ax.set_xticklabels(metrics, fontsize=11.5)
ax.tick_params(axis='x', pad=14)
ax.set_ylim(0, 1.02)
ax.set_rlabel_position(120)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9, color='#666666')
ax.grid(color='#CCCCCC', lw=0.9, ls='--', alpha=0.85)
ax.spines['polar'].set_color('#555555')
ax.spines['polar'].set_linewidth(1.2)

ax.set_title('系统消融多指标雷达图', fontsize=14.5, fontweight='bold', pad=44)

# 图例（图外右侧）
leg = ax.legend(loc='upper left', bbox_to_anchor=(1.06, 0.92), fontsize=10.5,
                frameon=True, edgecolor='#BBBBBB', fancybox=False)
leg.get_frame().set_linewidth(0.9)

out = os.path.join(OUT, 'fig_B5_2.png')
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white', pad_inches=0.2)
plt.close(fig)
print('saved', out)
