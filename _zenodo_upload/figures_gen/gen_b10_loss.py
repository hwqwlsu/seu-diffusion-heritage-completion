# -*- coding: utf-8 -*-
"""
重画 fig_B10_2：扩散先验补全优化过程损失曲线
优化点（最小改动布局，只让曲线更醒目）：
1. 所有曲线加粗；总损失与扩散先验数值接近，用 实线 vs 虚线 区分
2. 配色沿用论文主色板，提高饱和度与对比
3. 预热竖线与标注加粗加深
4. 图例字号加大、加边框
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, os, json

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.9

OUT = r'C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\figures'
DPI = 800

# 配色（与 gen_data_figs.py 的 C 一致，顺序：蓝橙绿紫红棕）
C = ['#2C5F8A', '#E07B39', '#4C9A6C', '#7A5C9E', '#B85450', '#C9922B']

jf = os.path.join(OUT, 'loss_history.json')
with open(jf) as f:
    h = json.load(f)
iters = np.arange(1, len(h['total']) + 1)

total  = np.array(h['total'], dtype=float)
diff   = np.array(h['diff'], dtype=float)
smooth = np.array(h['smooth'], dtype=float)
bnd    = np.array(h['bnd'], dtype=float)

fig, ax = plt.subplots(figsize=(9.5, 5.6))

# 总损失 E：深红实线（最粗，主曲线）
ax.plot(iters, total, lw=3.0, color=C[4], label='总损失 E', zorder=5)
# 扩散先验：紫色虚线（与总损失数值接近，用线型区分）
ax.plot(iters, diff, lw=2.4, color=C[3], ls='--', dashes=(6, 3),
        label=r'扩散先验 $\lambda_{diff}\!\cdot\!L_{diff}$', zorder=4)
# 平滑项：绿色实线
ax.plot(iters, smooth * 0.01, lw=2.4, color=C[2],
        label=r'平滑 $\lambda_{smooth}\!\cdot\!L_{smooth}$', zorder=3)
# 边界连续：橙色实线（真实数据中 bnd≈0，用极小下限使其在 log 轴底部可见）
bnd_plot = np.maximum(bnd * 0.5, 1e-7)
ax.plot(iters, bnd_plot, lw=2.4, color=C[1],
        label=r'边界连续 $\lambda_{bnd}\!\cdot\!L_{bnd}$ ($\approx$0)', zorder=3)

# 预热结束竖线（加粗、深色）
ax.axvline(20, ls='--', color='#555555', lw=1.6)
ax.annotate('预热结束\n（引入扩散先验）', xy=(22, ax.get_ylim()[1] * 0.72),
            fontsize=10, color='#333333', fontweight='bold')

ax.set_xlabel('优化迭代次数', fontsize=12)
ax.set_ylabel('损失值', fontsize=12)
ax.set_title('扩散先验补全优化过程损失曲线', fontsize=14, fontweight='bold')

# 图例：放回图内右侧（与原图位置一致），字号加大、加边框
leg = ax.legend(fontsize=10.5, loc='center right', frameon=True,
                edgecolor='#BBBBBB', fancybox=False)
leg.get_frame().set_linewidth(0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', ls='--', lw=0.6, alpha=0.5)
ax.set_axisbelow(True)
ax.set_yscale('log')

out = os.path.join(OUT, 'fig_B10_2.png')
fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white', pad_inches=0.12)
plt.close(fig)
print('saved', out)
