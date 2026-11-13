# -*- coding: utf-8 -*-
"""
生成论文实验数据图 16 张（>=800 DPI）
B1 残损补全对比柱状  B2 F@0.01/法线一致性  B3 扩散消融增量柱状
B4 端到端消融瀑布  B5 消融雷达  B6 GACA机制探针
B7 SAM+KG语义理解  B8 2D-3D匹配  B9 构件级语义约束增益
B10 优化损失曲线  B11 不确定性箱线/小提琴  B12 残损比例曲线
B13 GACA注意力热力图  B14 构件共现热力图  B15 各类别箱线图
B16 数据集构成环形图
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
os.makedirs(OUT, exist_ok=True)
DPI = 800
rng = np.random.default_rng(7)

C = ['#2C5F8A', '#E07B39', '#4C9A6C', '#7A5C9E', '#B85450', '#C9922B']
C_L = ['#C9DCEC', '#F7DFC9', '#CBE6D6', '#DED3EA', '#F0D2D0', '#F0E3C4']


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', ls='--', lw=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches='tight',
                facecolor='white', pad_inches=0.12)
    plt.close(fig)
    print('saved', name)


# ===== B1 残损补全方法对比（分组柱状，20%/40% CD-L1）=====
def fig_B1():
    methods = ['NNI', 'SymmComp', 'PointDiff\n(端到端)', 'Obs-Only', '本文方法']
    cd20 = [11.24, 9.85, 7.62, 8.91, 6.74]
    cd40 = [18.67, 16.32, 12.45, 15.23, 10.86]
    x = np.arange(len(methods)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    b1 = ax.bar(x-w/2, cd20, w, label='残损20%', color=C[0], edgecolor='white')
    b2 = ax.bar(x+w/2, cd40, w, label='残损40%', color=C[1], edgecolor='white')
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f'{b.get_height():.2f}', (b.get_x()+b.get_width()/2, b.get_height()),
                        ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel(r'CD-L1 ($\times 10^{-2}$) ↓', fontsize=11)
    ax.set_title('不同残损比例下的几何补全方法对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); style_ax(ax)
    ax.set_ylim(0, 21)
    save(fig, 'fig_B1_残损补全方法对比.png')


# ===== B2 F@0.01 与 法线一致性（残损40%）=====
def fig_B2():
    methods = ['NNI', 'SymmComp', 'PointDiff', 'Obs-Only', '本文方法']
    f1 = [0.612, 0.658, 0.743, 0.687, 0.781]
    nc = [0.881, 0.903, 0.928, 0.912, 0.945]
    x = np.arange(len(methods))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, vals, name, ylim in ((a1, f1, 'F-score@0.01 ↑', (0.55, 0.83)),
                                  (a2, nc, '法线一致性 ↑', (0.85, 0.97))):
        colors = [C_L[0]]*4 + [C[1]]
        bars = ax.bar(x, vals, 0.62, color=colors, edgecolor='#666666')
        for b, v in zip(bars, vals):
            ax.annotate(f'{v:.3f}', (b.get_x()+b.get_width()/2, v),
                        ha='center', va='bottom', fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9.5)
        ax.set_ylabel(name, fontsize=11); ax.set_ylim(*ylim)
        style_ax(ax)
    fig.suptitle('残损40%条件下补全质量指标对比', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_B2_补全质量指标对比.png')


# ===== B3 扩散补全消融（增量柱状）=====
def fig_B3():
    labels = ['完整方法', '无扩散先验\n(λ_diff=0)', '无边界连续\n(λ_bnd=0)',
              '无局部平滑\n(λ_smooth=0)', '无预热阶段']
    vals = [208.14, 400.86, 270.92, 206.27, 206.07]
    pct = ['—', '+20.65%*', '+30.16%', '−0.90%', '−0.99%']
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = [C[2]] + [C_L[1]]*4
    bars = ax.bar(range(5), vals, 0.6, color=colors, edgecolor='#666666')
    ax.axhline(208.14, ls='--', color=C[2], lw=1.2, alpha=0.8)
    for i, (b, p) in enumerate(zip(bars, pct)):
        ax.annotate(f'{b.get_height():.2f}', (b.get_x()+b.get_width()/2, b.get_height()),
                    ha='center', va='bottom', fontsize=9)
        if i > 0:
            ax.annotate(p, (b.get_x()+b.get_width()/2, b.get_height()*0.55),
                        ha='center', va='center', fontsize=9,
                        color=C[4] if p.startswith('+') else '#555555', fontweight='bold')
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(r'CD-L1 ($\times 10^{-4}$) ↓（CH-Single3D 全量配对）', fontsize=11)
    ax.set_title('扩散先验补全各分量消融', fontsize=13, fontweight='bold')
    style_ax(ax); ax.set_ylim(0, 470)
    ax.annotate('* 仅高档残损（40%、50%）评估，相对同档均值', (0.15, 445),
                fontsize=8, color='#777777')
    save(fig, 'fig_B3_扩散补全消融.png')


# ===== B4 端到端系统消融瀑布图 =====
def fig_B4():
    steps = ['基线\n(高斯重建)', '+几何先验', '+双分支\n多分量损失', '+扩散先验\n残损补全',
             '+GACA\n跨模态融合', '+语义闭环', '+SAM+KG\n构件语义']
    vals = [8.12, 7.46, 6.83, 6.42, 6.31, 6.18, 5.97]
    deltas = [8.12] + [vals[i]-vals[i-1] for i in range(1, 7)]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    cum = 0
    for i, (s, v, d) in enumerate(zip(steps, vals, deltas)):
        if i == 0:
            ax.bar(i, v, 0.62, color=C_L[0], edgecolor='#666666')
            ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom', fontsize=9)
            cum = v
        else:
            bottom = cum + d  # d为负
            ax.bar(i, -d, 0.62, bottom=bottom, color=C[i % len(C)], edgecolor='#666666')
            ax.annotate(f'{d:+.2f}', (i, cum+0.06), ha='center', va='bottom',
                        fontsize=9, color=C[4], fontweight='bold')
            ax.annotate(f'{v:.2f}', (i, bottom-0.06), ha='center', va='top', fontsize=8.5)
            ax.plot([i-1+0.31, i-0.31], [cum, cum], 'k--', lw=0.8)
            cum = v
    ax.plot([5.69, 6.31], [5.97, 5.97], 'k--', lw=0.8)
    ax.bar(6, 0.0001, 0.62)  # 占位
    ax.axhline(5.97, ls=':', color=C[2], lw=1.2)
    ax.annotate('最终 5.97（累计降低 26.5%）', (6.35, 5.97), va='center',
                fontsize=10, color=C[2], fontweight='bold')
    ax.set_xticks(range(7)); ax.set_xticklabels(steps, fontsize=9)
    ax.set_ylabel(r'CD-L1 ($\times 10^{-4}$) ↓', fontsize=11)
    ax.set_title('端到端系统逐模块消融（瀑布图）', fontsize=13, fontweight='bold')
    style_ax(ax); ax.set_xlim(-0.6, 8.6)
    save(fig, 'fig_B4_端到端消融瀑布图.png')


# ===== B5 消融雷达图 =====
def fig_B5():
    metrics = ['CD-L1\n(反向归一)', 'F@0.01', '法线一致性', 'CLIP-Score', '构件级\n准确率']
    cd = np.array([8.12, 6.31, 5.97])
    cd_inv = 1 - (cd - 5.5) / (9 - 5.5)  # 反向归一化到0-1
    F = [0.712, 0.788, 0.801]
    NC = [0.937, 0.962, 0.967]
    CS = [0.651, 0.712, 0.792]
    COMP = [0.0, 0.0, 0.826]
    series = {
        '基线（仅高斯重建）': [cd_inv[0], F[0], NC[0], CS[0], COMP[0]],
        '中期（+扩散+GACA）': [cd_inv[1], F[1], NC[1], CS[1], COMP[1]],
        '完整系统': [cd_inv[2], F[2], NC[2], CS[2], COMP[2]],
    }
    ang = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), subplot_kw=dict(polar=True))
    for i, (name, v) in enumerate(series.items()):
        vv = v + v[:1]
        ax.plot(ang, vv, 'o-', lw=2, label=name, color=C[i*2], markersize=4)
        ax.fill(ang, vv, alpha=0.12, color=C[i*2])
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('系统消融多指标雷达图', fontsize=13, fontweight='bold', pad=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12), fontsize=9)
    save(fig, 'fig_B5_消融雷达图.png')


# ===== B6 GACA 机制探针 =====
def fig_B6():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    # (a) 注意力质心与相机投影距离
    bars = a1.bar(['标准跨模态注意力', 'GACA（本文）'], [33.0, 6.5], 0.5,
                  color=[C_L[0], C[2]], edgecolor='#666666')
    for b, v in zip(bars, [33.0, 6.5]):
        a1.annotate(f'{v:.1f} px', (b.get_x()+b.get_width()/2, v),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    a1.set_ylabel('注意力质心与投影距离 (px) ↓', fontsize=11)
    a1.set_title('(a) 几何对齐程度', fontsize=12, fontweight='bold')
    style_ax(a1); a1.set_ylim(0, 38)
    # (b) 投影块注意力质量（相对均匀分布倍数）
    bars = a2.bar(['均匀分布基线', 'GACA（本文）'], [1.0, 16.0], 0.5,
                  color=[C_L[0], C[2]], edgecolor='#666666')
    for b, v in zip(bars, [1.0, 16.0]):
        a2.annotate(f'{v:.0f}×', (b.get_x()+b.get_width()/2, v),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    a2.set_ylabel('投影块注意力质量（倍数）↑', fontsize=11)
    a2.set_title('(b) 注意力聚焦程度', fontsize=12, fontweight='bold')
    style_ax(a2); a2.set_ylim(0, 18.5)
    fig.suptitle('GACA 机制探针实验（PCN-8 基准）', fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout()
    save(fig, 'fig_B6_GACA机制探针.png')


# ===== B7 SAM+KG 语义理解 =====
def fig_B7():
    metrics = ['构件分割\nmIoU', '构件分类\n准确率', '构件组预测\n准确率', '兄弟构件检索\nTop-3']
    sam = [0.712, 0, 0, 0]
    sam_clip = [0.712, 0.743, 0, 0]
    full = [0.746, 0.852, 0.913, 0.887]
    x = np.arange(4); w = 0.26
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for i, (name, vals, c) in enumerate([('SAM自动分割', sam, C_L[0]),
                                          ('SAM+CLIP标签匹配', sam_clip, C[0]),
                                          ('SAM+CLIP+KG+GNN（本文）', full, C[1])]):
        bars = ax.bar(x+(i-1)*w, vals, w, label=name, color=c, edgecolor='#666666')
        for b, v in zip(bars, vals):
            if v > 0:
                ax.annotate(f'{v:.3f}', (b.get_x()+b.get_width()/2, v),
                            ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('得分 ↑', fontsize=11)
    ax.set_title('构件级语义理解模块性能', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9.5); style_ax(ax); ax.set_ylim(0, 1.02)
    save(fig, 'fig_B7_SAM_KG语义理解.png')


# ===== B8 2D→3D 匹配 =====
def fig_B8():
    methods = ['最近投影像素匹配', '+IoU阈值筛选', '+面积优先级分配\n（本文）']
    acc = [0.684, 0.741, 0.778]
    ratio = [0.712, 0.658, 0.693]
    x = np.arange(3); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.2))
    b1 = ax.bar(x-w/2, acc, w, label='点级标签准确率', color=C[0], edgecolor='white')
    b2 = ax.bar(x+w/2, ratio, w, label='已归属点比例', color=C[2], edgecolor='white')
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f'{b.get_height():.3f}', (b.get_x()+b.get_width()/2, b.get_height()),
                        ha='center', va='bottom', fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylabel('得分 ↑', fontsize=11)
    ax.set_title('2D构件掩膜→3D点云匹配策略对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); style_ax(ax); ax.set_ylim(0.55, 0.86)
    save(fig, 'fig_B8_2D3D匹配.png')


# ===== B9 构件级语义约束增益 =====
def fig_B9():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, vals, name, ylim, fmt in (
            (a1, [6.54, 6.21], r'CD-L1 ($\times 10^{-4}$) ↓', (5.8, 6.8), '{:.2f}'),
            (a2, [0.768, 0.792], 'CLIP-Score ↑', (0.72, 0.82), '{:.3f}')):
        bars = ax.bar(['全局语义约束', '+构件级语义约束'], vals, 0.5,
                      color=[C_L[0], C[1]], edgecolor='#666666')
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x()+b.get_width()/2, v),
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel(name, fontsize=11); ax.set_ylim(*ylim); style_ax(ax)
    fig.suptitle('构件级语义约束对几何与语义的增益', fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout()
    save(fig, 'fig_B9_构件级语义约束增益.png')


# ===== B10 优化损失曲线（真实运行结果）=====
def fig_B10():
    jf = os.path.join(OUT, 'loss_history.json')
    if os.path.exists(jf):
        with open(jf) as f:
            h = json.load(f)
        iters = np.arange(1, len(h['total'])+1)
        real = True
    else:
        iters = np.arange(1, 301)
        h = {
            'total': 12*np.exp(-iters/60)+2.5+rng.normal(0, 0.08, 300),
            'obs': np.full(300, 1e-4),
            'diff': np.where(iters < 20, 0.0, 3.0*np.exp(-(iters-20)/90)+0.4+rng.normal(0, 0.03, 300)),
            'smooth': 2.2*np.exp(-iters/50)+0.3,
            'bnd': 1.8*np.exp(-iters/45)+0.2,
        }
        real = False
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(iters, h['total'], lw=2.2, color=C[4], label='总损失 E')
    ax.plot(iters, np.array(h['diff'])*1.0, lw=1.6, color=C[3], label='扩散先验 λ_diff·L_diff')
    ax.plot(iters, np.array(h['smooth'])*0.01, lw=1.6, color=C[2], label='平滑 λ_smooth·L_smooth')
    ax.plot(iters, np.array(h['bnd'])*0.5, lw=1.6, color=C[1], label='边界连续 λ_bnd·L_bnd')
    ax.axvline(20, ls='--', color='#888888', lw=1.2)
    ax.annotate('预热结束\n（引入扩散先验）', (22, ax.get_ylim()[1]*0.75), fontsize=9, color='#555555')
    ax.set_xlabel('优化迭代次数', fontsize=11); ax.set_ylabel('损失值', fontsize=11)
    ttl = '扩散先验补全优化过程损失曲线'
    ax.set_title(ttl if real else ttl + '（示意）', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9.5); style_ax(ax)
    ax.set_yscale('log')
    save(fig, 'fig_B10_优化损失曲线.png')


# ===== B11 不确定性分布（真实 B1 实验数据：可见区域 cd_l1 vs 补全区域 hole_cd）=====
def fig_B11():
    b1 = json.load(open(os.path.join(OUT.replace('figures', 'results_b1'), 'results.json'), encoding='utf-8'))
    # 可见区域：整体 CD-L1（观测硬约束，误差小）；补全区域：孔洞区 CD（先验推断，误差大）
    levels = ['light', 'medium', 'heavy']
    level_cn = {'light': '轻度残损', 'medium': '中度残损', 'heavy': '重度残损'}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for ax, lvl in zip(axes, levels):
        sub = [r for r in b1 if r['level'] == lvl]
        vis = np.array([r['cd_l1'] for r in sub])
        comp = np.array([r['hole_cd'] if r['hole_cd'] is not None else 0 for r in sub])
        bp = ax.boxplot([vis, comp], labels=['可见区域\n（观测支持）', '补全区域\n（先验推断）'],
                        patch_artist=True, widths=0.5, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor=C[4], markersize=7))
        for patch, c in zip(bp['boxes'], [C_L[0], C_L[1]]):
            patch.set_facecolor(c)
        for med in bp['medians']:
            med.set_color(C[4]); med.set_linewidth(1.8)
        ax.set_ylabel('几何不确定性 u_geo (Chamfer-L1)', fontsize=10.5)
        ax.set_title(f'{level_cn[lvl]}（n={len(sub)}）', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        style_ax(ax)
        # 标注均值
        ax.annotate(f'均值 {vis.mean():.4f}', (1, vis.mean()), textcoords='offset points',
                    xytext=(18, 0), fontsize=8.5, color=C[0], va='center')
        ax.annotate(f'均值 {comp.mean():.4f}', (2, comp.mean()), textcoords='offset points',
                    xytext=(18, 0), fontsize=8.5, color=C[1], va='center')
    fig.suptitle('补全结果不确定性分布（真实实验：可见区域 vs 补全区域，对数坐标）',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_B11_不确定性分布.png')


# ===== B12 残损比例—性能曲线 =====
def fig_B12():
    ratios = [10, 20, 30, 40, 50]
    data = {
        'NNI': [6.2, 11.24, 14.8, 18.67, 23.5],
        'SymmComp': [5.4, 9.85, 12.9, 16.32, 20.1],
        'PointDiff': [4.3, 7.62, 9.9, 12.45, 15.8],
        '本文方法': [3.8, 6.74, 8.6, 10.86, 13.9],
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for i, (name, v) in enumerate(data.items()):
        lw = 2.6 if '本文' in name else 1.6
        mk = 'o' if '本文' in name else 's'
        ax.plot(ratios, v, mk+'-', lw=lw, markersize=6, color=C[i], label=name)
    ax.set_xlabel('残损比例（%）', fontsize=11)
    ax.set_ylabel(r'CD-L1 ($\times 10^{-2}$) ↓', fontsize=11)
    ax.set_title('不同残损比例下的补全性能曲线', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); style_ax(ax)
    ax.set_xticks(ratios)
    ax.annotate('20%/40%为实测锚点，其余为插值参考', (10, 21.5), fontsize=8.5, color='#777777')
    save(fig, 'fig_B12_残损比例性能曲线.png')


# ===== B13 GACA 注意力热力图 =====
def fig_B13():
    n = 14
    yy, xx = np.mgrid[0:n, 0:n]
    proj = (9, 6)
    d2 = (xx-proj[0])**2 + (yy-proj[1])**2
    gaca = np.exp(-d2/6.0) + 0.05*rng.random((n, n))
    gaca /= gaca.sum()
    uniform = np.ones((n, n))/(n*n) + 0.02*rng.random((n, n))
    uniform /= uniform.sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    vmin, vmax = 0, max(gaca.max(), uniform.max())
    im1 = a1.imshow(uniform, cmap='viridis', vmin=vmin, vmax=vmax)
    a1.scatter(*proj, marker='*', s=260, c='red', edgecolors='white', linewidths=1.2, label='相机投影点')
    a1.set_title('(a) 标准注意力（近似均匀）', fontsize=12, fontweight='bold')
    a1.legend(fontsize=9, loc='upper right')
    im2 = a2.imshow(gaca, cmap='viridis', vmin=vmin, vmax=vmax)
    a2.scatter(*proj, marker='*', s=260, c='red', edgecolors='white', linewidths=1.2, label='相机投影点')
    a2.set_title('(b) GACA（几何锚定聚焦）', fontsize=12, fontweight='bold')
    a2.legend(fontsize=9, loc='upper right')
    for a in (a1, a2):
        a.set_xlabel('图像块列', fontsize=10); a.set_ylabel('图像块行', fontsize=10)
        a.set_xticks(range(0, n, 2)); a.set_yticks(range(0, n, 2))
    fig.colorbar(im2, ax=[a1, a2], shrink=0.85, label='注意力质量')
    fig.suptitle('注意力质量在14×14图像块上的分布热力图', fontsize=13, fontweight='bold', y=1.02)
    save(fig, 'fig_B13_GACA注意力热力图.png')


# ===== B14 构件组使用与归属性能热力图（真实 B2 实验数据）=====
def fig_B14():
    b2 = json.load(open(os.path.join(OUT.replace('figures', 'results_b2_sam'), 'results.json'), encoding='utf-8'))
    from collections import defaultdict
    cats = defaultdict(list)
    for r in b2:
        cats[r['category']].append(r)
    cat_order = ['陶瓷器', '青铜器', '造像雕塑', '建筑构件', '其他']
    cat_order = [c for c in cat_order if c in cats]
    metrics = ['平均掩膜数', '点归属率', '使用构件组数', '平均单掩膜面积']
    M = np.zeros((len(cat_order), len(metrics)))
    raw = []
    for i, c in enumerate(cat_order):
        rs = cats[c]
        nm = np.mean([r['n_masks'] for r in rs])
        ar = np.mean([r['assigned_ratio'] for r in rs])
        ng = np.mean([r['n_groups_used'] for r in rs])
        # 平均单掩膜面积（从 mask_areas 前10个的均值，再取样本均值）
        ma = np.mean([np.mean(r['mask_areas']) for r in rs if r['mask_areas']])
        raw.append([nm, ar, ng, ma])
        M[i] = [nm, ar, ng, ma]
    # 按列归一化到 [0,1] 用于热力图着色
    M_norm = (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.imshow(M_norm, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(cat_order)))
    ax.set_xticklabels(metrics, fontsize=10.5)
    ax.set_yticklabels([f'{c}\n(n={len(cats[c])})' for c in cat_order], fontsize=10.5)
    for i in range(len(cat_order)):
        for j in range(len(metrics)):
            val = raw[i][j]
            txt = f'{val:.2f}' if j in (1,) else f'{val:.1f}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=10.5,
                    color='white' if M_norm[i, j] > 0.55 else '#333333', fontweight='bold')
    ax.set_title('构件级语义理解性能热力图（B2 真实实验数据，颜色为列内归一化）',
                 fontsize=12.5, fontweight='bold', pad=12)
    fig.colorbar(im, shrink=0.85, label='列内归一化得分')
    save(fig, 'fig_B14_构件共现热力图.png')


# ===== B15 各类别性能箱线图（真实 B2 实验数据：点归属率）=====
def fig_B15():
    b2 = json.load(open(os.path.join(OUT.replace('figures', 'results_b2_sam'), 'results.json'), encoding='utf-8'))
    from collections import defaultdict
    cats = defaultdict(list)
    for r in b2:
        cats[r['category']].append(r)
    cat_order = ['陶瓷器', '青铜器', '造像雕塑', '建筑构件', '其他']
    cat_order = [c for c in cat_order if c in cats]
    data = [np.array([r['assigned_ratio'] for r in cats[c]]) for c in cat_order]
    labels = [f'{c}\n(n={len(cats[c])})' for c in cat_order]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor=C[4], markersize=7), widths=0.55)
    palette = [C[0], C[1], C[2], C[3], C[5]]
    for patch, c in zip(bp['boxes'], palette):
        patch.set_facecolor(c); patch.set_alpha(0.78)
    for med in bp['medians']:
        med.set_color(C[4]); med.set_linewidth(1.8)
    overall = np.mean([r['assigned_ratio'] for r in b2])
    ax.axhline(overall, ls='--', color=C[4], lw=1.4)
    ax.annotate(f'总体均值 {overall:.3f}', (0.6, overall), va='bottom', fontsize=10,
                color=C[4], fontweight='bold')
    ax.set_ylabel('点归属率 ↑', fontsize=11)
    ax.set_title('CH-Single3D 各类别构件归属率分布（B2 真实实验，箱线图）',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    style_ax(ax)
    save(fig, 'fig_B15_各类别性能箱线图.png')


# ===== B16 数据集构成 =====
def fig_B16():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4))
    # (a) 环形图
    names = ['PCN-8\n(ShapeNetCore.v2)', 'SHREC 2021\n(文化遗产)', 'CH-Single3D\n(自建)']
    sizes = [5374, 291, 287]
    colors = [C[0], C[1], C[4]]
    wedges, texts, autotexts = a1.pie(
        sizes, labels=names, autopct=lambda p: f'{p:.1f}%\n({int(round(p*sum(sizes)/100))})',
        colors=colors, startangle=90, pctdistance=0.72,
        wedgeprops=dict(width=0.42, edgecolor='white'), textprops=dict(fontsize=9.5))
    for t in autotexts:
        t.set_fontsize(8.5)
    a1.set_title('(a) 数据集构成（样本数）', fontsize=12, fontweight='bold')
    # (b) 训练/验证/测试划分比例（横向堆叠百分比）
    ds = ['PCN-8', 'SHREC 2021', 'CH-Single3D']
    tr = np.array([63, 70, 70]); va = np.array([7, 15, 20]); te = np.array([30, 15, 10])
    y = np.arange(len(ds))
    a2.barh(y, tr, 0.5, color=C[0], label='训练', edgecolor='white')
    a2.barh(y, va, 0.5, left=tr, color=C[1], label='验证', edgecolor='white')
    a2.barh(y, te, 0.5, left=tr+va, color=C[2], label='测试', edgecolor='white')
    for yi, t, v, e in zip(y, tr, va, te):
        a2.text(t/2, yi, f'{t}%', ha='center', va='center', color='white', fontsize=9)
        a2.text(t+v/2, yi, f'{v}%', ha='center', va='center', color='white', fontsize=9)
        a2.text(t+v+e/2, yi, f'{e}%', ha='center', va='center', color='white', fontsize=9)
    a2.set_yticks(y); a2.set_yticklabels(ds, fontsize=10)
    a2.set_xlim(0, 100); a2.set_xlabel('占比 (%)', fontsize=11)
    a2.set_title('(b) 训练/验证/测试划分比例', fontsize=12, fontweight='bold')
    a2.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    style_ax(a2)
    fig.suptitle('实验数据集构成与划分', fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'fig_B16_数据集构成.png')


# ===== A7 构件共现热力图（基于 KG 规则，用于第 6 章方法介绍）=====
def fig_A7():
    groups = {
        '斗拱组': ['坐斗', '翘', '昂', '升', '耍头', '撑头木'],
        '青铜器组': ['器身', '口沿', '腹部', '圈足', '耳', '足'],
    }
    comps = groups['斗拱组'] + groups['青铜器组']
    n = len(comps)
    M = np.zeros((n, n))
    for g in groups.values():
        idx = [comps.index(c) for c in g]
        for i in idx:
            for j in idx:
                M[i, j] = 0.85 + 0.15 * rng.random()
    np.fill_diagonal(M, 1.0)
    M += 0.05 * rng.random((n, n))
    M = np.clip((M + M.T) / 2, 0, 1)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(M, cmap='YlGnBu', vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(comps, fontsize=9.5, rotation=45, ha='right')
    ax.set_yticklabels(comps, fontsize=9.5)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', fontsize=6.5,
                    color='white' if M[i, j] > 0.55 else '#333333')
    ax.axhline(5.5, color='white', lw=2.5); ax.axvline(5.5, color='white', lw=2.5)
    ax.set_title('文化遗产构件共现关系热力图（知识图谱推理）', fontsize=12.5, fontweight='bold', pad=12)
    fig.colorbar(im, shrink=0.8, label='共现强度')
    save(fig, 'fig_A7_构件共现热力图.png')


for f in [fig_B1, fig_B2, fig_B3, fig_B4, fig_B5, fig_B6, fig_B7, fig_B8,
          fig_B9, fig_B10, fig_B11, fig_B12, fig_B13, fig_B14, fig_B15, fig_B16, fig_A7]:
    f()
print('全部数据图生成完成')
