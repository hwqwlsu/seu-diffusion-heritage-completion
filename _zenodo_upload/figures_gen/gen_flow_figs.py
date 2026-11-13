# -*- coding: utf-8 -*-
"""
生成论文流程/架构图 6 张（>=800 DPI）
A1 总体技术路线  A2 几何先验提取  A3 扩散先验补全
A4 GACA注意力机制  A5 SAM+KG语义理解  A6 语义-几何闭环
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np, os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\figures'
os.makedirs(OUT, exist_ok=True)
DPI = 800

# 学术配色
C_MAIN = '#2C5F8A'; C_MAIN_L = '#C9DCEC'
C_ORG  = '#E07B39'; C_ORG_L  = '#F7DFC9'
C_GRN  = '#4C9A6C'; C_GRN_L  = '#CBE6D6'
C_PUR  = '#7A5C9E'; C_PUR_L  = '#DED3EA'
C_GRY  = '#5A5A5A'; C_GRY_L  = '#E4E4E4'
C_RED  = '#B85450'; C_RED_L  = '#F0D2D0'


def box(ax, cx, cy, w, h, text, fc, ec=None, fs=11, tc='black', lw=1.4, bold=False, r=0.08):
    ec = ec or fc
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, zorder=3, fontweight='bold' if bold else 'normal',
            linespacing=1.35)
    return (cx, cy, w, h)


def arrow(ax, p1, p2, color=C_GRY, lw=2.0, style='-|>', rad=0.0, ls='-'):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                        color=color, lw=lw, zorder=1,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)


def right(b): return (b[0]+b[2]/2, b[1])
def left(b):  return (b[0]-b[2]/2, b[1])
def top(b):   return (b[0], b[1]+b[3]/2)
def bottom(b):return (b[0], b[1]-b[3]/2)


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=DPI, bbox_inches='tight',
                facecolor='white', pad_inches=0.15)
    plt.close(fig)
    print('saved', name)


# ============ A1 总体技术路线图 ============
def fig_A1():
    fig, ax = new_ax(15, 5.6)
    ax.text(50, 95, '基于多模态语义约束的单视图文化遗产三维复原与理解——总体技术路线',
            ha='center', fontsize=15, fontweight='bold', color=C_MAIN)

    stages = [
        ('第三章\n几何先验协同约束的\n三维高斯初始化', C_MAIN, C_MAIN_L),
        ('第四章\n扩散先验引导的\n残损区域几何补全', C_ORG, C_ORG_L),
        ('第五章\n几何锚定跨模态\n注意力点云补全', C_GRN, C_GRN_L),
        ('第六章\nSAM构件分割与\n知识图谱语义推理', C_PUR, C_PUR_L),
        ('闭环优化\n语义—几何联合优化\n+不确定性量化', C_RED, C_RED_L),
    ]
    xs = [11, 30.5, 50, 69.5, 89]
    boxes = []
    for x, (txt, ec, fc) in zip(xs, stages):
        boxes.append(box(ax, x, 55, 16, 26, txt, fc, ec, fs=10.5, bold=True))
    for i in range(4):
        arrow(ax, right(boxes[i]), left(boxes[i+1]), lw=2.4)

    box(ax, 11, 15, 16, 12, '输入\n单张RGB图像\n+文本描述', C_GRY_L, C_GRY, fs=10)
    box(ax, 50, 15, 30, 12, '多模态语义知识库\n（1287条文物知识条目）', C_GRY_L, C_GRY, fs=10)
    box(ax, 89, 15, 18, 12, '输出\n完整三维模型\n构件级语义+不确定性', C_GRY_L, C_GRY, fs=9.5)
    arrow(ax, (11, 21), (11, 41), lw=1.8)
    arrow(ax, (50, 21), (50, 38), lw=1.8, ls='--')
    arrow(ax, (89, 41), (89, 21), lw=1.8)
    save(fig, 'fig_A1_总体技术路线图.png')


# ============ A2 第三章 几何先验提取与双分支重建 ============
def fig_A2():
    fig, ax = new_ax(13, 7)
    ax.text(50, 96, '第三章  多类几何先验协同约束的三维复原流程', ha='center',
            fontsize=14, fontweight='bold', color=C_MAIN)

    b_in = box(ax, 10, 62, 14, 18, '单视图\nRGB图像', C_GRY_L, C_GRY, fs=11, bold=True)

    # 三类先验
    b_d = box(ax, 30, 80, 17, 12, '单目深度\nDepth Anything', C_MAIN_L, C_MAIN, fs=9.5)
    b_n = box(ax, 30, 62, 17, 12, '表面法线\n估计网络', C_MAIN_L, C_MAIN, fs=9.5)
    b_e = box(ax, 30, 44, 17, 12, '轮廓边界\n提取', C_MAIN_L, C_MAIN, fs=9.5)
    for b in (b_d, b_n, b_e):
        arrow(ax, right(b_in), left(b), lw=1.8, rad=0.05)

    b_joint = box(ax, 52, 62, 17, 16, '深度—法线\n联合优化\n（边缘保持）', C_ORG_L, C_ORG, fs=10, bold=True)
    for b in (b_d, b_n, b_e):
        arrow(ax, right(b), left(b_joint), lw=1.8)

    # 双分支
    b_p = box(ax, 74, 72, 18, 13, '点云分支\n几何特征提取', C_GRN_L, C_GRN, fs=10)
    b_m = box(ax, 74, 52, 18, 13, '网格分支\n拓扑特征提取', C_GRN_L, C_GRN, fs=10)
    arrow(ax, right(b_joint), left(b_p), lw=1.8, rad=-0.05)
    arrow(ax, right(b_joint), left(b_m), lw=1.8, rad=0.05)

    b_fuse = box(ax, 74, 30, 22, 12, '多分量联合损失\nChamfer+平滑+法线一致', C_ORG_L, C_ORG, fs=9.5)
    arrow(ax, bottom(b_p), top(b_fuse), lw=1.8)
    arrow(ax, bottom(b_m), top(b_fuse), lw=1.8)

    b_g = box(ax, 47, 10, 26, 12, '三维高斯初始化\n（密度自适应细分）', C_PUR_L, C_PUR, fs=10, bold=True)
    arrow(ax, left(b_fuse), right(b_g), lw=1.8)
    b_out = box(ax, 84, 10, 24, 12, '初始三维点云\n+高斯参数', C_GRY_L, C_GRY, fs=10, bold=True)
    arrow(ax, right(b_g), left(b_out), lw=1.8)

    ax.text(52, 92.5, '几何先验提取', fontsize=10, color=C_MAIN, ha='center')
    ax.text(74, 88, '双分支重建架构', fontsize=10, color=C_GRN, ha='center')
    save(fig, 'fig_A2_第三章_几何先验提取与双分支重建.png')


# ============ A3 第四章 扩散先验补全 ============
def fig_A3():
    fig, ax = new_ax(13.5, 7)
    ax.text(50, 96, '第四章  扩散先验引导的残损区域几何补全流程', ha='center',
            fontsize=14, fontweight='bold', color=C_ORG)

    b_in1 = box(ax, 11, 72, 15, 12, '部分点云\nP_obs', C_GRY_L, C_GRY, fs=10, bold=True)
    b_in2 = box(ax, 11, 52, 15, 12, '深度图 D\n相机K,T', C_GRY_L, C_GRY, fs=10, bold=True)

    b_det = box(ax, 32, 62, 19, 22, '残损区域检测\n· 可见性判定\n· 边界点提取\n· 缺失区域初始化', C_ORG_L, C_ORG, fs=9.5, bold=True)
    arrow(ax, right(b_in1), left(b_det), lw=1.8)
    arrow(ax, right(b_in2), left(b_det), lw=1.8)

    # 优化循环
    b_opt = box(ax, 62, 62, 22, 30,
                '能量最小化优化\n（仅优化缺失点）\n\nE = λ_obs·观测约束\n  + λ_diff·扩散先验\n  + λ_smooth·平滑\n  + λ_bnd·边界连续',
                C_MAIN_L, C_MAIN, fs=9.5, bold=True)
    arrow(ax, right(b_det), left(b_opt), lw=2.0)

    b_prior = box(ax, 62, 22, 24, 14, '预训练点云扩散模型\nscore function\ns_θ(x,t) ≈ -ε/√(1-ᾱ)', C_PUR_L, C_PUR, fs=9.5)
    arrow(ax, top(b_prior), bottom(b_opt), lw=1.8, ls='--')
    ax.text(69, 38, '形状先验正则', fontsize=8.5, color=C_PUR, ha='left')

    b_out = box(ax, 90, 62, 15, 22, '完整点云\n16384点\n+\n不确定性\n量化', C_GRN_L, C_GRN, fs=10, bold=True)
    arrow(ax, right(b_opt), left(b_out), lw=2.0)

    # 可见点冻结说明
    ax.text(62, 84.5, '可见点冻结 · 缺失点可优化', fontsize=9, color=C_RED,
            ha='center', fontweight='bold')
    save(fig, 'fig_A3_第四章_扩散先验补全流程.png')


# ============ A4 第五章 GACA 注意力机制 ============
def fig_A4():
    fig, ax = new_ax(13.5, 6.2)
    ax.text(50, 95, '第五章  几何锚定跨模态注意力（GACA）机制', ha='center',
            fontsize=14, fontweight='bold', color=C_GRN)

    b_q = box(ax, 11, 55, 15, 16, '点查询\nQ（三维点）', C_GRY_L, C_GRY, fs=10, bold=True)

    b_proj = box(ax, 33, 74, 20, 14, '相机投影\nπ(K,T,p)→(u,v)', C_MAIN_L, C_MAIN, fs=10)
    b_bias = box(ax, 33, 40, 22, 16, '可学习高斯偏置\nB∝exp(-d²/2σ²)\n（视锥外自动消隐）', C_ORG_L, C_ORG, fs=9.5)
    arrow(ax, right(b_q), left(b_proj), lw=1.8, rad=-0.08)
    arrow(ax, right(b_q), left(b_bias), lw=1.8, rad=0.08)

    b_img = box(ax, 57, 74, 20, 14, '图像块特征\nK/V（ResNet+FPN）', C_MAIN_L, C_MAIN, fs=10)
    arrow(ax, right(b_proj), left(b_img), lw=1.8)

    b_att = box(ax, 57, 40, 20, 14, '注意力logits\nQK^T/√d + B', C_ORG_L, C_ORG, fs=10, bold=True)
    arrow(ax, right(b_bias), left(b_att), lw=1.8)
    arrow(ax, bottom(b_img), top(b_att), lw=1.8)

    b_out = box(ax, 84, 55, 22, 18, '几何锚定\n跨模态融合特征\n（外观可覆盖先验）', C_GRN_L, C_GRN, fs=10, bold=True)
    arrow(ax, right(b_att), left(b_out), lw=2.0, rad=0.06)

    # 效果标注
    box(ax, 50, 12, 66, 12,
        '机制探针：注意力质心与投影距离 33.0px → 6.5px，投影块注意力质量提升至 16×',
        C_RED_L, C_RED, fs=10)
    save(fig, 'fig_A4_第五章_GACA注意力机制.png')


# ============ A5 第六章 SAM+KG 语义理解 ============
def fig_A5():
    fig, ax = new_ax(14, 7)
    ax.text(50, 96, '第六章  SAM构件分割与文化遗产知识图谱推理流程', ha='center',
            fontsize=14, fontweight='bold', color=C_PUR)

    b_img = box(ax, 9, 62, 13, 14, '输入\nRGB图像', C_GRY_L, C_GRY, fs=10, bold=True)
    b_sam = box(ax, 28, 62, 17, 16, 'SAM自动分割\n构件级掩膜\n（≤30个）', C_MAIN_L, C_MAIN, fs=9.5, bold=True)
    arrow(ax, right(b_img), left(b_sam), lw=1.8)

    b_clip = box(ax, 48, 74, 18, 12, 'CLIP标签匹配\n构件名称词表', C_ORG_L, C_ORG, fs=9.5)
    arrow(ax, right(b_sam), left(b_clip), lw=1.8, rad=-0.06)

    # KG + GNN
    b_kg = box(ax, 48, 44, 20, 18, '文化遗产知识图谱\n6类实体·6类关系\n（斗拱=坐斗+翘+昂+升）', C_PUR_L, C_PUR, fs=9)
    b_gnn = box(ax, 72, 44, 18, 16, 'GraphSAGE\n构件关系推理\n组预测/兄弟检索', C_PUR_L, C_PUR, fs=9.5)
    arrow(ax, right(b_kg), left(b_gnn), lw=1.8)

    b_fix = box(ax, 72, 74, 18, 12, '标签修正\n（消歧+补全）', C_ORG_L, C_ORG, fs=9.5)
    arrow(ax, right(b_clip), left(b_fix), lw=1.8)
    arrow(ax, top(b_gnn), bottom(b_fix), lw=1.8, ls='--')

    b_match = box(ax, 91, 58, 15, 20, '2D→3D\n投影匹配\n（面积优先\n级分配）', C_GRN_L, C_GRN, fs=9.5, bold=True)
    arrow(ax, right(b_fix), left(b_match), lw=1.8, rad=0.05)

    b_out = box(ax, 91, 24, 17, 14, '构件级语义\n标注点云', C_GRN_L, C_GRN, fs=10, bold=True)
    arrow(ax, bottom(b_match), top(b_out), lw=1.8)

    b_pts = box(ax, 28, 24, 18, 12, '完整三维点云\n（第四章输出）', C_GRY_L, C_GRY, fs=9.5)
    arrow(ax, right(b_pts), left(b_out), lw=1.8, ls='--')
    save(fig, 'fig_A5_第六章_SAM_KG语义理解流程.png')


# ============ A6 语义—几何闭环优化 ============
def fig_A6():
    fig, ax = new_ax(9.5, 7.5)
    ax.text(50, 96, '语义—几何闭环优化机制', ha='center',
            fontsize=14, fontweight='bold', color=C_RED)

    cx, cy, R = 50, 50, 30
    nodes = [
        ('渲染\n（高斯泼溅）', 90, C_MAIN, C_MAIN_L),
        ('语义对齐\nCLIP三级对齐', 18, C_ORG, C_ORG_L),
        ('差异评估\n语义损失', -54, C_PUR, C_PUR_L),
        ('参数更新\n几何+外观', -126, C_GRN, C_GRN_L),
        ('重新渲染', 162, C_MAIN, C_MAIN_L),
    ]
    centers = []
    for txt, ang, ec, fc in nodes:
        a = np.deg2rad(ang)
        x, y = cx+R*np.cos(a), cy+R*np.sin(a)*0.78
        centers.append((x, y))
        box(ax, x, y, 24, 15, txt, fc, ec, fs=10, bold=True)
    for i in range(len(centers)):
        arrow(ax, centers[i], centers[(i+1) % len(centers)], lw=2.2, rad=-0.22)

    box(ax, 50, 50, 26, 13, '闭环迭代\n直至收敛', C_RED_L, C_RED, fs=11, bold=True)

    box(ax, 13, 10, 22, 12, '不确定性量化\n区分观测/推断', C_GRY_L, C_GRY, fs=9)
    box(ax, 87, 10, 22, 12, '构件级语义约束\n（SAM+KG）', C_GRY_L, C_GRY, fs=9)
    arrow(ax, (24, 13), (38, 40), lw=1.5, ls='--', rad=-0.15)
    arrow(ax, (76, 13), (62, 40), lw=1.5, ls='--', rad=0.15)
    save(fig, 'fig_A6_语义几何闭环优化.png')


fig_A1(); fig_A2(); fig_A3(); fig_A4(); fig_A5(); fig_A6()
print('全部流程图生成完成')
