# -*- coding: utf-8 -*-
"""
生成定性可视化图（800 DPI）：
  Q1: B1 扩散补全过程对比（GT / 残损 / 补全，前6样本×3档，选代表性展示）
  Q2: B2 SAM+KG 构件掩膜叠加图（渲染RGB + 构件标签着色）
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

OUT = r'C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\figures'
os.makedirs(OUT, exist_ok=True)
DPI = 800

B1_DIR = r'C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\results_b1'
B2_DIR = r'C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\results_b2'


def fig_Q1_completion():
    """B1 补全过程对比：每样本三列（GT点云 / 残损点云 / 补全点云），选3个样本×3档"""
    levels = ['light', 'medium', 'heavy']
    level_cn = {'light': '轻度', 'medium': '中度', 'heavy': '重度'}
    # 取前3个样本（每个样本有3个level的npz）
    samples = sorted(set(
        os.path.basename(p).split('_')[1]
        for p in glob.glob(os.path.join(B1_DIR, 'sample_*_*.npz'))
    ))[:3]

    n_rows = len(samples)
    n_cols = len(levels) + 1  # GT + 3档补全
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for r, s in enumerate(samples):
        # GT 取该样本任意level的gt
        gt_path = os.path.join(B1_DIR, f'sample_{s}_{levels[0]}.npz')
        d = np.load(gt_path)
        gt = d['gt']
        axes[r, 0].scatter(gt[:, 0], gt[:, 1], c=gt[:, 2], s=0.3, cmap='viridis', rasterized=True)
        axes[r, 0].set_title('GT 完整点云', fontsize=10)
        axes[r, 0].set_aspect('equal')
        axes[r, 0].axis('off')
        for c, lvl in enumerate(levels, 1):
            p = os.path.join(B1_DIR, f'sample_{s}_{lvl}.npz')
            if not os.path.exists(p):
                axes[r, c].axis('off')
                continue
            d = np.load(p)
            comp = d['complete']
            axes[r, c].scatter(comp[:, 0], comp[:, 1], c=comp[:, 2], s=0.3, cmap='viridis', rasterized=True)
            axes[r, c].set_title(f'{level_cn[lvl]} 补全', fontsize=10)
            axes[r, c].set_aspect('equal')
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(f'样本 {int(s)+1}', fontsize=11, rotation=90)

    fig.suptitle('扩散先验残损补全过程对比（真实实验）', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_Q1_补全过程对比.png'), dpi=DPI, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close(fig)
    print('saved fig_Q1_补全过程对比.png')


def fig_Q2_masks():
    """B2 构件掩膜叠加：渲染RGB + 构件标签着色点云，前4个样本"""
    npzs = sorted(glob.glob(os.path.join(B2_DIR, 'sample_*.npz')))[:4]
    n = len(npzs)
    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    cmap = plt.cm.tab20

    for r, p in enumerate(npzs):
        d = np.load(p, allow_pickle=True)
        img = d['image']
        gt = d['gt']
        labels = d['labels']
        # 左：渲染RGB
        axes[r, 0].imshow(img)
        axes[r, 0].set_title('渲染 RGB', fontsize=10)
        axes[r, 0].axis('off')
        # 右：构件标签着色点云
        uniq = np.unique(labels[labels >= 0])
        colors = np.zeros((len(labels), 3))
        for i, lab in enumerate(uniq):
            colors[labels == lab] = cmap(i / max(len(uniq) - 1, 1))[:3]
        colors[labels < 0] = [0.85, 0.85, 0.85]
        axes[r, 1].scatter(gt[:, 0], gt[:, 1], c=colors, s=0.3, rasterized=True)
        axes[r, 1].set_title(f'构件标签着色（{len(uniq)} 组）', fontsize=10)
        axes[r, 1].set_aspect('equal')
        axes[r, 1].axis('off')

    fig.suptitle('SAM+KG 构件分割与 2D→3D 匹配结果（真实实验）', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_Q2_构件掩膜叠加.png'), dpi=DPI, bbox_inches='tight',
                facecolor='white', pad_inches=0.1)
    plt.close(fig)
    print('saved fig_Q2_构件掩膜叠加.png')


if __name__ == '__main__':
    fig_Q1_completion()
    fig_Q2_masks()
    print('定性可视化图生成完成')
