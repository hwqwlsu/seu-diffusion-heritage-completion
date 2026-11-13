# -*- coding: utf-8 -*-
"""六类别 × 四损伤档定性矩阵图（紧凑版）。

布局：6 行 × 4 列
  行 = 类别（Bronze/Ceramic/Grotto/Jade/Mural/Structure）
  列 = 损伤档（10%/20%/40%/50%）
  每个单元格 = 一张完整链图（GT → Damaged → Restored → 3D points）
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

ROOT = Path(r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01")
SRC = ROOT / "figures_selected"
OUT = ROOT / "fig_matrix_6cat_4lv.png"

CASES = [
    ("Bronze vessel",          "fig_bronze_3a6e31a"),
    ("Ceramic ware",           "fig_ceramic_033c94"),
    ("Grotto statue",          "fig_grotto_0c90ba77"),
    ("Jade",                   "fig_jade_1934240d"),
    ("Mural",                  "fig_mural_6bac7ea1"),
    ("Architectural component","fig_structure_af12c8a"),
]
LEVELS = [10, 20, 40, 50]
ROW_LABEL_W = 230
COL_LABEL_H = 88
GAP_W = 18
GAP_H = 26
TARGET_CELL_W = 1320  # 每列链图压到这个宽度
TITLE_H = 110
PAD = 40

def crop_chain(im):
    """裁掉顶部标题带和底部色条。"""
    w, h = im.size
    arr = np.asarray(im.convert("L"))
    ink = (arr < 245).mean(axis=1)
    # 顶部：第一个非白像素带底部 + gap 结束
    idx = np.where(ink > 0.02)[0]
    if len(idx) == 0:
        top = 120
    else:
        top = idx[0]
        # 跳过标题带（通常带深底色）
        for i in range(top, min(top + 180, h)):
            if ink[i] < 0.005:
                top = i + 4
                break
    # 底部
    idx2 = np.where(ink > 0.02)[0]
    bottom = h - 8
    if len(idx2):
        bottom = idx2[-1]
        # 如果尾部有浅色条，往上裁
        for i in range(bottom, max(bottom - 120, top), -1):
            if ink[i] < 0.005:
                bottom = i - 4
                break
    # 宽度：链图左侧有标题区要去掉
    col_ink = (arr < 245).mean(axis=0)
    ci = np.where(col_ink > 0.01)[0]
    left = ci[0] if len(ci) else 0
    right = ci[-1] + 6 if len(ci) else w
    return im.crop((left, top, right, bottom))


# 加载所有 24 张链图（缺失的跳过）
cells = {}  # (row_idx, col_idx) -> PIL.Image
for ri, (name, prefix) in enumerate(CASES):
    for ci, lv in enumerate(LEVELS):
        p = SRC / f"{prefix}_{lv}.png"
        if p.exists():
            im = Image.open(p).convert("RGB")
            cells[(ri, ci)] = crop_chain(im)
        else:
            cells[(ri, ci)] = None

# 统一缩放到 TARGET_CELL_W
scaled = {}
for (ri, ci), im in cells.items():
    if im is not None:
        w, h = im.size
        new_h = int(h * TARGET_CELL_W / w)
        scaled[(ri, ci)] = im.resize((TARGET_CELL_W, new_h), Image.LANCZOS)

# 每行用该行所有链图最高的那个高度
row_heights = []
for ri in range(len(CASES)):
    hs = [scaled[(ri, ci)].size[1] for ci in range(len(LEVELS)) if scaled.get((ri, ci)) is not None]
    row_heights.append(max(hs) if hs else 800)

# 画布尺寸
total_w = PAD * 2 + ROW_LABEL_W + TARGET_CELL_W * len(LEVELS) + GAP_W * (len(LEVELS) - 1)
total_h = PAD + TITLE_H + COL_LABEL_H + sum(row_heights) + GAP_H * (len(CASES) - 1) + PAD
print(f"画布: {total_w} × {total_h}")

canvas = Image.new("RGB", (total_w, total_h), "white")
draw = ImageDraw.Draw(canvas)

try:
    ft_title = ImageFont.truetype("arial.ttf", 64)
    ft_col = ImageFont.truetype("arial.ttf", 54)
    ft_row = ImageFont.truetype("arial.ttf", 48)
except IOError:
    ft_title = ft_col = ft_row = ImageFont.load_default()

# 主标题
title = "Qualitative Completion: Six Heritage Categories × Four Damage Levels"
bb = draw.textbbox((0, 0), title, font=ft_title)
tw = bb[2] - bb[0]
draw.text(((total_w - tw) // 2, 28), title, fill="#1A1A1A", font=ft_title)

# 损伤档列标题
col_x0 = PAD + ROW_LABEL_W
for ci, lv in enumerate(LEVELS):
    cx = col_x0 + ci * (TARGET_CELL_W + GAP_W) + TARGET_CELL_W // 2
    text = f"{lv}% damage"
    bb = draw.textbbox((0, 0), text, font=ft_col)
    tw = bb[2] - bb[0]
    draw.text((cx - tw // 2, TITLE_H + 16), text, fill="#3A3A3A", font=ft_col)

# 行标签 + 贴面板
cur_y = PAD + TITLE_H + COL_LABEL_H
for ri, (cat_name, _) in enumerate(CASES):
    rh = row_heights[ri]
    # 行标签（垂直居中）
    bb = draw.textbbox((0, 0), cat_name, font=ft_row)
    tw = bb[2] - bb[0]
    th = bb[3] - bb[1]
    draw.text((PAD + ROW_LABEL_W - tw - 10, cur_y + (rh - th) // 2),
              cat_name, fill="#1A1A1A", font=ft_row)

    # 贴 4 个链图
    cell_y = cur_y
    for ci, lv in enumerate(LEVELS):
        x = col_x0 + ci * (TARGET_CELL_W + GAP_W)
        im = scaled.get((ri, ci))
        if im is not None:
            canvas.paste(im, (x, cell_y))
        else:
            draw.rectangle([x, cell_y, x + TARGET_CELL_W, cell_y + rh],
                           fill="#F5F5F5", outline="#CCCCCC", width=2)
            text = "missing"
            bb = draw.textbbox((0, 0), text, font=ft_row)
            tw = bb[2] - bb[0]; th = bb[3] - bb[1]
            draw.text((x + (TARGET_CELL_W - tw) // 2, cell_y + (rh - th) // 2),
                      text, fill="#AAAAAA", font=ft_row)

    cur_y += rh + GAP_H

canvas.save(OUT, quality=95)
print(f"已保存: {OUT}")
