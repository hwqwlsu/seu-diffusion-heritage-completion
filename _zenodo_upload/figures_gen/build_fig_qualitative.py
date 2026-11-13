# -*- coding: utf-8 -*-
"""把已有的单案例定性链图拼成跨类别多行定性总图，不改变任何实验结果。"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01")
OUT = ROOT / "fig_ch5_qualitative_multirow.png"

# (文件名, 面板上边界, 面板下边界)
CASES = [
    ("fig_case0_arch_chain.png", 145, 816),
    ("fig_case1_grotto_chain.png", 130, 712),
    ("fig_case2_grotto2_chain.png", 145, 816),
]
COLUMN_TITLES = ("GT", "Damaged input", "Restored", "3D points")
KEEP_RATIO = 0.655      # 只保留前四列，去掉后两列语义理解结果
TARGET_W = 3600
PAD_LEFT = 120
GAP = 30
HEADER_H = 92


def load_row(name, top, bottom):
    image = Image.open(ROOT / name).convert("RGB")
    width, _ = image.size
    # 原图顶部的黑底列标题带不完整，先剔除，再由本脚本统一重绘
    gray = np.asarray(image.convert("L"))
    dark_rows = np.where((gray < 60).mean(axis=1) > 0.5)[0]
    if dark_rows.size:
        top = max(top, int(dark_rows[-1]) + 4)
    row = image.crop((int(width * 0.03), top, int(width * KEEP_RATIO), bottom))
    # 去掉原图遗留的 (a)-(d) 面板角标，避免与本图的行标签冲突
    return row.crop((0, 36, row.width, row.height))


def main():
    rows = []
    for name, top, bottom in CASES:
        row = load_row(name, top, bottom)
        rows.append(row.resize((TARGET_W, round(row.height * TARGET_W / row.width)), Image.LANCZOS))

    width = TARGET_W + PAD_LEFT + 10
    height = HEADER_H + sum(row.height for row in rows) + GAP * (len(rows) + 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 46)
    except OSError:
        font = ImageFont.load_default()

    col_w = TARGET_W / len(COLUMN_TITLES)
    for index, title in enumerate(COLUMN_TITLES):
        box = draw.textbbox((0, 0), title, font=font)
        x = PAD_LEFT + col_w * (index + 0.5) - (box[2] - box[0]) / 2
        draw.text((x, 26), title, fill="black", font=font)

    y = HEADER_H + GAP
    for letter, row in zip("abcd", rows):
        canvas.paste(row, (PAD_LEFT, y))
        draw.text((14, y + row.height // 2 - 28), f"({letter})", fill="black", font=font)
        y += row.height + GAP

    canvas.save(OUT, optimize=True)
    print("saved", OUT, canvas.size)


if __name__ == "__main__":
    main()
