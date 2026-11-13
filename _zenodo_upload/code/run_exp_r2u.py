# -*- coding: utf-8 -*-
"""
R2U 实验：复原→理解 端到端串联验证（Restoration-to-Understanding）

四组对照（同一渲染器、同一 SAM 权重、同一 CLIP）：
  A  GT 点云   → GP-SAM(几何提示) → Cloze-CLIP      （理解上限）
  B  复原点云  → GP-SAM(几何提示) → Cloze-CLIP      （核心串联：创新1+2）
  C  复原点云  → 全自动SAM        → Cloze-CLIP      （消融创新1）
  D  复原点云  → GP-SAM(几何提示) → 普通模板CLIP     （消融创新2）

创新1 GP-SAM：复原/真值点云经相机投影到图像，KMeans 聚类得构件中心，
      作为 SAM predictor 的点提示——复原几何直接参与语义分割。
创新2 Cloze-CLIP：构件识别采用完形填空式多模板文本提示集成，
      温度由模型 logit_scale 控制。

评估：
  cat_acc   整图文物类别 CLIP 分类准确率（有 GT：manifest 六类）
  cons      与 A 组的点级语义标签一致率（串联保真度，核心指标）
  assign    2D→3D 点归属率
  conf      CLIP top1 相似度均值
  kg_fix    KG 构件组校验修正比例（轻量闭环收益）

可视化（--viz，分辨率 ≥800，focal 随分辨率同步缩放，仅用于论文出图，
SAM/CLIP 推理仍走 448 链路）：
  fig1 triptych  缺损输入 | 复原结果 | GT 三点云渲染对比
  fig2 gpsam     GP-SAM 几何提示过程（投影点+簇中心）| GP-SAM 掩膜 | 全自动SAM 掩膜
  fig3 labels    GP-SAM 掩膜构件标注（KG 修正后标签）
  fig4 backproj  2D→3D 反投影：点云按构件标签着色 | 2D 标签图
  fig_grid_by_category.png  六类代表样本网格总图

用法：
  python run_exp_r2u.py --limit 6 --level 0.4 --smoke --viz --iters 300
  python run_exp_r2u.py --limit 36 --level 0.4 --viz --iters 300
"""
import argparse
import glob
import gzip
import json
import os
import sys
import time
import zlib

import numpy as np
import trimesh

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

if os.name == "nt":
    WS = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01"
    sys.path.insert(0, os.path.join(WS, "sam_pkg"))
    sys.path.insert(0, os.path.join(WS, "clip_pkg"))
    sys.path.insert(0, r"G:\大博士论文资料\代码")
else:
    # seetacloud 服务器：B3 全量部署目录（glbs/manifest/modules 都在此处）
    WS = "/root/cloud_deploy"
sys.path.insert(0, WS)

import run_exp_b3 as _b3  # noqa: E402
if os.name != "nt":
    _b3.WS = WS
    _glb_root = os.path.join(WS, "glbs")
    if glob.glob(os.path.join(_glb_root, "*", "*.glb")):
        _b3.CACHE_GLB = _glb_root
    else:  # 服务器 glbs 为平铺布局时换平铺索引
        def _flat_glb_index():
            return {os.path.splitext(os.path.basename(fp))[0]: fp
                    for fp in glob.glob(os.path.join(_glb_root, "*.glb"))}
        _b3.build_glb_index = _flat_glb_index

from run_exp_b3 import (load_gt_points, damage_to_ratio, splat_depth,  # noqa: E402
                        load_dataset, K, POSE, IMG, FOCAL,
                        trim_outliers, merge_resample)
from modules.diffusion_completion import DiffusionCompletionConfig, DiffusionGuidedCompletion  # noqa: E402
from modules.sam_kg import CulturalHeritageKG  # noqa: E402

SAM_CKPT = os.path.join(WS, "checkpoints", "sam_vit_b_01ec64.pth")
META_DIR = (r"C:\Users\黄文全\.cache\.objaverse\hf-objaverse-v1\metadata"
            if os.name == "nt" else os.path.join(WS, "metadata"))

# manifest 六类 → 英文 CLIP 类别词（短词，适配单色渲染图）
CAT_EN = {
    "石窟造像": "Buddha statue",
    "古建筑构件": "wooden bracket",
    "青铜器": "bronze vessel",
    "陶瓷器": "ceramic vase",
    "玉器": "jade carving",
    "壁画": "mural painting",
}

# KG 36 个中文构件 → 英文 CLIP 提示（标签落回 KG 中文名，保证图谱可用）
KG_COMP_EN = {
    "器身": "vessel body", "口沿": "mouth rim", "肩部": "shoulder part",
    "腹部": "belly part", "圈足": "ring foot", "羊首": "sheep head",
    "耳部": "ear handle", "足": "foot", "斗拱": "bracket set",
    "坐斗": "capital block", "翘": "cantilever arm", "昂": "angled lever",
    "升": "small bearing block", "耍头": "decorative head",
    "撑头木": "support timber", "头部": "head", "面部": "face",
    "躯干": "torso", "衣纹": "robe folds", "莲座": "lotus pedestal",
    "背光": "halo aureole", "颈部": "neck", "耳": "ear",
    "鋬": "loop handle", "口部": "mouth part", "底部": "bottom part",
    "台基": "platform base", "柱": "column", "梁": "beam",
    "枋": "architrave", "椽": "rafter", "飞椽": "flying rafter",
    "瓦当": "eaves tile", "纹饰区": "decorative pattern area",
    "穿孔": "hole perforation", "边缘": "edge region",
}
COMPONENTS_ZH = list(KG_COMP_EN.keys())

# 类别 → KG 候选构件（按构件层级限定）。若在 36 类全集上直接 argmax，
# "hole perforation" 这类与几何无关的短语会在任意局部裁剪上通吃，
# 导致所有样本所有掩膜标注成同一个词。限定候选集后标签才具备区分性。
KG_BY_CAT = {
    "石窟造像": ["头部", "面部", "颈部", "躯干", "衣纹", "莲座", "背光", "耳"],
    "古建筑构件": ["斗拱", "坐斗", "翘", "昂", "升", "耍头", "撑头木",
                   "台基", "柱", "梁", "枋", "椽", "飞椽", "瓦当"],
    "青铜器": ["器身", "口沿", "肩部", "腹部", "圈足", "羊首", "耳部",
               "足", "鋬", "口部", "底部", "纹饰区"],
    "陶瓷器": ["器身", "口沿", "肩部", "腹部", "圈足", "耳部", "底部", "纹饰区"],
    "玉器": ["器身", "穿孔", "边缘", "纹饰区"],
    "壁画": ["纹饰区", "边缘", "台基", "柱"],
}

# KG 中的文物类别节点（part_of 的顶层）；构件标签不应被归并到这一层
CAT_NODES = {"石窟造像", "古建筑", "青铜器", "陶瓷器", "玉器", "壁画"}

CLOZE_TEMPLATES = [
    "the {} part of a cultural relic",
    "a {} component of an ancient artifact",
    "the {} region of a heritage object",
    "close-up of the {} of an antique",
]
# 类别整图分类：域适配模板（渲染图不是照片）
CAT_TEMPLATES = [
    "a 3D rendering of a {}",
    "a shaded drawing of a {}",
    "a point cloud rendering of a {}",
    "a photo of a {}",
]
PLAINTEXT_TEMPLATES = ["a photo of a {}"]


# ---------------------------------------------------------------- 工具
def _frame_of(pts):
    """归一化坐标系（中心+尺度）。viz 时三种点云共用 GT 坐标系保证可比。"""
    c = pts.mean(axis=0)
    s = np.linalg.norm(pts - c, axis=1).max()
    return c.astype(np.float32), float(max(s, 1e-6))


def normalize(pts):
    c, s = _frame_of(pts)
    return ((pts - c) / s).astype(np.float32)


def render_points(points, img_size=IMG, focal=FOCAL, frame=None):
    """点云溅射渲染：法线 Lambert 明暗 + 深度衰减 + 砂岩底色 + 轮廓。"""
    import cv2
    from scipy.spatial import cKDTree
    if frame is None:
        frame = _frame_of(points)
    pts = ((points - frame[0]) / frame[1]).astype(np.float32)
    n = len(pts)
    # PCA 局部法线
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(16, n))
    normals = np.zeros_like(pts)
    for i in range(0, n):
        nb = pts[idx[i]] - pts[i]
        cov = nb.T @ nb
        w, vec = np.linalg.eigh(cov)
        nv = vec[:, 0]
        normals[i] = nv if nv[2] < 0 else -nv
    cam = pts + np.array([0, 0, 3.0], dtype=np.float32)
    z = cam[:, 2]
    u = np.round(focal * cam[:, 0] / z + img_size / 2).astype(int)
    v = np.round(focal * cam[:, 1] / z + img_size / 2).astype(int)
    inside = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
    u, v, z = u[inside], v[inside], z[inside]
    nrm = normals[inside]
    light = np.array([0.4, 0.5, -1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    lam = np.abs(nrm @ light)
    zn = (z - z.min()) / max(np.ptp(z), 1e-6)
    shade = (0.35 + 0.55 * lam + 0.10 * (1 - zn))[:, None]
    base = np.array([0.82, 0.74, 0.60], dtype=np.float32)
    colors = np.clip(base[None] * shade, 0, 1) * 255
    colors = colors.astype(np.uint8)
    img = np.full((img_size, img_size, 3), (245, 243, 238), dtype=np.uint8)
    order = np.argsort(-z)
    img[v[order], u[order]] = colors[order]
    fg = np.any(img != (245, 243, 238), axis=-1).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    edges = cv2.Canny(fg, 50, 150)
    img[edges > 0] = (90, 80, 70)
    return img


def render_points_colored(points, pt_colors, img_size, focal, frame):
    """按逐点颜色做 z-buffer 溅射渲染（2D→3D 反投影可视化用）。"""
    pts = ((points - frame[0]) / frame[1]).astype(np.float32)
    cam = pts + np.array([0, 0, 3.0], dtype=np.float32)
    z = np.clip(cam[:, 2], 1e-3, None)
    u = np.round(focal * cam[:, 0] / z + img_size / 2).astype(int)
    v = np.round(focal * cam[:, 1] / z + img_size / 2).astype(int)
    inside = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
    u, v, z = u[inside], v[inside], z[inside]
    cols = pt_colors[inside]
    img = np.full((img_size, img_size, 3), (245, 243, 238), dtype=np.uint8)
    order = np.argsort(-z)
    img[v[order], u[order]] = cols[order]
    return img


def project_points(points, img_size=IMG, focal=FOCAL, frame=None):
    """点云 → 图像像素坐标（返回 inside 的 u,v 与掩膜索引）"""
    if frame is None:
        frame = _frame_of(points)
    pts = ((points - frame[0]) / frame[1]).astype(np.float32)
    cam = pts + np.array([0, 0, 3.0], dtype=np.float32)
    z = np.clip(cam[:, 2], 1e-3, None)
    u = focal * cam[:, 0] / z + img_size / 2
    v = focal * cam[:, 1] / z + img_size / 2
    inside = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
    return np.stack([u[inside], v[inside]], 1), inside


# ---------------------------------------------------------------- GP-SAM
def _gp_prompts(points, k=5, seed=0):
    """GP-SAM 几何提示：投影点 uv + KMeans 簇中心 + 簇标签。"""
    from sklearn.cluster import KMeans
    uv, _ = project_points(points)
    if len(uv) == 0:
        return uv, np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
    if len(uv) < k:
        k = max(1, len(uv))
    km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(uv)
    return uv, km.cluster_centers_, km.labels_


def gp_sam_segment(predictor, image, points, k=5, seed=0):
    """几何提示 SAM：点云投影 → KMeans → 每簇 中心+簇内2点 三点正提示出掩膜。"""
    uv, centers, clab = _gp_prompts(points, k=k, seed=seed)
    if len(uv) == 0:
        return []
    rng = np.random.default_rng(seed)
    predictor.set_image(image)
    h, w = image.shape[:2]
    masks = []
    for ci, c in enumerate(centers):
        idxs = np.where(clab == ci)[0]
        extra = rng.choice(idxs, size=min(2, len(idxs)), replace=False)
        pts_xy = np.vstack([[c], uv[extra]]).astype(np.float32)
        pl = np.ones(len(pts_xy), dtype=np.int64)
        try:
            m, scores, _ = predictor.predict(
                point_coords=pts_xy, point_labels=pl, multimask_output=True)
            m = m[int(np.argmax(scores))]
        except Exception:
            continue
        area = int(m.sum())
        if area < 0.002 * h * w or area > 0.99 * h * w:
            continue
        masks.append(m.astype(bool))
    # 去重：IoU>0.90 合并
    keep = []
    for m in masks:
        dup = False
        for q in keep:
            inter = (m & q).sum()
            if inter / max((m | q).sum(), 1) > 0.90:
                dup = True
                break
        if not dup:
            keep.append(m)
    return keep


_AUTO_GEN = None


def auto_sam_segment(image):
    global _AUTO_GEN
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    if _AUTO_GEN is None:
        sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to("cuda")
        _AUTO_GEN = SamAutomaticMaskGenerator(sam, points_per_side=16,
                                              pred_iou_thresh=0.86,
                                              stability_score_thresh=0.92,
                                              min_mask_region_area=2000)
    out = _AUTO_GEN.generate(image)
    return [m["segmentation"].astype(bool) for m in out]


def _add_dotted(rgb, edge, color=(255, 0, 0)):
    """在 RGB 图上把边缘像素点彩成红色——点云域桥接增广。"""
    out = rgb.copy()
    out[edge > 0] = color
    return out


# ---------------------------------------------------------------- CLIP
class ClozeCLIP:
    """Cloze 式多模板 CLIP。文本特征按模板分组保存 (C,T,D)，
    推理时逐模板算 logit 再跨模板平均（logit 集成，区分度不被特征均值稀释）。"""

    def __init__(self, comp_templates, cat_templates=None):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        self.model = self.model.cuda().eval()
        self.tok = open_clip.get_tokenizer("ViT-B-32")
        cat_templates = cat_templates or comp_templates
        with torch.no_grad():
            self.text_feat = self._encode_texts(
                [t.format(KG_COMP_EN[c]) for c in COMPONENTS_ZH for t in comp_templates]
            ).reshape(len(COMPONENTS_ZH), len(comp_templates), -1)
            self.cat_feat = self._encode_texts(
                [t.format(v) for v in CAT_EN.values() for t in cat_templates]
            ).reshape(len(CAT_EN), len(cat_templates), -1)
        self.cat_keys = list(CAT_EN.keys())

    def _encode_texts(self, texts):
        torch = self.torch
        with torch.no_grad():
            f = self.model.encode_text(self.tok(texts).cuda())
        return (f / f.norm(dim=-1, keepdim=True)).float()

    def _encode_image(self, rgb):
        from PIL import Image
        torch = self.torch
        img = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).cuda()
        with torch.no_grad():
            f = self.model.encode_image(img)
        return (f / f.norm(dim=-1, keepdim=True)).float()

    def _logit_ens(self, f, text_feat):
        """f:(1,D)  text_feat:(C,T,D) → (C,) 跨模板平均 logit"""
        torch = self.torch
        return (100.0 * torch.einsum("d,ctd->ct", f[0], text_feat)).mean(-1)

    def classify_image(self, rgb, edge=None):
        """整图分类；给 edge 时叠加红点彩图做域桥接，取两次 prob 较高者。"""
        cands = [rgb]
        if edge is not None:
            cands.append(_add_dotted(rgb, edge))
        best_p, best_i = -1.0, 0
        for im in cands:
            f = self._encode_image(im)
            p = self._logit_ens(f, self.cat_feat).softmax(-1)
            i = int(p.argmax())
            if float(p[i]) > best_p:
                best_p, best_i = float(p[i]), i
        return self.cat_keys[best_i], best_p

    def label_mask(self, image, mask, allowed=None):
        """掩膜 bbox 裁剪 → 构件 top1 标签与置信度。
        allowed 给定时只在候选构件内 argmax（类别先验约束）。"""
        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None, 0.0
        h, w = image.shape[:2]
        y0, y1 = max(0, ys.min() - 8), min(h, ys.max() + 8)
        x0, x1 = max(0, xs.min() - 8), min(w, xs.max() + 8)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None, 0.0
        f = self._encode_image(crop)
        p = self._logit_ens(f, self.text_feat).softmax(-1)
        if allowed:
            keep = [COMPONENTS_ZH.index(a) for a in allowed if a in COMPONENTS_ZH]
            if keep:
                idx = self.torch.tensor(keep, device=p.device)
                masked = p.new_full(p.shape, -1.0)
                masked[idx] = p[idx]
                p = masked
        i = int(p.argmax())
        return COMPONENTS_ZH[i], float(p[i])


# ---------------------------------------------------------------- 评估
def assign_labels_to_points(points, masks, labels):
    """把掩膜标签投到点云：返回点级标签数组（-1 未归属）与归属率。"""
    uv, inside = project_points(points)
    lab = np.full(len(points), -1, dtype=np.int32)
    uvi = np.round(uv).astype(int)
    idxs = np.where(inside)[0]
    for mid, m in enumerate(masks):
        hit = m[uvi[:, 1], uvi[:, 0]]
        lab[idxs[hit]] = mid
    return lab, float((lab >= 0).mean())


def point_consistency(lab_a, lab_b):
    """A/B 两组掩膜标签场的一致率：贪心最大匹配近似。"""
    if len(lab_a) != len(lab_b):
        return 0.0
    both = (lab_a >= 0) & (lab_b >= 0)
    if both.sum() == 0:
        return 0.0
    la, lb = lab_a[both], lab_b[both]
    na, nb = la.max() + 1, lb.max() + 1
    conf = np.zeros((na, nb), dtype=np.int64)
    np.add.at(conf, (la, lb), 1)
    match = 0
    used_b = set()
    for i in range(na):
        j = int(np.argmax(conf[i]))
        if conf[i, j] > 0 and j not in used_b:
            match += conf[i, j]
            used_b.add(j)
    return float(match) / float(both.sum())


def kg_group_fix(labels, kg, cat_nodes=None):
    """KG 轻量闭环：同组构件碎片标签投票归并。
    KG 中 part_of 直接连到文物类别节点（穿孔→玉器、器身→青铜器），
    若照单全收会把构件标签升级成类别名，故类别节点不参与归并。"""
    cat_nodes = cat_nodes or CAT_NODES
    groups = []
    for l in labels:
        g = kg.get_component_group(l)
        groups.append(g if (g and g not in cat_nodes) else None)
    fixed = list(labels)
    n_fix = 0
    from collections import Counter
    cnt = Counter(g for g in groups if g)
    for i, g in enumerate(groups):
        if g and cnt[g] >= 2 and labels[i] != g:
            # 组内出现≥2次 → 归并到组名（坐斗/翘/昂 → dougong 组）
            fixed[i] = g
            n_fix += 1
    return fixed, n_fix / max(len(labels), 1)


# ---------------------------------------------------------------- 可视化
# 高对比调色板（RGB）
VIZ_PALETTE = np.array([
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40),
], dtype=np.uint8)
BG = (245, 243, 238)


def _put_title(img, title, sub=""):
    """顶部深色标题条（英文，规避服务器中文字体缺失）。"""
    import cv2
    out = img.copy()
    bar = max(64, img.shape[0] // 12)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar), (35, 35, 35), -1)
    fs = bar / 75.0
    cv2.putText(out, title, (14, int(bar * 0.52)), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(out, sub, (14, int(bar * 0.88)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs * 0.62, (200, 200, 200), 1,
                    cv2.LINE_AA)
    return out


def overlay_masks(image, masks, alpha=0.45):
    """掩膜半透明彩色叠加 + 轮廓描边。"""
    import cv2
    out = image.copy()
    for i, m in enumerate(masks):
        color = np.array(VIZ_PALETTE[i % len(VIZ_PALETTE)], dtype=np.float32)
        out[m] = np.clip(out[m].astype(np.float32) * (1 - alpha) + color * alpha,
                         0, 255).astype(np.uint8)
    for i, m in enumerate(masks):
        color = tuple(int(x) for x in VIZ_PALETTE[i % len(VIZ_PALETTE)])
        cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cont, -1, color, max(2, image.shape[0] // 400))
    return out


def annotate_masks(image, masks, labels):
    """掩膜叠加 + 质心处英文构件标签（带底色框）。"""
    import cv2
    out = overlay_masks(image, masks)
    fs = max(0.7, image.shape[0] / 900.0)
    th = max(2, image.shape[0] // 500)
    for i, (m, lb) in enumerate(zip(masks, labels)):
        ys, xs = np.where(m)
        if len(ys) == 0:
            continue
        cy, cx = int(ys.mean()), int(xs.mean())
        txt = KG_COMP_EN.get(lb, "region")
        color = tuple(int(x) for x in VIZ_PALETTE[i % len(VIZ_PALETTE)])
        (tw, thh), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x0 = min(max(4, cx - tw // 2), out.shape[1] - tw - 8)
        y0 = min(max(thh + 8, cy), out.shape[0] - 8)
        cv2.rectangle(out, (x0 - 4, y0 - thh - 6), (x0 + tw + 4, y0 + 6),
                      (255, 255, 255), -1)
        cv2.rectangle(out, (x0 - 4, y0 - thh - 6), (x0 + tw + 4, y0 + 6),
                      color, 2)
        cv2.putText(out, txt, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (20, 20, 20), th, cv2.LINE_AA)
    return out


def draw_prompts(image, uv, centers, clab):
    """GP-SAM 几何提示过程图：投影点（按簇着色）+ 簇中心十字。"""
    import cv2
    out = image.copy()
    s = image.shape[0] / IMG  # 448 → viz_res 的缩放
    r = max(1, image.shape[0] // 300)
    for (u, v), ci in zip(uv, clab):
        color = tuple(int(x) for x in VIZ_PALETTE[ci % len(VIZ_PALETTE)])
        cv2.circle(out, (int(u * s), int(v * s)), r, color, -1)
    for ci, c in enumerate(centers):
        color = tuple(int(x) for x in VIZ_PALETTE[ci % len(VIZ_PALETTE)])
        x, y = int(c[0] * s), int(c[1] * s)
        L = max(10, image.shape[0] // 45)
        cv2.drawMarker(out, (x, y), (255, 255, 255), cv2.MARKER_CROSS, L * 2,
                       max(3, image.shape[0] // 250))
        cv2.drawMarker(out, (x, y), color, cv2.MARKER_CROSS, L * 2,
                       max(2, image.shape[0] // 350))
    return out


def masks_label_image(masks, img_size):
    """2D 标签着色图（掩膜 idx → 调色板色），自适应掩膜自身分辨率。"""
    import cv2
    src = masks[0].shape[0] if masks else IMG
    lab_img = np.full((src, src, 3), BG, dtype=np.uint8)
    for i, m in enumerate(masks):
        lab_img[m] = VIZ_PALETTE[i % len(VIZ_PALETTE)]
    if src != img_size:
        lab_img = cv2.resize(lab_img, (img_size, img_size),
                             interpolation=cv2.INTER_NEAREST)
    return lab_img


def _up_masks(masks, res):
    """448 掩膜最近邻放大到出图分辨率。"""
    import cv2
    return [cv2.resize(m.astype(np.uint8), (res, res),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
            for m in masks]


def make_sample_figs(uid, cat, level, cd, gt, partial, comp, viz_data,
                     viz_res, fig_dir):
    """每个样本输出 4 张过程高清图；返回六类网格总图所需的一行 tiles。"""
    import cv2
    fv = FOCAL * viz_res / IMG
    frame = _frame_of(gt)
    cat_en = CAT_EN.get(cat, cat)

    img_gt_h = render_points(gt, viz_res, fv, frame=frame)
    img_rec_h = render_points(comp, viz_res, fv, frame=frame)
    img_dam_h = render_points(partial, viz_res, fv, frame=frame)

    # fig1：缺损 | 复原 | GT 三联
    fig1 = np.hstack([
        _put_title(img_dam_h, f"Damaged input", f"missing ratio {level:.0%}"),
        _put_title(img_rec_h, f"Restored (ours)", f"CD-L1 {cd:.4f}"),
        _put_title(img_gt_h, f"Ground truth", f"{cat_en}"),
    ])
    cv2.imwrite(os.path.join(fig_dir, f"{uid[:8]}_fig1_triptych.png"),
                cv2.cvtColor(fig1, cv2.COLOR_RGB2BGR))

    # fig2：GP-SAM 提示过程 | GP-SAM 掩膜 | 全自动SAM 掩膜
    img_rec448 = viz_data["B"]["img"]  # 448 推理链路同款渲染（self-frame）
    img_rec_h2 = cv2.resize(img_rec448, (viz_res, viz_res),
                            interpolation=cv2.INTER_LINEAR)
    uv, centers, clab = _gp_prompts(comp, k=viz_data["k_prompts"],
                                    seed=viz_data["seed"])
    masks_b = _up_masks(viz_data["B"]["masks"], viz_res)
    masks_c = _up_masks(viz_data["C"]["masks"], viz_res) if viz_data.get("C") else []
    fig2 = np.hstack([
        _put_title(draw_prompts(img_rec_h2, uv, centers, clab),
                   f"GP-SAM prompts", f"{len(centers)} clusters"),
        _put_title(overlay_masks(img_rec_h2, masks_b),
                   f"GP-SAM masks (B)", f"{len(masks_b)} masks"),
        _put_title(overlay_masks(img_rec_h2, masks_c),
                   f"Auto-SAM masks (C)", f"{len(masks_c)} masks"),
    ])
    cv2.imwrite(os.path.join(fig_dir, f"{uid[:8]}_fig2_gpsam.png"),
                cv2.cvtColor(fig2, cv2.COLOR_RGB2BGR))

    # fig3：构件标注（B 组，KG 修正后）
    fig3 = _put_title(annotate_masks(img_rec_h2, masks_b,
                                     viz_data["B"]["labels"]),
                      f"Component labels (B, after KG fix)",
                      f"{cat_en} | {len(masks_b)} components")
    cv2.imwrite(os.path.join(fig_dir, f"{uid[:8]}_fig3_labels.png"),
                cv2.cvtColor(fig3, cv2.COLOR_RGB2BGR))

    # fig4：2D→3D 反投影（B 组点级标签着色）
    lab_arr = viz_data["B"]["lab_arr"]
    pt_colors = np.full((len(comp), 3), (200, 200, 200), dtype=np.uint8)
    for mid in range(lab_arr.max() + 1 if len(lab_arr) else 0):
        pt_colors[lab_arr == mid] = VIZ_PALETTE[mid % len(VIZ_PALETTE)]
    cloud_col = render_points_colored(comp, pt_colors, viz_res, fv, frame)
    lab2d = masks_label_image(masks_b, viz_res)
    fig4 = np.hstack([
        _put_title(cloud_col, "3D points colored by 2D labels",
                   f"assign {viz_data['B']['assign']:.3f}"),
        _put_title(lab2d, "2D label map (GP-SAM)", ""),
    ])
    cv2.imwrite(os.path.join(fig_dir, f"{uid[:8]}_fig4_backproj.png"),
                cv2.cvtColor(fig4, cv2.COLOR_RGB2BGR))

    # 网格总图行：GT | 缺损 | 复原 | GP-SAM掩膜 | 构件标注
    row = [_put_title(img_gt_h, f"GT", cat_en),
           _put_title(img_dam_h, "Damaged", ""),
           _put_title(img_rec_h, "Restored", f"CD {cd:.3f}"),
           _put_title(overlay_masks(img_rec_h2, masks_b), "GP-SAM", ""),
           _put_title(annotate_masks(img_rec_h2, masks_b,
                                     viz_data["B"]["labels"]), "Labeled", "")]
    return [cv2.cvtColor(t, cv2.COLOR_RGB2BGR) for t in row]


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ch3d")
    ap.add_argument("--level", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--k_prompts", type=int, default=5)
    ap.add_argument("--iters", type=int, default=300,
                    help="扩散补全迭代数（高精度默认 300；缓存键含迭代数）")
    ap.add_argument("--prior_ckpt",
                    default=os.path.join(WS, "prior_step14000.pt"),
                    help="预训练扩散先验权重；文件不存在则回退内置轻量先验")
    ap.add_argument("--viz", action="store_true", help="输出高清过程图")
    ap.add_argument("--viz_res", type=int, default=832, help="出图分辨率(≥800)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(WS, "results_r2u"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache_dir = os.path.join(args.out, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fig_dir = os.path.join(args.out, "figs")
    if args.viz:
        os.makedirs(fig_dir, exist_ok=True)
    res_path = os.path.join(args.out, "results.json")

    # 样本：六类分层均匀抽样（manifest 前 65 个全是石窟造像，顺序取 smoke 无代表性）
    data_all = load_dataset(args.dataset)
    m = json.load(open(os.path.join(WS, "ch_single3d_manifest.json"), encoding="utf-8"))
    uid2cat = {it["uid"]: cat for cat, items in m["categories"].items() for it in items}
    per_cat = max(1, args.limit // 6)
    buckets = {}
    for uid, gp in data_all:
        buckets.setdefault(uid2cat.get(uid, "其他"), []).append((uid, gp))
    data = []
    for cat, items in buckets.items():
        data.extend(items[:per_cat])
    data = data[:args.limit]
    print(f"[data] 分层抽样 {len(data)} 个："
          + ", ".join(f"{c}×{min(per_cat, len(items))}" for c, items in buckets.items()),
          flush=True)

    print(f"[init] 加载扩散补全器（num_iters={args.iters}）…", flush=True)
    cfg = DiffusionCompletionConfig(num_iters=args.iters, device="cuda")
    if args.prior_ckpt and os.path.exists(args.prior_ckpt):
        cfg.prior_ckpt = args.prior_ckpt
        print(f"[init] 使用预训练扩散先验: {args.prior_ckpt}", flush=True)
    else:
        print(f"[init] 警告: 未找到先验权重 {args.prior_ckpt}，回退内置轻量先验",
              flush=True)
    completer = DiffusionGuidedCompletion(cfg)

    print("[init] 加载 SAM predictor…", flush=True)
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to("cuda")
    predictor = SamPredictor(sam)

    print("[init] 加载 CLIP…", flush=True)
    import cv2  # noqa
    clip_cloze = ClozeCLIP(CLOZE_TEMPLATES, cat_templates=CAT_TEMPLATES)
    clip_plain = ClozeCLIP(PLAINTEXT_TEMPLATES, cat_templates=PLAINTEXT_TEMPLATES)
    kg = CulturalHeritageKG()
    print(f"[KG] {kg.summary()}", flush=True)

    # smoke 模式重跑所有样本（复原点云缓存仍复用），非 smoke 才断点续跑
    results = []
    if not args.smoke and os.path.exists(res_path):
        try:
            results = json.load(open(res_path, encoding="utf-8"))
        except Exception:
            results = []
    done = {r["uid"] for r in results}

    grid_rows = {}  # cat → 网格总图行（每类留 CD 最小样本）
    grid_best = {}  # cat → 当前最优 CD

    for si, (uid, gpath) in enumerate(data):
        if uid in done:
            continue
        cat = uid2cat.get(uid, "其他")
        t0 = time.time()
        print(f"[{si+1}/{len(data)}] {uid[:8]} {cat}", flush=True)
        try:
            gt = load_gt_points(gpath)
            if gt is None:
                continue
            gt = normalize(gt)
            cache = os.path.join(cache_dir,
                                 f"{uid}_{args.level:g}_i{args.iters}_t2.npz")
            seed = zlib.crc32(f"{uid}|{args.level}".encode("utf-8"))
            if os.path.exists(cache):
                z = np.load(cache)
                comp = z["complete"]
                cd = float(z["cd_l1"])
                if args.viz:
                    partial, _, _, _, _, _ = damage_to_ratio(gt, args.level, seed)
            else:
                partial, hole_gt, hole, ratio, radius, hole_center = damage_to_ratio(
                    gt, args.level, seed)
                depth = splat_depth(partial)
                res = completer.complete(
                    partial, depth, K, POSE, verbose=False,
                    missing_hint=(hole_center, radius, int(round(len(gt) * ratio))))
                # 离群裁剪（B3 同口径）：剔除扩散补全的漂浮点，可见点全保留
                kept, _ = trim_outliers(res["completed"], k=8, sigma=2.0)
                comp = merge_resample(res["visible"], kept,
                                      len(res["complete"]), seed).astype(np.float32)
                # 与 B3 口径一致的 CD-L1
                from run_exp_b3 import f_score_pr  # noqa
                from scipy.spatial import cKDTree
                t = cKDTree(gt)
                cd = float(t.query(comp)[0].mean())
                np.savez_compressed(cache, complete=comp, cd_l1=cd)
        except Exception as e:
            print(f"  [skip] 复原失败 {e}", flush=True)
            continue

        rec = {"uid": uid, "cat": cat, "level": args.level, "cd_l1": round(cd, 5)}
        # 渲染（同一渲染器）
        img_gt = render_points(gt)
        img_rec = render_points(comp)
        viz_data = {"k_prompts": args.k_prompts, "seed": si} if args.viz else None

        # ---- 组别循环
        for grp, (src_pts, src_img, sam_mode, clip_model) in {
            "A": (gt, img_gt, "gp", clip_cloze),
            "B": (comp, img_rec, "gp", clip_cloze),
            "C": (comp, img_rec, "auto", clip_cloze),
            "D": (comp, img_rec, "gp", clip_plain),
        }.items():
            try:
                if sam_mode == "gp":
                    masks = gp_sam_segment(predictor, src_img, src_pts,
                                           k=args.k_prompts, seed=si)
                else:
                    masks = auto_sam_segment(src_img)
                edge = cv2.Canny(cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY), 40, 120)
                pred_cat, cat_p = clip_model.classify_image(src_img, edge=edge)
                if grp == "A":
                    f0 = clip_model._encode_image(src_img)
                    p0 = clip_model._logit_ens(f0, clip_model.cat_feat).softmax(-1)
                    order = p0.argsort(descending=True)[:3]
                    top = " ".join(f"{clip_model.cat_keys[i]}:{float(p0[i]):.2f}"
                                   for i in order)
                    areas = sorted([int(mm.sum()) for mm in masks], reverse=True)
                    print(f"  [oracle] 真类={cat} | top3 {top}", flush=True)
                    print(f"  [GP-SAM] masks={areas}", flush=True)
                labs_raw, confs = [], []
                for mk in masks:
                    lb, cf = clip_model.label_mask(src_img, mk,
                                                   allowed=KG_BY_CAT.get(cat))
                    labs_raw.append(lb if lb else "边缘")
                    confs.append(cf)
                fixed, fix_rate = kg_group_fix(labs_raw, kg)
                lab_arr, assign = assign_labels_to_points(src_pts, masks, labs_raw)
                rec[grp] = {
                    "n_masks": len(masks),
                    "assign": round(assign, 4),
                    "cat_pred": pred_cat,
                    "cat_ok": int(pred_cat == cat),
                    "cat_p": round(cat_p, 4),
                    "conf": round(float(np.mean(confs)) if confs else 0.0, 4),
                    "kg_fix": round(fix_rate, 4),
                    "labels": fixed,
                    "_lab_arr": lab_arr,  # 内部用，不落盘
                }
                if args.viz and grp in ("B", "C"):
                    viz_data[grp] = {"masks": masks, "labels": fixed,
                                     "img": src_img, "assign": assign,
                                     "lab_arr": lab_arr}
            except Exception as e:
                print(f"  [{grp}] 失败 {e}", flush=True)
                rec[grp] = None

        # ---- 高清过程图
        if args.viz and rec.get("B") and rec.get("C"):
            try:
                row = make_sample_figs(uid, cat, args.level, cd, gt, partial,
                                       comp, viz_data, args.viz_res, fig_dir)
                # 每类选 CD 最小（复原质量最高）的样本进网格总图
                if cd < grid_best.get(cat, 1e9):
                    grid_rows[cat] = row
                    grid_best[cat] = cd
                print(f"  [viz] figs → {fig_dir}", flush=True)
            except Exception as e:
                print(f"  [viz] 出图失败 {e}", flush=True)

        # 一致率：B/C/D vs A（点级掩膜结构一致性）
        if rec.get("A") and rec["A"]:
            la = rec["A"]["_lab_arr"]
            for grp in "BCD":
                if rec.get(grp):
                    rec[grp]["cons"] = round(point_consistency(la, rec[grp]["_lab_arr"]), 4)
        for grp in "ABCD":
            if rec.get(grp):
                rec[grp].pop("_lab_arr", None)
        results.append(rec)
        json.dump(results, open(res_path, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        ok = {g: (rec[g]["cat_ok"] if rec.get(g) else None) for g in "ABCD"}
        print(f"  cat_ok={ok} cons_B={rec.get('B',{}).get('cons') if rec.get('B') else None}"
              f" | {time.time()-t0:.1f}s", flush=True)
        if args.smoke and len(results) >= args.limit:
            break

    # ---------------- 六类网格总图
    if args.viz and grid_rows:
        import cv2
        order = [c for c in CAT_EN if c in grid_rows] + \
                [c for c in grid_rows if c not in CAT_EN]
        grid = np.vstack([np.hstack(grid_rows[c]) for c in order])
        gp = os.path.join(fig_dir, "fig_grid_by_category.png")
        cv2.imwrite(gp, grid)
        print(f"[viz] 六类网格总图 → {gp}", flush=True)

    # ---------------- 汇总
    print("\n==== R2U 汇总 ====")
    head = f"{'组':<3}{'cat_acc':>8}{'cons':>8}{'assign':>8}{'conf':>8}{'kg_fix':>8}{'masks':>7}"
    print(head)
    lines = [head]
    for grp in "ABCD":
        rs = [r[grp] for r in results if r.get(grp)]
        if not rs:
            continue
        row = (f"{grp:<3}{np.mean([r['cat_ok'] for r in rs]):>8.3f}"
               f"{np.mean([r.get('cons', 0) for r in rs]):>8.3f}" if grp != "A" else
               f"{grp:<3}{np.mean([r['cat_ok'] for r in rs]):>8.3f}{'—':>8}")
        row += (f"{np.mean([r['assign'] for r in rs]):>8.3f}"
                f"{np.mean([r['conf'] for r in rs]):>8.3f}"
                f"{np.mean([r['kg_fix'] for r in rs]):>8.3f}"
                f"{np.mean([r['n_masks'] for r in rs]):>7.1f}")
        print(row)
        lines.append(row)
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n已写出 {res_path}")


if __name__ == "__main__":
    main()
