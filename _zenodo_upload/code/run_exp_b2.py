"""
B2 实验：真实 GLB 上的 SAM 构件分割 + KG 推理 + 2D→3D 匹配

流程（每个样本）：
  1. trimesh 加载 GLB → GT 点云（16384）+ 稠密采样带法线用于渲染
  2. 软件渲染 Lambertian 着色 RGB 图（z-buffer 溅射）
  3. SAM（有权重）或 SLIC 退化分割 → 构件掩膜
  4. KG + GNN 构件组推理
  5. Component3DMatcher 2D→3D 匹配
  6. 指标：掩膜数、点归属率、构件组数；按名称关键词归类
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\sam_pkg")
sys.path.insert(0, r"G:\大博士论文资料\代码")
from modules.sam_kg import (  # noqa: E402
    SAMComponentSegmenter, CulturalHeritageKG,
    ComponentGNNReasoner, Component3DMatcher,
)

CACHE_GLB = r"C:\Users\黄文全\.cache\.objaverse\hf-objaverse-v1\glbs"
META_DIR = r"C:\Users\黄文全\.cache\.objaverse\hf-objaverse-v1\metadata"
N_GT = 16384
IMG = 448
FOCAL = 500.0
K = np.array([[FOCAL, 0, IMG / 2], [0, FOCAL, IMG / 2], [0, 0, 1]], dtype=np.float32)

# world→camera：相机位于 (0,0,-3) 朝 +Z（与 B1 溅射约定一致）
POSE_W2C = np.eye(4, dtype=np.float32)
POSE_W2C[:3, 3] = [0.0, 0.0, 3.0]

CATEGORY_KWS = {
    "陶瓷器": ["vase", "bowl", "jar", "jug", "plate", "teapot", "urn", "amphora",
             "pottery", "ceramic", "porcelain", "stoneware", "earthenware",
             "terracotta", "pitcher", "cup", "mug", "basin", "pot"],
    "造像雕塑": ["figurine", "statue", "buddha", "sculpture", "bust", "head",
              "figure", "idol", "deity", "guanyin"],
    "青铜器": ["bronze", "ding", "tripod", "censer", "cauldron"],
    "建筑构件": ["capital", "column", "cornice", "bracket", "frieze", "relief",
              "gable", "lintel"],
}


def classify(name):
    n = (name or "").lower()
    for cat, kws in CATEGORY_KWS.items():
        if any(re.search(r"\b" + kw + r"\b", n) for kw in kws):
            return cat
    return "其他"


def load_names(uids):
    """从 77 个元数据块中查找所需 UID 的 name"""
    need = set(uids)
    found = {}
    for fp in sorted(glob.glob(os.path.join(META_DIR, "*.json.gz"))):
        if not need:
            break
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            chunk = json.load(f)
        for uid in list(need):
            if uid in chunk:
                found[uid] = chunk[uid].get("name", "")
                need.discard(uid)
    return found


def load_mesh(glb_path):
    mesh = trimesh.load(glb_path, force="scene")
    if not isinstance(mesh, trimesh.Scene):
        return None
    mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None
    if len(mesh.faces) > 3_000_000:
        return None
    return mesh


def normalize(pts):
    c = pts.mean(axis=0)
    pts = pts - c
    s = np.linalg.norm(pts, axis=1).max()
    return pts / max(s, 1e-6)


def render_rgb(mesh, n_samples=60000):
    """Lambertian 着色 + z-buffer 溅射渲染"""
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_idx]
    pts = normalize(pts.astype(np.float32))

    light = np.array([0.4, 0.7, -0.8], dtype=np.float32)
    light /= np.linalg.norm(light)
    shade = 0.25 + 0.75 * np.clip(normals @ light, 0, 1)

    base = np.array([0.80, 0.72, 0.58], dtype=np.float32)  # 砂岩色
    try:
        mc = mesh.visual.main_color
        if mc is not None and len(mc) >= 3:
            base = np.array(mc[:3], dtype=np.float32) / 255.0
    except Exception:
        pass
    colors = (base[None] * shade[:, None] * 255).astype(np.uint8)

    cam = pts + np.array([0, 0, 3.0], dtype=np.float32)  # world→cam (R=I, t=(0,0,3))
    z = cam[:, 2]
    u = np.round(FOCAL * cam[:, 0] / z + IMG / 2).astype(int)
    v = np.round(FOCAL * cam[:, 1] / z + IMG / 2).astype(int)
    inside = (u >= 0) & (u < IMG) & (v >= 0) & (v < IMG)
    u, v, z, colors = u[inside], v[inside], z[inside], colors[inside]

    img = np.full((IMG, IMG, 3), (245, 243, 238), dtype=np.uint8)
    order = np.argsort(-z)
    img[v[order], u[order]] = colors[order]
    # 轻量闭运算填补溅射空洞
    import cv2
    fg = np.any(img != (245, 243, 238), axis=-1).astype(np.uint8) * 255
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hole = (fg == 0) & np.any(img == (245, 243, 238), axis=-1)
    img[hole] = (245, 243, 238)
    return img, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default=r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\results_b2")
    ap.add_argument("--sam_checkpoint", default=None)
    ap.add_argument("--save_npz", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    glbs = sorted(glob.glob(os.path.join(CACHE_GLB, "**", "*.glb"), recursive=True))[:args.limit * 2]
    uids = [os.path.splitext(os.path.basename(g))[0] for g in glbs]
    print(f"候选 GLB: {len(glbs)}，查询名称…")
    names = load_names(uids)

    segmenter = SAMComponentSegmenter(
        model_type="vit_b", checkpoint=args.sam_checkpoint, device="cuda", max_components=30)
    kg = CulturalHeritageKG()
    reasoner = ComponentGNNReasoner(kg, device="cuda")
    matcher = Component3DMatcher(K, POSE_W2C, iou_threshold=0.3)
    print(f"[KG] {kg.summary()}")

    results = []
    n_done = 0
    for g, uid in zip(glbs, uids):
        if n_done >= args.limit:
            break
        try:
            mesh = load_mesh(g)
            if mesh is None:
                continue
            img, gt = render_rgb(mesh)
        except Exception as e:
            print(f"[skip] {uid}: {e}")
            continue

        t0 = time.perf_counter()
        masks = segmenter.segment(img)
        groups = [reasoner.predict_group(f"component_{i}") for i in range(len(masks))]
        labels, label_names = matcher.match(gt, masks, [f"{gp}_{i}" for i, gp in enumerate(groups)])
        dt = time.perf_counter() - t0

        assigned = float((labels >= 0).mean())
        name = names.get(uid, "")
        rec = {
            "uid": uid, "name": name, "category": classify(name),
            "n_masks": len(masks), "assigned_ratio": assigned,
            "n_groups_used": len(set(labels[labels >= 0].tolist())) if (labels >= 0).any() else 0,
            "mask_areas": [int(m["area"]) for m in masks[:10]],
            "runtime_s": round(dt, 2),
        }
        results.append(rec)
        print(f"[{n_done:3d}] {uid[:8]}… {rec['category']}「{name[:24]}」 "
              f"masks={len(masks)} assigned={assigned:.2f} ({dt:.1f}s)")

        if n_done < args.save_npz:
            np.savez_compressed(
                os.path.join(args.out, f"sample_{n_done}.npz"),
                image=img, gt=gt, labels=labels,
                masks=np.array([m["mask"] for m in masks]))
        n_done += 1

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    cats = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in cats.items():
        print(f"== {cat}: n={len(rs)} 平均掩膜={np.mean([r['n_masks'] for r in rs]):.1f} "
              f"平均归属率={np.mean([r['assigned_ratio'] for r in rs]):.2f}")
    print(f"完成 {len(results)} 个样本 → {args.out}")


if __name__ == "__main__":
    main()
