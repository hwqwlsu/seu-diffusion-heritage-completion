"""
B1 实验：真实 GLB 数据上的扩散先验残损补全

流程（每个样本）：
  1. trimesh 加载 GLB → 表面均匀采样 16384 点作为 GT
  2. 归一化到单位球（质心原点、最大模长 1）
  3. 模拟残损：随机球域挖除（3 档严重度）
  4. 虚拟相机（cam2world: R=I, t=(0,0,-3)，朝 +Z 看），溅射深度图
  5. DiffusionGuidedCompletion.complete 补全
  6. 指标：CD-L1、F@0.01、孔洞区 CD
  7. 逐样本结果写 results_b1/results.json；前若干样本存 npz 供定性可视化
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, r"G:\大博士论文资料\代码")
from modules.diffusion_completion import DiffusionCompletionConfig, DiffusionGuidedCompletion  # noqa: E402

CACHE_GLB = r"C:\Users\黄文全\.cache\.objaverse\hf-objaverse-v1\glbs"
N_GT = 16384
IMG = 448
FOCAL = 500.0

# 相机：位于 (0,0,-3)，朝 +Z 看（与 detector 反投影约定一致：world = cam @ R.T + t）
K = np.array([[FOCAL, 0, IMG / 2], [0, FOCAL, IMG / 2], [0, 0, 1]], dtype=np.float32)
POSE = np.eye(4, dtype=np.float32)
POSE[:3, 3] = [0.0, 0.0, -3.0]

DAMAGE_LEVELS = {
    "light": {"n_balls": 1, "radius": 0.15},
    "medium": {"n_balls": 1, "radius": 0.25},
    "heavy": {"n_balls": 2, "radius": 0.25},
}


def load_gt_points(glb_path, n=N_GT):
    mesh = trimesh.load(glb_path, force="scene")
    if not isinstance(mesh, trimesh.Scene):
        return None
    mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None
    if len(mesh.faces) > 3_000_000:
        return None
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    pts = pts.astype(np.float32)
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.linalg.norm(pts, axis=1).max()
    if scale < 1e-6:
        return None
    return (pts / scale).astype(np.float32)


def simulate_damage(gt, level, seed):
    rng = np.random.default_rng(seed)
    spec = DAMAGE_LEVELS[level]
    hole_mask = np.zeros(len(gt), dtype=bool)
    for _ in range(spec["n_balls"]):
        c = gt[rng.integers(0, len(gt))]
        d = np.linalg.norm(gt - c, axis=1)
        hole_mask |= d < spec["radius"]
    partial = gt[~hole_mask]
    hole_gt = gt[hole_mask]
    return partial.astype(np.float32), hole_gt.astype(np.float32), hole_mask


def splat_depth(points):
    """按 detector 约定投影：x=(u-cx)/fx*d, y=(v-cy)/fy*d, z=d"""
    cam = points - POSE[:3, 3]  # R=I
    z = cam[:, 2]
    valid = z > 0.1
    cam = cam[valid]
    z = z[valid]
    u = np.round(FOCAL * cam[:, 0] / z + IMG / 2).astype(int)
    v = np.round(FOCAL * cam[:, 1] / z + IMG / 2).astype(int)
    inside = (u >= 0) & (u < IMG) & (v >= 0) & (v < IMG)
    u, v, z = u[inside], v[inside], z[inside]
    depth = np.zeros((IMG, IMG), dtype=np.float32)
    # 每像素取最近 z
    order = np.argsort(-z)  # 远的先写，近的覆盖
    depth[v[order], u[order]] = z[order]
    return depth


def chamfer_l1(p, q):
    tp, tq = cKDTree(p), cKDTree(q)
    return 0.5 * (tp.query(q)[0].mean() + tq.query(p)[0].mean())


def f_score(p, q, tau=0.01):
    tp, tq = cKDTree(p), cKDTree(q)
    prec = (tp.query(q)[0] < tau).mean()
    rec = (tq.query(p)[0] < tau).mean()
    return float(2 * prec * rec / max(prec + rec, 1e-12))


def run_one(completer, gt, level, seed):
    partial, hole_gt, _ = simulate_damage(gt, level, seed)
    depth = splat_depth(partial)
    t0 = time.perf_counter()
    res = completer.complete(partial, depth, K, POSE, verbose=False)
    dt = time.perf_counter() - t0
    comp = res["complete"]
    return {
        "cd_l1": float(chamfer_l1(comp, gt)),
        "f_score": f_score(comp, gt),
        "hole_cd": float(chamfer_l1(res["completed"], hole_gt)) if len(hole_gt) > 0 else None,
        "removed_ratio": float(1 - len(partial) / len(gt)),
        "n_completed": int(len(res["completed"])),
        "final_loss": {k: v[-1] for k, v in res["loss_history"].items()},
        "runtime_s": round(dt, 2),
    }, res, partial, hole_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--out", default=r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\results_b1")
    ap.add_argument("--save_npz", type=int, default=6, help="前 N 个样本保存 npz")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    glbs = sorted(glob.glob(os.path.join(CACHE_GLB, "**", "*.glb"), recursive=True))
    print(f"缓存 GLB: {len(glbs)} 个，取前 {args.limit} 个尝试")

    cfg = DiffusionCompletionConfig(num_iters=args.iters, device="cuda")
    completer = DiffusionGuidedCompletion(cfg)

    results = []
    n_done = 0
    for i, g in enumerate(glbs):
        if n_done >= args.limit:
            break
        uid = os.path.splitext(os.path.basename(g))[0]
        try:
            gt = load_gt_points(g)
            if gt is None:
                print(f"[skip] {uid}: 网格不可用")
                continue
        except Exception as e:
            print(f"[skip] {uid}: 加载失败 {e}")
            continue

        for j, level in enumerate(DAMAGE_LEVELS):
            seed = hash((uid, level)) % (2**31)
            try:
                metrics, res, partial, hole_gt = run_one(completer, gt, level, seed)
            except Exception as e:
                print(f"[fail] {uid}/{level}: {e}")
                continue
            rec = {"uid": uid, "level": level, **metrics}
            results.append(rec)
            print(f"[{n_done:3d}] {uid[:8]}… {level:6s} "
                  f"CD={metrics['cd_l1']:.4f} F={metrics['f_score']:.3f} "
                  f"hole_CD={metrics['hole_cd']:.4f} rm={metrics['removed_ratio']:.2f} "
                  f"({metrics['runtime_s']}s)")
            if n_done < args.save_npz:
                np.savez_compressed(
                    os.path.join(args.out, f"sample_{n_done}_{level}.npz"),
                    gt=gt, partial=partial, hole_gt=hole_gt,
                    complete=res["complete"], completed=res["completed"])
        n_done += 1

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # 汇总
    for level in DAMAGE_LEVELS:
        sub = [r for r in results if r["level"] == level]
        if sub:
            cd = np.mean([r["cd_l1"] for r in sub])
            fs = np.mean([r["f_score"] for r in sub])
            print(f"== {level}: n={len(sub)} CD-L1={cd:.4f} F@0.01={fs:.3f}")
    print(f"完成 {len(results)} 次补全 → {args.out}")


if __name__ == "__main__":
    main()
