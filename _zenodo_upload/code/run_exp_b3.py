"""
B3 主实验：在 SHREC 298 / CH-Single3D 287 上跑模块1（扩散先验残损补全）

与 B1 的区别：
  1. 残损率按论文口径显式标定到目标值（20% / 40%），而不是用固定球半径
     → 用二分搜索球半径，使实际 removed_ratio 逼近目标
  2. 支持从 manifest 读取数据集（SHREC / CH-Single3D）
  3. 支持消融：通过 --variant 关闭某个损失分量

用法：
  python run_exp_b3.py --dataset shrec --levels 0.2,0.4 --limit 298
  python run_exp_b3.py --dataset ch3d --levels 0.2,0.4 --limit 287
  python run_exp_b3.py --dataset ch3d --levels 0.4 --limit 100 --variant no_diff
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, r"G:\大博士论文资料\代码")
from modules.diffusion_completion import DiffusionCompletionConfig, DiffusionGuidedCompletion  # noqa: E402

WS = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01"
CACHE_GLB = r"C:\Users\黄文全\.cache\.objaverse\hf-objaverse-v1\glbs"
N_GT = 16384
IMG = 448
FOCAL = 500.0
K = np.array([[FOCAL, 0, IMG / 2], [0, FOCAL, IMG / 2], [0, 0, 1]], dtype=np.float32)
POSE = np.eye(4, dtype=np.float32)
POSE[:3, 3] = [0.0, 0.0, -3.0]

# 消融变体：对应论文表 2 / 图 B3
VARIANTS = {
    "full": {},
    "no_diff": {"diff_prior_weight": 0.0},
    "no_bnd": {"boundary_weight": 0.0},
    "no_smooth": {"smoothness_weight": 0.0},
    "no_warmup": {"warmup_iters": 0},
}


def build_glb_index():
    idx = {}
    for fp in glob.glob(os.path.join(CACHE_GLB, "*", "*.glb")):
        uid = os.path.splitext(os.path.basename(fp))[0]
        idx[uid] = fp
    return idx


def load_dataset(dataset):
    """返回 [(uid, glb_path), ...]"""
    glb_idx = build_glb_index()
    if dataset == "shrec":
        with open(os.path.join(WS, "shrec2021_alternative_uids.json"), encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            uids = raw.get("core_subset_uids") or raw.get("all_ceramic_uids") or []
        else:
            uids = raw
    elif dataset == "ch3d":
        with open(os.path.join(WS, "ch_single3d_manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        uids = [it["uid"] for items in m["categories"].values() for it in items]
    else:
        raise ValueError(dataset)
    out = []
    for u in uids:
        u = str(u).strip()
        p = glb_idx.get(u)
        if p:
            out.append((u, p))
    return out


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
    pts -= pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale < 1e-6:
        return None
    return (pts / scale).astype(np.float32)


def damage_with_radius(gt, radius, seed, n_balls=1):
    rng = np.random.default_rng(seed)
    hole = np.zeros(len(gt), dtype=bool)
    center = gt[0]
    for _ in range(n_balls):
        c = gt[rng.integers(0, len(gt))]
        center = c
        hole |= np.linalg.norm(gt - c, axis=1) < radius
    return hole, center


def damage_to_ratio(gt, target_ratio, seed, n_balls=1, iters=18):
    """二分搜索半径，使被挖除比例逼近 target_ratio"""
    lo, hi = 0.0, 1.6
    best = None
    for _ in range(iters):
        mid = (lo + hi) / 2
        hole, c = damage_with_radius(gt, mid, seed, n_balls)
        ratio = hole.mean()
        if best is None or abs(ratio - target_ratio) < abs(best[1] - target_ratio):
            best = (mid, ratio, hole, c)
        if ratio < target_ratio:
            lo = mid
        else:
            hi = mid
    radius, ratio, hole, center = best
    return gt[~hole].astype(np.float32), gt[hole].astype(np.float32), hole, float(ratio), radius, center


def splat_depth(points):
    cam = points - POSE[:3, 3]
    z = cam[:, 2]
    valid = z > 0.1
    cam, z = cam[valid], z[valid]
    u = np.round(FOCAL * cam[:, 0] / z + IMG / 2).astype(int)
    v = np.round(FOCAL * cam[:, 1] / z + IMG / 2).astype(int)
    inside = (u >= 0) & (u < IMG) & (v >= 0) & (v < IMG)
    u, v, z = u[inside], v[inside], z[inside]
    depth = np.zeros((IMG, IMG), dtype=np.float32)
    order = np.argsort(-z)
    depth[v[order], u[order]] = z[order]
    return depth


def chamfer_l1(p, q):
    tp, tq = cKDTree(p), cKDTree(q)
    return float(0.5 * (tp.query(q)[0].mean() + tq.query(p)[0].mean()))


def f_score(p, q, tau=0.01):
    tp, tq = cKDTree(p), cKDTree(q)
    prec = (tp.query(q)[0] < tau).mean()
    rec = (tq.query(p)[0] < tau).mean()
    return float(2 * prec * rec / max(prec + rec, 1e-12))


def trim_outliers(points, k=8, sigma=2.0, min_keep=0.5):
    """离群裁剪：剔除 kNN 平均间距显著高于中位数的漂浮点（稳健 MAD 阈值）。

    仅作用于扩散生成/优化的补全点，可见观测点一律保留。
    """
    n = len(points)
    if n < k + 5:
        return points, np.ones(n, dtype=bool)
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k + 1)
    dk = d[:, 1:].mean(axis=1)
    med = float(np.median(dk))
    mad = float(np.median(np.abs(dk - med)) + 1e-12)
    thr = med + sigma * 1.4826 * mad
    keep = dk <= thr
    # 防过度裁剪：至少保留 min_keep 比例
    if keep.sum() < max(int(n * min_keep), k + 5):
        keep = dk <= float(np.percentile(dk, 100 * min_keep))
    return points[keep], keep


def merge_resample(visible, missing, n_out, seed):
    """可见点 + 补全点 合并后重采样到固定输出点数（与主流程口径一致）"""
    comp = np.concatenate([visible, missing], axis=0)
    rng = np.random.default_rng(seed)
    if len(comp) >= n_out:
        idx = rng.choice(len(comp), n_out, replace=False)
    else:
        idx = rng.choice(len(comp), n_out, replace=True)
    return comp[idx]


def f_score_pr(p, q, tau=0.01):
    """返回 (precision, recall, f1)"""
    tp, tq = cKDTree(p), cKDTree(q)
    prec = float((tp.query(q)[0] < tau).mean())
    rec = float((tq.query(p)[0] < tau).mean())
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return prec, rec, float(f1)


def normal_consistency(p, q, k=16):
    """两点云法线一致性（用PCA法线近似）"""
    def est_normals(pts):
        tree = cKDTree(pts)
        _, idx = tree.query(pts, k=k)
        nb = pts[idx]  # (N,k,3)
        nb = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb)
        w, v = np.linalg.eigh(cov)
        return v[:, :, 0]
    nq = est_normals(q)
    tp = cKDTree(p)
    _, ni = tp.query(q)
    np_ = est_normals(p)[ni]
    dot = np.abs(np.sum(nq * np_, axis=1))
    return float(dot.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["shrec", "ch3d"], required=True)
    ap.add_argument("--levels", default="0.2,0.4")
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--variant", default="full", choices=list(VARIANTS.keys()))
    ap.add_argument("--out", default="")
    ap.add_argument("--skip", type=int, default=0, help="跳过前 N 个（断点续跑）")
    ap.add_argument("--prior_ckpt", default="", help="训练好的扩散先验权重路径")
    ap.add_argument("--trim_sigma", type=float, default=0.0,
                    help=">0 时对扩散补全点做离群裁剪，值为 MAD 阈值倍数（建议 2.0~3.0）")
    ap.add_argument("--trim_k", type=int, default=8, help="离群判定用的 kNN 邻居数")
    ap.add_argument("--mask_mode", choices=["known", "auto"], default="known",
                    help="known=残损区域由掩码给出（合成基准标准设定）；auto=由检测器自动判定")
    ap.add_argument("--diff_weight", type=float, default=-1.0,
                    help="覆盖扩散先验权重（>=0 生效）")
    ap.add_argument("--denoise_ts", type=float, default=-1.0,
                    help="覆盖去噪时间步 t/T（>0 生效）")
    ap.add_argument("--prior_init", action="store_true",
                    help="用扩散先验单步 x0 预测作为缺失点初值（替代随机空腔初始化）")
    ap.add_argument("--prior_init_t", type=float, default=0.3,
                    help="x0 预测所用加噪时间步 t/T")
    ap.add_argument("--prior_init_adaptive", type=float, default=-1.0,
                    help=">=0 时按实际残损率自适应启用 prior_init（actual_ratio >= 阈值才启用），"
                         "阈值由 CH3D 交叉实验标定为 0.35；覆盖 --prior_init")
    args = ap.parse_args()

    levels = [float(x) for x in args.levels.split(",")]
    tag = args.variant
    if args.prior_init:
        tag += f"_pinit{args.prior_init_t:g}"
    if args.prior_init_adaptive >= 0:
        tag += f"_adapt{args.prior_init_adaptive:g}"
    out_dir = args.out or os.path.join(WS, f"results_b3_{args.dataset}_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.json")

    data = load_dataset(args.dataset)
    print(f"[{args.dataset}] 可用样本 {len(data)}，variant={args.variant}", flush=True)
    # 分片：先 skip 再 limit，便于多进程切片并行
    data = data[args.skip : args.skip + args.limit] if args.limit else data[args.skip:]

    cfg = DiffusionCompletionConfig(num_iters=args.iters, device="cuda")
    if args.prior_ckpt:
        cfg.prior_ckpt = args.prior_ckpt
    for k, v in VARIANTS[args.variant].items():
        setattr(cfg, k, v)
    if args.diff_weight >= 0:
        cfg.diff_prior_weight = args.diff_weight
    if args.denoise_ts > 0:
        cfg.denoise_timestep = args.denoise_ts
    completer = DiffusionGuidedCompletion(cfg)

    # 断点续跑
    results = []
    done_keys = set()
    if os.path.exists(res_path):
        try:
            results = json.load(open(res_path, encoding="utf-8"))
            done_keys = {(r["uid"], r["target_ratio"]) for r in results}
            print(f"已有结果 {len(results)} 条，继续", flush=True)
        except Exception:
            pass

    t_start = time.time()
    n_done = 0
    for uid, gpath in data:
        try:
            gt = load_gt_points(gpath)
            if gt is None:
                continue
        except Exception as e:
            print(f"[skip] {uid[:8]} 加载失败 {e}", flush=True)
            continue
        for level in levels:
            if (uid, level) in done_keys:
                continue
            # 种子只依赖 (uid, level)，与 variant 无关：保证消融对比使用完全相同的挖洞模式
            # 用 crc32 而非内置 hash，避免跨进程随机化导致不可复现
            seed = zlib.crc32(f"{uid}|{level}".encode("utf-8"))
            try:
                partial, hole_gt, hole, ratio, radius, hole_center = damage_to_ratio(
                    gt, level, seed)
                depth = splat_depth(partial)
                t0 = time.perf_counter()
                # 按实际残损率自适应启用先验初值：低残损率下先验偏差大于收益（CD 劣化），
                # 高残损率（>= 阈值，CH3D 交叉实验标定 0.35）下初值带来结构性增益
                use_init = args.prior_init
                if args.prior_init_adaptive >= 0:
                    use_init = ratio >= args.prior_init_adaptive
                if args.mask_mode == "known":
                    n_miss = int(round(len(gt) * ratio))
                    res = completer.complete(partial, depth, K, POSE, verbose=False,
                                             missing_hint=(hole_center, radius, n_miss),
                                             prior_init=use_init,
                                             prior_init_t=args.prior_init_t)
                else:
                    res = completer.complete(partial, depth, K, POSE, verbose=False)
                dt = time.perf_counter() - t0
                comp = res["complete"]
                trim_info = {}
                if args.trim_sigma > 0:
                    kept, keep_mask = trim_outliers(res["completed"], k=args.trim_k,
                                                    sigma=args.trim_sigma)
                    comp = merge_resample(res["visible"], kept, len(res["complete"]), seed)
                    trim_info = {
                        "trim_ratio": round(float(1.0 - keep_mask.mean()), 4),
                        "n_kept": int(keep_mask.sum()),
                    }
                prec, rec_, _ = f_score_pr(comp, gt)
                rec = {
                    "uid": uid,
                    "dataset": args.dataset,
                    "variant": args.variant,
                    "mask_mode": args.mask_mode,
                    "prior_init": bool(use_init),
                    "prior_init_t": args.prior_init_t if use_init else None,
                    "diff_prior_weight": cfg.diff_prior_weight,
                    "target_ratio": level,
                    "actual_ratio": round(ratio, 4),
                    "radius": round(radius, 4),
                    "cd_l1": chamfer_l1(comp, gt),
                    "f_score": f_score(comp, gt),
                    "precision": round(prec, 4),
                    "recall": round(rec_, 4),
                    "normal_consistency": normal_consistency(comp, gt),
                    "hole_cd": chamfer_l1(res["completed"], hole_gt) if len(hole_gt) else None,
                    "n_completed": int(len(res["completed"])),
                    "runtime_s": round(dt, 2),
                }
                rec.update(trim_info)
                results.append(rec)
                n_done += 1
                print(f"[{n_done}] {uid[:8]} r={ratio:.1%} CD={rec['cd_l1']:.4f} "
                      f"F={rec['f_score']:.3f} NC={rec['normal_consistency']:.3f} "
                      f"({dt:.1f}s)", flush=True)
                if n_done % 10 == 0:
                    json.dump(results, open(res_path, "w", encoding="utf-8"),
                              ensure_ascii=False, indent=1)
            except Exception as e:
                print(f"[fail] {uid[:8]}/{level}: {e}", flush=True)

    json.dump(results, open(res_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 汇总
    print("\n=== 汇总 ===", flush=True)
    for level in levels:
        sub = [r for r in results if r["target_ratio"] == level]
        if not sub:
            continue
        cd = np.mean([r["cd_l1"] for r in sub]) * 1e4
        fs = np.mean([r["f_score"] for r in sub])
        nc = np.mean([r["normal_consistency"] for r in sub])
        ar = np.mean([r["actual_ratio"] for r in sub])
        print(f"残损{level:.0%}: n={len(sub)} 实际={ar:.1%} CD×1e4={cd:.2f} "
              f"F={fs:.3f} NC={nc:.3f}", flush=True)
    print(f"总耗时 {(time.time()-t_start)/60:.1f} min → {res_path}", flush=True)


if __name__ == "__main__":
    main()
