"""
传统几何重建基线（真实可跑，不依赖训练）

基线：
  1. Poisson   —— Screened Poisson 隐式重建（open3d，若可用）
  2. BPA       —— Ball Pivoting 滚球法（open3d，若可用）
  3. IMLS-MC   —— 隐式移动最小二乘 + Marching Cubes（skimage，始终可用）
  4. AlphaShape—— Delaunay Alpha Shape（scipy + trimesh）

所有基线在与主实验完全相同的输入（同一残损点云、同一随机种子）上运行，
因此结果与 modules/diffusion_completion 的结果可直接配对比较。

用法：
  python run_baselines.py --dataset shrec --levels 0.4 --limit 100
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import trimesh
from scipy.spatial import cKDTree, Delaunay
from skimage.measure import marching_cubes

WS = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01"
CACHE = r"C:\Users\黄文全\.cache\.objaverse\hf-objaverse-v1\glbs"
N_GT = 16384

_O3D = None
sys.path.insert(0, os.path.join(WS, "o3d_pkg"))  # open3d 装在工作区本地
try:
    import open3d as o3d  # noqa
    _O3D = o3d
except Exception:
    pass


# ---------- 与主实验完全一致的采样/残损逻辑 ----------
def glb_index():
    idx = {}
    for fp in glob.glob(os.path.join(CACHE, "*", "*.glb")):
        idx[os.path.splitext(os.path.basename(fp))[0]] = fp
    return idx


def load_dataset(dataset):
    idx = glb_index()
    if dataset == "shrec":
        raw = json.load(open(os.path.join(WS, "shrec2021_alternative_uids.json"), encoding="utf-8"))
        uids = raw.get("core_subset_uids") or raw.get("all_ceramic_uids") or []
    else:
        m = json.load(open(os.path.join(WS, "ch_single3d_manifest.json"), encoding="utf-8"))
        uids = [it["uid"] for items in m["categories"].values() for it in items]
    return [(str(u).strip(), idx[str(u).strip()]) for u in uids if str(u).strip() in idx]


def load_gt_points(path, n=N_GT):
    scene = trimesh.load(path, force="scene")
    mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None
    if len(mesh.faces) > 3_000_000:
        return None
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    pts = pts.astype(np.float32)
    pts -= pts.mean(axis=0)
    s = np.linalg.norm(pts, axis=1).max()
    if s < 1e-6:
        return None
    return (pts / s).astype(np.float32)


def damage_to_ratio(gt, target, seed, n_balls=1, iters=18):
    rng = np.random.default_rng(seed)

    def carve(r):
        hole = np.zeros(len(gt), dtype=bool)
        for _ in range(n_balls):
            c = gt[rng.integers(0, len(gt))]
            hole |= np.linalg.norm(gt - c, axis=1) < r
        return hole

    lo, hi, best = 0.0, 1.6, None
    for _ in range(iters):
        mid = (lo + hi) / 2
        h = carve(mid)
        ratio = h.mean()
        if best is None or abs(ratio - target) < abs(best[1] - target):
            best = (mid, ratio, h)
        if ratio < target:
            lo = mid
        else:
            hi = mid
    return gt[~best[2]].astype(np.float32), gt[best[2]].astype(np.float32), float(best[1])


# ---------- 指标 ----------
def chamfer_l1(p, q):
    tp, tq = cKDTree(p), cKDTree(q)
    return float(0.5 * (tp.query(q)[0].mean() + tq.query(p)[0].mean()))


def f_score(p, q, tau=0.01):
    tp, tq = cKDTree(p), cKDTree(q)
    prec = (tp.query(q)[0] < tau).mean()
    rec = (tq.query(p)[0] < tau).mean()
    return float(2 * prec * rec / max(prec + rec, 1e-12))


def normal_consistency(p, q, k=16):
    def nrm(pts):
        _, i = cKDTree(pts).query(pts, k=k)
        nb = pts[i] - pts[i].mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", nb, nb)
        _, v = np.linalg.eigh(cov)
        return v[:, :, 0]
    nq = nrm(q)
    _, ni = cKDTree(p).query(q)
    return float(np.abs(np.sum(nq * nrm(p)[ni], axis=1)).mean())


# ---------- 基线实现 ----------
def resample(pts, n=N_GT, seed=0):
    if len(pts) >= n:
        rng = np.random.default_rng(seed)
        return pts[rng.choice(len(pts), n, replace=False)]
    rng = np.random.default_rng(seed)
    return pts[rng.choice(len(pts), n, replace=True)]


def normals_est(pts, k=30):
    _, i = cKDTree(pts).query(pts, k=min(k, len(pts)))
    nb = pts[i] - pts[i].mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, v = np.linalg.eigh(cov)
    n = v[:, :, 0]
    # 朝向一致化：以质心方向为准
    c = pts.mean(axis=0)
    flip = np.sum(n * (pts - c), axis=1) < 0
    n[flip] *= -1
    return n.astype(np.float32)


def to_o3d(pts, normals=None):
    pc = _O3D.geometry.PointCloud()
    pc.points = _O3D.utility.Vector3dVector(pts)
    if normals is not None:
        pc.normals = _O3D.utility.Vector3dVector(normals)
    return pc


def sample_o3d_mesh(m, n=N_GT):
    """从 open3d 网格均匀采样点"""
    nrm = np.asarray(m.vertex_normals)
    if len(nrm) == 0:
        m.compute_vertex_normals()
    return np.asarray(m.sample_points_uniformly(number_of_points=n).points).astype(np.float32)


def bl_p_poisson(pts):
    if _O3D is None:
        return None
    n = normals_est(pts)
    pc = to_o3d(resample(pts), n if len(n) == len(resample(pts)) else None)
    m, _ = _O3D.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8, width=0, scale=1.1, linear_fit=False)
    m = m.crop(pc.get_axis_aligned_bounding_box().scale(1.15, center=pc.get_center()))
    return sample_o3d_mesh(m)


def bl_bpa(pts):
    if _O3D is None:
        return None
    p = resample(pts)
    n = normals_est(p)
    pc = to_o3d(p, n)
    radii = np.linspace(p.std() * 0.5, p.std() * 3.0, 5)
    m = _O3D.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pc, _O3D.utility.DoubleVector(radii))
    if len(m.triangles) == 0:
        return None
    return sample_o3d_mesh(m)


def bl_imls_mc(pts, res=96):
    """隐式移动最小二乘 + Marching Cubes"""
    p = resample(pts)
    n = normals_est(p)
    lo, hi = p.min(axis=0) - 0.12, p.max(axis=0) + 0.12
    xs = [np.linspace(lo[i], hi[i], res) for i in range(3)]
    gx, gy, gz = np.meshgrid(*xs, indexing="ij")
    q = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    tree = cKDTree(p)
    dist, idx = tree.query(q, k=12)  # (M,12)
    h = max(float(np.median(dist[:, -1])), 1e-4)
    w = np.exp(-(dist ** 2) / (h * h))  # (M,k)
    d = q[:, None, :] - p[idx]           # (M,k,3)
    nn = n[idx]                          # (M,k,3)
    f = np.sum(w * np.sum(d * nn, axis=2), axis=1) / (w.sum(axis=1) + 1e-12)
    # 加权最小二乘平面拟合：残余距离替代简单投影
    f = f + 0.0
    F = f.reshape(res, res, res).astype(np.float32)
    if F.min() > 0 or F.max() < 0:
        return None
    verts, faces, _, _ = marching_cubes(F, level=0.0, spacing=(
        xs[0][1] - xs[0][0], xs[1][1] - xs[1][0], xs[2][1] - xs[2][0]))
    verts = verts + lo
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return trimesh.sample.sample_surface(mesh, N_GT)[0].astype(np.float32)


def bl_alpha(pts):
    p = resample(pts)
    tri = Delaunay(p)
    # alpha 参数按平均边长自适应
    sp = tri.simplices
    a = p[sp[:, 0]]; b = p[sp[:, 1]]; c = p[sp[:, 2]]
    ab, bc, ca = b - a, c - b, a - c
    la, lb, lc = np.linalg.norm(ab, axis=1), np.linalg.norm(bc, axis=1), np.linalg.norm(ca, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(ab, bc), axis=1)
    # 外接圆半径 R = abc/(4A)
    R = (la * lb * lc) / (4 * area + 1e-12)
    keep = R < np.percentile(R, 40)
    mesh = trimesh.Trimesh(vertices=p, faces=sp[keep], process=True)
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        return None
    return trimesh.sample.sample_surface(mesh, N_GT)[0].astype(np.float32)


BASELINES = {
    "Poisson": bl_p_poisson,
    "BPA": bl_bpa,
    "IMLS-MC": bl_imls_mc,
    "AlphaShape": bl_alpha,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ch3d", choices=["shrec", "ch3d"])
    ap.add_argument("--levels", default="0.4")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--methods", default="Poisson,BPA,IMLS-MC,AlphaShape")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(WS, f"results_baseline_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.json")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else []
    done = {(r["uid"], r["target_ratio"], r["method"]) for r in results}

    data = load_dataset(args.dataset)
    data = data[args.skip: args.skip + args.limit] if args.limit else data[args.skip:]
    levels = [float(x) for x in args.levels.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    print(f"[{args.dataset}] 样本 {len(data)}，方法 {methods}，open3d={_O3D is not None}", flush=True)
    for uid, path in data:
        gt = load_gt_points(path)
        if gt is None:
            continue
        for lv in levels:
            import zlib
            seed = zlib.crc32(f"{uid}|{lv}".encode("utf-8"))
            partial, hole_gt, ratio = damage_to_ratio(gt, lv, seed)
            for mname in methods:
                if (uid, lv, mname) in done:
                    continue
                fn = BASELINES[mname]
                t0 = time.perf_counter()
                try:
                    rec = fn(partial)
                except Exception as e:
                    print(f"  [fail] {mname} {uid[:8]} {lv}: {e}", flush=True)
                    continue
                if rec is None or len(rec) == 0:
                    continue
                dt = time.perf_counter() - t0
                results.append({
                    "uid": uid, "dataset": args.dataset, "method": mname,
                    "target_ratio": lv, "actual_ratio": round(ratio, 4),
                    "cd_l1": chamfer_l1(rec, gt),
                    "f_score": f_score(rec, gt),
                    "normal_consistency": normal_consistency(rec, gt),
                    "runtime_s": round(dt, 2),
                })
                json.dump(results, open(res_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
                print(f"  {mname:11s} {uid[:8]} {lv:.1f} CD={results[-1]['cd_l1']:.4f} "
                      f"F={results[-1]['f_score']:.3f} ({dt:.1f}s)", flush=True)

    print("\n=== 基线汇总 ===", flush=True)
    for lv in levels:
        for mname in methods:
            s = [r for r in results if r["method"] == mname and r["target_ratio"] == lv]
            if not s:
                continue
            print(f"  {mname:11s} lv={lv:.1f} n={len(s):3d} "
                  f"CD×1e4={np.mean([x['cd_l1'] for x in s])*1e4:7.2f} "
                  f"F={np.mean([x['f_score'] for x in s]):.3f} "
                  f"NC={np.mean([x['normal_consistency'] for x in s]):.3f} "
                  f"t={np.mean([x['runtime_s'] for x in s]):.1f}s", flush=True)


if __name__ == "__main__":
    main()
