"""参数扫描：为残损区域检测选最优判据（离线用 GT 掩码评估，仅用于选型）。"""
import sys, zlib
sys.path.insert(0, r"G:\大博士论文资料\代码")
import numpy as np
from scipy.spatial import cKDTree
import run_exp_b3 as R
from modules.diffusion_completion.damaged_region_detector import DamagedRegionDetector


def knn_metric(pts, k=16):
    tree = cKDTree(pts)
    dd, _ = tree.query(pts, k=k + 1, workers=-1)
    return dd[:, 1:].mean(axis=1), float(np.median(dd[:, 1:].mean(axis=1)))


def main():
    data = R.load_dataset("ch3d")[:6]
    det = DamagedRegionDetector()
    rows = []
    for uid, gp in data:
        gt = R.load_gt_points(gp)
        for level in (0.2, 0.4):
            seed = zlib.crc32(f"{uid}|{level}".encode())
            partial, hole_gt, hole, ratio, rad = R.damage_to_ratio(gt, level, seed)
            spacing = float(np.median(cKDTree(partial).query(partial, k=9)[0][:, 1:].mean(axis=1)))
            d_hole = cKDTree(hole_gt).query(partial)[0]
            true_edge = d_hole < 0.03
            hole_center = gt[hole].mean(axis=0)
            hole_R = float(np.median(np.linalg.norm(gt[hole] - hole_center, axis=1)))
            dk, dk_med = knn_metric(partial, 16)

            for thr in (1.15, 1.3, 1.45):
                bm = dk > thr * dk_med
                bp = partial[bm]
                if len(bp) < 8 or true_edge.sum() == 0:
                    continue
                prec = (cKDTree(partial[true_edge]).query(bp)[0] < 0.04).mean()
                rec = (cKDTree(bp).query(partial[true_edge])[0] < 0.04).mean()
                ring, _ = det._locate_main_hole(partial, bp, spacing)
                c = np.median(ring, axis=0)
                dcenter = float(np.linalg.norm(c - hole_center)) / hole_R
                rows.append((f"knn{thr}", len(bp), prec, rec, dcenter))

            for thr in (0.35, 0.45):
                bm, _ = det._density_boundary(partial, ratio_thr=thr)
                bp = partial[bm]
                if len(bp) < 8 or true_edge.sum() == 0:
                    continue
                prec = (cKDTree(partial[true_edge]).query(bp)[0] < 0.04).mean()
                rec = (cKDTree(bp).query(partial[true_edge])[0] < 0.04).mean()
                ring, _ = det._locate_main_hole(partial, bp, spacing)
                c = np.median(ring, axis=0)
                dcenter = float(np.linalg.norm(c - hole_center)) / hole_R
                rows.append((f"dens{thr}", len(bp), prec, rec, dcenter))

    methods = sorted({r[0] for r in rows})
    print(f"{'method':12s} {'n':>4s} {'P':>6s} {'R':>6s} {'center_err/R':>12s} {'命中(<0.4R)':>10s}")
    for m in methods:
        sub = [r for r in rows if r[0] == m]
        print(f"{m:12s} {np.mean([r[1] for r in sub]):4.0f} "
              f"{100*np.mean([r[2] for r in sub]):5.1f}% {100*np.mean([r[3] for r in sub]):5.1f}% "
              f"{np.mean([r[4] for r in sub]):12.3f} "
              f"{100*np.mean([r[4] < 0.4 for r in sub]):9.0f}%", flush=True)


if __name__ == "__main__":
    main()
