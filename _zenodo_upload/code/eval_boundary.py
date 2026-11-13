"""离线评估洞边界检测：密度判据 vs 角度缺口判据（仅用于算法选型，不进入方法）。"""
import sys, zlib
sys.path.insert(0, r"G:\大博士论文资料\代码")
import numpy as np
from scipy.spatial import cKDTree
import run_exp_b3 as R
from modules.diffusion_completion.damaged_region_detector import DamagedRegionDetector


def density_boundary(pts, k_spacing=9, r_factor=3.0, ratio_thr=0.6):
    tree = cKDTree(pts)
    dd, _ = tree.query(pts, k=k_spacing, workers=-1)
    spacing = float(np.median(dd[:, 1:].mean(axis=1)))
    r = r_factor * spacing
    cnt = np.array([len(x) for x in tree.query_ball_point(pts, r, workers=-1)])
    med = np.median(cnt)
    return cnt < ratio_thr * med, spacing


def main():
    data = R.load_dataset("ch3d")[:3]
    det = DamagedRegionDetector()
    for uid, gp in data:
        gt = R.load_gt_points(gp)
        bm_d, sp = density_boundary(gt)
        bm_a = det._geometric_boundary(gt)

        seed = zlib.crc32(f"{uid}|0.4".encode())
        partial, hole_gt, hole, ratio, rad = R.damage_to_ratio(gt, 0.4, seed)
        d_hole = cKDTree(hole_gt).query(partial)[0]
        true_edge = d_hole < 0.03

        for name, bm in [("角度", det._geometric_boundary(partial)), ("密度", density_boundary(partial)[0])]:
            bp = partial[bm]
            prec = (cKDTree(partial[true_edge]).query(bp)[0] < 0.04).mean() if len(bp) else 0
            rec = (cKDTree(bp).query(partial[true_edge])[0] < 0.04).mean() if len(bp) else 0
            print(f"{uid[:8]} {name}: 完整假边界={bm_d.sum() if name=='密度' else bm_a.sum()} "
                  f"partial边界={len(bp)} 真实边缘={true_edge.sum()} "
                  f"P={100*prec:.1f}% R={100*rec:.1f}%", flush=True)
        print(f"   点间距={sp:.4f}", flush=True)


if __name__ == "__main__":
    main()
