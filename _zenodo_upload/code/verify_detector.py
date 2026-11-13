"""验证修复后的残损检测：主洞定位、初始化点位置（离线用 GT 掩码检查，不进入方法）。"""
import sys, zlib
sys.path.insert(0, r"G:\大博士论文资料\代码")
import numpy as np
from scipy.spatial import cKDTree
import run_exp_b3 as R
from modules.diffusion_completion.damaged_region_detector import DamagedRegionDetector


def main():
    data = R.load_dataset("ch3d")[:5]
    det = DamagedRegionDetector(visibility_threshold=0.5, boundary_knn=16)
    for uid, gp in data:
        gt = R.load_gt_points(gp)
        for level in (0.2, 0.4):
            seed = zlib.crc32(f"{uid}|{level}".encode())
            partial, hole_gt, hole, ratio, rad = R.damage_to_ratio(gt, level, seed)
            d = det.detect(partial, R.splat_depth(partial), R.K, R.POSE)
            mi = d["missing_init"]
            dh = cKDTree(gt[hole]).query(mi)[0]
            dv = cKDTree(partial).query(mi)[0]
            print(f"{uid[:8]} r={level:.0%} mode={d['detect_mode']} nb={d['n_boundary']} "
                  f"n_missing={len(mi)} 洞内<0.03={100*(dh<0.03).mean():.1f}% "
                  f"距洞中位={np.median(dh):.3f} 距观测中位={np.median(dv):.3f}", flush=True)


if __name__ == "__main__":
    main()
