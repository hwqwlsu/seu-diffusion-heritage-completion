"""
数据集统计表：为论文 6.1 节提供真实、可复核的数据集描述

输出：
  dataset_stats.json  —— 结构化统计（供绘图/表格引用）
  dataset_stats.md    —— 可直接粘贴进论文的表格
统计内容：
  - 每个数据集样本数、类别分布
  - 网格面数 / 顶点数 的均值、中位数、P95
  - 包围盒尺度、是否闭合、水密性
"""
import glob
import json
import os
import statistics as st

import numpy as np
import trimesh

WS = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01"
CACHE = r"C:\Users\黄文全\.cache\.objaverse\hf-objaverse-v1\glbs"


def glb_index():
    idx = {}
    for fp in glob.glob(os.path.join(CACHE, "*", "*.glb")):
        idx[os.path.splitext(os.path.basename(fp))[0]] = fp
    return idx


def load_uids(dataset):
    if dataset == "shrec":
        raw = json.load(open(os.path.join(WS, "shrec2021_alternative_uids.json"), encoding="utf-8"))
        uids = raw.get("core_subset_uids") or raw.get("all_ceramic_uids") or []
        cat = {"cat": "陶瓷器（SHREC 2021 替代集）", "uids": uids}
        return [cat]
    m = json.load(open(os.path.join(WS, "ch_single3d_manifest.json"), encoding="utf-8"))
    return [{"cat": k, "uids": [it["uid"] for it in v]} for k, v in m["categories"].items()]


def mesh_stats(path):
    scene = trimesh.load(path, force="scene")
    if isinstance(scene, trimesh.Scene):
        try:
            mesh = scene.to_geometry()
        except Exception:
            mesh = scene.dump(concatenate=True)
    else:
        mesh = scene
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        return None
    ext = mesh.bounds[1] - mesh.bounds[0]
    return {
        "faces": int(len(mesh.faces)),
        "verts": int(len(mesh.vertices)),
        "bbox_max": float(np.max(ext)),
        "watertight": bool(mesh.is_watertight),
        "volume": float(abs(mesh.volume)) if mesh.is_watertight else None,
    }


def summarize(rows):
    if not rows:
        return {}
    f = [r["faces"] for r in rows if r]
    v = [r["verts"] for r in rows if r]
    b = [r["bbox_max"] for r in rows if r]
    rows = [r for r in rows if r]
    return {
        "n_meshes": len(rows),
        "faces_mean": int(np.mean(f)),
        "faces_median": int(np.median(f)),
        "faces_p95": int(np.percentile(f, 95)),
        "verts_mean": int(np.mean(v)),
        "verts_median": int(np.median(v)),
        "bbox_mean": round(float(np.mean(b)), 4),
        "watertight_ratio": round(float(np.mean([r["watertight"] for r in rows])), 3),
    }


def main():
    idx = glb_index()
    out = {}
    for dataset in ("shrec", "ch3d"):
        out[dataset] = {}
        for group in load_uids(dataset):
            rows = []
            missing = 0
            for u in group["uids"]:
                p = idx.get(str(u).strip())
                if not p:
                    missing += 1
                    continue
                try:
                    rows.append(mesh_stats(p))
                except Exception:
                    missing += 1
            s = summarize(rows)
            s["declared"] = len(group["uids"])
            s["missing"] = missing
            out[dataset][group["cat"]] = s
            print(f"[{dataset}] {group['cat']}: n={s.get('n_meshes')} "
                  f"faces均值={s.get('faces_mean')} 水密={s.get('watertight_ratio')}", flush=True)

    json.dump(out, open(os.path.join(WS, "dataset_stats.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # 生成 Markdown 表
    lines = []
    lines.append("| 数据集 | 类别 | 样本数 | 面数(均值) | 面数(中位数) | 面数(P95) | 顶点(均值) | 归一化尺度 | 水密比例 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    label = {"shrec": "SHREC 2021 替代集", "ch3d": "CH-Single3D（自建）"}
    for ds, cats in out.items():
        for cat, s in cats.items():
            if not s:
                continue
            lines.append(
                f"| {label[ds]} | {cat} | {s['n_meshes']} | {s['faces_mean']:,} | "
                f"{s['faces_median']:,} | {s['faces_p95']:,} | {s['verts_mean']:,} | "
                f"{s['bbox_mean']} | {s['watertight_ratio']:.1%} |")
    # 合计行
    for ds, cats in out.items():
        tot = sum(s["n_meshes"] for s in cats.values() if s)
        fm = int(np.mean([s["faces_mean"] for s in cats.values() if s]))
        lines.append(f"| **{label[ds]} 合计** | — | **{tot}** | {fm:,} | — | — | — | — | — |")

    open(os.path.join(WS, "dataset_stats.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
