"""
CH-Single3D 自建数据集构建脚本（v2）

合并两个来源：
1. Objaverse UK + CH 文化遗产 UID（按名称关键词分类）
2. Objaverse LVIS 标注 UID（按 LVIS 类别直接映射到 CH-Single3D 类别）

LVIS 类别优先；关键词作为补充。
去重后检查 GLB 缓存，按目标分布选取 287 样本。

目标分布（论文 7.1.2）：
  石窟造像 65、古建筑构件 72、青铜器 58、陶瓷器 47、玉器 23、壁画 22
"""
import glob
import gzip
import json
import os
import random

_CACHE_HOME = r"C:\Users\黄文全\.cache"
os.environ["HOME"] = _CACHE_HOME
os.environ["USERPROFILE"] = _CACHE_HOME
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import objaverse
objaverse.BASE_PATH = os.path.join(_CACHE_HOME, ".objaverse")
objaverse._VERSIONED_PATH = os.path.join(objaverse.BASE_PATH, "hf-objaverse-v1")

META_DIR = os.path.join(objaverse._VERSIONED_PATH, "metadata")
GLB_ROOT = os.path.join(objaverse._VERSIONED_PATH, "glbs")

UID_FILES = [
    r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\objaverse_uk\uids.json",
    r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\objaverse_ch\uids_partial.json",
]
LVIS_UIDS_PATH = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\lvis_ch_uids.json"

CATEGORY_KWS = {
    "石窟造像": ["figurine", "statue", "buddha", "sculpture", "bust", "head",
              "figure", "idol", "deity", "guanyin", "bodhisattva", "arhat",
              "luohan", "dancer", "goddess", "warrior"],
    "古建筑构件": ["capital", "column", "cornice", "bracket", "frieze", "relief",
               "gable", "lintel", "architrave", "pediment", "entablature",
               "dougong", "corbel", "vault", "arch", "keystone", "pillar"],
    "青铜器": ["bronze", "ding", "tripod", "censer", "cauldron", "gong", "gu",
             "jue", "zhi", "zun", "you", "hu", "yi", "pan", "gui"],
    "陶瓷器": ["vase", "bowl", "jar", "jug", "plate", "teapot", "urn", "amphora",
             "pottery", "ceramic", "porcelain", "stoneware", "earthenware",
             "terracotta", "pitcher", "cup", "mug", "basin", "pot", "dish",
             "crock", "flask", "ewer"],
    "玉器": ["jade", "bi", "cong", "huang", "axe", "adze", "chisel", "pendant",
            "ornament", "bead", "amulet"],
    "壁画": ["mural", "fresco", "painting", "panel", "triptych", "icon",
            "parchment"],
}

TARGET_DIST = {
    "石窟造像": 65,
    "古建筑构件": 72,
    "青铜器": 58,
    "陶瓷器": 47,
    "玉器": 23,
    "壁画": 22,
}
SEED = 42


def load_ukch_uids():
    uids = []
    for fp in UID_FILES:
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            uids.extend(data)
        elif isinstance(data, dict) and "uids" in data:
            uids.extend(data["uids"])
    seen = set()
    out = []
    for u in uids:
        u = str(u).strip()
        if len(u) == 32 and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def load_names(uids):
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


def classify_by_name(name):
    import re
    n = (name or "").lower()
    for cat, kws in CATEGORY_KWS.items():
        if any(re.search(r"\b" + kw + r"\b", n) for kw in kws):
            return cat
    return None


def build_glb_index():
    """预建所有已缓存 GLB 的 uid->path 映射，避免逐个 glob"""
    index = {}
    for fp in glob.glob(os.path.join(GLB_ROOT, "*", "*.glb")):
        uid = os.path.splitext(os.path.basename(fp))[0]
        index[uid] = fp
    return index


def main():
    print("=== CH-Single3D 构建 (v2: UK+CH keywords + LVIS) ===")

    # 1. UK+CH UIDs + 关键词分类
    ukch_uids = load_ukch_uids()
    print(f"UK+CH UID 总数: {len(ukch_uids)}")
    names = load_names(ukch_uids)
    print(f"有名称: {len(names)}")

    uid_cat = {}  # uid -> category
    for uid, name in names.items():
        cat = classify_by_name(name)
        if cat:
            uid_cat[uid] = cat

    # 2. LVIS UIDs（类别直接映射）
    with open(LVIS_UIDS_PATH, encoding="utf-8") as f:
        lvis_data = json.load(f)
    for cat, uids in lvis_data.items():
        for uid in uids:
            uid = str(uid).strip()
            if len(uid) == 32:
                uid_cat[uid] = cat  # LVIS 覆盖关键词

    print(f"\n分类后 UID 总数: {len(uid_cat)}")
    for cat in TARGET_DIST:
        n = sum(1 for c in uid_cat.values() if c == cat)
        print(f"  {cat}: {n}")

    # 3. 检查 GLB 缓存（预建索引）
    print("\n构建 GLB 缓存索引...")
    glb_index = build_glb_index()
    print(f"缓存中 GLB 总数: {len(glb_index)}")
    print("\n已缓存 GLB：")
    available = {}
    for uid, cat in uid_cat.items():
        glb = glb_index.get(uid)
        if glb:
            available.setdefault(cat, []).append((uid, glb))
    total_avail = 0
    for cat in TARGET_DIST:
        n = len(available.get(cat, []))
        total_avail += n
        print(f"  {cat}: {n}")
    print(f"  合计: {total_avail}")

    # 4. 按目标选取
    random.seed(SEED)
    selected = {}
    for cat, target in TARGET_DIST.items():
        pool = available.get(cat, [])
        if len(pool) <= target:
            chosen = pool
        else:
            chosen = random.sample(pool, target)
        selected[cat] = chosen

    total = sum(len(v) for v in selected.values())
    print(f"\n选取: {total}/287")
    for cat in TARGET_DIST:
        print(f"  {cat}: {len(selected[cat])}/{TARGET_DIST[cat]}")

    # 5. 输出 manifest
    manifest = {
        "dataset": "CH-Single3D",
        "version": "v2",
        "total": total,
        "target_distribution": TARGET_DIST,
        "categories": {},
    }
    for cat, items in selected.items():
        manifest["categories"][cat] = [
            {"uid": uid, "glb_path": glb} for uid, glb in items
        ]
    out_path = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\ch_single3d_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"清单: {out_path}")

    # 6. 不足报告
    print("\n=== 不足类别 ===")
    short = False
    for cat, target in TARGET_DIST.items():
        avail = len(available.get(cat, []))
        if avail < target:
            short = True
            print(f"  {cat}: 可用 {avail} < 目标 {target}（缺 {target - avail}）")
    if not short:
        print("  无不足类别，所有目标均已满足！")


if __name__ == "__main__":
    main()
