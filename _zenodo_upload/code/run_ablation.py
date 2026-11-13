"""
消融实验编排器（后台守护）

流程：
  1. 轮询等待所有 run_exp_b3.py 主实验进程退出
  2. 合并 CH-Single3D 分片结果 → results_b3_ch3d_full/results.json
  3. 在 CH-Single3D 前 100 个样本、残损 40% 上，跑 5 组配对消融
     （full / no_diff / no_bnd / no_smooth / no_warmup）
     每组拆 3 片并行，逐组串行，保证任何时刻最多 3 个进程
  4. 汇总输出消融表
"""
import glob
import json
import os
import subprocess
import sys
import time

WS = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01"
PY = r"D:\anaconda3\python.exe"
SCRIPT = os.path.join(WS, "run_exp_b3.py")
VARIANTS = ["full", "no_diff", "no_bnd", "no_smooth", "no_warmup"]
N_SAMPLES = 100
LEVEL = "0.4"
N_SHARDS = 3


def pids_running_experiments():
    """返回仍在跑 run_exp_b3.py 的进程 PID 列表"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*run_exp_b3.py*' } | "
         "Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True)
    return [x.strip() for x in out.stdout.splitlines() if x.strip().isdigit()]


def wait_experiments():
    print("[1/3] 等待主实验结束...", flush=True)
    while True:
        p = pids_running_experiments()
        if not p:
            print("  主实验已全部结束", flush=True)
            return
        print(f"  仍有 {len(p)} 个进程，60s 后再查", flush=True)
        time.sleep(60)


def merge_shards():
    print("[2/3] 合并 CH-Single3D 分片结果", flush=True)
    dst_dir = os.path.join(WS, "results_b3_ch3d_full")
    os.makedirs(dst_dir, exist_ok=True)
    merged = {}
    files = [os.path.join(dst_dir, "results.json")]
    files += sorted(glob.glob(os.path.join(WS, "results_b3_ch3d_s?", "results.json")))
    for fp in files:
        if not os.path.exists(fp):
            continue
        try:
            for r in json.load(open(fp, encoding="utf-8")):
                merged[(r["uid"], r["target_ratio"])] = r
        except Exception as e:
            print(f"  跳过 {fp}: {e}", flush=True)
    rows = list(merged.values())
    json.dump(rows, open(os.path.join(dst_dir, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    uids = {r["uid"] for r in rows}
    print(f"  合并后 {len(rows)} 条 / {len(uids)} 个样本", flush=True)


def run_variant(v):
    print(f"[3/3] 变体 {v} ...", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    procs = []
    per = N_SAMPLES // N_SHARDS
    for s in range(N_SHARDS):
        skip = s * per
        limit = per if s < N_SHARDS - 1 else N_SAMPLES - skip
        out = os.path.join(WS, f"results_abl_{v}_s{s}")
        procs.append(subprocess.Popen(
            [PY, "-u", SCRIPT, "--dataset", "ch3d", "--levels", LEVEL,
             "--variant", v, "--skip", str(skip), "--limit", str(limit), "--out", out],
            cwd=WS, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    # 合并该变体分片
    merged = {}
    for s in range(N_SHARDS):
        fp = os.path.join(WS, f"results_abl_{v}_s{s}", "results.json")
        if os.path.exists(fp):
            for r in json.load(open(fp, encoding="utf-8")):
                merged[(r["uid"], r["target_ratio"])] = r
    os.makedirs(os.path.join(WS, f"results_abl_{v}"), exist_ok=True)
    json.dump(list(merged.values()),
              open(os.path.join(WS, f"results_abl_{v}", "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  {v}: {len(merged)} 条", flush=True)


def summarize():
    print("\n=== 消融汇总（CH-Single3D 前100样本, 残损40%）===", flush=True)
    base = None
    for v in VARIANTS:
        fp = os.path.join(WS, f"results_abl_{v}", "results.json")
        if not os.path.exists(fp):
            continue
        rows = json.load(open(fp, encoding="utf-8"))
        cd = sum(r["cd_l1"] for r in rows) / len(rows) * 1e4
        fs = sum(r["f_score"] for r in rows) / len(rows)
        nc = sum(r["normal_consistency"] for r in rows) / len(rows)
        if base is None:
            base = cd
        print(f"  {v:10s} n={len(rows):3d} CD×1e4={cd:7.2f} Δ={cd-base:+7.2f} "
              f"F={fs:.3f} NC={nc:.3f}", flush=True)


if __name__ == "__main__":
    wait_experiments()
    merge_shards()
    for v in VARIANTS:
        run_variant(v)
    summarize()
    print("\n[完成] 消融实验全部结束", flush=True)
