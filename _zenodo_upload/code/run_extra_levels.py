"""
补充残损档位：0.1 / 0.3 / 0.5

用途：支撑论文 B21「分档性能曲线」图（需要 ≥5 个档位才能画可信的单调性曲线）。
已在 0.2 / 0.4 上跑过，这里补中间与更重档位。

流程：
  1. 等待消融编排器（run_ablation.py）结束
  2. 在 CH-Single3D 前 100 样本上，按 3 片并行跑 0.1/0.3/0.5
  3. 合并 + 汇总
"""
import json
import os
import subprocess
import time

WS = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01"
PY = r"D:\anaconda3\python.exe"
SCRIPT = os.path.join(WS, "run_exp_b3.py")
LEVELS = "0.1,0.3,0.5"
N_SAMPLES = 100
N_SHARDS = 3


def abl_running():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -like '*run_ablation.py*' } | "
         "Select-Object -ExpandProperty ProcessId"],
        capture_output=True, text=True)
    return [x.strip() for x in out.stdout.splitlines() if x.strip().isdigit()]


print("等待消融实验结束...", flush=True)
while abl_running():
    time.sleep(120)
print("消融已结束，开始补档位", flush=True)

env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
procs = []
per = N_SAMPLES // N_SHARDS
for s in range(N_SHARDS):
    skip = s * per
    limit = per if s < N_SHARDS - 1 else N_SAMPLES - skip
    out = os.path.join(WS, f"results_lvl_s{s}")
    procs.append(subprocess.Popen(
        [PY, "-u", SCRIPT, "--dataset", "ch3d", "--levels", LEVELS,
         "--skip", str(skip), "--limit", str(limit), "--out", out],
        cwd=WS, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
for p in procs:
    p.wait()

merged = {}
for s in range(N_SHARDS):
    fp = os.path.join(WS, f"results_lvl_s{s}", "results.json")
    if os.path.exists(fp):
        for r in json.load(open(fp, encoding="utf-8")):
            merged[(r["uid"], r["target_ratio"])] = r
outdir = os.path.join(WS, "results_b3_ch3d_levels")
os.makedirs(outdir, exist_ok=True)
json.dump(list(merged.values()), open(os.path.join(outdir, "results.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n=== 补充档位汇总 ===", flush=True)
for lv in (0.1, 0.3, 0.5):
    s = [r for r in merged.values() if r["target_ratio"] == lv]
    if not s:
        continue
    cd = sum(r["cd_l1"] for r in s) / len(s) * 1e4
    fs = sum(r["f_score"] for r in s) / len(s)
    nc = sum(r["normal_consistency"] for r in s) / len(s)
    print(f"  残损{lv:.0%}: n={len(s)} CD×1e4={cd:.2f} F={fs:.3f} NC={nc:.3f}", flush=True)
print("[完成] 补充档位全部结束", flush=True)
