"""
扩散先验流水线接力器：等预处理 npy 足够 → 启动正式训练

后台运行：
  D:\\anaconda3\\python.exe -u run_prior_pipeline.py
"""
import glob
import os
import subprocess
import time

WS = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01"
NPY_DIR = os.path.join(WS, "prior_train_npy")
TARGET_NPY = 3500          # 预处理够这个数就启动训练
MAX_STEPS = 30000
BATCH = 64

print("[1/2] 等待预处理 npy 数据就绪 ...", flush=True)
while True:
    n = len(glob.glob(os.path.join(NPY_DIR, "*.npy")))
    print(f"  当前 {n} / {TARGET_NPY} 个 npy，60s 后再查", flush=True)
    if n >= TARGET_NPY:
        break
    time.sleep(60)

print(f"[2/2] npy 就绪（{n} 个），启动正式训练 max_steps={MAX_STEPS} batch={BATCH}", flush=True)
subprocess.run([
    r"D:\anaconda3\python.exe", "-u",
    os.path.join(WS, "train_diffusion_prior.py"),
    "--train", "--max_steps", str(MAX_STEPS), "--batch", str(BATCH),
], cwd=WS)

print("[完成] 扩散先验训练结束，ckpt 在 prior_ckpt/prior_latest.pt", flush=True)
