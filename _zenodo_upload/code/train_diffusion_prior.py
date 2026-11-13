"""
训练点云扩散先验 (PointCloudDenoiser)

目标：把"内置轻量先验（无预训练权重）"替换为在 Objaverse 缓存上真实训练的
形状先验，使消融实验中 full 优于 no_diff（当前随机先验为负贡献）。

流程：
  Phase 1 数据准备: 遍历缓存 GLB → 采样 4096 点 → 归一化(质心+最大模长)
                    → 存 npy (float32) 到 prior_train_npy/  （多进程，可续跑）
  Phase 2 训练:     ε-预测 MSE（与 diffusion_prior.training_loss 一致），
                    随机 2048 子采样 + 随机 SO(3) 旋转增强，AdamW + cosine，
                    EMA 权重，bf16 autocast，周期保存 ckpt。

checkpoint 格式与 PointCloudDiffusionPrior 加载约定一致：
  torch.save({"model": ema_state_dict, "step": step, ...})

用法：
  python train_diffusion_prior.py --prep --n_models 4000     # 仅数据准备
  python train_diffusion_prior.py --train --max_steps 30000  # 仅训练
  python train_diffusion_prior.py --smoke                    # 冒烟：64模型+60步
"""
import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

WS = r"C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01"
CACHE_GLB = r"C:\Users\黄文全\.cache\.objaverse\hf-objaverse-v1\glbs"
NPY_DIR = os.path.join(WS, "prior_train_npy")
N_SAMPLE = 4096      # 预处理采样点数（训练时子采样 2048）
N_PRIOR = 2048       # 先验网络输入点数（须与 prior_num_points 一致）


def _prep_one(args):
    """单模型预处理 worker：返回 (uid, ok)"""
    uid, gpath, out_dir = args
    out_path = os.path.join(out_dir, uid + ".npy")
    if os.path.exists(out_path):
        return uid, True
    try:
        import trimesh
        mesh = trimesh.load(gpath, force="scene")
        if not isinstance(mesh, trimesh.Scene):
            return uid, False
        mesh = mesh.dump(concatenate=True)
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            return uid, False
        if len(mesh.faces) > 3_000_000:
            return uid, False
        pts, _ = trimesh.sample.sample_surface(mesh, N_SAMPLE)
        pts = pts.astype(np.float32)
        pts -= pts.mean(axis=0)
        scale = np.linalg.norm(pts, axis=1).max()
        if scale < 1e-6 or not np.isfinite(scale):
            return uid, False
        pts = (pts / scale).astype(np.float32)
        if not np.isfinite(pts).all():
            return uid, False
        np.save(out_path, pts)
        return uid, True
    except Exception:
        return uid, False


def run_prep(n_models, workers=10):
    os.makedirs(NPY_DIR, exist_ok=True)
    glbs = sorted(glob.glob(os.path.join(CACHE_GLB, "*", "*.glb")))
    tasks = [(os.path.splitext(os.path.basename(g))[0], g, NPY_DIR) for g in glbs]
    if n_models:
        tasks = tasks[:n_models]
    print(f"[prep] 待处理 {len(tasks)} 个模型 → {NPY_DIR}", flush=True)
    t0 = time.time()
    ok = 0
    with Pool(workers) as pool:
        for i, (uid, success) in enumerate(pool.imap_unordered(_prep_one, tasks, chunksize=4)):
            ok += success
            if (i + 1) % 200 == 0:
                print(f"[prep] {i+1}/{len(tasks)} 成功 {ok} ({(time.time()-t0)/60:.1f}min)", flush=True)
    print(f"[prep] 完成：{ok}/{len(tasks)} 成功，耗时 {(time.time()-t0)/60:.1f} min", flush=True)


# ---------------- 训练 ----------------

def random_rotations(n, device):
    """生成 n 个随机旋转矩阵 (n,3,3)，QR 法"""
    m = torch.randn(n, 3, 3, device=device)
    q, r = torch.linalg.qr(m)
    # 保证 det=+1
    det = torch.linalg.det(m)
    q = q * torch.sign(det)[:, None, None]
    return q


def load_eval_uids():
    """评测集（CH-Single3D / SHREC 替代集）的 UID 并集，训练时必须排除以避免数据泄漏"""
    uids = set()
    m = json.load(open(os.path.join(WS, "ch_single3d_manifest.json"), encoding="utf-8"))
    uids |= {it["uid"] for items in m["categories"].values() for it in items}
    s = json.load(open(os.path.join(WS, "shrec2021_alternative_uids.json"), encoding="utf-8"))
    if isinstance(s, dict):
        uids |= set(s.get("core_subset_uids") or s.get("all_ceramic_uids") or [])
    else:
        uids |= set(s)
    return uids


def run_train(max_steps=30000, batch=64, lr=1e-3, val_n=200,
              save_every=2000, log_every=100, out_dir=None, workers=6,
              exclude_uids=None):
    import torch  # noqa: F811  (模块级已导入，此处仅为清晰)
    import torch.nn.functional as F  # noqa: F811

    sys.path.insert(0, r"G:\大博士论文资料\代码")
    from modules.diffusion_completion.diffusion_prior import (
        PointCloudDenoiser, cosine_beta_schedule)

    out_dir = out_dir or os.path.join(WS, "prior_ckpt")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda"

    npys = sorted(glob.glob(os.path.join(NPY_DIR, "*.npy")))
    if exclude_uids:
        before = len(npys)
        npys = [p for p in npys if os.path.splitext(os.path.basename(p))[0] not in exclude_uids]
        print(f"[train] 排除评测集 {before - len(npys)} 个样本，剩 {len(npys)}", flush=True)
    assert len(npys) >= 32, f"npy 不足：{len(npys)}"
    rng = np.random.default_rng(0)
    rng.shuffle(npys)
    val_paths_n = min(val_n, len(npys) // 10)
    val_paths, train_paths = npys[:val_paths_n], npys[val_paths_n:]
    print(f"[train] 训练 {len(train_paths)} / 验证 {len(val_paths)} 模型", flush=True)

    # 全部载入内存：~4000 × 4096 × 3 × 4B ≈ 200MB，一次读盘免除 IO
    def load_all(paths):
        return np.stack([np.load(p) for p in paths]).astype(np.float32)
    t0 = time.time()
    train_data = torch.from_numpy(load_all(train_paths)).to(device)  # (M,4096,3)
    val_data = torch.from_numpy(load_all(val_paths)).to(device)
    print(f"[train] 数据载入 {time.time()-t0:.0f}s  train{tuple(train_data.shape)}", flush=True)

    model = PointCloudDenoiser(num_points=N_PRIOR, hidden_dim=256).to(device)
    ema = PointCloudDenoiser(num_points=N_PRIOR, hidden_dim=256).to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    ema_decay = 0.999

    # 噪声调度（与先验一致：cosine, T=1000）
    T = 1000
    betas = cosine_beta_schedule(T)
    acp = torch.cumprod(1.0 - betas, dim=0).to(device)
    sqrt_acp = torch.sqrt(acp)
    sqrt_1macp = torch.sqrt(1 - acp)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_steps, eta_min=lr * 0.01)

    M = train_data.shape[0]
    loss_log = []
    t0 = time.time()
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, M, (batch,), device=device)
        x0_full = train_data[idx]                              # (B,4096,3)
        sub = torch.randint(0, N_SAMPLE, (batch, N_PRIOR), device=device)
        x0 = torch.gather(x0_full, 1, sub[..., None].expand(-1, -1, 3))
        rot = random_rotations(batch, device)
        x0 = torch.bmm(x0, rot.transpose(1, 2))                # 随机旋转增强

        t = torch.randint(0, T, (batch,), device=device)
        noise = torch.randn_like(x0)
        xt = sqrt_acp[t][:, None, None] * x0 + sqrt_1macp[t][:, None, None] * noise

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(xt, t)
            loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - ema_decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)

        if step % log_every == 0:
            loss_log.append((step, loss.item()))
            print(f"[train] step {step}/{max_steps} loss={loss.item():.5f} "
                  f"({(time.time()-t0)/60:.1f}min)", flush=True)

        if step % save_every == 0 or step == max_steps:
            # 验证（EMA）
            ema.eval()
            with torch.no_grad():
                nv = min(64, val_data.shape[0])
                vi = torch.randint(0, val_data.shape[0], (nv,), device=device)
                vp = torch.randint(0, N_SAMPLE, (nv, N_PRIOR), device=device)
                vx = torch.gather(val_data[vi], 1, vp[..., None].expand(-1, -1, 3))
                vt = torch.randint(0, T, (nv,), device=device)
                vn = torch.randn_like(vx)
                vxt = sqrt_acp[vt][:, None, None] * vx + sqrt_1macp[vt][:, None, None] * vn
                vloss = F.mse_loss(ema(vxt, vt), vn).item()
            ckpt = {"model": ema.state_dict(), "step": step,
                    "val_loss": vloss, "n_train": len(train_paths)}
            torch.save(ckpt, os.path.join(out_dir, f"prior_step{step}.pt"))
            torch.save(ckpt, os.path.join(out_dir, "prior_latest.pt"))
            print(f"[train] 保存 ckpt step={step} val_loss={vloss:.5f}", flush=True)

    json.dump(loss_log, open(os.path.join(out_dir, "loss_log.json"), "w"))
    print(f"[train] 完成 → {out_dir}  总耗时 {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_models", type=int, default=0, help="0=全部")
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="", help="checkpoint 输出目录")
    ap.add_argument("--exclude_eval", action="store_true",
                    help="训练时排除 CH-Single3D / SHREC 评测集样本（防数据泄漏）")
    args = ap.parse_args()

    if args.smoke:
        run_prep(n_models=64, workers=8)
        run_train(max_steps=60, batch=16, val_n=8, save_every=60, log_every=10,
                  out_dir=os.path.join(WS, "prior_ckpt_smoke"), workers=0)
        return
    if args.prep:
        run_prep(n_models=args.n_models)
    if args.train:
        run_train(max_steps=args.max_steps, batch=args.batch, lr=args.lr,
                  out_dir=(args.out_dir or None),
                  exclude_uids=load_eval_uids() if args.exclude_eval else None)


if __name__ == "__main__":
    main()
