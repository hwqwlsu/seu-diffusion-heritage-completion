# -*- coding: utf-8 -*-
"""运行一次真实的扩散补全优化，保存损失曲线 JSON（用于图 B10）"""
import sys, json
sys.path.insert(0, r'G:\大博士论文资料\代码')
import numpy as np
from modules.diffusion_completion import DiffusionCompletionConfig, DiffusionGuidedCompletion

np.random.seed(42)
theta = np.random.uniform(0, np.pi, 800)
phi = np.random.uniform(0, 2*np.pi, 800)
r = 0.4
x = r*np.sin(theta)*np.cos(phi); y = r*np.sin(theta)*np.sin(phi); z = r*np.cos(theta)
partial = np.stack([x,y,z], -1).astype(np.float32)
partial = partial[partial[:,2] > -0.15]  # 模拟残损

H, W = 64, 64
depth = np.full((H,W), 1.0, np.float32); depth[32,32] = 0.6
K = np.array([[32,0,32],[0,32,32],[0,0,1]], np.float32)
pose = np.eye(4, dtype=np.float32)

cfg = DiffusionCompletionConfig(device='cpu', prior_num_points=256, num_iters=300,
                                output_num_points=2048, warmup_iters=20)
comp = DiffusionGuidedCompletion(cfg)
res = comp.complete(partial, depth, K, pose, verbose=False)

with open(r'C:\Users\黄文全\Documents\Trae-CN\temp_20261031_01\figures\loss_history.json','w') as f:
    json.dump(res['loss_history'], f)
print('loss history saved, iters =', len(res['loss_history']['total']))
