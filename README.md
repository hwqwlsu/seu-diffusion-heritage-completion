# Score-Energy Uncertainty for Diffusion-Guided Cultural Heritage Point Cloud Completion

Supplementary code and data for the paper:

> Wenquan Huang, Teng Li, Qing Cheng, Lianqiang Yang. "Score-Energy Uncertainty for Diffusion-Guided Cultural Heritage Point Cloud Completion." *IEEE Transactions on Visualization and Computer Graphics*.

## Overview

This package contains the core implementation of a diffusion-guided point-cloud
completion framework that uses a pretrained point-cloud score function as an
energy regularizer (not an end-to-end generator). Observed points are frozen;
only missing points are optimized. A score-energy uncertainty (SEU) measure is
derived from the normalized diffusion score energy to quantify per-region
inferential confidence.

## Directory Structure

```
.
├── code/              Core algorithms
│   ├── train_diffusion_prior.py   Train the point-cloud diffusion score prior
│   ├── run_prior_pipeline.py      Main completion pipeline (freeze observed, optimize missing)
│   ├── run_ablation.py            Component ablation (boundary / smoothness / diffusion / warm-up)
│   ├── run_baselines.py           Geometry baselines (IMLS-MC, AlphaShape)
│   ├── run_exp_b1.py, b2.py, b3.py, r2u.py   Experiment runners
│   ├── run_extra_levels.py        Additional damage-level experiments
│   ├── eval_boundary.py           Boundary-continuity evaluation
│   ├── scan_boundary.py           Boundary detection utilities
│   ├── verify_detector.py         Visibility / damage-detector verification
│   ├── build_ch_single3d.py       Build the CH-Single3D dataset manifest
│   └── dataset_stats.py           Dataset statistics
├── data/              Manifests and result tables (no raw point clouds)
│   ├── ch_single3d_manifest.json  CH-Single3D sample manifest (281 samples)
│   ├── *_uids.json                Category UID lists
│   ├── sci_gap2_*.json            SEU–CD correlation and uncertainty results
│   ├── sci_gap3_results.json      Objaverse generalization results
│   ├── sci_gap1_baselines_objaverse.json  Geometry-baseline results on Objaverse
│   ├── fig2_6_data.json           Data behind Fig. 2/6
│   └── diagnose_trim.json         Diagnostic trimming statistics
├── figures/           All 9 figures used in the paper (PNG, ~800 dpi)
└── figures_gen/       Scripts that regenerate the figures
```

## Reproduction Steps

1. **Install dependencies**
   ```bash
   pip install torch numpy open3d scipy scikit-learn pandas matplotlib pillow
   ```

2. **Train the diffusion prior**
   ```bash
   python code/train_diffusion_prior.py --data_root <path-to-CH-Single3D>
   ```

3. **Run completion**
   ```bash
   python code/run_prior_pipeline.py --data_root <path-to-CH-Single3D> --ckpt <prior-ckpt>
   ```

4. **Run ablations and baselines**
   ```bash
   python code/run_ablation.py
   python code/run_baselines.py
   ```

5. **Regenerate figures**
   ```bash
   python figures_gen/make_en_figs.py
   python figures_gen/make_en_matrix.py
   python figures_gen/make_en_flow.py
   ```

## Data Availability

The raw point-cloud datasets are **not** included in this package due to size.

- **CH-Single3D** (281 cultural-heritage samples across six categories):
  available upon request from the authors; a public release is in preparation.
- **Objaverse** heritage and general-object subsets:
  downloaded from [Objaverse](https://objaverse.allenai.org/) using the UID
  lists in `data/*_uids.json`.

## Third-Party Weights

The following pretrained weights are used and must be obtained from their
original sources (not redistributed here):

- **SAM** (`sam_vit_b_01ec64.pth`): https://github.com/facebookresearch/segment-anything
- **SnowflakeNet**: https://github.com/AllenXiangX/SnowflakeNet
- **GACNet** (smoke-test checkpoint): https://github.com/chenhuiyuan/GACNet

## Citation

If you use this code or data, please cite:

```bibtex
@article{huang2026scoreenergy,
  title   = {Score-Energy Uncertainty for Diffusion-Guided Cultural Heritage Point Cloud Completion},
  author  = {Huang, Wenquan and Li, Teng and Cheng, Qing and Yang, Lianqiang},
  journal = {IEEE Transactions on Visualization and Computer Graphics},
  year    = {2026}
}
```

## License

- Code: MIT License (see `LICENSE`)
- Data and figures: Creative Commons Attribution 4.0 International (CC BY 4.0)
