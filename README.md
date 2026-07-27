# Continual Learning Enabled Vision Foundation Model for Defect Segmentation

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![W&B](https://img.shields.io/badge/Weights_&_Biases-FFCC3D?style=flat&logo=WeightsAndBiases&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

Official implementation of the Master's research project *"Continual Learning Enabled Vision Foundation Model for Shipping Container Defect Segmentation"* (University of Malaya, 2026).

A modular, parameter-efficient framework for adapting a **frozen DINOv3 Vision Foundation Model** to sequential, class-incremental dense segmentation without catastrophic forgetting and without full-model retraining. Six continual-learning strategies are benchmarked under an identical five-task protocol, over **three random seeds**, with a common per-class IoU metric.

> **Note (v2, 2026).** This README supersedes an earlier version of the results. The evaluation metric has been unified across all strategies, the trainable head upgraded from a 1×1 linear probe to a residual convolutional decoder, and all runs repeated over three seeds. The headline conclusion changed: **DER++, previously the weakest method under the linear probe, is now the strongest** — and understanding *why* is the central finding of the project (see below).

## 🌟 Overview

Industrial visual inspection systems operate in non-stationary environments where new defect categories emerge over time. Standard segmentation models suffer **catastrophic forgetting** when fine-tuned on new categories, while retraining large foundation models from scratch is computationally prohibitive.

This framework couples a **frozen DINOv3 (ViT-Large) backbone** with a lightweight, trainable **residual convolutional decoder** (1.44 M parameters, ≈ 0.5 % of the backbone). Because the backbone is frozen, its features are extracted **once and cached** (29× per-epoch speed-up), and the replay buffer stores cached features that can never go stale — eliminating the representation-ageing problem of conventional latent replay.

## 🎥 Video Demonstration

Sequential task learning in action: the model segments new repair categories on shipping containers while retaining previously learned ones.

👉 **[Watch the Full Video Demo Here](https://drive.google.com/file/d/1_RX9-k81P7kk2mH3w-cVPvy4uSth56cM/view?usp=sharing)**

## 🔬 Key Empirical Findings

1. **Replay can match joint training on a frozen VFM.** DER++'s final Average Accuracy (0.6165 ± 0.0021) is **statistically indistinguishable from the Joint upper bound** (0.6205 ± 0.0028) given seed variance, effectively closing the gap left by naive fine-tuning (0.3511 ± 0.0213) while cutting Forgetting by 80.5 %. The mechanism: buffered features from a frozen extractor never drift out of correspondence with the network that consumes them.
2. **Dark-knowledge distillation requires sufficient head capacity.** Under a 1×1 linear probe, DER++ collapsed to the naive baseline: six-channel near-one-hot logits carry no dark knowledge, and a β re-tune could not fix it (re-weighting an uninformative signal cannot make it informative). With the residual decoder, DER++ moves from **worst CL method to best** — logit *entropy*, not distillation *weight*, is the binding constraint.
3. **Replay is an order of magnitude more reproducible than regularization.** Across seeds, replay methods vary by ±0.002–0.015 AA while SI varies by ±0.062 — its path-integral importance estimate inherits the stochasticity of the training trajectory. Single-seed comparisons between these families should be read with caution.
4. **Cost follows pass counts exactly.** Per batch: Naive/Joint/SI = 1 pass, ER = 2, DER++ = 3, MER = 4 (S + 1). MER is the *most* expensive strategy, not the cheapest, and is outperformed by plain ER on every seed.

## 📊 Results Summary (5-task class-incremental, mean ± std over seeds 42/43/44)

| Strategy | Family | Avg Accuracy (mIoU) ↑ | Forgetting ↓ | Dice ↑ | Time (min) | Passes/batch |
|---|---|---|---|---|---|---|
| Naive | Baseline | 0.3511 ± 0.0213 | 0.4056 ± 0.0324 | 0.4583 | 5.2 | 1 |
| SI | Regularization | 0.4474 ± 0.0617 | 0.2378 ± 0.0707 | 0.5718 | 6.2 | 1 |
| MER | Replay | 0.5211 ± 0.0149 | 0.1653 ± 0.0120 | 0.6587 | 17.1 | 4 |
| ER | Replay | 0.5529 ± 0.0041 | 0.1463 ± 0.0029 | 0.6898 | 10.5 | 2 |
| **DER++** | **Replay** | **0.6165 ± 0.0021** | **0.0789 ± 0.0011** | **0.7499** | 12.7 | 3 |
| Joint | Ceiling | 0.6205 ± 0.0028 | — | 0.7541 | 5.1 | 1 |

The ordering **Naive < SI < MER < ER < DER++ < Joint** held on every seed without a single crossing. Training times are per-strategy decoder training with cached features (single GPU); one-time feature extraction ≈ 2 h. Energy figures from the earlier README have been removed — they were wall-clock × assumed TDP, i.e. an assumption presented as data.

![Average Accuracy](figures/figure_4_1_average_accuracy.png)

## ⚠️ Protocol qualification (read before citing)

The five-task split uses **class-sequential exposure with a stable background**, not a strictly label-disjoint split: masks retain the labels of *all* classes present in an image, and 14.2 % of annotated images contain more than one class. This isolates **parameter drift** as the cause of measured forgetting (no class is ever actively unlearned as background), but it does **not** test the canonical background-shift formulation of class-incremental segmentation. Results should be read under that qualification; a strictly label-disjoint variant is declared future work.

## 🏗️ Architecture & Methodology

- **Backbone:** DINOv3 ViT-Large — *strictly frozen*. Dense features on a 28×28 grid (patch 16), extracted once and cached as fp16 memory-maps.
- **Decoder:** Residual convolutional block, 1,445,126 params: 1×1 projection (1024→256) → two 3×3 convs with GroupNorm + GELU as a residual branch → 1×1 to 6 channels. Setting the residual branch to zero recovers the linear probe exactly.
- **Memory:** Reservoir buffer of 500 **cached features** (≈ 800 MB); DER++ additionally stores insertion-time logits. Sampler operates on annotated splits only.
- **Optimization:** AdamW (LR 1e-3, weight decay 1e-4), batch 8, 20 epochs/task at constant LR; Joint additionally uses cosine annealing. Identical decoder init across strategies within each seed (paired comparison).
- **Loss:** class-weighted CrossEntropy (median-frequency weights, capped at 5.0) + foreground soft Dice. Pure foreground Dice collapses to majority-class prediction; unweighted CE optimises a different quantity than the macro-IoU metric.
- **Metric:** per-class IoU on the target class for **every** strategy including Joint (the earlier macro-Jaccard/per-class mismatch invalidated cross-strategy comparison); supervision and evaluation both at the 28×28 patch grid.
- **Seeds:** 42, 43, 44 — controlling init, reservoir sampling and batch order. All executed runs are reported.

## 📁 Repository Structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── CL_full_multiseed.ipynb    # ⭐ main benchmark: 6 strategies × 3 seeds, cached features
│   └── CL_full.ipynb              # single-seed development version
├── legacy notebooks (v1, 1×1 head — kept for the DER++ capacity comparison):
│   ├── Naive_full.ipynb  ├── ER_full.ipynb  ├── DERpp_full.ipynb
│   ├── MER_full.ipynb    ├── SI_full.ipynb  └── Joint_UpperBound_full.ipynb
├── figures/                       # report figures (architecture, results, matrices)
└── docs/
    └── research_report_2026.pdf   # full research report
```

## 📊 Experiment Tracking

All runs logged to **Weights & Biases**: per-task loss dynamics, accuracy matrices, the legacy DER++ β-sweep, and multi-seed comparisons.

👉 **[Explore the W&B Workspace](https://wandb.ai/faqihahsyazwani23-university-malaya-medical-centre/dino-foreground-segmentation/workspace?nw=nwuserfaqihahsyazwani23)**

## 🚀 Installation & Setup

```bash
git clone https://github.com/FaqihahSyazwani99/CL-Foreground-Segmentation-DINOv3.git
cd CL-Foreground-Segmentation-DINOv3

conda create -n cl_vfm python=3.10 -y
conda activate cl_vfm
pip install -r requirements.txt
```

The dataset (11,133 images, 5 repair categories in COCO format) is proprietary to the industrial partner and is not distributed with this repository. The notebooks expect it under a Roboflow-style COCO export; adapt `DATASET_PATH` in the first cells.

## 📖 Citation

```bibtex
@mastersthesis{mutaliff2026clvfm,
  title  = {Continual Learning Enabled Vision Foundation Model for
            Shipping Container Defect Segmentation},
  author = {Faqihah Syazwani binti Abdul Mutaliff},
  school = {Faculty of Computer Science and Information Technology,
            University of Malaya},
  year   = {2026}
}
```

## 🙏 Acknowledgements

Supervised by Prof. Dr. Loo Chu Kiong, University of Malaya. Dataset courtesy of the industrial partner. Built on DINOv3 (Meta AI Research) and the Mammoth continual-learning protocol style.

