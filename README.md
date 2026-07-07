# Continual Learning Enabled Vision Foundation Model for Defect Segmentation

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![W&B](https://img.shields.io/badge/Weights_&_Biases-FFCC3D?style=flat&logo=WeightsAndBiases&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

Official implementation of the Master's research project: *"Continual Learning Enabled Vision Foundation Model for Shipping Container Defect Segmentation"* (University of Malaya). 

This repository provides a modular, parameter-efficient framework for adapting a frozen Vision Foundation Model (DINOv3) to sequential, class-incremental dense segmentation tasks without catastrophic forgetting and without full-model retraining.

## 🌟 Overview

Industrial visual inspection systems operate in non-stationary environments where new defect categories emerge over time. Standard segmentation models suffer from **catastrophic forgetting** when fine-tuned on new categories, while retraining large foundation models from scratch is computationally prohibitive.

This framework couples a **frozen DINOv3 (ViT-Large) backbone** with a lightweight, trainable **1x1 convolutional head**. It evaluates six Continual Learning (CL) strategies under an identical 5-task class-incremental protocol to find the optimal stability-plasticity-compute trade-off.

## 🎥 Video Demonstration

See the framework in action! The video demonstrates the sequential task learning process, showing how the model segments new repair categories on shipping containers while retaining previously learned defects using the Continual Learning strategies. 

👉 **[Watch the Full Video Demo Here](https://drive.google.com/file/d/1_RX9-k81P7kk2mH3w-cVPvy4uSth56cM/view?usp=sharing)**

## 🔬 Key Empirical Findings

1. **Replay > Regularization:** Replay-based methods (ER, MER) substantially outperform regularization (SI) and naive fine-tuning for dense prediction with frozen foundation models.
2. **MER is the Practical Default:** Meta-Experience Replay (MER) achieves the best stability (Lowest Forgetting: `0.083`) at roughly **60% of the compute cost** of standard Experience Replay (ER).
3. **The DER++ Anomaly:** Dark-knowledge distillation (DER++) fails when the trainable head is a low-capacity linear probe. The 1x1 head produces low-entropy, near-one-hot logits, rendering MSE-based logit distillation ineffective. Temperature-scaled KL-divergence is required for lightweight heads.

## 📊 Results Summary (5-Task Class-Incremental)

| Strategy | Family | Avg Accuracy (mIoU) ↑ | Forgetting ↓ | Training Time | Energy (kWh) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Naive | Baseline | 0.254 | 0.169 | 166 min | 0.828 |
| SI | Regularization | 0.268 | 0.109 | 167 min | 0.835 |
| DER++* | Replay | 0.259 | 0.167 | 145 min | 0.727 |
| **ER** | **Replay** | **0.353** | 0.096 | 370 min | 1.852 |
| **MER** | **Replay** | **0.353** | **0.083** | **224 min** | **1.118** |
| Joint | Ceiling | 0.469 | — | 103 min | 0.516 |

*\*DER++ collapsed to the naive baseline due to the low-entropy logit anomaly discussed in the paper.*

## 📊 Experiment Tracking & Live Results

All training runs, loss dynamics, and metric evaluations were logged in real-time using **Weights & Biases (W&B)**. To ensure full transparency and reproducibility, the complete experimental dashboard—including the per-task mIoU evolution, the stability-plasticity trade-offs, and the raw logs for the DER++ hyperparameter sweeps—is publicly available.

👉 **[Explore the Full W&B Workspace Here](https://wandb.ai/faqihahsyazwani23-university-malaya-medical-centre/dino-foreground-segmentation/workspace?nw=nwuserfaqihahsyazwani23)**

**What you will find in the dashboard:**
*   **Training Dynamics:** Per-task total loss curves showing rapid convergence and how replay stabilizes earlier-task loss.
*   **Strategy Comparison:** Head-to-head Average Accuracy (mIoU) and Forgetting metrics for all six strategies.
*   **The DER++ Anomaly:** Raw logs of the $\beta$ re-tune (0.5 $\rightarrow$ 0.1) demonstrating the low-entropy logit collapse.
*   **Compute Metrics:** Wall-clock training time and estimated energy consumption per strategy.

## 🏗️ Architecture & Methodology

*   **Backbone:** DINOv3 (ViT-Large) - *Strictly Frozen*. Extracts dense features on a 28x28 spatial grid (patch size 16).
*   **Decoder:** Single 1x1 Convolution (6 output channels). Acts as a per-patch linear probe.
*   **Memory:** Reservoir sampling buffer (Capacity: 500). *Note: The sampler operates only on annotated task splits to prevent buffer dilution by background-only images.*
*   **Optimization:** Adam (LR 1e-3), 10 epochs per task. Only the head is updated.
*   **Loss Function:** `CrossEntropy(all classes) + Dice(foreground)`. (Crucial: pure foreground Dice causes the model to collapse to predicting the majority class everywhere).

## 🚀 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/continual-learning-vfm-segmentation.git
cd continual-learning-vfm-segmentation

# Create conda environment
conda create -n cl_vfm python=3.10
conda activate cl_vfm

# Install dependencies
pip install -r requirements.txt
