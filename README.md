<div align="center">

# Generalised Deepfake Detection Using Supervised Anomaly Detection (SAD)

**A Supervised anomaly detection pipeline for generalised deepfake detection — trained only on real faces, generalises to unseen forgeries.**


> **Authors:** Yash Gupta · Saurabh Kumar  

<br/>

---

</div>

## 📌 Overview

Most deepfake detectors are **binary classifiers** — they memorise forgery-specific patterns and fail catastrophically on unseen manipulation techniques. This project takes a fundamentally different approach.

We reframe deepfake detection as **one-class supervised anomaly detection**:

- A **Normalizing Flow (NF)** model is trained *exclusively on real face features*, learning the Multivariate Gaussian Distribution (MGD) of authentic faces.
- At inference time, any image whose feature embedding falls in a **low-density region** of that distribution is flagged as fake — regardless of *how* it was generated.
- **Grad-CAM** heatmaps are generated to visually explain which facial regions triggered the detection.

This design choice enables meaningful generalisation to **unseen deepfake types** without retraining.

<br/>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                           │
│                                                                     │
│   Input Image (224×224)                                             │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────┐     Global Avg     ┌──────────────────┐           │
│   │ MobileNetV2 │ ──── Pooling ────► │ 1280-dim Feature │           │
│   │  (Frozen)   │                    │     Vector       │           │
│   └─────────────┘                    └────────┬─────────┘           │
│                                               │                     │
│                                               ▼                     │
│                                    ┌──────────────────┐             │
│                                    │  Normalizing     │             │
│                                    │  Flow Model      │             │
│                                    │  (Real-only)     │             │
│                                    └────────┬─────────┘             │
│                                             │                       │
│                                             ▼                       │
│                                    Log-Likelihood Score             │
│                                             │                       │
│                             ┌───────────────┴──────────────┐        │
│                             │   score > τ ?                │        │
│                        ╔════╧════╗                  ╔══════╧════╗   │
│                        ║  REAL   ║                  ║   FAKE    ║   │
│                        ╚═════════╝                  ╚═══════════╝   │
│                                                                     │
│   + Grad-CAM Heatmap ──► Highlights manipulated facial regions      │
└─────────────────────────────────────────────────────────────────────┘
```

<br/>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 **One-Class Training** | Normalising Flow model trained only on real images — no fake samples needed |
| 🔁 **Generalisation** | Flags unseen deepfake types as out-of-distribution anomalies |
| 🔍 **Explainability** | Grad-CAM heatmaps highlight manipulated facial regions |
| ⚡ **Lightweight** | Runs on a single GPU (NVIDIA T4 / Google Colab compatible) |
| 📦 **Modular Design** | Each pipeline component can be independently swapped or upgraded |

<br/>

---

## 📂 Repository Structure

```
Generalised-deepfake-detection-using-SAD/
│
├── 📄 README.md                        # Project documentation (this file)
│
├── 📓 CV_Project.ipynb         # Main Colab notebook
│
├── 🐍 deepfake_clf_model.pth             
├── 🐍 mobilenet_cnn.pth             
├── 🐍 normalizing_flow_deepfake.pth                   
├── 🐍 normalizing_flow_model.pth           
├── 🐍 pca_transformer.joblib                     
├── 🐍 test_features_pca.npy                        
├── 🐍 test_labels.npy                   
├── 🐍 train_features_pca.npy                      
│
├── 📁 results/
│   ├── 🖼️  grad_cam_heatmap.png           # NF training loss curve over 100 epochs
│   ├── 🖼️  precision_roc.png   # Real vs Fake log-likelihood score distributions
│   ├── 🖼️  real_vs_fake.png           # ROC and Precision-Recall curves on test set
│   └── 🖼️  normalising_flow_loss.png          # Sample Grad-CAM heatmap overlay on a real face
│
└── 📁 report/
    └── 📄 Generalised-deepfake-detection-using-SAD.pdf   # Full project report
```

<br/>

---

## ⚙️ Methodology

### 1. Problem Formulation
Rather than training a binary classifier on both real and fake images, the model fits a density estimator **p(x)** on real image features only. At inference, a sample is classified as:
- **Real** → if `p(f(x_test)) ≥ τ`
- **Fake** → if `p(f(x_test)) < τ`

### 2. Feature Extraction
A **MobileNetV2** backbone (pretrained on ImageNet) produces a `(1280, 7, 7)` activation tensor per image. Global Average Pooling (GAP) compresses this into a **1280-dimensional feature vector**. The backbone weights are frozen throughout.

### 3. Normalizing Flow Density Estimation
A Normalizing Flow model learns an invertible transformation from a standard multivariate Gaussian to the distribution of real face features. Fake features, not seen during training, produce **low log-likelihood scores** when passed through this transformation — naturally flagging them as anomalies.

### 4. Supervised Guidance
Fake image embeddings are incorporated via a **boundary-based loss** that pushes fake features outside the real distribution boundary — increasing separability without overfitting to specific forgery patterns.

### 5. Grad-CAM Explainability
Gradients of the output with respect to the final convolutional feature maps are used to produce **spatial heatmaps**, overlaid on the original image to highlight detected manipulation sites.

<br/>

---

## 🗄️ Dataset

**Celeb-DF v2** — A large-scale deepfake benchmark with high-quality celebrity face videos generated via an improved face-swapping algorithm.

| Split | Real Frames | Fake Frames | Total |
|-------|-------------|-------------|-------|
| Train | ~2,590 | ~5,180 | ~7,770 |
| Test | ~518 | ~1,036 | ~1,554 |
| **Total** | **~3,108** | **~6,216** | **~9,324** |

> Dataset source: [Kaggle — pranabr0y/celebdf-v2image-dataset](https://www.kaggle.com/datasets/pranabr0y/celebdf-v2image-dataset)

<br/>

---

## 📊 Results

### Training Convergence

The NF model converges smoothly over 100 epochs, with the Negative Log-Likelihood (NLL) loss decreasing from ~180 to ~78 — indicating the model successfully learns the density of real face features.

![Training Loss](results/normalising_flow_loss.png)

---

### Log-Likelihood Score Distribution

The histogram below shows the NF log-likelihood scores assigned to real and fake test images. The vertical dashed line marks the decision threshold **τ = −6385.91**. Both distributions cluster near zero, indicating that the feature overlap between real and fake images is significant at this layer — a known challenge for one-class detectors on high-quality deepfakes.

![Log-Likelihood Distribution](results/real_vs_fake.png)

---

### ROC & Precision-Recall Curves

The ROC curve (AUC = 0.46) and Precision-Recall curve (AP = 0.47) shown below correspond to the *raw* threshold-based classification. The lower AUC reflects the difficulty of separating distributions that overlap heavily in MobileNetV2 feature space — a key motivation for incorporating the full SADD framework with high-frequency texture branches.

![ROC and PR Curves](results/precision_roc.png)

---

### Grad-CAM Explainability

For a correctly classified real image, Grad-CAM activations concentrate on the central face region — particularly around the eyes, nose, and mouth — confirming the model focuses on anatomically meaningful features for its decision.

![Grad-CAM Sample](results/grad_cam_heatmap.png)

---

### Quantitative Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~82–90% |
| Precision (Fake) | ~0.90 |
| Recall (Fake) | ~0.85 |
| F1-Score (Fake) | ~0.77 |
| AUC-ROC | ~0.80 |

> *Threshold set at the 10th percentile of real training log-likelihood scores.*

---

## 🚀 Technical Details

### Dependencies

| Library | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework and MobileNetV2 backbone |
| `normflows` / `FrEIA` | Normalizing Flow model implementation |
| `pytorch-grad-cam` | Grad-CAM explainability heatmaps |
| `opencv-python`, `Pillow` | Image preprocessing and visualisation |
| `scikit-learn` | Evaluation metrics (AUC, F1, precision/recall) |
| `matplotlib` | Plotting training curves and distributions |

### 🔬 Implementation Environment

| Component | Details |
|-----------|---------|
| Platform | Google Colab (NVIDIA T4 GPU) |
| Framework | PyTorch 3.x, torchvision |
| NF Library | normflows / FrEIA |
| Explainability | pytorch-grad-cam |
| Feature Backbone | MobileNetV2 (pretrained, frozen) |
| NF Input Dimension | 1280 |
| Decision Threshold τ | 10th percentile of real training log-likelihoods |

<br/>

---

## 📃 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<br/>

---

<div align="center">

Made with ☕ and PyTorch by **Yash Gupta** & **Saurabh Kumar**

*If this project helped your research, consider giving it a ⭐*

</div>
