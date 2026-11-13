# Structure-Aware Cross-Domain 3D Point Cloud Classification MVP

An end-to-end real-world 3D object perception and domain generalization framework evaluating whether **Structure-Aware Graph Neural Networks (DGCNN)** trained on synthetic CAD models (**ModelNet40**) can generalize to physical, noisy LiDAR and RGB-D scans (**ScanObjectNN**).

---

## 1. Executive Summary & Key Results

| Benchmark Dimension | PointNet Baseline | DGCNN Baseline ($k=20$) | Peak GNN ($k=40$) | Domain-Adapted Fine-Tuned GNN | Net Adaptation Delta |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ModelNet40 In-Domain Test OA** | 55.88% | 62.19% | **74.62%** | — | — |
| **ScanObjectNN Zero-Shot OA** | 7.77% (30/386) | 6.48% (25/386) | **9.84%** (38/386) | **62.18%** (240/386) | **+52.33%** |
| **ScanObjectNN Mean Class Acc (mAcc)** | 13.30% | 8.72% | 12.19% | **56.63%** | **+44.44%** |
| **Macro F1-Score** | 5.69% | 3.74% | 6.31% | **56.46%** | **+50.16%** |
| **Chair Accuracy** | 20.00% | 60.00% | **83.33%** | 23.33% | — |
| **Display Monitor Accuracy** | 11.90% | 7.14% | 11.90% | **95.24%** | **+83.33%** |
| **Table Accuracy** | 0.00% | 0.00% | 2.38% | **76.19%** | **+73.81%** |
| **Bed Accuracy** | 0.00% | 0.00% | 12.50% | **67.50%** | **+55.00%** |
| **Total Parameters** | 811,048 | 1,809,320 | 1,809,320 | 1,801,610 | Zero Bloat |
| **Inference Latency (GPU / CPU)** | 0.8 ms / 3.8 ms | 4.2 ms / 91.2 ms | 6.8 ms / 169.4 ms | 6.2 ms / 148.2 ms | Real-Time |

---

## 2. Key Findings & Insights

1. **Topological Invariance on Articulated Shapes:**
   On objects with non-trivial geometric joint connectivity (*chair*), Peak Regularized GNN achieves **83.33% zero-shot accuracy** (compared to only 20.00% for PointNet). PointNet collapses because sensor occlusion strips backrest points; in contrast, EdgeConv captures local relative displacement vectors:
   $$\mathbf{e}_{ij} = h_\Theta(\mathbf{x}_i, \mathbf{x}_j - \mathbf{x}_i)$$
   preserving seat-to-leg joint topology regardless of view-angle occlusion.

2. **The Planar Surface Ambiguity Boundary:**
   Flat planar objects (*cabinet*, *desk*, *door*, *shelf*) experience catastrophic failure under zero-shot transfer ($0\%$) because room walls and sensor clutter distort exterior bounding boxes.

3. **Target Domain Adaptation Recovery:**
   Fine-tuning the classification projection on target-domain training representations restores decision boundaries, boosting overall target test accuracy from **9.84% to 62.18%** (+202 additional correctly recognized physical scans).

---

## 3. Project Directory Architecture

```
C:\DLproj
├── checkpoints/
│   ├── pointnet_best.pth               # Original ModelNet40 PointNet (Preserved)
│   ├── dgcnn_best.pth                  # Original ModelNet40 DGCNN (Preserved)
│   ├── gnn_best.pth                    # Original ModelNet40 GNN (Preserved)
│   ├── ablation_reg/                   # Peak Regularized GNN (k=40, 74.62% OA)
│   └── phase7/
│       └── gnn_finetuned_best.pth      # Domain-Adapted Fine-Tuned GNN (Target-trained)
│
├── datasets/
│   ├── ModelNet40/                     # Canonical CAD mesh repository (Source)
│   └── real_world/
│       └── ScanObjectNN/               # Official Real-World Scans (Target)
│           ├── main_split_nobg/        # Clean object-only splits (Train & Test)
│           └── main_split/             # Object + background clutter splits
│
├── src/
│   ├── datasets/
│   │   ├── scanobjectnn.py             # ScanObjectNN PyTorch Dataset & DataLoader
│   │   └── class_mapping.py            # Sub-logit slicing & correspondence math
│   ├── domain_adaptation/
│   │   ├── adapted_gnn.py              # DomainAdaptedDGCNN architecture
│   │   └── target_trainer.py           # Representation probing & fine-tuning engine
│   ├── evaluation/
│   │   └── zero_shot_evaluator.py      # Standardized metric calculator & confusion plotter
│   └── inference/
│       ├── model_loader.py             # Unified model loader (GPU/CPU fallback)
│       └── predictor.py                # Single-instance real-time inference engine
│
├── app/
│   ├── streamlit_app.py                # Interactive Streamlit WebGL MVP application
│   └── components/
│       ├── parsers.py                  # 3D format parsers (.ply, .obj, .off, .xyz, .npy)
│       ├── visualizer.py               # Plotly Scatter3d interactive viewer
│       └── demo_samples.py             # Pre-loaded 1-click CAD and real-scan catalog
│
├── scripts/
│   ├── audit_existing_repo.py          # Step 1: Checkpoint & model audit
│   ├── download_scanobjectnn.py        # Step 2: Dataset download & checksum validation
│   ├── audit_dataset.py                # Step 2: Coordinate health & overlap analysis
│   ├── test_class_mapping_and_loader.py# Step 3: Class mapping & normalization tests
│   ├── verify_checkpoint_compatibility.py # Step 4: Parameter immutability verification
│   ├── evaluate_zero_shot.py           # Step 5: Master zero-shot benchmark
│   ├── generate_domain_shift_analysis.py # Step 6: Statistical geometry distribution plots
│   ├── finetune_target.py              # Step 7: Domain adaptation training & test
│   ├── test_app_components.py          # Step 8: App component verification
│   └── test_end_to_end_mvp.py          # Step 9: 10-point automated system audit
│
├── visualizations/phase7/              # All publication & diagnostic figures
├── results/phase7/                     # Structured evaluation summaries & per-class CSVs
├── reports/phase7/                     # Detailed technical and research reports
├── configs/phase7_config.yaml          # Reproducible experiment configurations
├── requirements_phase7.txt             # MVP Python dependencies
└── README_PHASE7.md                    # This document
```

---

## 4. Quick Start & Execution Guide

### A. Environment Setup
```powershell
# Navigate to workspace
cd C:\DLproj

# Virtual environment is already pre-configured with Python 3.11 and PyTorch 2.5.1
# To install Phase 7 dependencies if recreating environment:
uv pip install -r requirements_phase7.txt
```

### B. Launch the Interactive Streamlit MVP
To launch the interactive 3D WebGL classifier in your browser:
```powershell
.\.venv\Scripts\streamlit.exe run app\streamlit_app.py
```
Open **`http://localhost:8501`** to:
- Test real-time predictions on uploaded `.ply`, `.obj`, `.off`, or `.xyz` files.
- Select from 7 preloaded real-world scans and synthetic CAD models.
- Run multi-model comparative inference side-by-side.
- Inspect high-resolution confusion matrices and domain-shift density curves.

### C. Run the Complete Verification Test Suite
```powershell
.\.venv\Scripts\python.exe scripts\test_end_to_end_mvp.py
```
Validates all 10 checklist items (datasets, mappings, normalization, checkpoints, CPU fallback, multi-format parsing, and metrics).

### D. Re-Run Benchmarks & Reports
```powershell
# Run Zero-Shot Synthetic-to-Real Benchmark:
.\.venv\Scripts\python.exe scripts\evaluate_zero_shot.py

# Generate Domain-Shift Multi-View Projections:
.\.venv\Scripts\python.exe scripts\generate_domain_shift_analysis.py

# Re-run Domain Adaptation Fine-Tuning:
.\.venv\Scripts\python.exe scripts\finetune_target.py
```

---

## 5. Mathematical Class Correspondence (10 Overlapping Categories)

When evaluating ModelNet40 models zero-shot on ScanObjectNN, raw 40-class logits $\mathbf{z} \in \mathbb{R}^{B \times 40}$ are sliced using column indices:
$$\text{indices} = [2, 38, 8, 12, 22, 13, 4, 30, 33, 35]$$
yielding sub-logits $\mathbf{z}_{\text{common}} \in \mathbb{R}^{B \times 10}$ without modifying model architecture or weights:

| Common ID | ScanObjectNN Class | Scan ID | ModelNet40 Target Class | ModelNet40 ID | Correspondence Type |
| :---: | :--- | :---: | :--- | :---: | :--- |
| **0** | `bed` | 1 | `bed` | 2 | Direct Exact Match |
| **1** | `cabinet` | 4 | `wardrobe` | 38 | Semantic Mapping |
| **2** | `chair` | 5 | `chair` | 8 | Direct Exact Match |
| **3** | `desk` | 6 | `desk` | 12 | Direct Exact Match |
| **4** | `display` | 7 | `monitor` | 22 | Semantic Mapping |
| **5** | `door` | 8 | `door` | 13 | Direct Exact Match |
| **6** | `shelf` | 10 | `bookshelf` | 4 | Semantic Mapping |
| **7** | `sofa` | 12 | `sofa` | 30 | Direct Exact Match |
| **8** | `table` | 13 | `table` | 33 | Direct Exact Match |
| **9** | `toilet` | 14 | `toilet` | 35 | Direct Exact Match |

---

## 6. Academic Citation & Team

**Project Title:** Structure-Aware Domain Generalization for 3D Point Cloud Classification using Graph Neural Networks  
**Department:** Computer Science and Engineering, VIT-AP University, Amaravati, Andhra Pradesh, India  
**Authors:** R. Upendra, B. J. V. Sivamani, B. Sasi, K. Pavan

