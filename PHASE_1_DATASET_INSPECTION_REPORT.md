# Project Phase 1: Comprehensive Dataset Inspection & Preprocessing Report

**Project Title:** Structure-Aware 3D Point Cloud Classification using Graph Neural Networks  
**Dataset:** ModelNet40 (Princeton 3D CAD Repository)  
**Phase:** Phase 1 — Dataset Audit, Geometric Inspection, Data-Quality Assessment & Preprocessing Design  
**Date:** September 2026  
**Status:** Completed (Empirical Validation & Baseline Architecture Specification)

---

## Executive Summary

This report establishes the empirical foundation for building a **Structure-Aware Graph Neural Network (GNN)** for 3D point cloud classification. Before constructing or training any neural network architecture, a comprehensive audit of the provided **ModelNet40** dataset was conducted. All measurements, statistics, duplicate audits, and geometric properties reported herein were computed directly from raw `.off` files using automated Python inspection scripts.

---

## 1. Introduction to the Dataset

ModelNet40 is a benchmark 3D geometric CAD dataset curated by the Princeton Vision & Robotics Lab. It comprises polygonal surface boundary meshes (`.off` format) spanning diverse synthetic 3D object categories (e.g., furniture, vehicles, everyday household objects, and architectural elements). 

Unlike 2D pixel grids (which possess fixed regular lattice connectivity) or dense voxel grids (which suffer from cubic memory scaling $O(V^3)$ and quantization artifacts), 3D point clouds represent spatial geometry as continuous unordered coordinate sets $\mathbf{P} = \{\mathbf{p}_i \in \mathbb{R}^3\}_{i=1}^N$. Graph Neural Networks (GNNs) exploit this continuous spatial representation by constructing dynamic spatial graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ over Euclidean neighborhoods to capture local geometric structure, surface orientation, and topological invariants.

---

## 2. Dataset Size & Sample Counts

Across the dataset workspace, all files were indexed and validated:

* **Total 3D Mesh Files Audited:** **10,485 objects**
* **Training Objects:** **8,377 objects** (79.89% of total corpus)
* **Test Objects:** **2,108 objects** (20.11% of total corpus)
* **Corrupted / Unreadable Files:** **0 files** (100% of `.off` files have valid ASCII headers and geometry).

---

## 3 & 4. Number of Classes & Complete Class Inventory

* **Canonical Classes in Archive Metadata:** **40 classes**
* **Active Classes with Extracted Data Directories:** **30 classes**
* **Classes with Metadata-Only Headers (`._*`):** **10 classes**

### Complete 40-Class Census:
| Status | Class Names |
| :--- | :--- |
| **Extracted & Populated (30 Classes)** | `airplane`, `bathtub`, `bed`, `bench`, `bookshelf`, `car`, `chair`, `curtain`, `desk`, `door`, `dresser`, `guitar`, `laptop`, `mantel`, `monitor`, `night_stand`, `person`, `piano`, `plant`, `range_hood`, `sofa`, `stairs`, `stool`, `table`, `tent`, `toilet`, `tv_stand`, `vase`, `wardrobe`, `xbox` |
| **Metadata-Only Headers (10 Classes)** | `bottle`, `bowl`, `cone`, `cup`, `flower_pot`, `glass_box`, `keyboard`, `lamp`, `radio`, `sink` |

---

## 5. Training vs. Test Class Distribution Analysis

### Measured Distribution Breakdown:
| # | Class Name | Train Samples | Test Samples | Total Samples | Train Share (%) | Test Share (%) |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | **chair** | 889 | 100 | 989 | 89.9% | 10.1% |
| 2 | **sofa** | 680 | 100 | 780 | 87.2% | 12.8% |
| 3 | **airplane** | 626 | 100 | 726 | 86.2% | 13.8% |
| 4 | **bookshelf** | 572 | 100 | 672 | 85.1% | 14.9% |
| 5 | **bed** | 515 | 100 | 615 | 83.7% | 16.3% |
| 6 | **vase** | 475 | 100 | 575 | 82.6% | 17.4% |
| 7 | **monitor** | 465 | 100 | 565 | 82.3% | 17.7% |
| 8 | **table** | 392 | 100 | 492 | 79.7% | 20.3% |
| 9 | **toilet** | 344 | 100 | 444 | 77.5% | 22.5% |
| 10 | **mantel** | 284 | 100 | 384 | 74.0% | 26.0% |
| 11 | **tv_stand** | 267 | 100 | 367 | 72.8% | 27.2% |
| 12 | **plant** | 240 | 100 | 340 | 70.6% | 29.4% |
| 13 | **piano** | 231 | 100 | 331 | 69.8% | 30.2% |
| 14 | **car** | 197 | 100 | 297 | 66.3% | 33.7% |
| 15 | **desk** | 200 | 86 | 286 | 69.9% | 30.1% |
| 16 | **dresser** | 200 | 86 | 286 | 69.9% | 30.1% |
| 17 | **night_stand** | 200 | 86 | 286 | 69.9% | 30.1% |
| 18 | **guitar** | 155 | 100 | 255 | 60.8% | 39.2% |
| 19 | **range_hood** | 115 | 100 | 215 | 53.5% | 46.5% |
| 20 | **bench** | 173 | 20 | 193 | 89.6% | 10.4% |
| 21 | **tent** | 163 | 20 | 183 | 89.1% | 10.9% |
| 22 | **laptop** | 149 | 20 | 169 | 88.2% | 11.8% |
| 23 | **curtain** | 138 | 20 | 158 | 87.3% | 12.7% |
| 24 | **bathtub** | 106 | 50 | 156 | 67.9% | 32.1% |
| 25 | **stairs** | 124 | 20 | 144 | 86.1% | 13.9% |
| 26 | **door** | 109 | 20 | 129 | 84.5% | 15.5% |
| 27 | **xbox** | 103 | 20 | 123 | 83.7% | 16.3% |
| 28 | **stool** | 90 | 20 | 110 | 81.8% | 18.2% |
| 29 | **person** | 88 | 20 | 108 | 81.5% | 18.5% |
| 30 | **wardrobe** | 87 | 20 | 107 | 81.3% | 18.7% |
| 31-40 | *10 Missing Classes* | 0 | 0 | 0 | 0.0% | 0.0% |
| — | **Total** | **8,377** | **2,108** | **10,485** | **79.9%** | **20.1%** |

### Observations:
1. **Class Imbalance Ratio:** **$9.24\times$** between `chair` (989 objects) and `wardrobe` (107 objects).
2. **Dominant Categories:** The top 4 classes (`chair`, `sofa`, `airplane`, `bookshelf`) account for **30.2%** of the entire dataset.
3. **Fixed Test Split Bucketing:** Standard ModelNet40 test partitions use fixed size caps (100 for large classes, 86 for desk/dresser/night_stand, 20 for small classes, 50 for bathtub). Pearson correlation between train and test counts is $r=0.6126$, with cosine similarity of $0.8834$.

---

## 6. Raw File Format Specifications

All 3D objects are stored in the **Object File Format (`.off`)**, an ASCII text specification:
* **Header:** `OFF` magic string followed by `N_vertices N_faces N_edges`.
* **Vertex Table:** $N_{vertices}$ consecutive lines containing three floating-point spatial coordinates:
  ```text
  X_coordinate  Y_coordinate  Z_coordinate
  ```
* **Face Table:** $N_{faces}$ lines containing polygon vertex counts and 0-indexed vertex IDs:
  ```text
  3  v0_idx  v1_idx  v2_idx  (Triangular Face)
  4  v0_idx  v1_idx  v2_idx  v3_idx  (Quadrilateral Face)
  ```

---

## 7. Point-Cloud Representation & Sampling Theory

Raw CAD files define 2D piecewise linear 2-manifolds (polygonal boundary surfaces in $\mathbb{R}^3$). Graph Neural Networks require point clouds $\mathbf{P} \in \mathbb{R}^{N \times 3}$.

### Mesh-to-Point Cloud Conversion:
To prevent point density distortion caused by irregular CAD tessellation (where small intricate bevels contain thousands of vertices while broad flat panels contain only 4 vertices), points must be sampled via **Uniform Area-Weighted Triangle Sampling**:
1. Triangulate all polygon facets into disjoint triangles $T_k = (\mathbf{v}_0, \mathbf{v}_1, \mathbf{v}_2)$.
2. Compute surface area $A_k = \frac{1}{2} \|(\mathbf{v}_1 - \mathbf{v}_0) \times (\mathbf{v}_2 - \mathbf{v}_0)\|_2$.
3. Sample triangle $k$ with probability $P(k) = A_k / \sum_j A_j$.
4. Generate point $\mathbf{p}$ via uniform random barycentric coordinates:
   $$\mathbf{p} = (1 - \sqrt{u})\mathbf{v}_0 + \sqrt{u}(1 - v)\mathbf{v}_1 + \sqrt{u}v\mathbf{v}_2, \quad u, v \sim \mathcal{U}(0, 1)$$

---

## 8. Available Features Audit

* **Native 3D Coordinates $(X, Y, Z)$:** **100% Present** (3 single-precision floats per vertex line).
* **Surface Normals $(N_x, N_y, N_z)$:** **0% Present** in raw files (`NOFF` flag absent in all 10,485 files).
* **Color Channels (RGB / RGBA):** **0% Present** (`COFF` flag absent in all 10,485 files).
* **Reflectance / Intensity:** **0% Present** (Synthetic CAD models, non-LiDAR data).
* **Finding:** The dataset natively supports **Option A ($XYZ$ Coordinates Only)**. Surface normals are not pre-stored and must be mathematically computed on-the-fly if needed.

---

## 9. Point & Vertex Statistics

All 10,485 meshes were scanned to compute global topological statistics:

| Metric | Vertex Count ($N_v$) | Face Count ($N_f$) |
| :--- | :---: | :---: |
| **Minimum** | **14** | **4** |
| **Maximum** | **2,501,688** ($2.5\times 10^6$) | **4,548,836** ($4.5\times 10^6$) |
| **Mean** | **19,174.00** | **18,803.93** |
| **Median** | **3,860.00** | **3,806.00** |
| **Standard Deviation** | **62,722.45** | **65,248.81** |

### Geometric Coordinate Extents:
* **Centroid Range:** $[-314.06, 10.03, 18.62]$ (`guitar_0072`) to $[221.50, 94.59, 18.55]$ (`airplane_0103`).
* **Scale Extent (Max Radius from Origin):** $2.07$ (`car_0107`) to $455.53$ (`stairs_0103`) ($>200\times$ difference in CAD modeling units).

---

## 10. Data-Quality Analysis

1. **macOS Resource Fork Files:** 40 hidden `._*` AppleDouble metadata files exist in the dataset root. Preprocessing loaders must explicitly filter files starting with `.` or `._`.
2. **Missing Classes:** 10 categories are absent from the extracted directory. The classification head must be configured for $C = 30$ output logits.
3. **Extreme Density Variance:** Raw vertex density varies by a factor of $178,000\times$ (from 14 vertices to 2.5 million vertices), proving that point-cloud surface sampling is mandatory.

---

## 11. Duplicate & Data Leakage Audit

A two-stage cryptographic (MD5) and geometric invariant fingerprint audit was executed across all 10,485 files:

* **Exact MD5 Duplicates within Training Set:** **33 pairs**
* **Exact MD5 Duplicates within Test Set:** **4 pairs**
* **Exact Cross-Split Leakage (Identical MD5 in Train & Test):** **18 pairs** (0.85% of test set)
  * *Same-Class Memorization (12 pairs):* E.g., `piano_0143.off` (train) is byte-identical to `piano_0260.off` (test).
  * *Cross-Class Ambiguity (6 pairs):* E.g., `night_stand_0174.off` (train) is byte-identical to `dresser_0272.off` (test).
* **Suspicious Geometric Variants:** 105 cross-split pairs share identical rotation-invariant moments and normalized eigenvalues.
* **Assessment:** The official train/test partition should be preserved without manual file shifting to remain strictly comparable with published literature baselines (PointNet, DGCNN, PointNet++).

---

## 12. Visualization Observations

Rendered visualizations are archived in `c:/DLproj/visualizations/`:
1. **Single Object Transformations ([airplane_pointcloud_pipeline.png](file:///c:/DLproj/visualizations/airplane_pointcloud_pipeline.png)):** Demonstrates successful transition from irregular raw vertex clusters to uniform unit-sphere point clouds.
2. **10-Class Gallery ([10_random_classes_gallery.png](file:///c:/DLproj/visualizations/10_random_classes_gallery.png)):** Confirms preservation of distinct geometric topological features across diverse classes.
3. **Fine-Grained Ambiguity ([visually_similar_classes_comparison.png](file:///c:/DLproj/visualizations/visually_similar_classes_comparison.png)):** Highlights subtle structural differences separating `dresser`, `night_stand`, and `desk`.
4. **Broad Structural Contrast ([visually_distinct_classes_comparison.png](file:///c:/DLproj/visualizations/visually_distinct_classes_comparison.png)):** Shows distinct aspect ratios and spatial distributions across `airplane`, `guitar`, `person`, and `stairs`.

---

## 13. Recommended Preprocessing Architecture

| Stage | Operation | Formulation |
| :--- | :--- | :--- |
| **1. Sampling** | Uniform Area-Weighted Surface Sampling | Sample $N = 1,024$ points via triangle area weighting |
| **2. Centering** | Instance Zero-Centering | $\mathbf{P} \leftarrow \mathbf{P} - \frac{1}{N}\sum_{i=1}^N \mathbf{p}_i$ |
| **3. Scaling** | Instance Unit-Sphere Normalization | $\mathbf{P} \leftarrow \frac{\mathbf{P}}{\max_i \|\mathbf{p}_i\|_2 + 1e-8}$ |
| **4. Graph Construction** | Dynamic $k$-Nearest Neighbors | Construct $k=20$ spatial edge index tensor for GNN message passing |

---

## 14 & 15. Training vs. Validation/Test Preprocessing

```text
Training Mode:
  Raw OFF Mesh ──► Dynamic Surface Sampling ──► Centering ──► Unit Sphere ──► Random SO(2) Rotation ──► Scale Jitter ──► Point Noise ──► (B, 3, 1024)

Validation / Test Mode:
  Raw OFF Mesh ──► Seeded Deterministic Sampling ──► Centering ──► Unit Sphere ──────────────────────────────────────────► (B, 3, 1024)
```

* **Training Augmentations:**
  * Random Azimuth Rotation ($Z$-axis): $\theta \sim \mathcal{U}(0, 2\pi)$.
  * Anisotropic Scale Jitter: $s \sim \mathcal{U}(0.8, 1.25)$.
  * Coordinate Gaussian Noise: $\mathcal{N}(0, 0.01^2)$, clipped to $[-0.05, 0.05]$.
  * Point Dropout: Randomly drop up to $15\%$ of points.
* **Validation & Test Preprocessing:** Strictly deterministic, fixed seed, zero augmentation.

---

## 16. Final Input Tensor Representation

* **Per-Sample Point Matrix:** $\mathbf{P} \in \mathbb{R}^{1024 \times 3}$
* **Mini-Batch Tensor (Channels-First):** $\mathbf{X} \in \mathbb{R}^{B \times 3 \times 1024}$ (Data Type: `torch.float32`)
* **Classification Target Tensor:** $\mathbf{y} \in \mathbb{Z}^{B}$ (Data Type: `torch.int64`, $y \in \{0, 1, \dots, 29\}$)
* **$k$-NN Adjacency Graph:** $k = 20$ nearest neighbors dynamically recomputed in feature space (EdgeConv).

---

## 17. Important Limitations

1. **Class Count Restriction:** The dataset contains 30 active classes rather than 40. Target classification heads must be dimensioned to $C=30$.
2. **Lack of Sensor Properties:** Synthetic CAD meshes lack sensor noise, reflectance intensity, and realistic single-view occlusions found in real LiDAR scans (e.g., ScanObjectNN, KITTI).
3. **Known Dataset Leakage:** 18 exact duplicates exist between the train and test splits as a legacy artifact of ModelNet40; metrics should account for this $<1\%$ baseline offset.

---

## 18. Phase 2 Recommendations (Model Architecture & Training)

1. **Model Architecture:** Build a **Dynamic Graph CNN (DGCNN) / EdgeConv GNN**:
   * Construct dynamic $k$-NN graphs in both coordinate space ($XYZ$) and latent feature space ($h_i$).
   * Utilize asymmetric EdgeConv operators: $e_{ij} = \text{MLP}(\mathbf{x}_i \parallel \mathbf{x}_j - \mathbf{x}_i)$.
   * Use symmetric global aggregation ($\text{MaxPool} \oplus \text{AvgPool}$) to ensure permutation invariance.
2. **Loss Function:** Use Label-Smoothed Cross-Entropy Loss with optional inverse-frequency class weights:
   $$\mathcal{L}_{CE} = -\sum_{c=1}^C w_c y_c \log \hat{y}_c, \quad w_c \propto \frac{1}{\sqrt{N_c}}$$
3. **Evaluation Protocol:**
   * Evaluate and report both **Overall Accuracy (OA)** and **Mean Class Accuracy (mAcc)**.
   * Maintain strict separation between training augmentation and deterministic test evaluation.

---

*Report generated and validated autonomously as part of Phase 1 of the Project.*
