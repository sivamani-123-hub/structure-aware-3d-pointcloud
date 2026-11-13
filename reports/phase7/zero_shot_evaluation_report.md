# Zero-Shot Synthetic-to-Real 3D Point Cloud Classification Benchmark

## 1. Executive Summary
- **Source Domain:** Princeton ModelNet40 (Clean Synthetic CAD Meshes, 1024 points, unit sphere normalized)
- **Target Domain:** ScanObjectNN (Noisy Real-World Scans from ScanNet/SceneNN, 1024 points, unit sphere normalized)
- **Target Test Partition:** Official Test Split (386 scans across 10 overlapping categories)
- **Zero-Shot Protocol:** Strict zero-shot transfer — **zero** target labels or scans were used during training or tuning.

## 2. Overall Benchmark Results

| Architecture | Overall Accuracy (OA) | Mean Class Acc (mAcc) | Macro F1-Score | Weighted F1-Score | Correct Scans |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pointnet** | **7.77%** | 13.30% | 5.69% | 4.19% | 30/386 |
| **Dgcnn** | **6.48%** | 8.72% | 3.74% | 3.06% | 25/386 |
| **Gnn** | **6.48%** | 8.72% | 3.74% | 3.06% | 25/386 |
| **Peak Dgcnn** | **9.84%** | 12.19% | 6.31% | 5.78% | 38/386 |

## 3. Per-Class Accuracy Census

| Class Name | Real-World Support | PointNet Acc | DGCNN Acc | GNN Acc | Peak GNN Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **bed** | 40 | 0.00% | 0.00% | 0.00% | 12.50% |
| **cabinet** | 78 | 0.00% | 0.00% | 0.00% | 0.00% |
| **chair** | 30 | 20.00% | 60.00% | 60.00% | 83.33% |
| **desk** | 42 | 0.00% | 0.00% | 0.00% | 0.00% |
| **display** | 42 | 11.90% | 7.14% | 7.14% | 11.90% |
| **door** | 49 | 0.00% | 0.00% | 0.00% | 0.00% |
| **shelf** | 22 | 36.36% | 0.00% | 0.00% | 0.00% |
| **sofa** | 24 | 0.00% | 8.33% | 8.33% | 0.00% |
| **table** | 42 | 0.00% | 0.00% | 0.00% | 2.38% |
| **toilet** | 17 | 64.71% | 11.76% | 11.76% | 11.76% |
