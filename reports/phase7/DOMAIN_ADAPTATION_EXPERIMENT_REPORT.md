# Domain Adaptation & Fine-Tuning Experiment Report

## 1. Executive Summary
- **Base Architecture:** Peak Regularized Structure-Aware GNN ($k=40$, Dropout $0.3$, Weight Decay $10^{-3}$, pre-trained on ModelNet40)
- **Target Domain Data:** ScanObjectNN clean object split (`main_split_nobg`)
- **Data Isolation:** Target Training partition (1375 train, 242 validation). Target Test partition (386 scans) strictly untouched during all training phases.
- **Checkpoint Path:** `checkpoints/phase7/gnn_finetuned_best.pth` (Zero mutation of original pre-trained checkpoints).

## 2. Quantitative Comparison: Zero-Shot vs. Domain-Adapted GNN

| Metric | Zero-Shot Transfer | Domain-Adapted Fine-Tuned | Net Improvement (Delta) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy (OA)** | **9.84%** | **62.18%** | **+52.33%** |
| **Mean Class Accuracy (mAcc)** | 12.19% | 56.63% | **+44.44%** |
| **Macro F1-Score** | 6.31% | 56.46% | **+50.16%** |
| **Weighted F1-Score** | 5.78% | 61.48% | **+55.70%** |
| **Correct Real Scans** | 38 / 386 | **240 / 386** | **+202 correctly recognized scans** |

## 3. Per-Class Accuracy Analysis

| Class Name | Real Scans | Zero-Shot Acc (%) | Fine-Tuned Acc (%) | Accuracy Gain (Delta) |
| :--- | :---: | :---: | :---: | :---: |
| **bed** | 40 | 12.50% | **67.50%** | **+55.00%** |
| **cabinet** | 78 | 0.00% | **66.67%** | **+66.67%** |
| **chair** | 30 | 83.33% | **23.33%** | **-60.00%** |
| **desk** | 42 | 0.00% | **64.29%** | **+64.29%** |
| **display** | 42 | 11.90% | **95.24%** | **+83.33%** |
| **door** | 49 | 0.00% | **67.35%** | **+67.35%** |
| **shelf** | 22 | 0.00% | **45.45%** | **+45.45%** |
| **sofa** | 24 | 0.00% | **25.00%** | **+25.00%** |
| **table** | 42 | 2.38% | **76.19%** | **+73.81%** |
| **toilet** | 17 | 11.76% | **35.29%** | **+23.53%** |
