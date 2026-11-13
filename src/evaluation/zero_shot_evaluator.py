"""
Zero-Shot Cross-Domain Evaluation Engine.

Evaluates ModelNet40-trained models on ScanObjectNN real-world scans.
Computes:
- Overall Accuracy (OA)
- Mean Class Accuracy (mAcc)
- Macro & Weighted Precision, Recall, F1
- Per-class metrics
- Annotated Confusion Matrices
- Misclassification error diagnostics
"""

import os
import sys
import json
import csv
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# DLproj imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.datasets.class_mapping import (
    COMMON_CLASSES_INFO,
    COMMON_CLASS_NAMES,
    MODELNET40_CLASSES,
    select_common_logits,
    compute_zero_shot_probabilities
)


def evaluate_zero_shot(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str = "dgcnn",
    eval_mode: str = "constrained",  # 'constrained' (10 common classes) or 'unconstrained' (40 classes)
    save_cm_path: Optional[str] = None,
    save_gallery_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes rigorous zero-shot evaluation on the target test dataloader.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_confidences = []
    sample_records = []
    
    class_names = COMMON_CLASS_NAMES if eval_mode == "constrained" else MODELNET40_CLASSES
    num_classes = len(class_names)

    with torch.no_grad():
        for batch_idx, (points, labels, meta) in enumerate(dataloader):
            points = points.to(device, non_blocking=True)
            
            # Forward pass through ModelNet40-trained architecture
            logits_40, _ = model(points)
            
            if eval_mode == "constrained":
                # Sub-softmax over 10 common classes
                sub_logits = select_common_logits(logits_40)
                probs = F.softmax(sub_logits, dim=-1)
                confidences, preds = probs.max(dim=-1)
                target_batch = labels.numpy()
            else:
                # Open-world 40-class argmax
                probs = F.softmax(logits_40, dim=-1)
                confidences, preds = probs.max(dim=-1)
                target_batch = meta["mn40_label"].numpy()

            preds_np = preds.cpu().numpy()
            confs_np = confidences.cpu().numpy()

            all_preds.extend(preds_np)
            all_targets.extend(target_batch)
            all_confidences.extend(confs_np)

            # Store sample diagnostics
            for i in range(len(preds_np)):
                p_idx = int(preds_np[i])
                t_idx = int(target_batch[i])
                is_correct = bool(p_idx == t_idx)
                
                # Save first 50 points coordinates for sample visualization if needed
                sample_records.append({
                    "sample_idx": len(sample_records),
                    "true_class_id": t_idx,
                    "true_class_name": class_names[t_idx] if t_idx < len(class_names) else f"cls_{t_idx}",
                    "pred_class_id": p_idx,
                    "pred_class_name": class_names[p_idx] if p_idx < len(class_names) else f"cls_{p_idx}",
                    "confidence": float(confs_np[i]),
                    "is_correct": is_correct
                })

    all_preds = np.array(all_preds, dtype=np.int64)
    all_targets = np.array(all_targets, dtype=np.int64)
    all_confidences = np.array(all_confidences, dtype=np.float32)

    total_samples = len(all_targets)
    correct_mask = (all_preds == all_targets)
    num_correct = int(correct_mask.sum())
    num_incorrect = total_samples - num_correct
    overall_accuracy = float(num_correct / total_samples)

    # 1. Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(all_targets, all_preds):
        if t < num_classes and p < num_classes:
            cm[t, p] += 1

    # 2. Per-Class Metrics
    per_class_metrics = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    for c in range(num_classes):
        c_name = class_names[c]
        support = int((all_targets == c).sum())
        correct = int(((all_targets == c) & (all_preds == c)).sum())
        acc = float(correct / support) if support > 0 else 0.0

        pred_total = int((all_preds == c).sum())
        prec = float(correct / pred_total) if pred_total > 0 else 0.0
        rec = acc
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0

        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
            supports.append(support)

        per_class_metrics.append({
            "class_id": c,
            "class_name": c_name,
            "support": support,
            "correct": correct,
            "incorrect": support - correct,
            "accuracy": acc,
            "accuracy_pct": f"{acc * 100:.2f}%",
            "precision": prec,
            "precision_pct": f"{prec * 100:.2f}%",
            "recall": rec,
            "recall_pct": f"{rec * 100:.2f}%",
            "f1_score": f1,
            "f1_score_pct": f"{f1 * 100:.2f}%"
        })

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    macro_accuracy = float(np.mean([d["accuracy"] for d in per_class_metrics if d["support"] > 0]))

    total_supp = sum(supports)
    weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total_supp) if total_supp > 0 else 0.0
    weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total_supp) if total_supp > 0 else 0.0
    weighted_f1 = float(sum(f * s for f, s in zip(f1_scores, supports)) / total_supp) if total_supp > 0 else 0.0

    # 3. Plot & Save Annotated Confusion Matrix
    if save_cm_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_cm_path)), exist_ok=True)
        cm_norm = np.zeros_like(cm, dtype=np.float32)
        row_sums = cm.sum(axis=1)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                cm_norm[i] = cm[i] / row_sums[i]

        fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
        cmap = plt.cm.Blues if "pointnet" in model_name.lower() else plt.cm.Greens if "dgcnn" in model_name.lower() else plt.cm.Purples
        cax = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9, fontweight='bold')
        ax.set_yticklabels(class_names, fontsize=9, fontweight='bold')

        # Annotate cells with count and percentage
        thresh = cm_norm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100
                if count > 0:
                    text_str = f"{pct:.0f}%\n({count})" if num_classes <= 15 else f"{pct:.0f}%"
                    ax.text(j, i, text_str,
                            ha="center", va="center",
                            color="white" if cm_norm[i, j] > thresh else "black",
                            fontsize=7 if num_classes <= 15 else 5,
                            fontweight='bold')

        ax.set_ylabel('True Real-World Class (ScanObjectNN)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Class (ModelNet40-Trained)', fontsize=11, fontweight='bold')
        title_str = f"Zero-Shot Cross-Domain Confusion Matrix: {model_name.upper()}\nOverall Accuracy: {overall_accuracy*100:.2f}% | Mean Class Acc: {macro_accuracy*100:.2f}%"
        ax.set_title(title_str, fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(save_cm_path, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix plot to: {save_cm_path}")

    # 4. Extract Misclassification Diagnostics
    incorrect_cases = [s for s in sample_records if not s["is_correct"]]
    correct_cases = [s for s in sample_records if s["is_correct"]]

    results = {
        "model_name": model_name,
        "evaluation_mode": eval_mode,
        "total_test_samples": total_samples,
        "correct_samples": num_correct,
        "incorrect_samples": num_incorrect,
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_pct": f"{overall_accuracy * 100:.2f}%",
        "mean_class_accuracy": macro_accuracy,
        "mean_class_accuracy_pct": f"{macro_accuracy * 100:.2f}%",
        "macro_precision": macro_precision,
        "macro_precision_pct": f"{macro_precision * 100:.2f}%",
        "macro_recall": macro_recall,
        "macro_recall_pct": f"{macro_recall * 100:.2f}%",
        "macro_f1": macro_f1,
        "macro_f1_pct": f"{macro_f1 * 100:.2f}%",
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "weighted_f1_pct": f"{weighted_f1 * 100:.2f}%",
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "sample_records": sample_records,
        "num_errors": len(incorrect_cases),
        "error_rate_pct": f"{(len(incorrect_cases) / total_samples) * 100:.2f}%"
    }

    return results

