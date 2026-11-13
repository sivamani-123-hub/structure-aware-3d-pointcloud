import os
import sys
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from data.dataloader import get_modelnet40_dataloaders
from models.dgcnn import DGCNN

def run_detailed_dgcnn_test_evaluation(
    checkpoint_path=r"c:\DLproj\checkpoints\dgcnn_best.pth",
    dataset_root=r"c:\DLproj\dataset\ModelNet40",
    vis_dir=r"c:\DLproj\visualizations",
    out_dir=r"c:\DLproj\checkpoints",
    k=20,
    num_points=1024,
    num_classes=40,
    batch_size=32
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"DETAILED DGCNN TEST SET EVALUATION (Device: {device})")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # 1. Load Test Dataloader (Untouched test set, deterministic, zero augmentation)
    _, _, test_loader, class_names = get_modelnet40_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_points=num_points,
        num_classes=num_classes,
        val_split_ratio=0.1,
        num_workers=0,
        seed=42
    )
    
    # 2. Load DGCNN Best Checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DGCNN(num_classes=num_classes, k=k, dropout=0.5).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    loaded_epoch = checkpoint.get("epoch", "N/A")
    val_acc_recorded = checkpoint.get("val_acc", 0.0)
    print(f"Loaded DGCNN checkpoint from Epoch {loaded_epoch} (Validation OA: {val_acc_recorded*100:.2f}%)")
    
    # 3. Test Inference
    all_preds = []
    all_targets = []
    all_confidences = []
    all_points = []
    
    with torch.no_grad():
        for points, labels in test_loader:
            pts_cpu = points.numpy()
            points = points.to(device)
            logits, _ = model(points)
            probs = F.softmax(logits, dim=-1)
            confidences, preds = probs.max(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
            for i in range(points.size(0)):
                all_points.append(pts_cpu[i])
                
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    
    total_samples = len(all_targets)
    correct_mask = (all_preds == all_targets)
    num_correct = int(correct_mask.sum())
    num_incorrect = total_samples - num_correct
    overall_accuracy = num_correct / total_samples
    
    # 4. Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
        
    # 5. Per-Class Metrics
    per_class_data = []
    active_class_ids = [c for c in range(num_classes) if (all_targets == c).sum() > 0]
    
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for c in active_class_ids:
        c_name = class_names[c]
        support = int((all_targets == c).sum())
        correct = int(((all_targets == c) & (all_preds == c)).sum())
        acc = correct / support if support > 0 else 0.0
        
        pred_total = int((all_preds == c).sum())
        prec = correct / pred_total if pred_total > 0 else 0.0
        rec = acc
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)
        supports.append(support)
        
        per_class_data.append({
            "class_id": c,
            "class_name": c_name,
            "support": support,
            "correct": correct,
            "incorrect": support - correct,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        })
        
    # 6. Multiclass Averaging
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))
    macro_accuracy = float(np.mean([d["accuracy"] for d in per_class_data]))
    
    total_supp = sum(supports)
    weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total_supp)
    weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total_supp)
    weighted_f1 = float(sum(f * s for f, s in zip(f1_scores, supports)) / total_supp)
    
    # 7. Identify Highest and Lowest Performing Classes
    sorted_by_acc = sorted(per_class_data, key=lambda x: x["accuracy"], reverse=True)
    top_5_classes = sorted_by_acc[:5]
    bottom_5_classes = sorted_by_acc[-5:]
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    # 8. Confusion Matrix Plot
    cm_plot_path = os.path.join(vis_dir, "dgcnn_test_confusion_matrix.png")
    active_names = [class_names[c] for c in active_class_ids]
    sub_cm = cm[np.ix_(active_class_ids, active_class_ids)]
    sub_cm_norm = sub_cm.astype('float') / sub_cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(15, 13), dpi=200)
    plt.imshow(sub_cm_norm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('DGCNN Baseline: Normalized Confusion Matrix on ModelNet40 Test Set', fontsize=13, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(active_names))
    plt.xticks(tick_marks, active_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, active_names, fontsize=8)
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved DGCNN confusion matrix to: {cm_plot_path}")
    
    # 9. Per-Class Accuracy Chart
    per_class_bar_path = os.path.join(vis_dir, "dgcnn_per_class_accuracy.png")
    plt.figure(figsize=(16, 6), dpi=200)
    classes_names_sorted = [d["class_name"] for d in per_class_data]
    accs_sorted = [d["accuracy"] * 100 for d in per_class_data]
    
    bars = plt.bar(range(len(classes_names_sorted)), accs_sorted, color='#2ECC71', edgecolor='#27AE60', alpha=0.85)
    plt.axhline(y=overall_accuracy * 100, color='#E74C3C', linestyle='--', linewidth=1.5, label=f'Overall Accuracy: {overall_accuracy*100:.2f}%')
    plt.axhline(y=macro_accuracy * 100, color='#3498DB', linestyle=':', linewidth=1.5, label=f'Macro Mean Accuracy: {macro_accuracy*100:.2f}%')
    
    plt.ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    plt.title('DGCNN Baseline: Per-Class Accuracy on ModelNet40 Test Set', fontsize=13, fontweight='bold')
    plt.xticks(range(len(classes_names_sorted)), classes_names_sorted, rotation=90, fontsize=8)
    plt.ylim(0, 105)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(per_class_bar_path, bbox_inches='tight')
    plt.close()
    print(f"Saved DGCNN per-class accuracy bar chart to: {per_class_bar_path}")
    
    # 10. Example Correct Predictions (3D Gallery)
    correct_indices = np.where(correct_mask)[0]
    selected_correct = []
    seen_correct_classes = set()
    for idx in correct_indices:
        t_cls = all_targets[idx]
        if t_cls not in seen_correct_classes:
            selected_correct.append(idx)
            seen_correct_classes.add(t_cls)
        if len(selected_correct) >= 8:
            break
            
    correct_plot_path = os.path.join(vis_dir, "dgcnn_correct_examples_gallery.png")
    fig = plt.figure(figsize=(18, 9), dpi=200)
    fig.suptitle('DGCNN Baseline: Example Correct Predictions on Test Set', fontsize=15, fontweight='bold', y=0.98)
    
    for plot_idx, sample_idx in enumerate(selected_correct):
        ax = fig.add_subplot(2, 4, plot_idx + 1, projection='3d')
        pts = all_points[sample_idx]
        cls_name = class_names[all_targets[sample_idx]]
        conf = all_confidences[sample_idx] * 100
        
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='viridis', s=4, alpha=0.8)
        ax.set_title(f"Class: {cls_name}\nConfidence: {conf:.1f}%", fontsize=10, fontweight='bold', color='#1E824C')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        
    plt.tight_layout()
    plt.savefig(correct_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved correct 3D gallery to: {correct_plot_path}")
    
    # 11. Example Incorrect Predictions (3D Gallery)
    incorrect_indices = np.where(~correct_mask)[0]
    selected_incorrect = []
    seen_incorrect_classes = set()
    for idx in incorrect_indices:
        t_cls = all_targets[idx]
        if t_cls not in seen_incorrect_classes:
            selected_incorrect.append(idx)
            seen_incorrect_classes.add(t_cls)
        if len(selected_incorrect) >= 8:
            break
            
    incorrect_plot_path = os.path.join(vis_dir, "dgcnn_incorrect_examples_gallery.png")
    fig = plt.figure(figsize=(18, 9), dpi=200)
    fig.suptitle('DGCNN Baseline: Example Incorrect Predictions on Test Set', fontsize=15, fontweight='bold', y=0.98)
    
    for plot_idx, sample_idx in enumerate(selected_incorrect):
        ax = fig.add_subplot(2, 4, plot_idx + 1, projection='3d')
        pts = all_points[sample_idx]
        t_name = class_names[all_targets[sample_idx]]
        p_name = class_names[all_preds[sample_idx]]
        conf = all_confidences[sample_idx] * 100
        
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap='plasma', s=4, alpha=0.8)
        ax.set_title(f"True: {t_name}\nPred: {p_name} ({conf:.1f}%)", fontsize=10, fontweight='bold', color='#D9381E')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.grid(False)
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        
    plt.tight_layout()
    plt.savefig(incorrect_plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved incorrect 3D gallery to: {incorrect_plot_path}")
    
    # 12. Save Per-Class CSV
    csv_path = os.path.join(out_dir, "dgcnn_per_class_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class ID", "Class Name", "Support", "Correct", "Incorrect", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"])
        for d in per_class_data:
            writer.writerow([
                d["class_id"], d["class_name"], d["support"], d["correct"], d["incorrect"],
                f"{d['accuracy']*100:.2f}", f"{d['precision']*100:.2f}", f"{d['recall']*100:.2f}", f"{d['f1_score']*100:.2f}"
            ])
        writer.writerow([])
        writer.writerow(["-", "MACRO AVERAGE", total_samples, num_correct, num_incorrect, f"{macro_accuracy*100:.2f}", f"{macro_precision*100:.2f}", f"{macro_recall*100:.2f}", f"{macro_f1*100:.2f}"])
        writer.writerow(["-", "WEIGHTED AVERAGE", total_samples, num_correct, num_incorrect, f"{overall_accuracy*100:.2f}", f"{weighted_precision*100:.2f}", f"{weighted_recall*100:.2f}", f"{weighted_f1*100:.2f}"])
    print(f"Saved DGCNN per-class CSV metrics to: {csv_path}")
    
    # 13. Save Full Detailed JSON Report
    json_path = os.path.join(out_dir, "dgcnn_detailed_test_report.json")
    results = {
        "model": f"DGCNN Baseline (Dynamic EdgeConv k={k} + Dual Max/Avg Pooling)",
        "evaluation_split": "ModelNet40 Official Test Set (Untouched)",
        "loaded_checkpoint": {
            "path": checkpoint_path,
            "best_validation_epoch": loaded_epoch,
            "best_validation_oa": float(val_acc_recorded)
        },
        "overall_summary": {
            "total_test_samples": total_samples,
            "correct_predictions": num_correct,
            "incorrect_predictions": num_incorrect,
            "test_accuracy_oa": float(overall_accuracy),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1_score": macro_f1,
            "macro_accuracy_macc": macro_accuracy,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1_score": weighted_f1
        },
        "top_5_classes": top_5_classes,
        "bottom_5_classes": bottom_5_classes,
        "per_class_metrics": per_class_data
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved detailed DGCNN JSON report to: {json_path}")
    
    return results

if __name__ == "__main__":
    run_detailed_dgcnn_test_evaluation()
