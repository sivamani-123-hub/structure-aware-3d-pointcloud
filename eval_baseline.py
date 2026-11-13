import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from data.dataloader import get_modelnet40_dataloaders
from models.pointnet import PointNetClassifier
from training.config import TrainConfig

def evaluate_baseline_on_test(checkpoint_path=r"c:\DLproj\checkpoints\pointnet_best.pth",
                              dataset_root=r"c:\DLproj\dataset\ModelNet40",
                              vis_dir=r"c:\DLproj\visualizations",
                              num_points=1024,
                              num_classes=40,
                              batch_size=32):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"EVALUATING POINTNET BASELINE ON OFFICIAL TEST SET (Device: {device})")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # 1. Load Test Dataloader (Deterministic, NO augmentations)
    _, _, test_loader, class_names = get_modelnet40_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_points=num_points,
        num_classes=num_classes,
        val_split_ratio=0.1,
        num_workers=0,
        seed=42
    )
    
    # 2. Instantiate Model and Load Checkpoint
    model = PointNetClassifier(num_classes=num_classes, use_tnet=False, dropout=0.3).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')} "
          f"(Recorded Val Acc: {checkpoint.get('val_acc', 0.0)*100:.2f}%)")
    
    # 3. Inference on Test Set
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for points, labels in test_loader:
            points = points.to(device)
            logits, _ = model(points)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 4. Overall Accuracy
    total_test = len(all_targets)
    correct_test = (all_preds == all_targets).sum()
    overall_accuracy = correct_test / total_test
    
    # 5. Per-class metrics
    per_class_stats = {}
    class_accuracies = []
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for t, p in zip(all_targets, all_preds):
        confusion_matrix[t, p] += 1
        
    for c in range(num_classes):
        c_name = class_names[c]
        c_total = (all_targets == c).sum()
        if c_total > 0:
            c_correct = ((all_targets == c) & (all_preds == c)).sum()
            c_acc = c_correct / c_total
            c_pred_total = (all_preds == c).sum()
            precision = c_correct / c_pred_total if c_pred_total > 0 else 0.0
            recall = c_acc
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_accuracies.append(c_acc)
            per_class_stats[c_name] = {
                "class_id": c,
                "support": int(c_total),
                "correct": int(c_correct),
                "accuracy": float(c_acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1)
            }
        else:
            per_class_stats[c_name] = {
                "class_id": c,
                "support": 0,
                "correct": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
            
    mean_class_acc = np.mean(class_accuracies) if class_accuracies else 0.0
    
    # 6. Top Confused Pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion_matrix[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "pred_class": class_names[j],
                    "count": int(confusion_matrix[i, j]),
                    "true_support": int((all_targets == i).sum()),
                    "error_rate": float(confusion_matrix[i, j] / (all_targets == i).sum())
                })
                
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)
    
    print("\n" + "=" * 80)
    print("OFFICIAL TEST EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Test Samples Evaluated: {total_test}")
    print(f"PointNet Test Overall Accuracy (OA): {overall_accuracy * 100:.2f}%")
    print(f"PointNet Test Mean Class Accuracy (mAcc): {mean_class_acc * 100:.2f}%")
    print("=" * 80)
    
    # 7. Print top 10 best and worst performing classes
    active_classes = [c for c in per_class_stats.items() if c[1]["support"] > 0]
    active_classes.sort(key=lambda x: x[1]["accuracy"], reverse=True)
    
    print("\nTop 5 Performing Classes on Test Set:")
    for name, stats in active_classes[:5]:
        print(f"  - {name:<15}: Acc: {stats['accuracy']*100:5.1f}% | F1: {stats['f1_score']*100:5.1f}% | Support: {stats['support']}")
        
    print("\nBottom 5 Performing Classes on Test Set (Candidates for Graph Neural Network improvements):")
    for name, stats in active_classes[-5:]:
        print(f"  - {name:<15}: Acc: {stats['accuracy']*100:5.1f}% | F1: {stats['f1_score']*100:5.1f}% | Support: {stats['support']}")
        
    print("\nTop 5 Most Common Confused Class Pairs:")
    for pair in confused_pairs[:5]:
        print(f"  - True: {pair['true_class']:<12} --> Pred: {pair['pred_class']:<12} "
              f"({pair['count']} samples, {pair['error_rate']*100:.1f}% of class)")
              
    # 8. Plot & Save Confusion Matrix
    os.makedirs(vis_dir, exist_ok=True)
    cm_path = os.path.join(vis_dir, "pointnet_confusion_matrix.png")
    
    # Filter confusion matrix for active classes with support > 0
    active_indices = [i for i in range(num_classes) if (all_targets == i).sum() > 0]
    active_names = [class_names[i] for i in active_indices]
    sub_cm = confusion_matrix[np.ix_(active_indices, active_indices)]
    
    # Normalize by row (true labels)
    sub_cm_norm = sub_cm.astype('float') / sub_cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(14, 12), dpi=200)
    plt.imshow(sub_cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('PointNet Baseline: Normalized Confusion Matrix on ModelNet40 Test Set', fontsize=12, fontweight='bold')
    plt.colorbar(fraction=0.046, pad=0.04)
    
    tick_marks = np.arange(len(active_names))
    plt.xticks(tick_marks, active_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, active_names, fontsize=8)
    plt.ylabel('True Class', fontsize=10, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_path}")
    
    # 9. Save Summary JSON
    eval_json_path = os.path.join(os.path.dirname(checkpoint_path), "pointnet_test_evaluation.json")
    results = {
        "overall_accuracy": float(overall_accuracy),
        "mean_class_accuracy": float(mean_class_acc),
        "total_test_samples": int(total_test),
        "best_epoch_loaded": checkpoint.get("epoch", None),
        "per_class_stats": per_class_stats,
        "top_confused_pairs": confused_pairs[:15]
    }
    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved test evaluation JSON to: {eval_json_path}")
    
    return results

if __name__ == "__main__":
    evaluate_baseline_on_test()
