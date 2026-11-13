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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataloader import get_modelnet40_dataloaders
from models.pointnet import PointNetClassifier
from models.dgcnn import DGCNN

def evaluate_model_on_split(
    model,
    dataloader,
    device,
    class_names,
    num_classes=40,
    save_cm_path=None,
    save_json_path=None,
    save_csv_path=None,
    title="Model Evaluation"
):
    model.eval()
    all_preds = []
    all_targets = []
    all_confidences = []
    
    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device, non_blocking=True)
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits, _ = model(points)
                    probs = F.softmax(logits, dim=-1)
            else:
                logits, _ = model(points)
                probs = F.softmax(logits, dim=-1)
                
            confidences, preds = probs.max(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confidences = np.array(all_confidences)
    
    total_samples = len(all_targets)
    correct_mask = (all_preds == all_targets)
    num_correct = int(correct_mask.sum())
    num_incorrect = total_samples - num_correct
    overall_accuracy = num_correct / total_samples
    
    # Confusion Matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
        
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
        
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1_scores))
    macro_accuracy = float(np.mean([d["accuracy"] for d in per_class_data]))
    
    total_supp = sum(supports)
    weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total_supp)
    weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total_supp)
    weighted_f1 = float(sum(f * s for f, s in zip(f1_scores, supports)) / total_supp)
    
    # Save Confusion Matrix Plot
    if save_cm_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_cm_path)), exist_ok=True)
        active_names = [class_names[c] for c in active_class_ids]
        sub_cm = cm[np.ix_(active_class_ids, active_class_ids)]
        sub_cm_norm = sub_cm.astype('float') / sub_cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(15, 13), dpi=200)
        plt.imshow(sub_cm_norm, interpolation='nearest', cmap=plt.cm.Greens if "DGCNN" in title else plt.cm.Blues)
        plt.title(f'{title}: Normalized Confusion Matrix', fontsize=13, fontweight='bold')
        plt.colorbar(fraction=0.046, pad=0.04)
        
        tick_marks = np.arange(len(active_names))
        plt.xticks(tick_marks, active_names, rotation=90, fontsize=8)
        plt.yticks(tick_marks, active_names, fontsize=8)
        plt.ylabel('True Class', fontsize=11, fontweight='bold')
        plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_cm_path, bbox_inches='tight')
        plt.close()
        print(f"Saved confusion matrix to: {save_cm_path}")
        
    # Save CSV
    if save_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_csv_path)), exist_ok=True)
        with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
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
        print(f"Saved CSV metrics to: {save_csv_path}")
        
    # Results dictionary
    results = {
        "title": title,
        "total_samples": total_samples,
        "correct_samples": num_correct,
        "incorrect_samples": num_incorrect,
        "overall_accuracy": float(overall_accuracy),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_accuracy": macro_accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_metrics": per_class_data
    }
    
    if save_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_json_path)), exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved JSON report to: {save_json_path}")
        
    return results

def evaluate_checkpoint(
    model_type="dgcnn",
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
    print(f"Evaluating {model_type.upper()} checkpoint on test set (Device: {device})")
    
    _, _, test_loader, class_names = get_modelnet40_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_points=num_points,
        num_classes=num_classes,
        val_split_ratio=0.1,
        num_workers=0,
        seed=42
    )
    
    if model_type.lower() == "dgcnn":
        model = DGCNN(num_classes=num_classes, k=k, dropout=0.5).to(device)
    else:
        model = PointNetClassifier(num_classes=num_classes, use_tnet=False, dropout=0.3).to(device)
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    title = f"{model_type.upper()} Baseline Test Set Evaluation"
    cm_path = os.path.join(vis_dir, f"{model_type.lower()}_confusion_matrix.png")
    json_path = os.path.join(out_dir, f"{model_type.lower()}_test_evaluation.json")
    csv_path = os.path.join(out_dir, f"{model_type.lower()}_per_class_metrics.csv")
    
    return evaluate_model_on_split(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        num_classes=num_classes,
        save_cm_path=cm_path,
        save_json_path=json_path,
        save_csv_path=csv_path,
        title=title
    )

if __name__ == "__main__":
    evaluate_checkpoint(model_type="dgcnn")
