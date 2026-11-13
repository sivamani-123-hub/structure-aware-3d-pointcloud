import os
import sys
import json
import csv
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.scanobjectnn import ScanObjectNNDataset, get_scanobjectnn_dataloader
from src.datasets.class_mapping import COMMON_CLASS_NAMES
from src.inference.model_loader import load_model, get_device
from src.domain_adaptation.adapted_gnn import DomainAdaptedDGCNN
from src.domain_adaptation.target_trainer import TargetDomainTrainer
from src.evaluation.zero_shot_evaluator import evaluate_zero_shot

def plot_training_curves(history, save_path):
    """Plots training and validation loss and accuracy curves."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

    # Loss
    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss', markersize=4, linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], 'r-s', label='Val Loss', markersize=4, linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Domain Adaptation Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Accuracy
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], 'b-o', label='Train Accuracy', markersize=4, linewidth=1.5)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], 'r-s', label='Val Accuracy', markersize=4, linewidth=1.5)
    ax2.axvline(history["best_epoch"], color='green', linestyle='--', label=f"Best Val Epoch ({history['best_epoch']})")
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title(f"Target Val Accuracy (Peak: {history['best_val_acc']*100:.2f}%)", fontsize=12, fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Domain Adaptation on ScanObjectNN: Training Dynamics", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to: {save_path}")


def plot_zeroshot_vs_finetuned(zeroshot_res, finetuned_res, save_path):
    """Generates comparison bar chart comparing Zero-Shot vs. Fine-Tuned GNN."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    classes = COMMON_CLASS_NAMES
    y = np.arange(len(classes))
    height = 0.35

    zs_accs = [d["accuracy"] * 100 for d in zeroshot_res["per_class_metrics"]]
    ft_accs = [d["accuracy"] * 100 for d in finetuned_res["per_class_metrics"]]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    rects1 = ax.barh(y - height/2, zs_accs, height, label=f"Zero-Shot GNN (OA: {zeroshot_res['overall_accuracy']*100:.2f}%)", color='#E74C3C', alpha=0.85, edgecolor='black')
    rects2 = ax.barh(y + height/2, ft_accs, height, label=f"Fine-Tuned GNN (OA: {finetuned_res['overall_accuracy']*100:.2f}%)", color='#2ECC71', alpha=0.85, edgecolor='black')

    ax.set_xlabel('Test Accuracy on Real-World ScanObjectNN (%)', fontsize=11, fontweight='bold')
    ax.set_title('Zero-Shot vs. Fine-Tuned Domain Adaptation on ScanObjectNN Test Set', fontsize=13, fontweight='bold', pad=12)
    ax.set_yticks(y)
    ax.set_yticklabels([c.title() for c in classes], fontsize=10, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    # Annotate deltas
    for i in range(len(classes)):
        delta = ft_accs[i] - zs_accs[i]
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        ax.annotate(delta_str,
                    xy=(max(ft_accs[i], zs_accs[i]) + 1.5, y[i]),
                    va='center', fontsize=8, fontweight='bold',
                    color='#27AE60' if delta > 0 else '#C0392B')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved zero-shot vs fine-tuned comparison chart to: {save_path}")


def main():
    print("=" * 80)
    print("PHASE D / STEP 7: TARGET DOMAIN ADAPTATION & FINE-TUNING")
    print("=" * 80)

    device = get_device("auto")
    print(f"Target Execution Device: {device}")

    # Paths
    save_ckpt_dir = r"c:\DLproj\checkpoints\phase7"
    save_ckpt_path = os.path.join(save_ckpt_dir, "gnn_finetuned_best.pth")
    vis_dir = r"c:\DLproj\visualizations\phase7"
    res_dir = r"c:\DLproj\results\phase7"
    rep_dir = r"c:\DLproj\reports\phase7"
    os.makedirs(save_ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    # 1. Load Pre-Trained GNN Base Model
    print("\n1. Initializing from Pre-Trained Peak Regularized GNN...")
    base_model, base_meta = load_model("peak_dgcnn", device=device)
    print(f"  • Base Model: {base_meta['description']}")
    print(f"  • Checkpoint ModelNet40 Val OA: {base_meta['val_acc']*100:.2f}%")

    # 2. Instantiate Adapted Architecture (Warm-started 10-Class Head)
    adapted_model = DomainAdaptedDGCNN(
        base_model=base_model,
        num_target_classes=10,
        dropout=0.4,
        freeze_backbone=True
    ).to(device)

    # 3. Prepare Target Training Data (Strictly isolated from test data)
    print("\n2. Loading Target Training Partition (ScanObjectNN clean split)...")
    full_target_train = ScanObjectNNDataset(
        split="train",
        num_points=1024,
        common_classes_only=True,
        apply_augmentation=True,
        seed=42
    )

    total_tr_samples = len(full_target_train)
    val_size = int(total_tr_samples * 0.15)
    train_size = total_tr_samples - val_size
    train_sub, val_sub = random_split(
        full_target_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_sub, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)

    print(f"  • Target Total Training Samples: {total_tr_samples} objects")
    print(f"  • Adaptation Train Set:          {train_size} objects (85%)")
    print(f"  • Model Selection Val Set:       {val_size} objects (15%)")
    print(f"  • Target Test Set (UNTRAINED):   386 objects (Strictly Held-Out)")

    # 4. Train Target Adaptation Head (or load existing trained state if already saved)
    if os.path.exists(save_ckpt_path):
        print(f"\nTrained checkpoint already exists at: {save_ckpt_path}. Loading trained state...")
        best_ckpt = torch.load(save_ckpt_path, map_location=device, weights_only=False)
        history = best_ckpt["history"]
    else:
        trainer = TargetDomainTrainer(
            adapted_model=adapted_model,
            device=device,
            lr=0.001,
            weight_decay=1e-4,
            num_epochs=30,
            patience=8
        )

        history = trainer.fit_adapted_head(
            train_loader=train_loader,
            val_loader=val_loader,
            save_checkpoint_path=save_ckpt_path
        )
        best_ckpt = torch.load(save_ckpt_path, map_location=device, weights_only=False)

    # Plot training curves
    curves_path = os.path.join(vis_dir, "finetuning_curves.png")
    plot_training_curves(history, curves_path)

    # 5. Load Best Checkpoint and Evaluate on UNTOUCHED Target Test Set
    print("\n3. Evaluating Best Adapted Checkpoint on UNTOUCHED Target Test Set...")
    adapted_model.load_state_dict(best_ckpt["model_state_dict"])
    adapted_model.eval()

    test_loader = get_scanobjectnn_dataloader(
        split="test",
        batch_size=32,
        num_points=1024,
        common_classes_only=True,
        seed=42
    )

    # Helper wrapper for evaluation
    class AdaptedModelWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            logits, _ = self.m(x)
            # Create a 40-dim padded tensor so it passes through evaluate_zero_shot cleanly,
            # or evaluate directly
            return logits, None

    # Evaluate directly on the 10 classes
    all_preds = []
    all_targets = []
    all_confs = []
    
    with torch.no_grad():
        for pts, labels, _ in test_loader:
            pts = pts.to(device)
            logits, _ = adapted_model(pts)
            probs = F.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_confs = np.array(all_confs)

    total_test = len(all_targets)
    num_correct = int((all_preds == all_targets).sum())
    ft_oa = num_correct / total_test

    # Compute metrics
    num_classes = len(COMMON_CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1

    per_class_metrics = []
    precisions, recalls, f1_scores, supports = [], [], [], []

    for c in range(num_classes):
        support = int((all_targets == c).sum())
        correct = int(((all_targets == c) & (all_preds == c)).sum())
        acc = correct / support if support > 0 else 0.0
        pred_tot = int((all_preds == c).sum())
        prec = correct / pred_tot if pred_tot > 0 else 0.0
        rec = acc
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1_scores.append(f1)
            supports.append(support)

        per_class_metrics.append({
            "class_id": c,
            "class_name": COMMON_CLASS_NAMES[c],
            "support": support,
            "correct": correct,
            "accuracy": acc,
            "accuracy_pct": f"{acc * 100:.2f}%",
            "precision": prec,
            "precision_pct": f"{prec * 100:.2f}%",
            "recall": rec,
            "recall_pct": f"{rec * 100:.2f}%",
            "f1_score": f1,
            "f1_score_pct": f"{f1 * 100:.2f}%"
        })

    ft_macc = float(np.mean(recalls))
    ft_macro_f1 = float(np.mean(f1_scores))
    tot_s = sum(supports)
    ft_weighted_f1 = float(sum(f * s for f, s in zip(f1_scores, supports)) / tot_s)

    finetuned_results = {
        "model_name": "fine_tuned_gnn",
        "overall_accuracy": ft_oa,
        "overall_accuracy_pct": f"{ft_oa * 100:.2f}%",
        "mean_class_accuracy": ft_macc,
        "mean_class_accuracy_pct": f"{ft_macc * 100:.2f}%",
        "macro_f1": ft_macro_f1,
        "macro_f1_pct": f"{ft_macro_f1 * 100:.2f}%",
        "weighted_f1": ft_weighted_f1,
        "weighted_f1_pct": f"{ft_weighted_f1 * 100:.2f}%",
        "correct_samples": num_correct,
        "total_test_samples": total_test,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics
    }

    # Plot & Save Fine-Tuned Confusion Matrix
    cm_path = os.path.join(vis_dir, "finetuned_gnn_confusion_matrix.png")
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    cax = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Greens, vmin=0.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(COMMON_CLASS_NAMES, rotation=45, ha="right", fontsize=9, fontweight='bold')
    ax.set_yticklabels(COMMON_CLASS_NAMES, fontsize=9, fontweight='bold')
    thresh = cm_norm.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            c_cnt = cm[i, j]
            p_val = cm_norm[i, j] * 100
            if c_cnt > 0:
                ax.text(j, i, f"{p_val:.0f}%\n({c_cnt})", ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black", fontsize=7, fontweight='bold')
    ax.set_ylabel('True Real-World Class', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Class (Fine-Tuned GNN)', fontsize=11, fontweight='bold')
    ax.set_title(f"Fine-Tuned Domain-Adapted GNN Confusion Matrix\nTest OA: {ft_oa*100:.2f}% | Mean Class Acc: {ft_macc*100:.2f}%",
                 fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"Saved fine-tuned confusion matrix to: {cm_path}")

    # Load zero-shot peak results for comparison
    zs_summary_path = os.path.join(res_dir, "zero_shot_evaluation_summary.json")
    with open(zs_summary_path, "r", encoding="utf-8") as f:
        zs_data = json.load(f)
    zs_peak = zs_data["models"]["peak_dgcnn"]

    # Plot Zero-Shot vs Fine-Tuned Comparison
    comp_path = os.path.join(vis_dir, "zeroshot_vs_finetuned_comparison.png")
    plot_zeroshot_vs_finetuned(zs_peak, finetuned_results, comp_path)

    # 6. Print Benchmark Summary Table
    print("\n" + "=" * 80)
    print("DOMAIN ADAPTATION BENCHMARK RESULTS (ScanObjectNN Test Set, 386 Scans)")
    print("=" * 80)
    print(f"{'Metric':30s} | {'Zero-Shot GNN':15s} | {'Fine-Tuned GNN':15s} | {'Delta Gain':12s}")
    print("-" * 80)
    delta_oa = (ft_oa - zs_peak["overall_accuracy"]) * 100
    delta_macc = (ft_macc - zs_peak["mean_class_accuracy"]) * 100
    delta_f1 = (ft_macro_f1 - zs_peak["macro_f1"]) * 100
    delta_wf1 = (ft_weighted_f1 - zs_peak["weighted_f1"]) * 100
    print(f"{'Overall Accuracy (OA)':30s} | {zs_peak['overall_accuracy_pct']:15s} | {finetuned_results['overall_accuracy_pct']:15s} | {f'+{delta_oa:.2f}%':12s}")
    print(f"{'Mean Class Acc (mAcc)':30s} | {zs_peak['mean_class_accuracy_pct']:15s} | {finetuned_results['mean_class_accuracy_pct']:15s} | {f'+{delta_macc:.2f}%':12s}")
    print(f"{'Macro F1-Score':30s} | {zs_peak['macro_f1_pct']:15s} | {finetuned_results['macro_f1_pct']:15s} | {f'+{delta_f1:.2f}%':12s}")
    print(f"{'Weighted F1-Score':30s} | {zs_peak['weighted_f1_pct']:15s} | {finetuned_results['weighted_f1_pct']:15s} | {f'+{delta_wf1:.2f}%':12s}")
    zs_correct_str = f"{zs_peak['correct_samples']}/{total_test}"
    ft_correct_str = f"{num_correct}/{total_test}"
    diff_scans = num_correct - zs_peak["correct_samples"]
    diff_str = f"+{diff_scans} scans" if diff_scans > 0 else f"{diff_scans} scans"
    print(f"{'Correctly Classified Scans':30s} | {zs_correct_str:15s} | {ft_correct_str:15s} | {diff_str:12s}")

    # Per-Class Table
    print("\nPer-Class Accuracy Gains:")
    print(f"{'Class Name':12s} | {'Support':7s} | {'Zero-Shot Acc':15s} | {'Fine-Tuned Acc':15s} | {'Delta Gain':12s}")
    print("-" * 72)
    for c_idx, c_name in enumerate(COMMON_CLASS_NAMES):
        supp = per_class_metrics[c_idx]["support"]
        zs_c = zs_peak["per_class_metrics"][c_idx]["accuracy"] * 100
        ft_c = per_class_metrics[c_idx]["accuracy"] * 100
        d_c = ft_c - zs_c
        d_str = f"+{d_c:.2f}%" if d_c > 0 else f"{d_c:.2f}%"
        print(f"{c_name:12s} | {supp:7d} | {zs_c:13.2f}% | {ft_c:13.2f}% | {d_str:12s}")

    # Save Results JSON
    full_output = {
        "benchmark": "Domain Adaptation & Fine-Tuning on ScanObjectNN",
        "base_model": "Peak Regularized GNN (ModelNet40 pre-trained)",
        "adaptation_protocol": "Representation Probing & Target Head Fine-Tuning (Warm-Started)",
        "history": history,
        "test_results": finetuned_results,
        "comparison_vs_zeroshot": {
            "zeroshot_oa": zs_peak["overall_accuracy"],
            "finetuned_oa": ft_oa,
            "delta_oa": float(delta_oa),
            "zeroshot_macc": zs_peak["mean_class_accuracy"],
            "finetuned_macc": ft_macc,
            "delta_macc": float(delta_macc),
            "zeroshot_correct": zs_peak["correct_samples"],
            "finetuned_correct": num_correct,
            "additional_correct_scans": int(num_correct - zs_peak["correct_samples"])
        }
    }

    out_json = os.path.join(res_dir, "finetuning_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nSaved fine-tuning results JSON to: {out_json}")

    # Save Markdown Report
    out_rep = os.path.join(rep_dir, "DOMAIN_ADAPTATION_EXPERIMENT_REPORT.md")
    with open(out_rep, "w", encoding="utf-8") as f:
        f.write("# Domain Adaptation & Fine-Tuning Experiment Report\n\n")
        f.write("## 1. Executive Summary\n")
        f.write("- **Base Architecture:** Peak Regularized Structure-Aware GNN ($k=40$, Dropout $0.3$, Weight Decay $10^{-3}$, pre-trained on ModelNet40)\n")
        f.write("- **Target Domain Data:** ScanObjectNN clean object split (`main_split_nobg`)\n")
        f.write(f"- **Data Isolation:** Target Training partition ({train_size} train, {val_size} validation). Target Test partition ({total_test} scans) strictly untouched during all training phases.\n")
        f.write(f"- **Checkpoint Path:** `checkpoints/phase7/gnn_finetuned_best.pth` (Zero mutation of original pre-trained checkpoints).\n\n")
        f.write("## 2. Quantitative Comparison: Zero-Shot vs. Domain-Adapted GNN\n\n")
        f.write("| Metric | Zero-Shot Transfer | Domain-Adapted Fine-Tuned | Net Improvement (Delta) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Overall Accuracy (OA)** | **{zs_peak['overall_accuracy_pct']}** | **{finetuned_results['overall_accuracy_pct']}** | **+{delta_oa:.2f}%** |\n")
        f.write(f"| **Mean Class Accuracy (mAcc)** | {zs_peak['mean_class_accuracy_pct']} | {finetuned_results['mean_class_accuracy_pct']} | **+{delta_macc:.2f}%** |\n")
        f.write(f"| **Macro F1-Score** | {zs_peak['macro_f1_pct']} | {finetuned_results['macro_f1_pct']} | **+{delta_f1:.2f}%** |\n")
        f.write(f"| **Weighted F1-Score** | {zs_peak['weighted_f1_pct']} | {finetuned_results['weighted_f1_pct']} | **+{delta_wf1:.2f}%** |\n")
        f.write(f"| **Correct Real Scans** | {zs_peak['correct_samples']} / {total_test} | **{num_correct} / {total_test}** | **+{num_correct - zs_peak['correct_samples']} correctly recognized scans** |\n\n")
        f.write("## 3. Per-Class Accuracy Analysis\n\n")
        f.write("| Class Name | Real Scans | Zero-Shot Acc (%) | Fine-Tuned Acc (%) | Accuracy Gain (Delta) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for c_idx, c_name in enumerate(COMMON_CLASS_NAMES):
            supp = per_class_metrics[c_idx]["support"]
            zs_c = zs_peak["per_class_metrics"][c_idx]["accuracy"] * 100
            ft_c = per_class_metrics[c_idx]["accuracy"] * 100
            d_c = ft_c - zs_c
            f.write(f"| **{c_name}** | {supp} | {zs_c:.2f}% | **{ft_c:.2f}%** | **{'+' if d_c > 0 else ''}{d_c:.2f}%** |\n")

    print(f"Saved domain adaptation markdown report to: {out_rep}")
    print("\n" + "=" * 80)
    print("STEP 7 COMPLETE: DOMAIN ADAPTATION TRAINING & EVALUATION COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
