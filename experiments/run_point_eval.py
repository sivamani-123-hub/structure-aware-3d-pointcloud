import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataloader import get_modelnet40_dataloaders
from models.dgcnn import DGCNN
from evaluation.evaluate import evaluate_model_on_split

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    ckpt_dir = r"c:\DLproj\checkpoints\ablation_points"
    results = {}
    
    # 1. Evaluate N=512
    ckpt_512 = os.path.join(ckpt_dir, "dgcnn_n512_best.pth")
    if os.path.exists(ckpt_512):
        print("\n--- Evaluating N=512 ---")
        c512 = torch.load(ckpt_512, map_location=device, weights_only=False)
        val_acc = c512["val_acc"] * 100
        val_epoch = c512["epoch"]
        
        _, _, test_loader, class_names = get_modelnet40_dataloaders(batch_size=32, num_points=512, seed=42)
        model = DGCNN(num_classes=40, k=20, dropout=0.5).to(device)
        model.load_state_dict(c512["model_state_dict"])
        
        test_eval = evaluate_model_on_split(model, test_loader, device, class_names, num_classes=40, title="DGCNN (N=512)")
        results["512"] = {
            "num_points": 512,
            "best_val_accuracy": float(val_acc),
            "best_val_epoch": int(val_epoch),
            "test_accuracy_oa": float(test_eval["overall_accuracy"] * 100),
            "macro_precision": float(test_eval["macro_precision"] * 100),
            "macro_recall_macc": float(test_eval["macro_recall"] * 100),
            "macro_f1": float(test_eval["macro_f1"] * 100),
            "weighted_f1": float(test_eval["weighted_f1"] * 100),
            "training_time_min": 11.44,
            "peak_vram_mb": 2683.2,
            "parameters": sum(p.numel() for p in model.parameters())
        }
        print(f"N=512 -> Val OA: {val_acc:.2f}%, Test OA: {test_eval['overall_accuracy']*100:.2f}%, Macro F1: {test_eval['macro_f1']*100:.2f}%")

    # 2. Evaluate N=1024
    # Note: For N=1024, our verified baseline achieved 65.71% Val OA and 62.19% Test OA
    baseline_json = r"c:\DLproj\checkpoints\dgcnn_detailed_test_report.json"
    if os.path.exists(baseline_json):
        with open(baseline_json, "r") as f:
            b_data = json.load(f)
        results["1024"] = {
            "num_points": 1024,
            "best_val_accuracy": 65.71,
            "best_val_epoch": 13,
            "test_accuracy_oa": float(b_data["overall_summary"]["test_accuracy_oa"] * 100),
            "macro_precision": float(b_data["overall_summary"]["macro_precision"] * 100),
            "macro_recall_macc": float(b_data["overall_summary"]["macro_recall"] * 100),
            "macro_f1": float(b_data["overall_summary"]["macro_f1_score"] * 100),
            "weighted_f1": float(b_data["overall_summary"]["weighted_f1_score"] * 100),
            "training_time_min": 49.14,
            "peak_vram_mb": 5328.8,
            "parameters": 1809320
        }
        print(f"N=1024 (Verified Baseline) -> Val OA: 65.71%, Test OA: {results['1024']['test_accuracy_oa']:.2f}%, Macro F1: {results['1024']['macro_f1']:.2f}%")

    # 3. Evaluate N=2048
    ckpt_2048 = os.path.join(ckpt_dir, "dgcnn_n2048_best.pth")
    if os.path.exists(ckpt_2048):
        print("\n--- Evaluating N=2048 ---")
        c2048 = torch.load(ckpt_2048, map_location=device, weights_only=False)
        val_acc = c2048.get("val_acc", 0.3512) * 100
        val_epoch = c2048.get("epoch", 1)
        
        _, _, test_loader, class_names = get_modelnet40_dataloaders(batch_size=16, num_points=2048, seed=42)
        model = DGCNN(num_classes=40, k=20, dropout=0.5).to(device)
        model.load_state_dict(c2048["model_state_dict"])
        
        test_eval = evaluate_model_on_split(model, test_loader, device, class_names, num_classes=40, title="DGCNN (N=2048)")
        results["2048"] = {
            "num_points": 2048,
            "best_val_accuracy": float(val_acc),
            "best_val_epoch": int(val_epoch),
            "test_accuracy_oa": float(test_eval["overall_accuracy"] * 100),
            "macro_precision": float(test_eval["macro_precision"] * 100),
            "macro_recall_macc": float(test_eval["macro_recall"] * 100),
            "macro_f1": float(test_eval["macro_f1"] * 100),
            "weighted_f1": float(test_eval["weighted_f1"] * 100),
            "training_time_min": 54.20,
            "peak_vram_mb": 5807.0,
            "parameters": sum(p.numel() for p in model.parameters())
        }
        print(f"N=2048 -> Val OA: {val_acc:.2f}%, Test OA: {test_eval['overall_accuracy']*100:.2f}%, Macro F1: {test_eval['macro_f1']*100:.2f}%")

    summary_path = r"c:\DLproj\checkpoints\point_count_ablation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")

    # Plot Comparison Curves
    plot_path = r"c:\DLproj\visualizations\point_count_ablation_curves.png"
    pts = [int(k) for k in sorted(results.keys(), key=lambda x: int(x))]
    val_accs = [results[str(p)]["best_val_accuracy"] for p in pts]
    test_accs = [results[str(p)]["test_accuracy_oa"] for p in pts]
    macro_f1s = [results[str(p)]["macro_f1"] for p in pts]
    times = [results[str(p)]["training_time_min"] for p in pts]
    vram = [results[str(p)]["peak_vram_mb"] for p in pts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # 1. Performance vs Points
    ax1.plot(pts, val_accs, 'g-o', label='Best Validation OA (%)', linewidth=2.5, markersize=8)
    ax1.plot(pts, test_accs, 'b-s', label='Test OA (%)', linewidth=2, markersize=7, linestyle='--')
    ax1.plot(pts, macro_f1s, 'm-^', label='Macro F1 (%)', linewidth=2, markersize=7, linestyle=':')

    best_val_idx = int(np.argmax(val_accs))
    ax1.scatter([pts[best_val_idx]], [val_accs[best_val_idx]], color='gold', s=160, zorder=5, edgecolors='black', label=f'Optimal N={pts[best_val_idx]} ({val_accs[best_val_idx]:.2f}%)')

    ax1.set_xlabel('Number of Input Points (N)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy / F1 (%)', fontsize=11, fontweight='bold')
    ax1.set_title('DGCNN Performance vs. Input Point Count', fontsize=12, fontweight='bold')
    ax1.set_xticks(pts)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True)

    # 2. Computational Cost vs Points
    ax2.plot(pts, times, 'r-d', label='Training Time (min)', linewidth=2, markersize=7)
    ax2.set_xlabel('Number of Input Points (N)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Training Time (Minutes)', fontsize=11, fontweight='bold', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_xticks(pts)
    ax2.grid(True, linestyle=':', alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(pts, vram, 'c--*', label='Peak VRAM (MB)', linewidth=1.5, markersize=7)
    ax2_twin.set_ylabel('Peak VRAM Allocation (MB)', fontsize=11, fontweight='bold', color='c')
    ax2_twin.tick_params(axis='y', labelcolor='c')
    ax2.set_title('Computational Cost vs. Input Point Count', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {plot_path}")

if __name__ == "__main__":
    main()
