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

def compile_point_count_results():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = r"c:\DLproj\checkpoints\ablation_points"
    results = {}
    
    # 512, 1024, 2048
    point_configs = [
        (512, 11.44, 2683.2),
        (1024, 30.45, 5328.8),
        (2048, 54.20, 5374.0)
    ]
    
    for n_pts, train_time, vram in point_configs:
        ckpt_path = os.path.join(ckpt_dir, f"dgcnn_n{n_pts}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for N={n_pts}: {ckpt_path}")
            continue
            
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        val_acc = ckpt["val_acc"] * 100
        val_epoch = ckpt["epoch"]
        
        # Load dataset
        batch_size = 16 if n_pts >= 2048 else 32
        _, _, test_loader, class_names = get_modelnet40_dataloaders(
            batch_size=batch_size,
            num_points=n_pts,
            num_classes=40,
            seed=42
        )
        
        model = DGCNN(num_classes=40, k=20, dropout=0.5).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        total_params = sum(p.numel() for p in model.parameters())
        
        test_eval = evaluate_model_on_split(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            num_classes=40,
            title=f"DGCNN (N={n_pts})"
        )
        
        test_oa = test_eval["overall_accuracy"] * 100
        test_macc = test_eval["macro_accuracy"] * 100
        macro_prec = test_eval["macro_precision"] * 100
        macro_rec = test_eval["macro_recall"] * 100
        macro_f1 = test_eval["macro_f1"] * 100
        weighted_f1 = test_eval["weighted_f1"] * 100
        
        results[str(n_pts)] = {
            "num_points": n_pts,
            "best_val_accuracy": float(val_acc),
            "best_val_epoch": int(val_epoch),
            "test_accuracy_oa": float(test_oa),
            "macro_precision": float(macro_prec),
            "macro_recall_macc": float(macro_rec),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "training_time_min": float(train_time),
            "peak_vram_mb": float(vram),
            "parameters": int(total_params),
            "checkpoint_path": ckpt_path
        }
        
    summary_path = r"c:\DLproj\checkpoints\point_count_ablation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to: {summary_path}")
    
    # Plot curves
    plot_path = r"c:\DLproj\visualizations\point_count_ablation_curves.png"
    pts = [int(k) for k in sorted(results.keys(), key=lambda x: int(x))]
    val_accs = [results[str(p)]["best_val_accuracy"] for p in pts]
    test_accs = [results[str(p)]["test_accuracy_oa"] for p in pts]
    macro_f1s = [results[str(p)]["macro_f1"] for p in pts]
    times = [results[str(p)]["training_time_min"] for p in pts]
    vram = [results[str(p)]["peak_vram_mb"] for p in pts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    
    # Plot 1: Accuracy & F1
    ax1.plot(pts, val_accs, 'g-o', label='Best Validation OA (%)', linewidth=2.5, markersize=8)
    ax1.plot(pts, test_accs, 'b-s', label='Test OA (%)', linewidth=2, markersize=7, linestyle='--')
    ax1.plot(pts, macro_f1s, 'm-^', label='Macro F1 (%)', linewidth=2, markersize=7, linestyle=':')
    
    best_idx = int(np.argmax(val_accs))
    ax1.scatter([pts[best_idx]], [val_accs[best_idx]], color='gold', s=160, zorder=5, edgecolors='black', label=f'Best Val N={pts[best_idx]} ({val_accs[best_idx]:.2f}%)')
    
    ax1.set_xlabel('Number of Input Points (N)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy / F1 (%)', fontsize=11, fontweight='bold')
    ax1.set_title('DGCNN Performance vs. Input Point Count (ModelNet40)', fontsize=12, fontweight='bold')
    ax1.set_xticks(pts)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True)
    
    # Plot 2: Compute & VRAM
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
    print(f"Saved plot to: {plot_path}")

if __name__ == "__main__":
    compile_point_count_results()
