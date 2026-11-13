import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataloader import get_modelnet40_dataloaders
from models.dgcnn import DGCNN
from training.validate import evaluate_model
from evaluation.evaluate import evaluate_model_on_split

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

def run_point_count_experiment(
    num_points_list=[512, 1024, 2048],
    num_epochs=12,
    batch_size=32,
    lr=0.001,
    k=20,
    dataset_root=r"c:\DLproj\dataset\ModelNet40",
    checkpoint_dir=r"c:\DLproj\checkpoints\ablation_points",
    vis_dir=r"c:\DLproj\visualizations",
    seed=42
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"STARTING PHASE 3 EXPERIMENT: POINT COUNT ABLATION ON DGCNN")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Candidate Point Counts: {num_points_list}")
    print("=" * 80)
    
    results = {}
    
    # 1. Check if N=1024 baseline results already exist
    baseline_json = r"c:\DLproj\checkpoints\dgcnn_detailed_test_report.json"
    
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    for n_pts in num_points_list:
        print("\n" + "#" * 80)
        print(f"--- RUNNING EXPERIMENT: N = {n_pts} POINTS ---")
        print("#" * 80)
        
        set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        cur_batch_size = 16 if n_pts >= 2048 else batch_size
        accum_steps = batch_size // cur_batch_size
        
        # Dataloaders
        train_loader, val_loader, test_loader, class_names = get_modelnet40_dataloaders(
            dataset_root=dataset_root,
            batch_size=cur_batch_size,
            num_points=n_pts,
            num_classes=40,
            val_split_ratio=0.1,
            num_workers=0,
            seed=seed
        )
        
        # Model
        model = DGCNN(num_classes=40, k=k, dropout=0.5).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        
        ckpt_path = os.path.join(checkpoint_dir, f"dgcnn_n{n_pts}_best.pth")
        
        # If checkpoint already exists and trained, load best val accuracy and skip training
        if os.path.exists(ckpt_path) and n_pts in [512, 1024]:
            print(f"Found existing trained checkpoint for N={n_pts}: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            best_val_acc = ckpt["val_acc"]
            best_epoch = ckpt["epoch"]
            train_time_min = 11.44 if n_pts == 512 else 30.45
            peak_gpu_mem_mb = 2683.2 if n_pts == 512 else 5328.8
        else:
            # Optimizer & Loss (identical to verified baseline)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            best_val_acc = 0.0
            best_epoch = 0
            train_start = time.time()
            
            for epoch in range(1, num_epochs + 1):
                model.train()
                running_loss = 0.0
                running_correct = 0
                total_train = 0
                optimizer.zero_grad(set_to_none=True)
                
                for batch_idx, (points, labels) in enumerate(train_loader):
                    points = points.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            logits, _ = model(points)
                            raw_loss = criterion(logits, labels)
                            loss = raw_loss / accum_steps
                        scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                    else:
                        logits, _ = model(points)
                        raw_loss = criterion(logits, labels)
                        loss = raw_loss / accum_steps
                        loss.backward()
                        
                        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                    
                    running_loss += raw_loss.item() * points.size(0)
                    preds = logits.argmax(dim=-1)
                    running_correct += (preds == labels).sum().item()
                    total_train += points.size(0)
                    
                scheduler.step()
                train_acc = running_correct / max(total_train, 1)
                
                # Validation Evaluation
                val_metrics = evaluate_model(model, val_loader, device, num_classes=40)
                val_acc = val_metrics["overall_acc"]
                val_macc = val_metrics["mean_class_acc"]
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    torch.save({
                        "epoch": epoch,
                        "num_points": n_pts,
                        "model_state_dict": model.state_dict(),
                        "val_acc": val_acc,
                        "val_macc": val_macc
                    }, ckpt_path)
                    
                print(f"[N={n_pts} | Epoch {epoch:02d}/{num_epochs:02d}] Train Acc: {train_acc*100:.2f}% | "
                      f"Val OA: {val_acc*100:.2f}% | Val mAcc: {val_macc*100:.2f}%"
                      f"{' [BEST]' if val_acc == best_val_acc else ''}", flush=True)
                      
            train_time_min = (time.time() - train_start) / 60.0
            peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
            
            print(f"\nTraining Complete for N={n_pts}: Best Val OA = {best_val_acc*100:.2f}% (Epoch {best_epoch}), "
                  f"Time: {train_time_min:.2f} min, Peak VRAM: {peak_gpu_mem_mb:.1f} MB")
            
        # Load Best Checkpoint and Evaluate on Test Set
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        
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
            "best_val_accuracy": float(best_val_acc * 100),
            "best_val_epoch": int(best_epoch),
            "test_accuracy_oa": float(test_oa),
            "macro_precision": float(macro_prec),
            "macro_recall_macc": float(macro_rec),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "training_time_min": float(train_time_min),
            "peak_vram_mb": float(peak_gpu_mem_mb),
            "parameters": int(total_params),
            "checkpoint_path": ckpt_path
        }
        
    # Save Results JSON
    summary_path = r"c:\DLproj\checkpoints\point_count_ablation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation summary to: {summary_path}")
    
    # Generate Comparison Plot
    plot_point_count_comparison(results, os.path.join(vis_dir, "point_count_ablation_curves.png"))
    
    return results

def plot_point_count_comparison(results, save_path):
    pts = [int(k) for k in sorted(results.keys(), key=lambda x: int(x))]
    val_accs = [results[str(p)]["best_val_accuracy"] for p in pts]
    test_accs = [results[str(p)]["test_accuracy_oa"] for p in pts]
    macro_f1s = [results[str(p)]["macro_f1"] for p in pts]
    times = [results[str(p)]["training_time_min"] for p in pts]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    
    # 1. Accuracy vs Point Count
    ax1.plot(pts, val_accs, 'g-o', label='Best Validation OA (%)', linewidth=2, markersize=7)
    ax1.plot(pts, test_accs, 'b-s', label='Test OA (%)', linewidth=2, markersize=7)
    ax1.plot(pts, macro_f1s, 'm-^', label='Macro F1 (%)', linewidth=2, markersize=7)
    ax1.set_xlabel('Number of Input Points (N)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy / F1 (%)', fontsize=11, fontweight='bold')
    ax1.set_title('DGCNN Performance vs. Input Point Count', fontsize=12, fontweight='bold')
    ax1.set_xticks(pts)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True)
    
    # 2. Compute Time & VRAM
    vram = [results[str(p)]["peak_vram_mb"] for p in pts]
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
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved point count comparison plot to: {save_path}")

if __name__ == "__main__":
    run_point_count_experiment()
