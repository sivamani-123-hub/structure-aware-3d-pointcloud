import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset import ModelNet40Dataset, CANONICAL_MODELNET40_CLASSES
from models.dgcnn import DGCNN
from training.validate import evaluate_model
from evaluation.evaluate import evaluate_model_on_split

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_dataloaders(num_points=1024, batch_size=32, seed=42):
    base_train = ModelNet40Dataset(split="train", num_points=num_points, apply_augmentation=False, seed=seed)
    n_total = len(base_train)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    
    gen = torch.Generator().manual_seed(seed)
    train_indices, val_indices = torch.utils.data.random_split(range(n_total), [n_train, n_val], generator=gen)
    train_indices = train_indices.indices
    val_indices = val_indices.indices
    
    train_subset = Subset(base_train, train_indices)
    val_subset = Subset(base_train, val_indices)
    test_ds = ModelNet40Dataset(split="test", num_points=num_points, apply_augmentation=False, seed=seed + 100)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, CANONICAL_MODELNET40_CLASSES

def plot_regularization_curves(results, save_path):
    epochs = range(1, 13)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11), dpi=200)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    for idx, (exp_name, data) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        lbl = data.get("label", exp_name)
        hist = data["history"]
        ax1.plot(epochs, hist["train_loss"], '-o', label=lbl, color=c, linewidth=1.8, markersize=3.5)
        ax2.plot(epochs, hist["val_loss"], '-s', label=lbl, color=c, linewidth=1.8, markersize=3.5)
        ax3.plot(epochs, [a * 100 for a in hist["train_acc"]], '-o', label=lbl, color=c, linewidth=1.8, markersize=3.5)
        ax4.plot(epochs, [a * 100 for a in hist["val_acc"]], '-s', label=lbl, color=c, linewidth=1.8, markersize=3.5)
        
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training Loss: Dropout & Weight Decay Impact', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Validation Loss: Generalization & Overfitting Check', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Training Accuracy (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Training Accuracy Trajectory', fontsize=12, fontweight='bold')
    ax3.legend(frameon=True, fontsize=8.5)
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Validation Accuracy Progression', fontsize=12, fontweight='bold')
    ax4.legend(frameon=True, fontsize=8.5)
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('DGCNN Regularization Ablation (Dropout & Weight Decay)', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved regularization curves plot to: {save_path}")

def run_single_experiment(
    dropout_p,
    weight_decay,
    label,
    train_loader,
    val_loader,
    test_loader,
    class_names,
    device,
    num_points=1024,
    k_val=40,
    lr=0.001,
    num_epochs=12,
    checkpoint_dir=r"c:\DLproj\checkpoints\ablation_reg",
    seed=42
):
    print("\n" + "#" * 80)
    print(f"--- RUNNING REGULARIZATION EXPERIMENT: {label} (Dropout = {dropout_p}, Weight Decay = {weight_decay}) ---")
    print("#" * 80)
    
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    model = DGCNN(num_classes=40, k=k_val, dropout=dropout_p).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    ckpt_name = f"dgcnn_drop{str(dropout_p).replace('.', '')}_wd{str(weight_decay).replace('.', '').replace('-', 'neg')}_best.pth"
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    
    # Check if baseline (drop=0.5, wd=1e-4) already trained
    lr_ckpt_path = r"c:\DLproj\checkpoints\ablation_lr\dgcnn_lr_0.001_best.pth"
    lr_results_path = r"c:\DLproj\checkpoints\lr_ablation_results.json"
    if dropout_p == 0.5 and weight_decay == 1e-4 and os.path.exists(lr_ckpt_path) and os.path.exists(lr_results_path):
        print(f"Reusing verified baseline run (Dropout=0.5, WD=1e-4) from: {lr_ckpt_path}")
        with open(lr_results_path, "r") as f:
            lr_res = json.load(f)
        baseline_data = lr_res["0.001"]
        return {
            "label": label,
            "dropout": dropout_p,
            "weight_decay": weight_decay,
            "num_points": num_points,
            "k": k_val,
            "best_val_accuracy": baseline_data["best_val_accuracy"],
            "best_val_epoch": baseline_data["best_val_epoch"],
            "test_accuracy_oa": baseline_data["test_accuracy_oa"],
            "macro_precision": baseline_data["macro_precision"],
            "macro_recall_macc": baseline_data["macro_recall_macc"],
            "macro_f1": baseline_data["macro_f1"],
            "weighted_f1": baseline_data["weighted_f1"],
            "final_train_loss": baseline_data["final_train_loss"],
            "final_val_loss": baseline_data["final_val_loss"],
            "final_train_acc": baseline_data["final_train_acc"],
            "final_val_acc": baseline_data["final_val_acc"],
            "training_time_min": baseline_data["training_time_min"],
            "peak_vram_mb": baseline_data["peak_vram_mb"],
            "parameters": total_params,
            "checkpoint_path": lr_ckpt_path,
            "history": baseline_data["history"]
        }
        
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macc": [],
        "epoch_times": []
    }
    
    train_start = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_train = 0
        
        for batch_idx, (points, labels) in enumerate(train_loader):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, _ = model(points)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(points)
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            running_loss += loss.item() * points.size(0)
            preds = logits.argmax(dim=-1)
            running_correct += (preds == labels).sum().item()
            total_train += points.size(0)
            
        scheduler.step()
        train_loss = running_loss / max(total_train, 1)
        train_acc = running_correct / max(total_train, 1)
        
        # Validation Evaluation
        val_metrics = evaluate_model(model, val_loader, device, num_classes=40)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["overall_acc"]
        val_macc = val_metrics["mean_class_acc"]
        
        epoch_time = time.time() - epoch_start
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macc"].append(val_macc)
        history["epoch_times"].append(epoch_time)
        
        is_best = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            is_best = " [BEST]"
            torch.save({
                "epoch": epoch,
                "dropout": dropout_p,
                "weight_decay": weight_decay,
                "k": k_val,
                "num_points": num_points,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_macc": val_macc
            }, ckpt_path)
            
        print(f"--> [{label} | Epoch {epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val mAcc: {val_macc*100:.2f}%{is_best}", flush=True)
              
    train_time_min = (time.time() - train_start) / 60.0
    peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    
    print(f"\nTraining Complete for {label}: Best Val OA = {best_val_acc*100:.2f}% (Epoch {best_epoch}), "
          f"Time: {train_time_min:.2f} min, Peak VRAM: {peak_gpu_mem_mb:.1f} MB")
          
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_eval = evaluate_model_on_split(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        num_classes=40,
        title=f"DGCNN {label}"
    )
    
    test_oa = test_eval["overall_accuracy"] * 100
    test_macc = test_eval["macro_accuracy"] * 100
    macro_prec = test_eval["macro_precision"] * 100
    macro_rec = test_eval["macro_recall"] * 100
    macro_f1 = test_eval["macro_f1"] * 100
    weighted_f1 = test_eval["weighted_f1"] * 100
    
    return {
        "label": label,
        "dropout": dropout_p,
        "weight_decay": weight_decay,
        "num_points": num_points,
        "k": k_val,
        "best_val_accuracy": float(best_val_acc * 100),
        "best_val_epoch": int(best_epoch),
        "test_accuracy_oa": float(test_oa),
        "macro_precision": float(macro_prec),
        "macro_recall_macc": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_train_acc": float(history["train_acc"][-1] * 100),
        "final_val_acc": float(history["val_acc"][-1] * 100),
        "training_time_min": float(train_time_min),
        "peak_vram_mb": float(peak_gpu_mem_mb),
        "parameters": int(total_params),
        "checkpoint_path": ckpt_path,
        "history": history
    }

def run_regularization_ablation(
    num_points=1024,
    k_val=40,
    lr=0.001,
    num_epochs=12,
    batch_size=32,
    checkpoint_dir=r"c:\DLproj\checkpoints\ablation_reg",
    vis_dir=r"c:\DLproj\visualizations",
    seed=42
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print("=" * 80)
    print("STARTING PHASE 3 EXPERIMENT: REGULARIZATION (DROPOUT & WEIGHT DECAY) ABLATION")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Base: N = {num_points}, k = {k_val}, LR = {lr}, Augmentation = Config A (None)")
    print("=" * 80)
    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        num_points=num_points,
        batch_size=batch_size,
        seed=seed
    )
    
    results = {}
    
    # -------------------------------------------------------------
    # Stage 1: Controlled Dropout Investigation (Fixed WD = 1e-4)
    # -------------------------------------------------------------
    dropout_experiments = [
        (0.3, 1e-4, "Dropout 0.3 (WD 1e-4)"),
        (0.5, 1e-4, "Dropout 0.5 [Baseline] (WD 1e-4)"),
        (0.7, 1e-4, "Dropout 0.7 (WD 1e-4)")
    ]
    
    for drop_p, wd, lbl in dropout_experiments:
        res = run_single_experiment(
            dropout_p=drop_p,
            weight_decay=wd,
            label=lbl,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            num_points=num_points,
            k_val=k_val,
            lr=lr,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            seed=seed
        )
        results[f"drop_{drop_p}_wd_{wd}"] = res
        
    # Select best dropout based on validation accuracy
    best_drop_key = max(results.keys(), key=lambda k: results[k]["best_val_accuracy"])
    best_drop_p = results[best_drop_key]["dropout"]
    print(f"\n>>> Selected Best Dropout from Stage 1: p = {best_drop_p} (Val OA = {results[best_drop_key]['best_val_accuracy']:.2f}%) <<<")
    
    # -------------------------------------------------------------
    # Stage 2: Controlled Weight Decay Investigation (Fixed Best Dropout)
    # -------------------------------------------------------------
    wd_experiments = [
        (best_drop_p, 0.0, f"Dropout {best_drop_p} (WD 0)"),
        (best_drop_p, 1e-5, f"Dropout {best_drop_p} (WD 1e-5)"),
        # (best_drop_p, 1e-4) is already in results from Stage 1!
        (best_drop_p, 1e-3, f"Dropout {best_drop_p} (WD 1e-3)")
    ]
    
    for drop_p, wd, lbl in wd_experiments:
        key = f"drop_{drop_p}_wd_{wd}"
        if key in results:
            continue
        res = run_single_experiment(
            dropout_p=drop_p,
            weight_decay=wd,
            label=lbl,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            device=device,
            num_points=num_points,
            k_val=k_val,
            lr=lr,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            seed=seed
        )
        results[key] = res
        
    # Save Results JSON
    summary_path = r"c:\DLproj\checkpoints\regularization_ablation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved regularization ablation summary to: {summary_path}")
    
    # Generate Curves Plot
    plot_regularization_curves(results, os.path.join(vis_dir, "regularization_ablation_curves.png"))
    
    return results

if __name__ == "__main__":
    run_regularization_ablation()
