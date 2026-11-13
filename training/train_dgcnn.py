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

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

def plot_dgcnn_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
    
    # 1. Loss Curves
    ax1.plot(epochs, history["train_loss"], 'b-o', label='Training Loss', linewidth=1.5, markersize=4)
    ax1.plot(epochs, history["val_loss"], 'g-s', label='Validation Loss', linewidth=1.5, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('DGCNN Baseline: Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 2. Accuracy Curves
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], 'b-o', label='Training Accuracy', linewidth=1.5, markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], 'g-s', label='Validation Accuracy', linewidth=1.5, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Overall Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_title('DGCNN Baseline: Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved DGCNN training curves to: {save_path}", flush=True)

def train_dgcnn(
    dataset_root=r"c:\DLproj\dataset\ModelNet40",
    checkpoint_dir=r"c:\DLproj\checkpoints",
    vis_dir=r"c:\DLproj\visualizations",
    num_classes=40,
    num_points=1024,
    k=20,
    batch_size=32,
    num_epochs=12,
    learning_rate=0.001,
    weight_decay=1e-4,
    val_split_ratio=0.1,
    seed=42
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80, flush=True)
    print(f"STARTING DGCNN BASELINE TRAINING (Device: {device})", flush=True)
    print(f"Hyperparameters: k={k}, Batch Size={batch_size}, LR={learning_rate}, "
          f"Epochs={num_epochs}, Points={num_points}, Classes={num_classes}", flush=True)
    print("=" * 80, flush=True)
    
    # Dataloaders (Same split & preprocessing as PointNet)
    train_loader, val_loader, _, _ = get_modelnet40_dataloaders(
        dataset_root=dataset_root,
        batch_size=batch_size,
        num_points=num_points,
        num_classes=num_classes,
        val_split_ratio=val_split_ratio,
        num_workers=0,
        seed=seed
    )
    
    total_batches = len(train_loader)
    print(f"Data Loaded: Train={len(train_loader.dataset)} samples ({total_batches} batches), "
          f"Val={len(val_loader.dataset)} samples ({len(val_loader)} batches)", flush=True)
    
    # Model
    model = DGCNN(num_classes=num_classes, k=k, dropout=0.5).to(device)
    
    # Standard Cross Entropy Loss & Adam Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.7)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macc": [],
        "epoch_times": []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    best_checkpoint_path = os.path.join(checkpoint_dir, "dgcnn_best.pth")
    
    start_total_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        running_correct = 0
        total_train_samples = 0
        
        for batch_idx, (points, labels) in enumerate(train_loader):
            points = points.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(points)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * points.size(0)
            preds = logits.argmax(dim=-1)
            running_correct += (preds == labels).sum().item()
            total_train_samples += points.size(0)
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                cur_loss = running_loss / total_train_samples
                cur_acc = (running_correct / total_train_samples) * 100
                print(f"  [Epoch {epoch:02d} | Batch {batch_idx+1:03d}/{total_batches:03d}] "
                      f"Loss: {cur_loss:.4f} | Acc: {cur_acc:.1f}%", flush=True)
                      
        scheduler.step()
        
        train_loss = running_loss / max(total_train_samples, 1)
        train_acc = running_correct / max(total_train_samples, 1)
        
        # Validation Evaluation
        val_metrics = evaluate_model(model, val_loader, device, num_classes=num_classes)
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
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_macc": val_macc,
                "k": k,
                "num_classes": num_classes
            }, best_checkpoint_path)
            
        print(f"--> Epoch [{epoch:02d}/{num_epochs:02d}] Summary ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val mAcc: {val_macc*100:.2f}%{is_best}\n", flush=True)
              
    total_time = time.time() - start_total_time
    print("=" * 80, flush=True)
    print(f"DGCNN TRAINING COMPLETE in {total_time/60:.2f} minutes.", flush=True)
    print(f"Best Validation Overall Accuracy: {best_val_acc*100:.2f}% (Achieved at Epoch {best_epoch})", flush=True)
    print(f"Saved Best DGCNN Checkpoint to: {best_checkpoint_path}", flush=True)
    print("=" * 80, flush=True)
    
    # Save curves and history
    curve_path = os.path.join(vis_dir, "dgcnn_training_curves.png")
    plot_dgcnn_curves(history, curve_path)
    
    history_path = os.path.join(checkpoint_dir, "dgcnn_training_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_acc": float(best_val_acc),
            "history": history
        }, f, indent=2)
        
    return {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
        "checkpoint_path": best_checkpoint_path
    }

if __name__ == "__main__":
    train_dgcnn()
