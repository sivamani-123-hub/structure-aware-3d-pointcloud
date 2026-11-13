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
from data.transforms import PointCloudTransforms
from models.dgcnn import DGCNN
from training.validate import evaluate_model
from evaluation.evaluate import evaluate_model_on_split

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class AugmentedModelNet40Dataset(ModelNet40Dataset):
    """
    Subclass of ModelNet40Dataset allowing surgical control of augmentation stages A-F.
    """
    def __init__(self, aug_mode="none", **kwargs):
        super().__init__(**kwargs)
        self.aug_mode = aug_mode
        
    def __getitem__(self, idx):
        fpath, label, cls_name = self.samples[idx]
        from data.dataset import parse_off_file_fast
        verts, _ = parse_off_file_fast(fpath, max_v=max(2048, self.num_points))
        
        # 1. Point Sampling to fixed N points
        if len(verts) >= self.num_points:
            if self.split == "train":
                choice_idx = np.random.choice(len(verts), self.num_points, replace=False)
            else:
                np.random.seed((self.seed + idx * 7) % 2147483647)
                choice_idx = np.random.choice(len(verts), self.num_points, replace=False)
            pts = verts[choice_idx]
        else:
            choice_idx = np.random.choice(len(verts), self.num_points, replace=True)
            pts = verts[choice_idx]
            
        # 2. Instance Normalization (Zero-Centering & Unit-Sphere Scaling)
        pts = PointCloudTransforms.normalize_instance(pts)
        
        # 3. Training Augmentations (TRAINING SPLIT ONLY)
        if self.split == "train" and self.apply_augmentation:
            if self.aug_mode == "none":
                pass # Config A: No augmentation
            elif self.aug_mode == "rotation": # Config B
                pts = PointCloudTransforms.random_rotation_z(pts)
            elif self.aug_mode == "rotation_scale": # Config C
                pts = PointCloudTransforms.random_rotation_z(pts)
                pts = PointCloudTransforms.random_anisotropic_scale(pts, low=0.8, high=1.25)
            elif self.aug_mode == "rotation_scale_trans": # Config D
                pts = PointCloudTransforms.random_rotation_z(pts)
                pts = PointCloudTransforms.random_anisotropic_scale(pts, low=0.8, high=1.25)
                pts = PointCloudTransforms.random_translation(pts, shift_range=0.02)
            elif self.aug_mode == "rotation_scale_trans_jitter": # Config E
                pts = PointCloudTransforms.random_rotation_z(pts)
                pts = PointCloudTransforms.random_anisotropic_scale(pts, low=0.8, high=1.25)
                pts = PointCloudTransforms.random_translation(pts, shift_range=0.02)
                pts = PointCloudTransforms.random_jitter(pts, sigma=0.01, clip=0.05)
            elif self.aug_mode == "all_with_dropout": # Config F
                pts = PointCloudTransforms.random_rotation_z(pts)
                pts = PointCloudTransforms.random_anisotropic_scale(pts, low=0.8, high=1.25)
                pts = PointCloudTransforms.random_translation(pts, shift_range=0.02)
                pts = PointCloudTransforms.random_jitter(pts, sigma=0.01, clip=0.05)
                pts = PointCloudTransforms.random_point_dropout(pts, max_dropout_ratio=0.15)
                
        points_tensor = torch.from_numpy(pts).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return points_tensor, label_tensor

def get_augmentation_dataloaders(aug_mode="none", num_points=1024, batch_size=32, seed=42):
    # Base training dataset for slicing indices
    base_train = ModelNet40Dataset(split="train", num_points=num_points, apply_augmentation=False, seed=seed)
    n_total = len(base_train)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    
    gen = torch.Generator().manual_seed(seed)
    train_indices, val_indices = torch.utils.data.random_split(range(n_total), [n_train, n_val], generator=gen)
    train_indices = train_indices.indices
    val_indices = val_indices.indices
    
    # Train dataset with specified augmentation
    train_ds = AugmentedModelNet40Dataset(aug_mode=aug_mode, split="train", num_points=num_points, apply_augmentation=True, seed=seed)
    train_subset = Subset(train_ds, train_indices)
    
    # Validation dataset (STRICTLY NO AUGMENTATION, deterministic)
    val_ds = AugmentedModelNet40Dataset(aug_mode="none", split="train", num_points=num_points, apply_augmentation=False, seed=seed)
    val_subset = Subset(val_ds, val_indices)
    
    # Test dataset (STRICTLY NO AUGMENTATION, deterministic)
    test_ds = AugmentedModelNet40Dataset(aug_mode="none", split="test", num_points=num_points, apply_augmentation=False, seed=seed + 100)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, CANONICAL_MODELNET40_CLASSES

def run_augmentation_experiment(
    num_points=1024,
    k_val=20,
    num_epochs=12,
    batch_size=32,
    lr=0.001,
    checkpoint_dir=r"c:\DLproj\checkpoints\ablation_augmentation",
    vis_dir=r"c:\DLproj\visualizations",
    seed=42
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    configs = [
        ("A", "none", "No Augmentation (Raw Centered)"),
        ("B", "rotation", "Random Rotation (Z-axis)"),
        ("C", "rotation_scale", "Rotation + Anisotropic Scaling"),
        ("D", "rotation_scale_trans", "Rotation + Scaling + Translation"),
        ("E", "rotation_scale_trans_jitter", "Rotation + Scaling + Trans + Gaussian Jitter"),
        ("F", "all_with_dropout", "Rotation + Scale + Trans + Jitter + Point Dropout")
    ]
    
    print("=" * 80)
    print(f"STARTING PHASE 3 EXPERIMENT: DATA AUGMENTATION ABLATION ON DGCNN")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Base Configuration: Points N = {num_points}, Neighbors k = {k_val}")
    print("=" * 80)
    
    results = {}
    
    torch.backends.cudnn.benchmark = True
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    for cfg_code, aug_mode, desc in configs:
        print("\n" + "#" * 80)
        print(f"--- RUNNING EXPERIMENT CONFIG {cfg_code}: {desc} ---")
        print("#" * 80)
        
        set_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        train_loader, val_loader, test_loader, class_names = get_augmentation_dataloaders(
            aug_mode=aug_mode,
            num_points=num_points,
            batch_size=batch_size,
            seed=seed
        )
        
        model = DGCNN(num_classes=40, k=k_val, dropout=0.5).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        ckpt_path = os.path.join(checkpoint_dir, f"dgcnn_aug_{cfg_code.lower()}_best.pth")
        
        best_val_acc = 0.0
        best_epoch = 0
        train_start = time.time()
        
        for epoch in range(1, num_epochs + 1):
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
            train_acc = running_correct / max(total_train, 1)
            
            # Validation Evaluation (Strictly Deterministic & Clean)
            val_metrics = evaluate_model(model, val_loader, device, num_classes=40)
            val_acc = val_metrics["overall_acc"]
            val_macc = val_metrics["mean_class_acc"]
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "config": cfg_code,
                    "aug_mode": aug_mode,
                    "num_points": num_points,
                    "k": k_val,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_macc": val_macc
                }, ckpt_path)
                
            print(f"[Config {cfg_code} | Epoch {epoch:02d}/{num_epochs:02d}] Train Acc: {train_acc*100:.2f}% | "
                  f"Val OA: {val_acc*100:.2f}% | Val mAcc: {val_macc*100:.2f}%"
                  f"{' [BEST]' if val_acc == best_val_acc else ''}", flush=True)
                  
        train_time_min = (time.time() - train_start) / 60.0
        peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        
        print(f"\nTraining Complete for Config {cfg_code}: Best Val OA = {best_val_acc*100:.2f}% (Epoch {best_epoch}), "
              f"Time: {train_time_min:.2f} min, Peak VRAM: {peak_gpu_mem_mb:.1f} MB")
        
        # Load Best Validation Checkpoint and Evaluate on Test Set
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        
        test_eval = evaluate_model_on_split(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names,
            num_classes=40,
            title=f"DGCNN Augmentation Config {cfg_code}"
        )
        
        test_oa = test_eval["overall_accuracy"] * 100
        test_macc = test_eval["macro_accuracy"] * 100
        macro_prec = test_eval["macro_precision"] * 100
        macro_rec = test_eval["macro_recall"] * 100
        macro_f1 = test_eval["macro_f1"] * 100
        weighted_f1 = test_eval["weighted_f1"] * 100
        
        results[cfg_code] = {
            "config_code": cfg_code,
            "aug_mode": aug_mode,
            "description": desc,
            "num_points": num_points,
            "k": k_val,
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
    summary_path = r"c:\DLproj\checkpoints\augmentation_ablation_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved augmentation ablation summary to: {summary_path}")
    
    # Generate Comparison Plot
    plot_augmentation_comparison(results, os.path.join(vis_dir, "augmentation_ablation_chart.png"))
    
    return results

def plot_augmentation_comparison(results, save_path):
    codes = list(results.keys())
    labels = [f"Config {c}\n({results[c]['aug_mode']})" for c in codes]
    val_accs = [results[c]["best_val_accuracy"] for c in codes]
    test_accs = [results[c]["test_accuracy_oa"] for c in codes]
    macro_f1s = [results[c]["macro_f1"] for c in codes]
    
    x = np.arange(len(codes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    rects1 = ax.bar(x - width, val_accs, width, label='Best Val OA (%)', color='#2ecc71', edgecolor='black', alpha=0.9)
    rects2 = ax.bar(x, test_accs, width, label='Test OA (%)', color='#3498db', edgecolor='black', alpha=0.9)
    rects3 = ax.bar(x + width, macro_f1s, width, label='Macro F1 (%)', color='#9b59b6', edgecolor='black', alpha=0.9)
    
    ax.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax.set_title('DGCNN Performance Across Training Data Augmentations (A-F)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.set_ylim([0, 100])
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved augmentation comparison plot to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_points", type=int, default=1024, help="Point count")
    parser.add_argument("--k", type=int, default=20, help="k neighbors")
    args = parser.parse_args()
    run_augmentation_experiment(num_points=args.num_points, k_val=args.k)
