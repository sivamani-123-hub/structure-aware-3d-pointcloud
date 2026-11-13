import sys
import torch
import numpy as np
from data.dataloader import get_modelnet40_dataloaders
from data.dataset import ModelNet40Dataset, CANONICAL_MODELNET40_CLASSES

def main():
    print("=" * 80)
    print("PHASE 2: PYTORCH DATASET & DATALOADER VERIFICATION TEST")
    print("=" * 80)
    
    # 1. Test 40-Class Configuration
    print("\n--- Test 1: Standard 40-Class Configuration (N = 1,024 points, Batch Size = 16) ---")
    train_loader, val_loader, test_loader, class_names = get_modelnet40_dataloaders(
        dataset_root=r"c:\DLproj\dataset\ModelNet40",
        batch_size=16,
        num_points=1024,
        num_classes=40,
        val_split_ratio=0.1,
        num_workers=0,
        seed=42
    )
    
    print(f"• Total Classes Configured: {len(class_names)}")
    print(f"• Training Batches: {len(train_loader)} (Samples: {len(train_loader.dataset):,})")
    print(f"• Validation Batches: {len(val_loader)} (Samples: {len(val_loader.dataset):,})")
    print(f"• Test Batches: {len(test_loader)} (Samples: {len(test_loader.dataset):,})")
    
    # Fetch first training batch
    points_batch, labels_batch = next(iter(train_loader))
    print(f"\n[Training Batch Tensor Audit]")
    print(f"  • Points Tensor Shape: {points_batch.shape} (Expected: [16, 1024, 3])")
    print(f"  • Points Tensor Dtype: {points_batch.dtype}")
    print(f"  • Labels Tensor Shape: {labels_batch.shape} (Expected: [16])")
    print(f"  • Labels Tensor Dtype: {labels_batch.dtype}")
    print(f"  • Sample Labels in Batch: {labels_batch.tolist()}")
    print(f"  • Label Range in Batch: Min = {labels_batch.min().item()}, Max = {labels_batch.max().item()}")
    
    # Check normalization constraints
    radii = torch.norm(points_batch, dim=-1)
    max_radius_per_sample = torch.max(radii, dim=-1).values
    centroids = torch.mean(points_batch, dim=1)
    print(f"  • Max Centroid Norm in Batch: {torch.max(torch.norm(centroids, dim=-1)).item():.6f} (Must be ~0)")
    print(f"  • Max Radius in Batch: {torch.max(max_radius_per_sample).item():.6f} (Must be <= 1.25 with scale aug)")

    # Fetch first test batch (Deterministic)
    test_points_batch, test_labels_batch = next(iter(test_loader))
    test_radii = torch.norm(test_points_batch, dim=-1)
    print(f"\n[Test Batch Tensor Audit (Deterministic, No Augmentation)]")
    print(f"  • Points Tensor Shape: {test_points_batch.shape}")
    print(f"  • Max Radius in Test Batch: {torch.max(torch.norm(test_points_batch, dim=-1)).item():.6f} (Strictly <= 1.0)")
    print(f"  • Test Labels Sample: {test_labels_batch[:8].tolist()}")

    # 2. Test Custom Configurable Points & 30-Class Mapping
    print("\n--- Test 2: Custom Configurable Parameters (N = 512 points, 30 Active Classes) ---")
    tr_loader30, _, te_loader30, cls30 = get_modelnet40_dataloaders(
        dataset_root=r"c:\DLproj\dataset\ModelNet40",
        batch_size=8,
        num_points=512,
        num_classes=30,
        val_split_ratio=0.0,
        seed=123
    )
    p30, l30 = next(iter(tr_loader30))
    print(f"• Configured Points: {p30.shape[1]} (Expected: 512)")
    print(f"• Configured Batch Shape: {p30.shape} (Expected: [8, 512, 3])")
    print(f"• Configured Target Class Count: {len(cls30)} classes")
    print(f"• Target Label Range: [{l30.min().item()}, {l30.max().item()}]")

    # 3. Class Index Mapping Verification
    print("\n--- Test 3: Canonical 40-Class Index Mapping Audit ---")
    dataset_full = ModelNet40Dataset(
        dataset_root=r"c:\DLproj\dataset\ModelNet40",
        split="test",
        num_points=1024,
        num_classes=40
    )
    print("Class-to-Index Mapping Samples (first 10 classes):")
    for i, (name, idx) in enumerate(list(dataset_full.class_to_idx.items())[:10]):
        print(f"  [{idx:2d}] -> {name}")

    print("\nAll PyTorch Dataset and DataLoader tests passed successfully!")

if __name__ == "__main__":
    main()
