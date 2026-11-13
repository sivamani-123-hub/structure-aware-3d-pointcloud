import os
import sys
import torch
import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.scanobjectnn import ScanObjectNNDataset, get_scanobjectnn_dataloader
from src.datasets.class_mapping import (
    COMMON_CLASSES_INFO,
    COMMON_CLASS_NAMES,
    select_common_logits,
    compute_zero_shot_probabilities,
    COMMON_MN40_INDICES_TENSOR
)

def test_loader_and_mapping():
    print("=" * 80)
    print("TESTING SCANOBJECTNN DATASET, DATALOADER, AND CLASS MAPPING (STEP 3)")
    print("=" * 80)

    # 1. Test Common-Classes Test Dataset (Zero-Shot Target Set)
    print("\n--- Test 1: Common Classes Test Loader (10 Classes, N = 1024 points) ---")
    test_loader = get_scanobjectnn_dataloader(
        split="test",
        batch_size=16,
        num_points=1024,
        common_classes_only=True,
        seed=42
    )

    test_dataset = test_loader.dataset
    print(f"• Total filtered test samples (common classes): {len(test_dataset)}")
    print(f"• Total raw samples in test split: {test_dataset.total_raw_samples}")
    print(f"• Common categories ({len(test_dataset.class_names)}): {test_dataset.class_names}")

    # Inspect first batch
    points, labels, meta = next(iter(test_loader))
    print(f"\n[Batch Tensor Audit]")
    print(f"  • Points Tensor Shape: {list(points.shape)} (Expected: [16, 1024, 3])")
    print(f"  • Points Tensor Dtype: {points.dtype}")
    print(f"  • Labels Tensor Shape: {list(labels.shape)} (Expected: [16])")
    print(f"  • Labels Tensor Dtype: {labels.dtype}")
    print(f"  • Sample Target Labels: {labels[:8].tolist()}")
    print(f"  • Sample ModelNet40 Target Labels: {meta['mn40_label'][:8].tolist()}")

    # Coordinate & Normalization Constraints Audit
    radii = torch.norm(points, dim=-1) # [16, 1024]
    max_radius = torch.max(radii).item()
    centroids = torch.mean(points, dim=1) # [16, 3]
    max_centroid_norm = torch.max(torch.norm(centroids, dim=-1)).item()

    print(f"\n[Geometry & Normalization Audit]")
    print(f"  • Max Centroid Norm across batch: {max_centroid_norm:.8f} (Must be ~0)")
    print(f"  • Max Radius across batch:        {max_radius:.8f} (Must be <= 1.000000)")
    assert max_centroid_norm < 1e-4, f"Centroid not centered: {max_centroid_norm}"
    assert max_radius <= 1.000001, f"Radius exceeds unit sphere: {max_radius}"
    print("  --> Normalization constraints: PASSED [VERIFIED]")

    # 2. Test Full 15-Class Training Loader (for Domain Adaptation)
    print("\n--- Test 2: Full 15-Class Training Loader (N = 1024 points, Augmented) ---")
    train_loader = get_scanobjectnn_dataloader(
        split="train",
        batch_size=32,
        num_points=1024,
        common_classes_only=False,
        apply_augmentation=True,
        seed=42
    )
    tr_dataset = train_loader.dataset
    print(f"• Total full train samples: {len(tr_dataset)} (Expected: 2,309)")
    print(f"• Total classes: {tr_dataset.num_classes} classes")
    tr_points, tr_labels, _ = next(iter(train_loader))
    print(f"• Train batch points shape: {list(tr_points.shape)} (Expected: [32, 1024, 3])")
    print(f"• Train batch labels min/max: [{tr_labels.min().item()}, {tr_labels.max().item()}]")

    # 3. Test Mathematical Sub-Logit Selection
    print("\n--- Test 3: Mathematical Logit Selection Verification ---")
    batch_size = 4
    num_mn40_classes = 40
    # Create synthetic logits tensor with unique recognizable values
    dummy_logits_40 = torch.zeros(batch_size, num_mn40_classes)
    for i in range(num_mn40_classes):
        dummy_logits_40[:, i] = float(i * 10)

    # Select common logits
    sub_logits = select_common_logits(dummy_logits_40) # [4, 10]
    print(f"  • Raw 40-class dummy shape: {list(dummy_logits_40.shape)}")
    print(f"  • Extracted 10-class sub-logits shape: {list(sub_logits.shape)}")
    
    # Check that extracted values match exactly the expected ModelNet40 IDs
    expected_mn40_ids = [item["mn40_id"] for item in COMMON_CLASSES_INFO]
    extracted_sample = sub_logits[0].tolist()
    expected_sample = [float(cid * 10) for cid in expected_mn40_ids]
    print(f"  • Expected ModelNet40 IDs: {expected_mn40_ids}")
    print(f"  • Extracted common values: {extracted_sample}")
    assert extracted_sample == expected_sample, f"Logit mismatch: {extracted_sample} vs {expected_sample}"
    print("  --> Mathematical index selection: PASSED [VERIFIED]")

    # 4. Test Zero-Shot Probability Computation (Constrained vs Unconstrained)
    print("\n--- Test 4: Zero-Shot Probability Modes Verification ---")
    # Make chair (ModelNet40 ID 8) the highest logit
    dummy_logits_40[0, :] = 0.0
    dummy_logits_40[0, 8] = 10.0 # chair
    
    # In constrained mode: chair is common_id 2
    probs_c, preds_c = compute_zero_shot_probabilities(dummy_logits_40, mode="constrained")
    print(f"  • Constrained mode prediction for sample 0: common_id {preds_c[0].item()} ({COMMON_CLASS_NAMES[preds_c[0].item()]})")
    print(f"  • Constrained probability: {probs_c[0, preds_c[0]].item():.4f}")
    assert preds_c[0].item() == 2, f"Expected common_id 2 (chair), got {preds_c[0].item()}"

    # In unconstrained mode: chair is ModelNet40 ID 8
    probs_u, preds_u = compute_zero_shot_probabilities(dummy_logits_40, mode="unconstrained")
    print(f"  • Unconstrained mode prediction for sample 0: mn40_id {preds_u[0].item()}")
    print(f"  • Unconstrained probability: {probs_u[0, preds_u[0]].item():.4f}")
    assert preds_u[0].item() == 8, f"Expected mn40_id 8 (chair), got {preds_u[0].item()}"
    print("  --> Probability computation: PASSED [VERIFIED]")

    # 5. Deterministic Reproducibility Check
    print("\n--- Test 5: Deterministic DataLoader Reproducibility ---")
    loader_a = get_scanobjectnn_dataloader(split="test", batch_size=8, seed=42)
    loader_b = get_scanobjectnn_dataloader(split="test", batch_size=8, seed=42)
    pts_a, lbl_a, _ = next(iter(loader_a))
    pts_b, lbl_b, _ = next(iter(loader_b))
    assert torch.equal(pts_a, pts_b), "Test dataloader is not deterministic!"
    assert torch.equal(lbl_a, lbl_b), "Test dataloader labels are not deterministic!"
    print("  --> Test loader determinism: PASSED [VERIFIED]")

    print("\n" + "=" * 80)
    print("ALL STEP 3 TESTS PASSED DETERMINISTICALLY AND MATHEMATICALLY")
    print("=" * 80)

if __name__ == "__main__":
    test_loader_and_mapping()

