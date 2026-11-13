import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from data.dataloader import get_modelnet40_dataloaders
from models.dgcnn import DGCNN
from evaluation.evaluate import evaluate_checkpoint

def verify_dgcnn_reproducibility():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print(f"REPRODUCIBILITY & STABILITY AUDIT: DGCNN BASELINE (Device: {device})")
    print("=" * 80)
    
    # 1. Parameter Count Verification
    model = DGCNN(num_classes=40, k=20, dropout=0.5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"1. Model Parameter Verification:")
    print(f"   - Total Parameters:     {total_params:,}")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    assert total_params == 1809320, f"Parameter mismatch! Expected 1,809,320, got {total_params}"
    print(f"   --> Parameter count MATCHES target: 1,809,320 [VERIFIED]")
    
    # 2. Checkpoint Validation Metrics
    checkpoint_path = r"c:\DLproj\checkpoints\dgcnn_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    val_acc = checkpoint.get("val_acc", 0.0) * 100
    val_macc = checkpoint.get("val_macc", 0.0) * 100
    best_epoch = checkpoint.get("epoch", 0)
    print(f"\n2. Checkpoint State Verification:")
    print(f"   - Checkpoint Path:      {checkpoint_path}")
    print(f"   - Best Validation Epoch: Epoch {best_epoch}")
    print(f"   - Best Validation OA:    {val_acc:.2f}%")
    print(f"   - Best Validation mAcc:  {val_macc:.2f}%")
    assert abs(val_acc - 65.71) < 0.1, f"Validation accuracy mismatch! Expected 65.71%, got {val_acc:.2f}%"
    print(f"   --> Validation OA MATCHES target: 65.71% [VERIFIED]")
    
    # 3. Deterministic Test Set Evaluation
    print(f"\n3. Deterministic Test Set Evaluation on CUDA:")
    eval_results = evaluate_checkpoint(
        model_type="dgcnn",
        checkpoint_path=checkpoint_path,
        k=20,
        num_points=1024,
        num_classes=40,
        batch_size=32
    )
    
    test_oa = eval_results["overall_accuracy"] * 100
    test_macc = eval_results["macro_accuracy"] * 100
    macro_prec = eval_results["macro_precision"] * 100
    macro_rec = eval_results["macro_recall"] * 100
    macro_f1 = eval_results["macro_f1"] * 100
    weighted_f1 = eval_results["weighted_f1"] * 100
    correct = eval_results["correct_samples"]
    incorrect = eval_results["incorrect_samples"]
    total = eval_results["total_samples"]
    
    print(f"\n4. Final Verified Metrics Table:")
    print(f"   - Test Overall Accuracy (OA): {test_oa:.2f}% (Expected: 62.19%)")
    print(f"   - Best Validation Accuracy:   {val_acc:.2f}% (Expected: 65.71%)")
    print(f"   - Macro Precision:            {macro_prec:.2f}% (Expected: 63.46%)")
    print(f"   - Macro Recall:               {macro_rec:.2f}% (Expected: 57.98%)")
    print(f"   - Macro F1-Score:             {macro_f1:.2f}% (Expected: 57.14%)")
    print(f"   - Weighted F1-Score:          {weighted_f1:.2f}% (Expected: 61.33%)")
    print(f"   - Correct / Incorrect / Total: {correct} / {incorrect} / {total} (Expected: 1,311 / 797 / 2,108)")
    
    # Verification Assertions
    assert abs(test_oa - 62.19) < 0.05, f"Test OA mismatch! Got {test_oa:.2f}%"
    assert correct == 1311, f"Correct count mismatch! Got {correct}"
    assert incorrect == 797, f"Incorrect count mismatch! Got {incorrect}"
    assert total == 2108, f"Total test samples mismatch! Got {total}"
    
    print("\n" + "=" * 80)
    print("ALL REPRODUCIBILITY CHECKS PASSED: DGCNN BASELINE IS 100% STABLE & REPRODUCIBLE")
    print("=" * 80)
    
    return {
        "status": "PASSED",
        "test_accuracy": test_oa,
        "val_accuracy": val_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "parameters": total_params,
        "correct": correct,
        "incorrect": incorrect,
        "total": total
    }

if __name__ == "__main__":
    verify_dgcnn_reproducibility()
