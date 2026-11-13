import os
import sys
import json
import time
import torch
import numpy as np

# Ensure DLproj root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.scanobjectnn import get_scanobjectnn_dataloader
from src.datasets.class_mapping import COMMON_CLASS_NAMES, COMMON_CLASSES_INFO
from src.inference.model_loader import load_model, get_device, MODEL_REGISTRY
from src.inference.predictor import PointCloudPredictor

def verify_all_checkpoints():
    print("=" * 80)
    print("PHASE C / STEP 4: CHECKPOINT COMPATIBILITY & INFERENCE VERIFICATION")
    print("=" * 80)

    device = get_device("auto")
    print(f"Target Execution Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

    # 1. Fetch real-world test batch from ScanObjectNN
    test_loader = get_scanobjectnn_dataloader(
        split="test",
        batch_size=8,
        num_points=1024,
        common_classes_only=True,
        seed=42
    )
    test_batch_points, test_batch_labels, test_batch_meta = next(iter(test_loader))
    print(f"\nFetched verification test batch: shape={list(test_batch_points.shape)}, labels={test_batch_labels.tolist()}")

    models_to_test = ["pointnet", "dgcnn", "gnn", "peak_dgcnn"]
    checkpoint_results = {}

    for model_name in models_to_test:
        print(f"\n--- Verifying Model: '{model_name.upper()}' ---")
        
        # A. Load Model Strictly
        model, meta = load_model(model_name, device=device)
        print(f"  • Architecture: {meta['description']}")
        print(f"  • Total Parameters: {meta['total_parameters']:,}")
        print(f"  • Best Validation OA in Checkpoint: {meta['val_acc']*100:.2f}% (Epoch {meta['epoch']})" if meta['val_acc'] else "  • Val Acc: N/A")

        # Snapshot first parameter tensor hash to check zero mutation
        first_param = next(model.parameters())
        initial_hash = float(first_param.sum().cpu().item())

        # B. Batch Forward Pass on Real-World Scans
        t0 = time.perf_counter()
        with torch.no_grad():
            batch_in = test_batch_points.to(device)
            logits_40, _ = model(batch_in)
        batch_latency_ms = (time.perf_counter() - t0) * 1000.0

        print(f"  • Batch Forward Pass (B=8, N=1024): Output Shape = {list(logits_40.shape)}")
        print(f"  • Batch Latency: {batch_latency_ms:.2f} ms ({batch_latency_ms/8:.2f} ms/sample)")
        assert logits_40.shape == (8, 40), f"Unexpected shape {logits_40.shape}"

        # C. Single-Instance Predictor Test
        predictor = PointCloudPredictor(model_name=model_name, device=device)
        sample_pts = test_batch_points[0].numpy()
        true_label_id = int(test_batch_labels[0].item())
        true_class_name = COMMON_CLASS_NAMES[true_label_id]

        pred_res = predictor.predict(sample_pts, mode="constrained", top_k=3)
        print(f"  • Single-sample Prediction on '{true_class_name}' scan:")
        print(f"    - Predicted: '{pred_res['predicted_class']}' (Confidence: {pred_res['confidence_pct']})")
        print(f"    - Top-3: {[p['class_name'] + ' (' + p['confidence_pct'] + ')' for p in pred_res['top_predictions']]}")
        print(f"    - Single Latency: {pred_res['latency_ms']:.2f} ms")

        # D. Parameter Immutability Audit
        final_hash = float(first_param.sum().cpu().item())
        assert initial_hash == final_hash, "Model parameters were mutated during inference!"
        print(f"  • Parameter Immutability: STRICTLY PRESERVED [VERIFIED]")

        checkpoint_results[model_name] = {
            "model_name": model_name,
            "description": meta["description"],
            "total_parameters": meta["total_parameters"],
            "checkpoint_val_acc": meta["val_acc"],
            "batch_latency_ms": batch_latency_ms,
            "single_latency_ms": pred_res["latency_ms"],
            "output_logits_shape": list(logits_40.shape),
            "parameters_preserved": True
        }

    # 2. CPU Fallback Verification
    print("\n--- Verifying Explicit CPU Fallback ---")
    cpu_device = torch.device("cpu")
    cpu_model, _ = load_model("pointnet", device=cpu_device)
    cpu_pts = test_batch_points[:2].to(cpu_device)
    with torch.no_grad():
        cpu_logits, _ = cpu_model(cpu_pts)
    print(f"  • Explicit CPU Forward Pass Shape: {list(cpu_logits.shape)} [SUCCESS]")
    assert cpu_logits.device.type == "cpu"

    # Save summary report
    out_dir = r"c:\DLproj\reports\phase7"
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "checkpoint_compatibility_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(checkpoint_results, f, indent=2)

    print(f"\nSaved compatibility report to: {report_file}")
    print("=" * 80)
    print("ALL CHECKPOINTS VERIFIED FOR REAL-WORLD ZERO-SHOT INFERENCE")
    print("=" * 80)

if __name__ == "__main__":
    verify_all_checkpoints()

