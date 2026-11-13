"""
Model loading utility for PointNet, DGCNN, Structure-Aware GNN, and Fine-Tuned GNN checkpoints.
"""

import os
import sys
from typing import Dict, Tuple, Any, Optional
import torch

# DLproj imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.pointnet import PointNetClassifier
from models.dgcnn import DGCNN
from src.domain_adaptation.adapted_gnn import DomainAdaptedDGCNN

MODEL_REGISTRY = {
    "pointnet": {
        "class": PointNetClassifier,
        "default_checkpoint": r"c:\DLproj\checkpoints\pointnet_best.pth",
        "kwargs": {"num_classes": 40, "use_tnet": False, "dropout": 0.3},
        "description": "PointNet Baseline (Shared pointwise MLPs + Global Max-Pooling)",
        "num_classes": 40
    },
    "dgcnn": {
        "class": DGCNN,
        "default_checkpoint": r"c:\DLproj\checkpoints\dgcnn_best.pth",
        "kwargs": {"num_classes": 40, "k": 20, "dropout": 0.5},
        "description": "DGCNN Baseline (Dynamic EdgeConv, k=20, Dual Max/Avg Pooling)",
        "num_classes": 40
    },
    "gnn": {
        "class": DGCNN,
        "default_checkpoint": r"c:\DLproj\checkpoints\gnn_best.pth",
        "kwargs": {"num_classes": 40, "k": 20, "dropout": 0.5},
        "description": "Structure-Aware GNN (Multi-Scale EdgeConv with Dynamic Feature Graphs)",
        "num_classes": 40
    },
    "peak_dgcnn": {
        "class": DGCNN,
        "default_checkpoint": r"c:\DLproj\checkpoints\ablation_reg\dgcnn_drop03_wd0001_best.pth",
        "kwargs": {"num_classes": 40, "k": 40, "dropout": 0.3},
        "description": "Peak Regularized GNN (k=40, Dropout 0.3, Weight Decay 1e-3, 74.62% ModelNet40 OA)",
        "num_classes": 40
    },
    "finetuned_gnn": {
        "class": DomainAdaptedDGCNN,
        "default_checkpoint": r"c:\DLproj\checkpoints\phase7\gnn_finetuned_best.pth",
        "kwargs": {"num_target_classes": 10},
        "description": "Domain-Adapted Fine-Tuned GNN (ScanObjectNN adapted, 62.18% Target OA)",
        "num_classes": 10
    }
}


def get_device(preference: str = "auto") -> torch.device:
    """Selects CUDA device if available, otherwise falls back gracefully to CPU."""
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preference == "cpu":
        return torch.device("cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Instantiates model architecture and strictly loads trained weights.
    """
    key = model_name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[key]
    path = checkpoint_path or config["default_checkpoint"]
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    dev = device or get_device("auto")

    # Instantiate model
    if key == "finetuned_gnn":
        base_dgcnn = DGCNN(num_classes=40, k=40, dropout=0.3).to(dev)
        model = DomainAdaptedDGCNN(base_model=base_dgcnn, num_target_classes=10).to(dev)
    else:
        model = config["class"](**config["kwargs"]).to(dev)

    # Load state dict
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Freeze weights for inference
    for p in model.parameters():
        p.requires_grad = False

    metadata = {
        "model_name": key,
        "checkpoint_path": path,
        "device": str(dev),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "epoch": ckpt.get("epoch", "N/A"),
        "val_acc": ckpt.get("val_acc", ckpt.get("best_val_acc", None)),
        "description": config["description"],
        "num_classes": config["num_classes"]
    }

    return model, metadata
