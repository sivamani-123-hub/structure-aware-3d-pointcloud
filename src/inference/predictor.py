"""
Inference predictor for 3D Point Cloud Classification.

Handles:
- Input point cloud preprocessing and normalization.
- Forward inference through loaded checkpoints.
- Top-K probability extraction for constrained (10-class) and unconstrained (40-class) modes.
- Latency profiling.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# DLproj imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data.transforms import PointCloudTransforms
from src.datasets.class_mapping import (
    COMMON_CLASSES_INFO,
    COMMON_CLASS_NAMES,
    MODELNET40_CLASSES,
    select_common_logits,
    compute_zero_shot_probabilities
)
from src.inference.model_loader import load_model, get_device


class PointCloudPredictor:
    """
    Standardized inference engine for single-instance or mini-batch 3D point cloud classification.
    """
    def __init__(
        self,
        model_name: str = "dgcnn",
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_points: int = 1024
    ):
        self.device = device or get_device("auto")
        self.num_points = num_points
        self.model, self.metadata = load_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            device=self.device
        )

    def preprocess_point_cloud(self, points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepares raw point cloud coordinates [N, 3] for inference:
        1. Downsamples/samples to self.num_points.
        2. Applies per-instance unit-sphere normalization.
        3. Converts to [1, num_points, 3] float tensor.
        """
        if isinstance(points, torch.Tensor):
            pts = points.detach().cpu().numpy()
        else:
            pts = np.array(points, dtype=np.float32)

        # Handle 2D [N, 3] or 3D [1, N, 3]
        if pts.ndim == 3:
            pts = pts[0]

        if pts.shape[1] != 3:
            raise ValueError(f"Expected point cloud shape [N, 3], got {pts.shape}")

        num_raw = len(pts)
        if num_raw >= self.num_points:
            # Deterministic uniform stride or centered choice
            idx = np.linspace(0, num_raw - 1, self.num_points, dtype=int)
            pts = pts[idx]
        else:
            # Upsample with replacement if points < num_points
            rng = np.random.RandomState(42)
            idx = rng.choice(num_raw, self.num_points, replace=True)
            pts = pts[idx]

        # Unit sphere normalization
        pts = PointCloudTransforms.normalize_instance(pts)
        tensor = torch.from_numpy(pts).unsqueeze(0).float().to(self.device) # [1, 1024, 3]
        return tensor

    def predict(
        self,
        points: Union[np.ndarray, torch.Tensor],
        mode: str = "constrained",
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes forward inference on a single 3D point cloud.
        
        Args:
            points: [N, 3] coordinate array or tensor.
            mode: 'constrained' (10 common real-world classes) or 'unconstrained' (40 ModelNet40 classes).
            top_k: Number of top predictions to return.
            
        Returns:
            Dictionary with prediction results, probabilities, top-k list, and latency.
        """
        input_tensor = self.preprocess_point_cloud(points) # [1, 1024, 3]

        start_time = time.perf_counter()
        with torch.no_grad():
            raw_logits, _ = self.model(input_tensor)
            if self.metadata.get("num_classes", 40) == 10:
                probs = F.softmax(raw_logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                mode = "constrained"
            else:
                probs, preds = compute_zero_shot_probabilities(raw_logits, mode=mode)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        probs_np = probs[0].cpu().numpy()
        pred_idx = int(preds[0].item())

        class_list = COMMON_CLASS_NAMES if mode == "constrained" else MODELNET40_CLASSES
        predicted_class = class_list[pred_idx]
        confidence = float(probs_np[pred_idx])

        # Top-K rankings
        top_indices = np.argsort(probs_np)[::-1][:top_k]
        top_predictions = [
            {
                "rank": rank + 1,
                "class_name": class_list[idx],
                "class_id": int(idx),
                "probability": float(probs_np[idx]),
                "confidence_pct": f"{probs_np[idx] * 100:.2f}%"
            }
            for rank, idx in enumerate(top_indices)
        ]

        return {
            "model_name": self.metadata["model_name"],
            "mode": mode,
            "predicted_class": predicted_class,
            "predicted_id": pred_idx,
            "confidence": confidence,
            "confidence_pct": f"{confidence * 100:.2f}%",
            "top_predictions": top_predictions,
            "latency_ms": latency_ms,
            "num_candidate_classes": len(class_list),
            "device": str(self.device),
            "all_probabilities": {class_list[i]: float(probs_np[i]) for i in range(len(class_list))}
        }

