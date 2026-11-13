"""
Inference package for model loading and real-time point cloud prediction.
"""

from .model_loader import load_model, get_device, MODEL_REGISTRY
from .predictor import PointCloudPredictor

__all__ = ["load_model", "get_device", "MODEL_REGISTRY", "PointCloudPredictor"]

