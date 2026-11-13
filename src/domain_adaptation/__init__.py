"""
Domain adaptation package for real-world target fine-tuning.
"""

from .adapted_gnn import DomainAdaptedDGCNN
from .target_trainer import TargetDomainTrainer

__all__ = ["DomainAdaptedDGCNN", "TargetDomainTrainer"]

