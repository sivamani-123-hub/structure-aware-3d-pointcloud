"""
Domain-Adapted Graph Neural Network Architecture.

Adapts the ModelNet40-trained Structure-Aware GNN (DGCNN) for real-world ScanObjectNN
classification via warm-started feature projections and fine-tuning.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# DLproj imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from models.dgcnn import DGCNN
from src.datasets.class_mapping import COMMON_CLASSES_INFO


class DomainAdaptedDGCNN(nn.Module):
    """
    Domain-Adapted DGCNN architecture.
    
    Backbone: Inherited directly from pre-trained ModelNet40 DGCNN (EdgeConv L1..L4 + 1D Conv Fusion).
    Dual Global Pooling: Max-Pool (1024) + Avg-Pool (1024) -> 2048 dimensions.
    Classification Head: Initialized with pre-trained weights for the 10 common target classes.
    """
    def __init__(
        self,
        base_model: DGCNN,
        num_target_classes: int = 10,
        dropout: float = 0.4,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.num_target_classes = num_target_classes
        self.k = base_model.k
        self.freeze_backbone = freeze_backbone

        # 1. Inherit feature extraction backbone
        self.edgeconv1 = base_model.edgeconv1
        self.edgeconv2 = base_model.edgeconv2
        self.edgeconv3 = base_model.edgeconv3
        self.edgeconv4 = base_model.edgeconv4
        self.conv_fusion = base_model.conv_fusion

        if freeze_backbone:
            for p in self.get_backbone_parameters():
                p.requires_grad = False

        # 2. Adaptation Classification Head
        self.fc1 = nn.Linear(2048, 512, bias=False)
        self.fc1.weight.data.copy_(base_model.fc1.weight.data)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn1.load_state_dict(base_model.bn1.state_dict())
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc2.weight.data.copy_(base_model.fc2.weight.data)
        
        self.bn2 = nn.BatchNorm1d(256)
        self.bn2.load_state_dict(base_model.bn2.state_dict())
        self.drop2 = nn.Dropout(p=dropout)

        # 3. Target Classification Head (10 Classes)
        self.fc3 = nn.Linear(256, num_target_classes)
        # Warm-start weights from the corresponding ModelNet40 class rows
        common_mn40_ids = [item["mn40_id"] for item in COMMON_CLASSES_INFO]
        with torch.no_grad():
            self.fc3.weight.data.copy_(base_model.fc3.weight.data[common_mn40_ids])
            if base_model.fc3.bias is not None:
                self.fc3.bias.data.copy_(base_model.fc3.bias.data[common_mn40_ids])

    def get_backbone_parameters(self):
        for m in [self.edgeconv1, self.edgeconv2, self.edgeconv3, self.edgeconv4, self.conv_fusion]:
            for p in m.parameters():
                yield p

    def get_head_parameters(self):
        for m in [self.fc1, self.bn1, self.fc2, self.bn2, self.fc3]:
            for p in m.parameters():
                yield p

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts 2048-dimensional shape embedding through the EdgeConv backbone.
        Input: [B, N, 3] or [B, 3, N]
        Output: [B, 2048]
        """
        if x.dim() == 3 and x.size(2) == 3 and x.size(1) != 3:
            x = x.transpose(2, 1) # [B, 3, N]

        batch_size = x.size(0)
        x1 = self.edgeconv1(x)
        x2 = self.edgeconv2(x1)
        x3 = self.edgeconv3(x2)
        x4 = self.edgeconv4(x3)

        x_concat = torch.cat((x1, x2, x3, x4), dim=1) # [B, 512, N]
        x_fused = self.conv_fusion(x_concat)          # [B, 1024, N]

        x_max = F.adaptive_max_pool1d(x_fused, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x_fused, 1).view(batch_size, -1)
        global_feat = torch.cat((x_max, x_avg), dim=1) # [B, 2048]
        return global_feat

    def forward_from_features(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head given pre-extracted [B, 2048] features.
        """
        x = F.leaky_relu(self.bn1(self.fc1(feat)), negative_slope=0.2)
        x = self.drop1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = self.drop2(x)
        logits = self.fc3(x)
        return logits

    def forward(self, x: torch.Tensor):
        feat = self.extract_features(x)
        logits = self.forward_from_features(feat)
        return logits, feat

