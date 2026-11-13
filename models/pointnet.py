import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """
    Spatial Transformer Network (T-Net) for aligning 3D point coordinates or feature spaces.
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0] # [B, 1024]
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        identity = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x

class PointNetBackbone(nn.Module):
    """
    PointNet Feature Extractor:
    Input coordinates -> Shared MLP (64) -> Shared MLP (128, 1024) -> Symmetric Global MaxPool
    """
    def __init__(self, use_tnet=False):
        super().__init__()
        self.use_tnet = use_tnet
        if self.use_tnet:
            self.input_tnet = TNet(k=3)
            self.feature_tnet = TNet(k=64)
            
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x: [B, 3, N]
        trans_input = None
        trans_feat = None
        
        if self.use_tnet:
            trans_input = self.input_tnet(x)
            x = torch.bmm(trans_input, x)
            
        x = F.relu(self.bn1(self.conv1(x))) # [B, 64, N]
        
        if self.use_tnet:
            trans_feat = self.feature_tnet(x)
            x = torch.bmm(trans_feat, x)
            
        x = F.relu(self.bn2(self.conv2(x))) # [B, 128, N]
        x = self.bn3(self.conv3(x)) # [B, 1024, N]
        
        # Symmetric Global Max Pooling across all N points
        global_features = torch.max(x, 2, keepdim=False)[0] # [B, 1024]
        
        return global_features, trans_input, trans_feat

class PointNetClassifier(nn.Module):
    """
    Full PointNet Classification Network.
    Maps point clouds [B, N, 3] (or [B, 3, N]) to class logits [B, num_classes].
    """
    def __init__(self, num_classes=40, use_tnet=False, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = PointNetBackbone(use_tnet=use_tnet)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Allow input shape [B, N, 3] or [B, 3, N]
        if x.dim() == 3 and x.size(2) == 3 and x.size(1) != 3:
            x = x.transpose(2, 1) # [B, 3, N]
            
        global_features, trans_input, trans_feat = self.backbone(x)
        
        x = F.relu(self.bn1(self.fc1(global_features)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        logits = self.fc3(x) # [B, num_classes]
        
        return logits, trans_feat

def feature_transform_regularizer(trans):
    """
    Orthogonal regularization loss for Feature T-Net: ||I - A A^T||_F^2
    """
    if trans is None:
        return 0.0
    d = trans.size(1)
    batch_size = trans.size(0)
    I = torch.eye(d, device=trans.device).unsqueeze(0).repeat(batch_size, 1, 1)
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
