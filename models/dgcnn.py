import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k=20):
    """
    Computes k-nearest neighbors in Euclidean or feature space.
    Input:
        x: [B, C, N] tensor
    Output:
        idx: [B, N, k] indices of k-nearest neighbors for each point
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # [B, N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True)       # [B, 1, N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # [B, N, N] (negated distance for topk)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1]     # [B, N, k]
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Constructs edge features e_ij = [x_i, x_j - x_i] for all k neighbors of each point.
    Input:
        x: [B, C, N] tensor
        k: number of nearest neighbors
        idx: optional precomputed neighbor indices [B, N, k]
    Output:
        edge_features: [B, 2*C, N, k] tensor
    """
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    
    if idx is None:
        idx = knn(x, k=k) # [B, N, k]
        
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    x_flat = x.transpose(2, 1).contiguous().view(batch_size * num_points, -1)
    feature = x_flat[idx, :].view(batch_size, num_points, k, num_dims) # [B, N, k, C] (neighbors x_j)
    
    x_expanded = x.transpose(2, 1).unsqueeze(2).expand(-1, -1, k, -1) # [B, N, k, C] (central points x_i)
    
    # Edge feature: [x_j - x_i, x_i] (relative displacement followed by central point coordinates)
    edge_feature = torch.cat((feature - x_expanded, x_expanded), dim=3).permute(0, 3, 1, 2).contiguous() # [B, 2*C, N, k]
    return edge_feature

class EdgeConv(nn.Module):
    """
    Standard EdgeConv Block:
    1. Recomputes dynamic k-NN graph in feature space
    2. Builds edge features [x_i, x_j - x_i]
    3. Shared Conv2d + BatchNorm2d + LeakyReLU
    4. Channel-wise Max-Pooling over neighborhood
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        # x: [B, C, N]
        edge_feat = get_graph_feature(x, k=self.k) # [B, 2*C, N, k]
        edge_feat = self.conv(edge_feat)           # [B, out_channels, N, k]
        out = edge_feat.max(dim=-1, keepdim=False)[0] # [B, out_channels, N]
        return out

class DGCNN(nn.Module):
    """
    Dynamic Graph CNN (DGCNN) Baseline Architecture for 3D Point Cloud Classification.
    
    Architecture:
      Input: [B, N, 3] or [B, 3, N]
      EdgeConv 1: 3   -> 64 (k-NN in spatial coordinate space)
      EdgeConv 2: 64  -> 64 (Dynamic k-NN in feature space 1)
      EdgeConv 3: 64  -> 128 (Dynamic k-NN in feature space 2)
      EdgeConv 4: 128 -> 256 (Dynamic k-NN in feature space 3)
      Multi-Scale Fusion: (64 + 64 + 128 + 256 = 512) -> 1024 1D Conv
      Dual Global Pooling: Adaptive MaxPool (1024) + Adaptive AvgPool (1024) = 2048 dims
      Classification Head: Linear(2048, 512) -> Linear(512, 256) -> Linear(256, num_classes)
    """
    def __init__(self, num_classes=40, k=20, emb_dims=1024, dropout=0.5):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        
        # 4 Hierarchical EdgeConv Layers
        self.edgeconv1 = EdgeConv(in_channels=3, out_channels=64, k=k)
        self.edgeconv2 = EdgeConv(in_channels=64, out_channels=64, k=k)
        self.edgeconv3 = EdgeConv(in_channels=64, out_channels=128, k=k)
        self.edgeconv4 = EdgeConv(in_channels=128, out_channels=256, k=k)
        
        # Multi-scale 1D Convolution Fusion
        self.conv_fusion = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        
        # Classification Head (Dual Pooling 1024*2 = 2048)
        self.fc1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=dropout)
        
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Permute [B, N, 3] to [B, 3, N] if necessary
        if x.dim() == 3 and x.size(2) == 3 and x.size(1) != 3:
            x = x.transpose(2, 1) # [B, 3, N]
            
        batch_size = x.size(0)
        
        # Layer 1 (spatial k-NN)
        x1 = self.edgeconv1(x)       # [B, 64, N]
        
        # Layer 2 (dynamic feature k-NN)
        x2 = self.edgeconv2(x1)      # [B, 64, N]
        
        # Layer 3 (dynamic feature k-NN)
        x3 = self.edgeconv3(x2)      # [B, 128, N]
        
        # Layer 4 (dynamic feature k-NN)
        x4 = self.edgeconv4(x3)      # [B, 256, N]
        
        # Multi-scale concatenation & 1D conv fusion
        x_concat = torch.cat((x1, x2, x3, x4), dim=1) # [B, 512, N]
        x_fused = self.conv_fusion(x_concat)          # [B, 1024, N]
        
        # Dual Global Pooling: Max-Pool + Avg-Pool
        x_max = F.adaptive_max_pool1d(x_fused, 1).view(batch_size, -1) # [B, 1024]
        x_avg = F.adaptive_avg_pool1d(x_fused, 1).view(batch_size, -1) # [B, 1024]
        global_feat = torch.cat((x_max, x_avg), dim=1)                 # [B, 2048]
        
        # Fully Connected Classification Head
        x = F.leaky_relu(self.bn1(self.fc1(global_feat)), negative_slope=0.2)
        x = self.drop1(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.2)
        x = self.drop2(x)
        logits = self.fc3(x) # [B, num_classes]
        
        return logits, None
