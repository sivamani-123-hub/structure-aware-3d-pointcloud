import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k=20):
    """
    Computes k-nearest neighbors in feature or coordinate space.
    Input:
        x: [B, C, N] tensor
    Output:
        idx: [B, N, k] indices of nearest neighbors for each point
    """
    # pairwise distance: ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * <x_i, x_j>
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # [B, N, N]
    xx = torch.sum(x ** 2, dim=1, keepdim=True) # [B, 1, N]
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # [B, N, N] (negative distance for topk)
    
    idx = pairwise_distance.topk(k=k, dim=-1)[1] # [B, N, k]
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Constructs edge features e_ij = [x_j - x_i, x_i] for all k neighbors.
    Input:
        x: [B, C, N] tensor
        k: number of neighbors
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
    
    # Reshape x to [B*N, C] to gather
    x_flat = x.transpose(2, 1).contiguous().view(batch_size * num_points, -1)
    feature = x_flat[idx, :].view(batch_size, num_points, k, num_dims) # [B, N, k, C]
    
    x_expanded = x.transpose(2, 1).unsqueeze(2).repeat(1, 1, k, 1) # [B, N, k, C]
    
    # Concatenate [x_j - x_i, x_i] along feature dimension
    edge_feature = torch.cat((feature - x_expanded, x_expanded), dim=3).permute(0, 3, 1, 2).contiguous() # [B, 2*C, N, k]
    return edge_feature

class EdgeConvBlock(nn.Module):
    """
    EdgeConv Layer:
    1. Builds dynamic k-NN graph in feature space
    2. Constructs edge feature [x_j - x_i, x_i]
    3. Applies shared MLP (Conv2d + BatchNorm2d + LeakyReLU)
    4. Aggregates across neighborhood using channel-wise max pooling
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
        edge_feat = self.conv(edge_feat) # [B, out_channels, N, k]
        out = edge_feat.max(dim=-1, keepdim=False)[0] # [B, out_channels, N]
        return out
