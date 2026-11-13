import time
import torch
from models.pointnet import PointNetClassifier
from models.dgcnn import DGCNN

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on: {device} ({torch.cuda.get_device_name(0)})")
    
    p = PointNetClassifier(40).to(device).eval()
    d = DGCNN(40, k=20).to(device).eval()
    
    p_params = sum(x.numel() for x in p.parameters())
    d_params = sum(x.numel() for x in d.parameters())
    
    x = torch.randn(32, 1024, 3, device=device)
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            p(x)
            d(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # PointNet Timing
        t0 = time.time()
        for _ in range(50):
            p(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        p_time = (time.time() - t0) / 50 * 1000
        
        # DGCNN Timing
        t0 = time.time()
        for _ in range(50):
            d(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        d_time = (time.time() - t0) / 50 * 1000
        
    print(f"PointNet Parameters: {p_params:,}")
    print(f"DGCNN Parameters:    {d_params:,}")
    print(f"PointNet Inference:  {p_time:.2f} ms/batch (32) | {p_time/32:.3f} ms/sample")
    print(f"DGCNN Inference:     {d_time:.2f} ms/batch (32) | {d_time/32:.3f} ms/sample")

if __name__ == "__main__":
    benchmark()
