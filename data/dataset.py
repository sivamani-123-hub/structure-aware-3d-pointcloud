import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import PointCloudTransforms

CANONICAL_MODELNET40_CLASSES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
    'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
    'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
    'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
    'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
]

def parse_off_file_fast(file_path, max_v=2048):
    """
    Reads header and streams vertices uniformly across the entire mesh up to max_v.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        line = f.readline()
        while not line.strip() or line.strip().startswith('#'):
            line = f.readline()
        line = line.strip()
        if line.startswith('OFF'):
            rem = line[3:].strip()
            if rem:
                parts = rem.split()
            else:
                line2 = f.readline()
                while not line2.strip() or line2.strip().startswith('#'):
                    line2 = f.readline()
                parts = line2.strip().split()
        else:
            parts = line.split()
            
        n_verts = int(parts[0])
        n_faces = int(parts[1])
        
        if n_verts <= max_v:
            vertices = []
            for _ in range(n_verts):
                vl = f.readline().strip()
                while not vl or vl.startswith('#'):
                    vl = f.readline().strip()
                p = vl.split()
                if len(p) >= 3:
                    vertices.append([float(p[0]), float(p[1]), float(p[2])])
        else:
            # Stride uniformly across all n_verts to preserve full 3D geometry
            all_lines = [f.readline().strip() for _ in range(n_verts)]
            stride_indices = np.linspace(0, n_verts - 1, max_v, dtype=int)
            vertices = []
            for s_idx in stride_indices:
                vl = all_lines[s_idx]
                if vl and not vl.startswith('#'):
                    p = vl.split()
                    if len(p) >= 3:
                        vertices.append([float(p[0]), float(p[1]), float(p[2])])
            
        return np.array(vertices, dtype=np.float32), []

class ModelNet40Dataset(Dataset):
    """
    High-performance PyTorch Dataset for ModelNet40 Point Clouds.
    """
    def __init__(
        self,
        dataset_root=r"c:\DLproj\dataset\ModelNet40",
        split="train",
        num_points=1024,
        num_classes=40,
        apply_augmentation=True,
        cache_in_memory=False,
        seed=42
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split.lower()
        self.num_points = num_points
        self.num_classes = num_classes
        self.apply_augmentation = apply_augmentation and (self.split == "train")
        self.seed = seed
        
        # Build canonical class mapping
        if self.num_classes == 40:
            self.classes = CANONICAL_MODELNET40_CLASSES
        else:
            present_dirs = sorted([
                d for d in os.listdir(self.dataset_root) 
                if os.path.isdir(os.path.join(self.dataset_root, d)) and not d.startswith('.')
            ])
            self.classes = present_dirs[:self.num_classes]
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.dataset_root, cls_name, self.split)
            if os.path.exists(cls_dir):
                files = sorted([
                    f for f in os.listdir(cls_dir) 
                    if f.endswith('.off') and not f.startswith('.')
                ])
                for fname in files:
                    fpath = os.path.join(cls_dir, fname)
                    label_idx = self.class_to_idx[cls_name]
                    self.samples.append((fpath, label_idx, cls_name))
                    
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid .off files found for split '{self.split}'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label, cls_name = self.samples[idx]
        
        verts, faces = parse_off_file_fast(fpath, max_v=max(2048, self.num_points))
            
        # 1. Point Sampling to fixed N points
        if len(verts) >= self.num_points:
            if self.split == "train":
                choice_idx = np.random.choice(len(verts), self.num_points, replace=False)
            else:
                np.random.seed((self.seed + idx * 7) % 2147483647)
                choice_idx = np.random.choice(len(verts), self.num_points, replace=False)
            pts = verts[choice_idx]
        else:
            choice_idx = np.random.choice(len(verts), self.num_points, replace=True)
            pts = verts[choice_idx]
            
        # 2. Instance Normalization (Zero-centering & unit-sphere scaling)
        pts = PointCloudTransforms.normalize_instance(pts)
        
        # 3. Training Augmentations (Train Split Only)
        if self.apply_augmentation:
            pts = PointCloudTransforms.random_rotation_z(pts)
            pts = PointCloudTransforms.random_anisotropic_scale(pts, low=0.8, high=1.25)
            pts = PointCloudTransforms.random_translation(pts, shift_range=0.02)
            pts = PointCloudTransforms.random_jitter(pts, sigma=0.01, clip=0.05)
            pts = PointCloudTransforms.random_point_dropout(pts, max_dropout_ratio=0.15)
            
        points_tensor = torch.from_numpy(pts).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return points_tensor, label_tensor
