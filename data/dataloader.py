import os
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import ModelNet40Dataset, CANONICAL_MODELNET40_CLASSES

def get_modelnet40_dataloaders(
    dataset_root=r"c:\DLproj\dataset\ModelNet40",
    batch_size=32,
    num_points=1024,
    num_classes=40,
    val_split_ratio=0.1,
    num_workers=0,
    pin_memory=False,
    cache_in_memory=True,
    seed=42
):
    """
    Constructs PyTorch DataLoaders for Training, Validation, and Test sets.
    """
    torch.manual_seed(seed)
    
    # 1. Full Training Dataset
    full_train_dataset = ModelNet40Dataset(
        dataset_root=dataset_root,
        split="train",
        num_points=num_points,
        num_classes=num_classes,
        apply_augmentation=True,
        cache_in_memory=cache_in_memory,
        seed=seed
    )
    
    # 2. Validation Sub-split (strictly carved from train)
    if val_split_ratio > 0.0:
        val_size = int(len(full_train_dataset) * val_split_ratio)
        train_size = len(full_train_dataset) - val_size
        train_subset, val_subset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
        
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            full_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        val_loader = None
        
    # 3. Test Dataset (Deterministic, ZERO augmentation)
    test_dataset = ModelNet40Dataset(
        dataset_root=dataset_root,
        split="test",
        num_points=num_points,
        num_classes=num_classes,
        apply_augmentation=False,
        cache_in_memory=cache_in_memory,
        seed=seed + 100
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, full_train_dataset.classes
