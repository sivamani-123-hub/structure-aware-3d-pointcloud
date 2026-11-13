"""
Target-Domain Training and Adaptation Engine.

Implements:
1. Feature extraction and caching for rapid adaptation.
2. Adam optimization with Cosine Annealing learning rate schedule.
3. Validation-driven early stopping for model selection.
4. Separate checkpoint saving (zero mutation of original weights).
"""

import os
import sys
import time
import copy
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# DLproj imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.domain_adaptation.adapted_gnn import DomainAdaptedDGCNN


class TargetDomainTrainer:
    """
    Manages fine-tuning and domain adaptation of GNN on ScanObjectNN target data.
    """
    def __init__(
        self,
        adapted_model: DomainAdaptedDGCNN,
        device: torch.device,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        num_epochs: int = 25,
        patience: int = 7
    ):
        self.model = adapted_model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience

    def extract_and_cache_dataset_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        desc: str = "Split"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts 2048-dim global feature embeddings through the frozen GNN backbone.
        """
        self.model.eval()
        all_feats = []
        all_labels = []

        print(f"Extracting 2048-dim GNN representations for {desc} ({len(dataloader.dataset)} samples)...", flush=True)
        t0 = time.time()
        
        with torch.no_grad():
            for points, labels, _ in dataloader:
                points = points.to(self.device)
                feats = self.model.extract_features(points) # [B, 2048]
                all_feats.append(feats.cpu())
                all_labels.append(labels)

        all_feats_tensor = torch.cat(all_feats, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)
        elapsed = time.time() - t0
        print(f"  Extracted {len(all_feats_tensor)} embeddings in {elapsed:.2f}s ({elapsed/len(all_feats_tensor)*1000:.2f} ms/scan).", flush=True)
        return all_feats_tensor, all_labels_tensor

    def fit_adapted_head(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        save_checkpoint_path: str
    ) -> Dict[str, Any]:
        """
        Trains the adaptation classification head using cached representations.
        """
        os.makedirs(os.path.dirname(os.path.abspath(save_checkpoint_path)), exist_ok=True)

        # 1. Extract representations
        train_feats, train_labels = self.extract_and_cache_dataset_features(train_loader, "Target Train")
        val_feats, val_labels = self.extract_and_cache_dataset_features(val_loader, "Target Val")

        feat_train_loader = DataLoader(
            TensorDataset(train_feats, train_labels),
            batch_size=32,
            shuffle=True
        )
        feat_val_loader = DataLoader(
            TensorDataset(val_feats, val_labels),
            batch_size=32,
            shuffle=False
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.get_head_parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-5
        )

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "best_epoch": 0,
            "best_val_acc": 0.0
        }

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        print(f"\nStarting Domain Adaptation Head Training ({self.num_epochs} Epochs, lr={self.lr})...", flush=True)

        for epoch in range(1, self.num_epochs + 1):
            # Training Phase
            self.model.train()
            total_train_loss = 0.0
            correct_train = 0
            total_train = 0

            for feats_b, labels_b in feat_train_loader:
                feats_b = feats_b.to(self.device)
                labels_b = labels_b.to(self.device)

                optimizer.zero_grad()
                logits = self.model.forward_from_features(feats_b)
                loss = criterion(logits, labels_b)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * len(labels_b)
                preds = logits.argmax(dim=-1)
                correct_train += (preds == labels_b).sum().item()
                total_train += len(labels_b)

            scheduler.step()

            epoch_train_loss = total_train_loss / total_train
            epoch_train_acc = correct_train / total_train

            # Validation Phase
            self.model.eval()
            total_val_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for feats_b, labels_b in feat_val_loader:
                    feats_b = feats_b.to(self.device)
                    labels_b = labels_b.to(self.device)

                    logits = self.model.forward_from_features(feats_b)
                    loss = criterion(logits, labels_b)

                    total_val_loss += loss.item() * len(labels_b)
                    preds = logits.argmax(dim=-1)
                    correct_val += (preds == labels_b).sum().item()
                    total_val += len(labels_b)

            epoch_val_loss = total_val_loss / total_val
            epoch_val_acc = correct_val / total_val

            history["train_loss"].append(float(epoch_train_loss))
            history["val_loss"].append(float(epoch_val_loss))
            history["train_acc"].append(float(epoch_train_acc))
            history["val_acc"].append(float(epoch_val_acc))

            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                history["best_val_acc"] = float(best_val_acc)
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                improved_tag = " [BEST]"
            else:
                patience_counter += 1
                improved_tag = ""

            if epoch % 5 == 0 or epoch == 1 or improved_tag:
                print(f"Epoch {epoch:2d}/{self.num_epochs} | "
                      f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:5.2f}% | "
                      f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:5.2f}%{improved_tag}", flush=True)

            if patience_counter >= self.patience:
                print(f"Early stopping triggered at Epoch {epoch} (No improvement for {self.patience} epochs).")
                break

        # Load best state
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save checkpoint
        checkpoint = {
            "epoch": history["best_epoch"],
            "model_state_dict": self.model.state_dict(),
            "best_val_acc": history["best_val_acc"],
            "history": history,
            "config": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "num_classes": self.model.num_target_classes
            }
        }
        torch.save(checkpoint, save_checkpoint_path)
        print(f"\nSaved best fine-tuned GNN checkpoint to: {save_checkpoint_path}")
        print(f"Best Validation Accuracy: {history['best_val_acc']*100:.2f}% (Epoch {history['best_epoch']})")

        return history

