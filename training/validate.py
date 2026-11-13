import torch
import torch.nn.functional as F
import numpy as np

def evaluate_model(model, dataloader, device, num_classes=40):
    """
    Evaluates model on validation/test DataLoader.
    Computes Loss, Overall Accuracy (OA), and Mean Class Accuracy (mAcc).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for points, labels in dataloader:
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass with AMP autocast
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    logits, _ = model(points)
                    loss = F.cross_entropy(logits, labels)
            else:
                logits, _ = model(points)
                loss = F.cross_entropy(logits, labels)
            
            total_loss += loss.item() * points.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total_samples += points.size(0)
            
            for p, l in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                if l < num_classes:
                    class_total[l] += 1
                    if p == l:
                        class_correct[l] += 1
                        
    avg_loss = total_loss / max(total_samples, 1)
    overall_acc = correct / max(total_samples, 1)
    
    # Calculate Mean Class Accuracy (mAcc) over classes with >0 test samples
    valid_classes = class_total > 0
    if np.any(valid_classes):
        mean_class_acc = np.mean(class_correct[valid_classes] / class_total[valid_classes])
    else:
        mean_class_acc = 0.0
        
    return {
        "loss": float(avg_loss),
        "overall_acc": float(overall_acc),
        "mean_class_acc": float(mean_class_acc),
        "class_accuracies": (class_correct / np.maximum(class_total, 1)).tolist()
    }
