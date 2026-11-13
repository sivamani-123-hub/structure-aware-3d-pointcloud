import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    dataset_root = r"c:\DLproj\dataset\ModelNet40"
    vis_dir = r"c:\DLproj\visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # 40 canonical classes
    all_entries = os.listdir(dataset_root)
    mac_artifacts = sorted([e for e in all_entries if e.startswith("._")])
    all_40_classes = sorted([e.replace("._", "") for e in mac_artifacts])
    
    if len(all_40_classes) != 40:
        # Fallback to standard canonical ModelNet40 list if needed
        all_40_classes = sorted([
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
            'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
            'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
            'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
            'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
        ])
        
    class_data = []
    
    for cls in all_40_classes:
        cls_dir = os.path.join(dataset_root, cls)
        train_dir = os.path.join(cls_dir, "train")
        test_dir = os.path.join(cls_dir, "test")
        
        n_train = 0
        n_test = 0
        is_extracted = os.path.isdir(cls_dir)
        
        if is_extracted and os.path.exists(train_dir):
            n_train = len([f for f in os.listdir(train_dir) if f.endswith('.off') and not f.startswith('.')])
        if is_extracted and os.path.exists(test_dir):
            n_test = len([f for f in os.listdir(test_dir) if f.endswith('.off') and not f.startswith('.')])
            
        n_total = n_train + n_test
        split_ratio = (n_train / n_total * 100) if n_total > 0 else 0.0
        
        class_data.append({
            "class": cls,
            "is_extracted": is_extracted,
            "train": n_train,
            "test": n_test,
            "total": n_total,
            "train_pct": split_ratio,
            "test_pct": 100.0 - split_ratio if n_total > 0 else 0.0
        })
        
    # Sort by total count descending
    class_data_sorted = sorted(class_data, key=lambda x: x["total"], reverse=True)
    
    # Statistical analysis of present classes
    present_classes = [c for c in class_data if c["is_extracted"]]
    missing_classes = [c for c in class_data if not c["is_extracted"]]
    
    train_counts = np.array([c["train"] for c in present_classes])
    test_counts = np.array([c["test"] for c in present_classes])
    total_counts = np.array([c["total"] for c in present_classes])
    
    # Train / Test Consistency: Pearson Correlation and cosine similarity
    train_prob = train_counts / train_counts.sum()
    test_prob = test_counts / test_counts.sum()
    
    correlation = np.corrcoef(train_counts, test_counts)[0, 1]
    cosine_sim = np.dot(train_prob, test_prob) / (np.linalg.norm(train_prob) * np.linalg.norm(test_prob))
    
    # Imbalance Ratio
    max_cls = present_classes[np.argmax(total_counts)]
    min_cls = present_classes[np.argmin(total_counts)]
    imbalance_ratio = max_cls["total"] / min_cls["total"]
    
    print(f"=== CLASS DISTRIBUTION SUMMARY ===")
    print(f"Total Canonical Classes: {len(all_40_classes)}")
    print(f"Present Classes: {len(present_classes)}")
    print(f"Missing Classes: {len(missing_classes)}")
    print(f"Total Training Samples: {train_counts.sum()}")
    print(f"Total Test Samples: {test_counts.sum()}")
    print(f"Total Samples: {total_counts.sum()}")
    print(f"Max Class: {max_cls['class']} ({max_cls['total']} samples)")
    print(f"Min Class: {min_cls['class']} ({min_cls['total']} samples)")
    print(f"Imbalance Ratio (Max / Min among present): {imbalance_ratio:.2f}x")
    print(f"Train/Test Distribution Correlation: {correlation:.4f}")
    print(f"Train/Test Cosine Similarity: {cosine_sim:.4f}")
    
    # Generate Visualizations
    
    # 1. Bar Chart: All 40 Classes (Train vs Test)
    fig, ax = plt.subplots(figsize=(16, 7), dpi=200)
    
    all_names = [c["class"] for c in class_data_sorted]
    all_tr = [c["train"] for c in class_data_sorted]
    all_te = [c["test"] for c in class_data_sorted]
    
    x = np.arange(len(all_names))
    width = 0.42
    
    rects1 = ax.bar(x - width/2, all_tr, width, label=f'Train Split (Total: {train_counts.sum():,})', color='#1f77b4', edgecolor='black', linewidth=0.5)
    rects2 = ax.bar(x + width/2, all_te, width, label=f'Test Split (Total: {test_counts.sum():,})', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('ModelNet40 Classes (Ranked by Total Sample Count)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Number of 3D Objects', fontsize=11, fontweight='bold')
    ax.set_title('ModelNet40 Detailed Class Distribution: Training vs. Test Samples (All 40 Classes)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=60, ha='right', fontsize=9, fontweight='semibold')
    ax.legend(frameon=True, fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Mark missing classes visually
    for i, c in enumerate(class_data_sorted):
        if not c["is_extracted"]:
            ax.text(i, 15, "0 (Missing)", rotation=90, ha='center', va='bottom', fontsize=7, color='red', fontweight='bold')
            
    plt.tight_layout()
    barchart_path = os.path.join(vis_dir, "modelnet40_all_40_classes_distribution.png")
    plt.savefig(barchart_path)
    plt.close()
    print(f"Saved bar chart: {barchart_path}")
    
    # 2. Train vs Test Ratio Consistency Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
    
    # Scatter plot showing proportion alignment
    ax1.scatter(train_prob * 100, test_prob * 100, color='#2ca02c', s=60, edgecolor='black', zorder=3)
    lims = [0, max((train_prob * 100).max(), (test_prob * 100).max()) + 0.5]
    ax1.plot(lims, lims, 'r--', alpha=0.75, label='Perfect Split Proportionality (1:1)')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel('Class Share in Training Set (%)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Class Share in Test Set (%)', fontsize=10, fontweight='bold')
    ax1.set_title(f'Train vs Test Class Proportions\n(Pearson r = {correlation:.4f}, Cosine Sim = {cosine_sim:.4f})', fontsize=11, fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Label top classes
    for i, c in enumerate(present_classes):
        if c["total"] > 400 or c["total"] < 120:
            ax1.annotate(c["class"], (train_prob[i]*100, test_prob[i]*100), fontsize=7, alpha=0.8,
                         textcoords="offset points", xytext=(5, 5))
            
    # Train/Test percentage per class
    ratios = [c["train"] / c["total"] * 100 for c in present_classes]
    p_names = [c["class"] for c in present_classes]
    ax2.axhline(80, color='r', linestyle='--', label='Theoretical 80/20 Split')
    ax2.bar(range(len(p_names)), ratios, color='#3b528b', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(p_names)))
    ax2.set_xticklabels(p_names, rotation=65, ha='right', fontsize=8)
    ax2.set_ylabel('Training Set Percentage (%)', fontsize=10, fontweight='bold')
    ax2.set_title('Train Split Proportion per Present Class', fontsize=11, fontweight='bold')
    ax2.set_ylim([70, 90])
    ax2.legend(frameon=True)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    consistency_path = os.path.join(vis_dir, "train_test_split_consistency.png")
    plt.savefig(consistency_path)
    plt.close()
    print(f"Saved consistency chart: {consistency_path}")
    
    # Save output data to json
    out_dict = {
        "summary": {
            "canonical_classes": len(all_40_classes),
            "present_classes": len(present_classes),
            "missing_classes": len(missing_classes),
            "total_train_samples": int(train_counts.sum()),
            "total_test_samples": int(test_counts.sum()),
            "total_samples": int(total_counts.sum()),
            "imbalance_ratio": float(imbalance_ratio),
            "train_test_correlation": float(correlation),
            "train_test_cosine_similarity": float(cosine_sim)
        },
        "all_classes_table": class_data_sorted,
        "missing_class_names": [c["class"] for c in missing_classes]
    }
    with open(r"c:\DLproj\class_distribution_analysis.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print("Class distribution analysis complete!")

if __name__ == "__main__":
    main()
