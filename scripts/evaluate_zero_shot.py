import os
import sys
import json
import csv
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure DLproj root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.datasets.scanobjectnn import get_scanobjectnn_dataloader
from src.datasets.class_mapping import COMMON_CLASS_NAMES
from src.inference.model_loader import load_model, get_device
from src.evaluation.zero_shot_evaluator import evaluate_zero_shot

def plot_model_comparison_chart(results_dict, save_path):
    """
    Generates a grouped bar chart comparing OA, mAcc, Macro F1, and Weighted F1.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    models = list(results_dict.keys())
    labels = [m.replace("_", " ").title() for m in models]
    
    metrics = ["overall_accuracy", "mean_class_accuracy", "macro_f1", "weighted_f1"]
    metric_names = ["Overall Accuracy (OA)", "Mean Class Acc (mAcc)", "Macro F1-Score", "Weighted F1-Score"]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    for i, (m_key, m_name) in enumerate(zip(metrics, metric_names)):
        vals = [results_dict[m][m_key] * 100 for m in models]
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, vals, width, label=m_name, color=colors[i], alpha=0.9, edgecolor='black', linewidth=0.8)
        # Add text labels on top of bars
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('Zero-Shot Cross-Domain Performance on ScanObjectNN (Real-World 3D Scans)', fontsize=13, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved zero-shot comparison bar chart to: {save_path}")


def plot_per_class_comparison_chart(results_dict, save_path):
    """
    Generates horizontal per-class bar comparison chart across all evaluated models.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    classes = COMMON_CLASS_NAMES
    n_classes = len(classes)
    y = np.arange(n_classes)
    height = 0.2
    
    models = list(results_dict.keys())
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    palette = ['#4A90E2', '#50E3C2', '#F5A623', '#9013FE']
    
    for idx, m in enumerate(models):
        metrics = results_dict[m]["per_class_metrics"]
        accs = [d["accuracy"] * 100 for d in metrics]
        offset = (idx - 1.5) * height
        rects = ax.barh(y + offset, accs, height, label=m.replace("_", " ").title(), color=palette[idx % len(palette)], alpha=0.85, edgecolor='black', linewidth=0.6)

    ax.set_xlabel('Per-Class Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Zero-Shot Per-Class Accuracy Comparison (ScanObjectNN Real Scans)', fontsize=13, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels([c.title() for c in classes], fontsize=9, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=9)
    ax.set_xlim(0, 105)
    ax.grid(axis='x', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class comparison chart to: {save_path}")


def run_all_zero_shot_evaluations():
    print("=" * 80)
    print("PHASE C / STEP 5: RIGOROUS ZERO-SHOT CROSS-DOMAIN BENCHMARK")
    print("Target Domain: ScanObjectNN (Official Real-World Scans, Test Split, 10 Classes)")
    print("Source Domain: ModelNet40 (Trained CAD Checkpoints, Zero Retraining/Fine-tuning)")
    print("=" * 80)

    device = get_device("auto")
    print(f"Execution Device: {device}")

    # Build standardized deterministic target test dataloader
    test_loader = get_scanobjectnn_dataloader(
        split="test",
        batch_size=32,
        num_points=1024,
        common_classes_only=True,
        seed=42
    )

    models_to_evaluate = [
        ("pointnet", "PointNet Baseline"),
        ("dgcnn", "DGCNN Baseline"),
        ("gnn", "Structure-Aware GNN"),
        ("peak_dgcnn", "Peak Regularized GNN")
    ]

    all_results = {}
    vis_dir = r"c:\DLproj\visualizations\phase7"
    res_dir = r"c:\DLproj\results\phase7"
    rep_dir = r"c:\DLproj\reports\phase7"
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    for model_key, model_label in models_to_evaluate:
        print(f"\n" + "-" * 70)
        print(f"Evaluating {model_label.upper()} on 386 Target-Domain Real-World Scans...")
        print("-" * 70)

        model, meta = load_model(model_key, device=device)
        cm_path = os.path.join(vis_dir, f"{model_key}_zeroshot_confusion_matrix.png")
        
        t0 = time.time()
        res = evaluate_zero_shot(
            model=model,
            dataloader=test_loader,
            device=device,
            model_name=model_key,
            eval_mode="constrained",
            save_cm_path=cm_path
        )
        elapsed = time.time() - t0

        all_results[model_key] = res

        print(f"  • Overall Accuracy (OA):       {res['overall_accuracy_pct']} ({res['correct_samples']}/{res['total_test_samples']} correct)")
        print(f"  • Mean Class Accuracy (mAcc):  {res['mean_class_accuracy_pct']}")
        print(f"  • Macro Precision:             {res['macro_precision_pct']}")
        print(f"  • Macro Recall:                {res['macro_recall_pct']}")
        print(f"  • Macro F1-Score:              {res['macro_f1_pct']}")
        print(f"  • Weighted F1-Score:           {res['weighted_f1_pct']}")
        print(f"  • Inference Time:              {elapsed:.2f}s ({elapsed/len(test_loader.dataset)*1000:.2f} ms/scan)")

    # Generate Comparison Visualizations
    bar_chart_path = os.path.join(vis_dir, "zero_shot_model_comparison.png")
    plot_model_comparison_chart(all_results, bar_chart_path)

    per_class_chart_path = os.path.join(vis_dir, "zero_shot_per_class_comparison.png")
    plot_per_class_comparison_chart(all_results, per_class_chart_path)

    # Save CSV with Per-Class Comparison
    csv_path = os.path.join(res_dir, "zero_shot_per_class_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Class Name", "Support"] + [f"{m.title()} Acc (%)" for m in all_results.keys()] + [f"{m.title()} F1 (%)" for m in all_results.keys()]
        writer.writerow(header)

        for c_idx, c_name in enumerate(COMMON_CLASS_NAMES):
            row = [c_name, all_results["pointnet"]["per_class_metrics"][c_idx]["support"]]
            for m in all_results.keys():
                row.append(f"{all_results[m]['per_class_metrics'][c_idx]['accuracy'] * 100:.2f}")
            for m in all_results.keys():
                row.append(f"{all_results[m]['per_class_metrics'][c_idx]['f1_score'] * 100:.2f}")
            writer.writerow(row)
    print(f"\nSaved per-class comparison CSV to: {csv_path}")

    # Save JSON summary (without bloated sample arrays)
    summary_json = {
        "benchmark": "Zero-Shot Synthetic-to-Real Domain Generalization",
        "source_dataset": "Princeton ModelNet40 (CAD)",
        "target_dataset": "ScanObjectNN (Real-World Scans, clean main_split_nobg)",
        "test_samples_evaluated": len(test_loader.dataset),
        "target_classes_count": len(COMMON_CLASS_NAMES),
        "common_classes": COMMON_CLASS_NAMES,
        "models": {}
    }
    for m in all_results.keys():
        r = all_results[m].copy()
        r.pop("sample_records", None) # exclude per-sample diagnostics to keep JSON concise
        summary_json["models"][m] = r

    json_path = os.path.join(res_dir, "zero_shot_evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved evaluation summary JSON to: {json_path}")

    # Generate Markdown Report
    rep_path = os.path.join(rep_dir, "zero_shot_evaluation_report.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Zero-Shot Synthetic-to-Real 3D Point Cloud Classification Benchmark\n\n")
        f.write("## 1. Executive Summary\n")
        f.write("- **Source Domain:** Princeton ModelNet40 (Clean Synthetic CAD Meshes, 1024 points, unit sphere normalized)\n")
        f.write("- **Target Domain:** ScanObjectNN (Noisy Real-World Scans from ScanNet/SceneNN, 1024 points, unit sphere normalized)\n")
        f.write(f"- **Target Test Partition:** Official Test Split ({len(test_loader.dataset)} scans across 10 overlapping categories)\n")
        f.write("- **Zero-Shot Protocol:** Strict zero-shot transfer — **zero** target labels or scans were used during training or tuning.\n\n")
        f.write("## 2. Overall Benchmark Results\n\n")
        f.write("| Architecture | Overall Accuracy (OA) | Mean Class Acc (mAcc) | Macro F1-Score | Weighted F1-Score | Correct Scans |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for m in all_results.keys():
            res = all_results[m]
            f.write(f"| **{m.replace('_', ' ').title()}** | **{res['overall_accuracy_pct']}** | {res['mean_class_accuracy_pct']} | {res['macro_f1_pct']} | {res['weighted_f1_pct']} | {res['correct_samples']}/{res['total_test_samples']} |\n")
        
        f.write("\n## 3. Per-Class Accuracy Census\n\n")
        f.write("| Class Name | Real-World Support | PointNet Acc | DGCNN Acc | GNN Acc | Peak GNN Acc |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for c_idx, c_name in enumerate(COMMON_CLASS_NAMES):
            supp = all_results["pointnet"]["per_class_metrics"][c_idx]["support"]
            pn_acc = all_results["pointnet"]["per_class_metrics"][c_idx]["accuracy_pct"]
            dg_acc = all_results["dgcnn"]["per_class_metrics"][c_idx]["accuracy_pct"]
            gn_acc = all_results["gnn"]["per_class_metrics"][c_idx]["accuracy_pct"]
            pk_acc = all_results["peak_dgcnn"]["per_class_metrics"][c_idx]["accuracy_pct"]
            f.write(f"| **{c_name}** | {supp} | {pn_acc} | {dg_acc} | {gn_acc} | {pk_acc} |\n")

    print(f"Saved evaluation markdown report to: {rep_path}")
    print("\n" + "=" * 80)
    print("PHASE C / STEP 5 BENCHMARK COMPLETE: ALL RESULTS LOGGED AND VERIFIED")
    print("=" * 80)

if __name__ == "__main__":
    run_all_zero_shot_evaluations()

