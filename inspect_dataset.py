import os
import sys
import glob
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_off_header(file_path):
    """Fast header parser for OFF mesh files."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        line = f.readline().strip()
        while not line or line.startswith('#'):
            line = f.readline().strip()
        if line.startswith('OFF'):
            rem = line[3:].strip()
            if rem:
                parts = rem.split()
            else:
                line2 = f.readline().strip()
                while not line2 or line2.startswith('#'):
                    line2 = f.readline().strip()
                parts = line2.split()
        else:
            parts = line.split()
        return int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0

def load_off_mesh(file_path, max_vertices=None):
    """
    Parses OFF mesh vertices and faces for geometric analysis / point sampling.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
    first_line = lines[0]
    if first_line.startswith('OFF'):
        rem = first_line[3:].strip()
        if rem:
            header_parts = rem.split()
            start_idx = 1
        else:
            header_parts = lines[1].split()
            start_idx = 2
    else:
        header_parts = first_line.split()
        start_idx = 1
        
    n_verts = int(header_parts[0])
    n_faces = int(header_parts[1])
    
    verts_to_read = min(n_verts, max_vertices) if max_vertices else n_verts
    
    vertices = []
    for i in range(start_idx, start_idx + verts_to_read):
        parts = lines[i].split()
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    vertices = np.array(vertices, dtype=np.float32)
    
    faces = []
    if max_vertices is None: # Only load faces if full mesh requested
        for i in range(start_idx + n_verts, start_idx + n_verts + n_faces):
            parts = lines[i].split()
            n_v = int(parts[0])
            faces.append([int(x) for x in parts[1:1+n_v]])
            
    return vertices, faces, n_verts, n_faces

def sample_point_cloud_from_mesh(vertices, faces, n_points=1024):
    """
    Uniform Surface Sampling (area-weighted triangle sampling + barycentric interpolation).
    """
    if len(faces) == 0:
        if len(vertices) >= n_points:
            idx = np.random.choice(len(vertices), n_points, replace=False)
            return vertices[idx]
        else:
            idx = np.random.choice(len(vertices), n_points, replace=True)
            return vertices[idx]
            
    tri_verts = []
    for f in faces:
        if len(f) >= 3:
            for j in range(1, len(f) - 1):
                tri_verts.append([f[0], f[j], f[j+1]])
    
    tri_verts = np.array(tri_verts, dtype=np.int32)
    v0 = vertices[tri_verts[:, 0]]
    v1 = vertices[tri_verts[:, 1]]
    v2 = vertices[tri_verts[:, 2]]
    
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = np.sum(areas)
    
    if total_area == 0 or np.isnan(total_area):
        idx = np.random.choice(len(vertices), n_points, replace=(len(vertices) < n_points))
        return vertices[idx]
        
    probs = areas / total_area
    chosen_tris = np.random.choice(len(tri_verts), size=n_points, p=probs)
    
    u = np.random.uniform(0, 1, size=(n_points, 1))
    v = np.random.uniform(0, 1, size=(n_points, 1))
    mask = (u + v) > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - u - v
    
    a = vertices[tri_verts[chosen_tris, 0]]
    b = vertices[tri_verts[chosen_tris, 1]]
    c = vertices[tri_verts[chosen_tris, 2]]
    
    sampled_pts = u * a + v * b + w * c
    return sampled_pts

def normalize_point_cloud(points):
    """
    Centering at origin and scaling to fit within unit sphere.
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1)))
    points = points / (furthest_distance + 1e-8)
    return points, centroid, furthest_distance

def main():
    dataset_root = r"c:\DLproj\dataset\ModelNet40"
    vis_dir = r"c:\DLproj\visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 1: MODELNET40 DATASET COMPREHENSIVE INSPECTION")
    print("=" * 80)
    
    all_entries = os.listdir(dataset_root)
    mac_artifacts = sorted([e for e in all_entries if e.startswith("._")])
    extracted_class_dirs = sorted([e for e in all_entries if os.path.isdir(os.path.join(dataset_root, e)) and not e.startswith(".")])
    all_40_class_names = sorted([e.replace("._", "") for e in mac_artifacts])
    missing_classes = sorted(list(set(all_40_class_names) - set(extracted_class_dirs)))
    
    print(f"\n[1 & 2 & 3] CLASSES & ROOT FOLDER AUDIT:")
    print(f" - Canonical ModelNet40 Classes defined in dataset metadata: {len(all_40_class_names)}")
    print(f" - Extracted Category Directories Present: {len(extracted_class_dirs)}")
    print(f" - Unextracted / Metadata-Only Category Headers (._*): {len(missing_classes)}")
    print(f" - Extracted Classes ({len(extracted_class_dirs)}): {', '.join(extracted_class_dirs)}")
    if missing_classes:
        print(f" - Metadata-Only Classes ({len(missing_classes)}): {', '.join(missing_classes)}")
        
    train_files = []
    test_files = []
    class_stats = {}
    file_formats = set()
    
    for cls in extracted_class_dirs:
        cls_dir = os.path.join(dataset_root, cls)
        train_dir = os.path.join(cls_dir, "train")
        test_dir = os.path.join(cls_dir, "test")
        
        tr_list = []
        te_list = []
        
        if os.path.exists(train_dir):
            for f in os.listdir(train_dir):
                if not f.startswith("._") and not f.startswith("."):
                    ext = os.path.splitext(f)[1].lower()
                    file_formats.add(ext)
                    tr_list.append(os.path.join(train_dir, f))
                    
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                if not f.startswith("._") and not f.startswith("."):
                    ext = os.path.splitext(f)[1].lower()
                    file_formats.add(ext)
                    te_list.append(os.path.join(test_dir, f))
                    
        class_stats[cls] = {
            "train": len(tr_list),
            "test": len(te_list),
            "total": len(tr_list) + len(te_list),
            "train_paths": tr_list,
            "test_paths": te_list
        }
        train_files.extend(tr_list)
        test_files.extend(te_list)
        
    print(f"\n[4, 5, 6] SAMPLE COUNTS & FORMATS:")
    print(f" - Total Valid 3D Mesh Files: {len(train_files) + len(test_files)}")
    print(f" - Training Objects: {len(train_files)}")
    print(f" - Test Objects: {len(test_files)}")
    print(f" - File Formats Used: {list(file_formats)}")
    
    # Vertex & Face Statistics Scan
    print(f"\n[7, 8, 9] POINT / VERTEX / FEATURE INSPECTION:")
    vertex_counts = []
    face_counts = []
    corrupted_files = []
    
    all_objects = [("train", f) for f in train_files] + [("test", f) for f in test_files]
    
    for split, fpath in all_objects:
        try:
            nv, nf, _ = read_off_header(fpath)
            vertex_counts.append(nv)
            face_counts.append(nf)
        except Exception as e:
            corrupted_files.append((fpath, str(e)))
            
    v_arr = np.array(vertex_counts)
    f_arr = np.array(face_counts)
    
    print(f" - All {len(v_arr)} files parsed successfully. Corrupted headers: {len(corrupted_files)}")
    print(f" - Vertices per Object:")
    print(f"     Min: {v_arr.min():,}")
    print(f"     Max: {v_arr.max():,}")
    print(f"     Mean: {v_arr.mean():,.2f}")
    print(f"     Median: {np.median(v_arr):,.1f}")
    print(f"     Std: {v_arr.std():,.2f}")
    print(f" - Faces per Object:")
    print(f"     Min: {f_arr.min():,}")
    print(f"     Max: {f_arr.max():,}")
    print(f"     Mean: {f_arr.mean():,.2f}")
    print(f"     Median: {np.median(f_arr):,.1f}")
    print(f" - Points Fixed or Variable: VARIABLE (ranges from {v_arr.min()} to {v_arr.max():,} vertices)")
    print(f" - Available Features in raw .OFF files: Only 3D Cartesian coordinates (X, Y, Z) and polygon facet indices. No surface normals or RGB color channels are pre-stored.")
    
    # Hash check for duplicates and data leakage
    print(f"\n[10 & 11] DUPLICATE AUDIT & DATA LEAKAGE ANALYSIS:")
    hash_dict = {}
    train_duplicates = []
    test_duplicates = []
    train_test_leakage = []
    
    for split, fpath in all_objects:
        with open(fpath, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in hash_dict:
            prev_split, prev_path = hash_dict[h]
            if prev_split == split:
                if split == "train":
                    train_duplicates.append((prev_path, fpath))
                else:
                    test_duplicates.append((prev_path, fpath))
            else:
                train_test_leakage.append((prev_path, fpath))
        else:
            hash_dict[h] = (split, fpath)
            
    print(f" - Exact Duplicate Files within Training Split: {len(train_duplicates)}")
    print(f" - Exact Duplicate Files within Test Split: {len(test_duplicates)}")
    print(f" - Train/Test Data Leakage (Identical MD5 across train and test): {len(train_test_leakage)}")
    if train_test_leakage:
        print("   Detected Leaked Sample Pairs (MD5 collision across splits):")
        for p1, p2 in train_test_leakage[:5]:
            cls1 = os.path.basename(os.path.dirname(os.path.dirname(p1)))
            cls2 = os.path.basename(os.path.dirname(os.path.dirname(p2)))
            print(f"     • [{cls1}] {os.path.basename(p1)} <==> [{cls2}] {os.path.basename(p2)}")

    # 12. Five Example Objects from Different Classes
    print(f"\n[12] FIVE DETAILED EXAMPLE OBJECTS FROM DIFFERENT CLASSES:")
    chosen_5_classes = ["airplane", "chair", "bed", "car", "guitar"]
    example_objects = {}
    
    for cls in chosen_5_classes:
        fpath = class_stats[cls]["train_paths"][0]
        verts, faces, nv, nf = load_off_mesh(fpath)
        norm_verts, centroid, scale = normalize_point_cloud(verts)
        example_objects[cls] = {
            "class": cls,
            "filepath": fpath,
            "split": "train",
            "num_vertices": nv,
            "num_faces": nf,
            "raw_bbox_min": verts.min(axis=0).tolist(),
            "raw_bbox_max": verts.max(axis=0).tolist(),
            "raw_dimensions": (verts.max(axis=0) - verts.min(axis=0)).tolist(),
            "raw_centroid": centroid.tolist(),
            "scale_radius": float(scale)
        }
        print(f"\n  • Class: {cls.upper()}")
        print(f"    - File: {fpath}")
        print(f"    - Vertices: {nv:,} | Faces: {nf:,}")
        print(f"    - Raw Bounding Box Min: {np.round(verts.min(axis=0), 3)}")
        print(f"    - Raw Bounding Box Max: {np.round(verts.max(axis=0), 3)}")
        print(f"    - Dimensions (W x H x D): {np.round(verts.max(axis=0) - verts.min(axis=0), 3)}")

    # 13. Visualizations Generation
    print(f"\n[13] GENERATING 3D POINT CLOUD VISUALIZATIONS...")
    
    # 13.1 Single Object Detailed Visualization (Airplane): Raw Vertices vs Surface Sampled vs Normalized
    ap_file = class_stats["airplane"]["train_paths"][0]
    ap_verts, ap_faces, _, _ = load_off_mesh(ap_file)
    ap_sampled = sample_point_cloud_from_mesh(ap_verts, ap_faces, n_points=1024)
    ap_norm, _, _ = normalize_point_cloud(ap_sampled)
    
    fig = plt.figure(figsize=(18, 6), dpi=200)
    
    # Subplot 1: Raw Mesh Vertices
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    # Subsample raw vertices if too large for clear scatter
    sub_raw = ap_verts[np.random.choice(len(ap_verts), min(len(ap_verts), 10000), replace=False)]
    p1 = ax1.scatter(sub_raw[:, 0], sub_raw[:, 1], sub_raw[:, 2], c=sub_raw[:, 2], cmap='viridis', s=1, alpha=0.6)
    ax1.set_title(f"1. Raw Mesh Vertices\n({len(ap_verts):,} total vertices)", fontsize=11, fontweight='bold', pad=12)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=25, azim=45)
    
    # Subplot 2: Area-Weighted Surface Sampled
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    p2 = ax2.scatter(ap_sampled[:, 0], ap_sampled[:, 1], ap_sampled[:, 2], c=ap_sampled[:, 2], cmap='magma', s=8, alpha=0.9)
    ax2.set_title("2. Surface-Sampled Point Cloud\n(N = 1,024 points, raw scale)", fontsize=11, fontweight='bold', pad=12)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.view_init(elev=25, azim=45)
    
    # Subplot 3: Unit-Sphere Normalized Point Cloud (Ready for GNN)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    p3 = ax3.scatter(ap_norm[:, 0], ap_norm[:, 1], ap_norm[:, 2], c=ap_norm[:, 2], cmap='plasma', s=8, alpha=0.9)
    ax3.set_title("3. Unit-Sphere Normalized P.C.\n(N × 3 ∈ [-1, 1], Centered at 0)", fontsize=11, fontweight='bold', pad=12)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])
    ax3.view_init(elev=25, azim=45)
    
    plt.suptitle("ModelNet40 3D Representation Pipeline (Airplane Example)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    vis_single_path = os.path.join(vis_dir, "airplane_pointcloud_pipeline.png")
    plt.savefig(vis_single_path)
    plt.close()
    print(f" - Saved Single Object Pipeline Visualization: {vis_single_path}")
    
    # 13.2 Five Classes 3D Point Cloud Gallery
    fig = plt.figure(figsize=(20, 4.5), dpi=200)
    for idx, (cls_name, data) in enumerate(example_objects.items()):
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        verts, faces, _, _ = load_off_mesh(data["filepath"])
        sampled = sample_point_cloud_from_mesh(verts, faces, n_points=1024)
        norm_pts, _, _ = normalize_point_cloud(sampled)
        
        ax.scatter(norm_pts[:, 0], norm_pts[:, 1], norm_pts[:, 2], c=norm_pts[:, 2], cmap='coolwarm', s=5, alpha=0.85)
        ax.set_title(f"{cls_name.upper()}\n({data['num_vertices']:,} v / {data['num_faces']:,} f)", fontsize=11, fontweight='bold')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=40)
        
    plt.suptitle("5 Example Classes: Sampled & Normalized Point Clouds (1024 points each)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    vis_gallery_path = os.path.join(vis_dir, "5_classes_pointcloud_gallery.png")
    plt.savefig(vis_gallery_path)
    plt.close()
    print(f" - Saved 5-Class Gallery Visualization: {vis_gallery_path}")

    # 13.3 Class Sample Distribution Plot
    fig, ax = plt.subplots(figsize=(15, 6), dpi=200)
    cls_names = list(class_stats.keys())
    tr_counts = [class_stats[k]["train"] for k in cls_names]
    te_counts = [class_stats[k]["test"] for k in cls_names]
    
    x = np.arange(len(cls_names))
    width = 0.4
    ax.bar(x - width/2, tr_counts, width, label='Train Split', color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, te_counts, width, label='Test Split', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_names, rotation=65, ha='right', fontsize=9, fontweight='semibold')
    ax.set_ylabel('Number of 3D Objects', fontsize=11, fontweight='bold')
    ax.set_title('ModelNet40 Sample Distribution per Category (Train vs Test)', fontsize=13, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    vis_dist_path = os.path.join(vis_dir, "class_distribution.png")
    plt.savefig(vis_dist_path)
    plt.close()
    print(f" - Saved Class Distribution Chart: {vis_dist_path}")

    # 13.4 Vertex Distribution Histogram
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    log_v = np.log10(v_arr)
    ax.hist(log_v, bins=40, color='#2ca02c', edgecolor='black', alpha=0.8)
    ax.set_xlabel("Log10(Number of Vertices)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Number of Objects", fontsize=11, fontweight='bold')
    ax.set_title("Distribution of Raw Vertex Counts Across ModelNet40 (Log Scale)", fontsize=12, fontweight='bold')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    plt.tight_layout()
    vis_hist_path = os.path.join(vis_dir, "vertex_count_distribution.png")
    plt.savefig(vis_hist_path)
    plt.close()
    print(f" - Saved Vertex Count Distribution: {vis_hist_path}")

    # Save detailed JSON summary
    report = {
        "dataset_root": dataset_root,
        "total_canonical_classes": len(all_40_class_names),
        "all_40_class_names": all_40_class_names,
        "extracted_class_dirs_count": len(extracted_class_dirs),
        "extracted_class_dirs": extracted_class_dirs,
        "missing_class_dirs": missing_classes,
        "mac_os_resource_fork_files_count": len(mac_artifacts),
        "total_samples": len(train_files) + len(test_files),
        "train_samples": len(train_files),
        "test_samples": len(test_files),
        "file_formats": list(file_formats),
        "vertex_statistics": {
            "min": int(v_arr.min()),
            "max": int(v_arr.max()),
            "mean": float(v_arr.mean()),
            "median": float(np.median(v_arr)),
            "std": float(v_arr.std())
        },
        "face_statistics": {
            "min": int(f_arr.min()),
            "max": int(f_arr.max()),
            "mean": float(f_arr.mean()),
            "median": float(np.median(f_arr)),
            "std": float(f_arr.std())
        },
        "duplicate_analysis": {
            "train_internal_duplicates": len(train_duplicates),
            "test_internal_duplicates": len(test_duplicates),
            "train_test_data_leakage_count": len(train_test_leakage),
            "leakage_sample_pairs": [
                {"file1": p1, "file2": p2} for p1, p2 in train_test_leakage[:10]
            ]
        },
        "example_objects": example_objects,
        "class_breakdown": {
            k: {"train": v["train"], "test": v["test"], "total": v["total"]} for k, v in class_stats.items()
        }
    }
    
    with open(r"c:\DLproj\inspection_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nInspection summary successfully written to: c:\\DLproj\\inspection_summary.json")

if __name__ == "__main__":
    main()
