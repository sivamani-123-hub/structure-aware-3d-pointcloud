import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_off_header(file_path):
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

def load_off_vertices_and_faces(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
        
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
    
    vertices = []
    for i in range(start_idx, start_idx + n_verts):
        parts = lines[i].split()
        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    vertices = np.array(vertices, dtype=np.float32)
    
    faces = []
    for i in range(start_idx + n_verts, start_idx + n_verts + n_faces):
        parts = lines[i].split()
        nv = int(parts[0])
        faces.append([int(x) for x in parts[1:1+nv]])
        
    return vertices, faces

def sample_point_cloud_from_mesh(vertices, faces, n_points=1024):
    if len(faces) == 0:
        idx = np.random.choice(len(vertices), n_points, replace=(len(vertices) < n_points))
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
    
    return u * a + v * b + w * c

def compute_geometry_metrics(verts):
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    bbox_dims = max_xyz - min_xyz
    centroid = verts.mean(axis=0)
    
    dist_origin = np.linalg.norm(verts, axis=1)
    max_dist_origin = np.max(dist_origin)
    
    dist_centroid = np.linalg.norm(verts - centroid, axis=1)
    avg_dist_centroid = np.mean(dist_centroid)
    max_dist_centroid = np.max(dist_centroid)
    
    return {
        "min_x": float(min_xyz[0]), "max_x": float(max_xyz[0]),
        "min_y": float(min_xyz[1]), "max_y": float(max_xyz[1]),
        "min_z": float(min_xyz[2]), "max_z": float(max_xyz[2]),
        "centroid": centroid.tolist(),
        "bbox_dimensions": bbox_dims.tolist(),
        "aspect_ratios": (bbox_dims / (np.max(bbox_dims) + 1e-8)).tolist(),
        "max_dist_origin": float(max_dist_origin),
        "avg_dist_centroid": float(avg_dist_centroid),
        "max_dist_centroid": float(max_dist_centroid),
        "num_vertices": int(len(verts))
    }

def main():
    np.random.seed(42)
    dataset_root = r"c:\DLproj\dataset\ModelNet40"
    vis_dir = r"c:\DLproj\visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    classes_to_sample = [
        "airplane", "chair", "bed", "car", "guitar",
        "toilet", "laptop", "stairs", "person", "table"
    ]
    
    results = {}
    loaded_objects = {}
    
    print("=" * 80)
    print("PHASE 1: 3D GEOMETRY INSPECTION & METRICS CALCULATION")
    print("=" * 80)
    
    for cls in classes_to_sample:
        train_dir = os.path.join(dataset_root, cls, "train")
        files = [f for f in os.listdir(train_dir) if f.endswith('.off') and not f.startswith('.')]
        chosen_file = np.random.choice(files)
        fpath = os.path.join(train_dir, chosen_file)
        
        verts, faces = load_off_vertices_and_faces(fpath)
        metrics = compute_geometry_metrics(verts)
        metrics["file"] = fpath
        metrics["class"] = cls
        metrics["filename"] = chosen_file
        metrics["num_faces"] = len(faces)
        
        results[cls] = metrics
        loaded_objects[cls] = (verts, faces, metrics)
        
        print(f"\n[{cls.upper()}] - {chosen_file}")
        print(f"  • Vertices: {metrics['num_vertices']:,} | Faces: {len(faces):,}")
        print(f"  • X range: [{metrics['min_x']:.2f}, {metrics['max_x']:.2f}] (Span: {metrics['bbox_dimensions'][0]:.2f})")
        print(f"  • Y range: [{metrics['min_y']:.2f}, {metrics['max_y']:.2f}] (Span: {metrics['bbox_dimensions'][1]:.2f})")
        print(f"  • Z range: [{metrics['min_z']:.2f}, {metrics['max_z']:.2f}] (Span: {metrics['bbox_dimensions'][2]:.2f})")
        print(f"  • Centroid: [{metrics['centroid'][0]:.2f}, {metrics['centroid'][1]:.2f}, {metrics['centroid'][2]:.2f}]")
        print(f"  • Max Distance from Origin: {metrics['max_dist_origin']:.2f}")
        print(f"  • Avg Distance from Centroid: {metrics['avg_dist_centroid']:.2f}")
        print(f"  • Max Distance from Centroid: {metrics['max_dist_centroid']:.2f}")

    # Generate Visualizations
    # 1. 3D Multi-Class Point Cloud Comparison (5 Core Classes)
    five_core = ["airplane", "chair", "bed", "car", "guitar"]
    fig = plt.figure(figsize=(20, 8), dpi=200)
    
    # Row 1: Raw Un-normalized Mesh Vertices
    for i, cls in enumerate(five_core):
        verts, faces, m = loaded_objects[cls]
        ax = fig.add_subplot(2, 5, i + 1, projection='3d')
        # Subsample raw vertices if large for clean visualization
        sub_v = verts[np.random.choice(len(verts), min(len(verts), 4000), replace=False)]
        ax.scatter(sub_v[:, 0], sub_v[:, 1], sub_v[:, 2], c=sub_v[:, 2], cmap='viridis', s=2, alpha=0.7)
        ax.set_title(f"RAW: {cls.capitalize()}\nCentroid: [{m['centroid'][0]:.1f}, {m['centroid'][1]:.1f}, {m['centroid'][2]:.1f}]\nScale: {m['max_dist_origin']:.1f}", fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=6)
        ax.view_init(elev=22, azim=45)
        
    # Row 2: Normalized & Uniformly Sampled Point Clouds (N = 1,024)
    for i, cls in enumerate(five_core):
        verts, faces, m = loaded_objects[cls]
        sampled = sample_point_cloud_from_mesh(verts, faces, n_points=1024)
        # Unit sphere normalization
        c = np.mean(sampled, axis=0)
        norm_sampled = sampled - c
        norm_sampled = norm_sampled / np.max(np.sqrt(np.sum(norm_sampled ** 2, axis=-1)))
        
        ax = fig.add_subplot(2, 5, i + 6, projection='3d')
        ax.scatter(norm_sampled[:, 0], norm_sampled[:, 1], norm_sampled[:, 2], c=norm_sampled[:, 2], cmap='plasma', s=6, alpha=0.9)
        ax.set_title(f"NORMALIZED: {cls.capitalize()}\n(N = 1,024, Unit Sphere)", fontsize=9, fontweight='bold')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.tick_params(labelsize=6)
        ax.view_init(elev=22, azim=45)
        
    plt.suptitle("Geometric Comparison: Raw Un-aligned Offsets vs. Normalized Point Clouds", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    vis_geom_path = os.path.join(vis_dir, "geometry_raw_vs_normalized_comparison.png")
    plt.savefig(vis_geom_path)
    plt.close()
    print(f"\nSaved geometry comparison plot: {vis_geom_path}")
    
    # 2. Multi-view 3D projection for a single object (Chair & Airplane)
    for target_cls in ["chair", "airplane"]:
        verts, faces, m = loaded_objects[target_cls]
        sampled = sample_point_cloud_from_mesh(verts, faces, n_points=1024)
        c = np.mean(sampled, axis=0)
        norm_pts = (sampled - c) / np.max(np.sqrt(np.sum((sampled - c) ** 2, axis=-1)))
        
        fig = plt.figure(figsize=(16, 4), dpi=200)
        views = [(20, 45, "Perspective (Elev: 20°, Azim: 45°)"),
                 (90, 0, "Top View (XY Plane, Elev: 90°)"),
                 (0, 0, "Front View (XZ Plane, Elev: 0°)"),
                 (0, 90, "Side View (YZ Plane, Azim: 90°)")]
        for idx, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
            ax.scatter(norm_pts[:, 0], norm_pts[:, 1], norm_pts[:, 2], c=norm_pts[:, 2], cmap='coolwarm', s=5, alpha=0.85)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.view_init(elev=elev, azim=azim)
        plt.suptitle(f"Multi-View 3D Point-Cloud Representations ({target_cls.capitalize()})", fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout()
        mv_path = os.path.join(vis_dir, f"{target_cls}_multiview_projections.png")
        plt.savefig(mv_path)
        plt.close()
        print(f"Saved multi-view plot: {mv_path}")

    with open(r"c:\DLproj\geometry_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Geometry analysis complete!")

if __name__ == "__main__":
    main()
