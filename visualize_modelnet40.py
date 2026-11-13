import os
import glob
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

def load_off_mesh(file_path):
    """
    Parses OFF mesh vertices and faces.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
        
    l0 = lines[0]
    if l0.startswith('OFF'):
        rem = l0[3:].strip()
        if rem:
            parts = rem.split()
            start_v = 1
        else:
            parts = lines[1].split()
            start_v = 2
    else:
        parts = l0.split()
        start_v = 1
        
    n_verts = int(parts[0])
    n_faces = int(parts[1])
    
    vertices = []
    for i in range(start_v, start_v + n_verts):
        p = lines[i].split()
        vertices.append([float(p[0]), float(p[1]), float(p[2])])
    vertices = np.array(vertices, dtype=np.float32)
    
    faces = []
    for i in range(start_v + n_verts, start_v + n_verts + n_faces):
        p = lines[i].split()
        nv = int(p[0])
        faces.append([int(x) for x in p[1:1+nv]])
        
    return vertices, faces

def sample_point_cloud(vertices, faces, n_points=1024):
    """
    Area-weighted triangle surface sampling + barycentric coordinate interpolation.
    """
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

def normalize_point_cloud(pts):
    """
    Centers at origin and scales to unit sphere (preserving true geometric aspect ratio).
    """
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    max_radius = np.max(np.sqrt(np.sum(pts ** 2, axis=-1))) + 1e-8
    pts = pts / max_radius
    return pts, centroid, max_radius

def set_equal_3d_axes(ax, pts):
    """
    Sets equal aspect ratio for 3D axis so geometric proportions are not distorted.
    """
    max_range = np.array([pts[:, 0].max()-pts[:, 0].min(), 
                          pts[:, 1].max()-pts[:, 1].min(), 
                          pts[:, 2].max()-pts[:, 2].min()]).max() / 2.0
    mid_x = (pts[:, 0].max()+pts[:, 0].min()) * 0.5
    mid_y = (pts[:, 1].max()+pts[:, 1].min()) * 0.5
    mid_z = (pts[:, 2].max()+pts[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def visualize_object(file_path, num_points=1024, save_path=None, title_extra="", elev=25, azim=45):
    """
    Loads one 3D object and visualizes its raw mesh vertices and sampled point cloud.
    """
    class_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    filename = os.path.basename(file_path)
    
    raw_verts, raw_faces = load_off_mesh(file_path)
    sampled_pts = sample_point_cloud(raw_verts, raw_faces, n_points=num_points)
    norm_pts, centroid, max_radius = normalize_point_cloud(sampled_pts)
    
    x_span = raw_verts[:, 0].max() - raw_verts[:, 0].min()
    y_span = raw_verts[:, 1].max() - raw_verts[:, 1].min()
    z_span = raw_verts[:, 2].max() - raw_verts[:, 2].min()
    
    fig = plt.figure(figsize=(14, 6), dpi=200)
    
    # 1. Perspective View
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    p1 = ax1.scatter(norm_pts[:, 0], norm_pts[:, 1], norm_pts[:, 2], 
                     c=norm_pts[:, 2], cmap='plasma', s=8, alpha=0.9, edgecolors='none')
    set_equal_3d_axes(ax1, norm_pts)
    ax1.set_title(f"Perspective 3D View (Elev: {elev}°, Azim: {azim}°)", fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel("X (norm)")
    ax1.set_ylabel("Y (norm)")
    ax1.set_zlabel("Z (norm)")
    ax1.view_init(elev=elev, azim=azim)
    
    # 2. Side / Alternate View
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    p2 = ax2.scatter(norm_pts[:, 0], norm_pts[:, 1], norm_pts[:, 2], 
                     c=norm_pts[:, 0], cmap='viridis', s=8, alpha=0.9, edgecolors='none')
    set_equal_3d_axes(ax2, norm_pts)
    ax2.set_title("Alternate Angle View (Elev: 10°, Azim: 120°)", fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel("X (norm)")
    ax2.set_ylabel("Y (norm)")
    ax2.set_zlabel("Z (norm)")
    ax2.view_init(elev=10, azim=120)
    
    stats_text = (
        f"Object: {filename}  |  Class: {class_name.upper()} {title_extra}\n"
        f"Raw Vertices: {len(raw_verts):,}  |  Faces: {len(raw_faces):,}  |  Sampled Points: {num_points:,}\n"
        f"Raw Range X: [{raw_verts[:,0].min():.2f}, {raw_verts[:,0].max():.2f}] (Δ={x_span:.2f})  |  "
        f"Y: [{raw_verts[:,1].min():.2f}, {raw_verts[:,1].max():.2f}] (Δ={y_span:.2f})  |  "
        f"Z: [{raw_verts[:,2].min():.2f}, {raw_verts[:,2].max():.2f}] (Δ={z_span:.2f})"
    )
    plt.suptitle(stats_text, fontsize=10, fontweight='semibold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved visualization to: {save_path}")
    plt.close()
    return fig

def visualize_random_samples(num_samples=10, dataset_root=r"c:\DLproj\dataset\ModelNet40", save_path=None):
    """
    Randomly selects objects across distinct classes and displays them in a gallery grid.
    """
    class_dirs = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')])
    chosen_classes = np.random.choice(class_dirs, size=min(num_samples, len(class_dirs)), replace=False)
    
    cols = 5
    rows = int(np.ceil(len(chosen_classes) / cols))
    
    fig = plt.figure(figsize=(20, 4.2 * rows), dpi=200)
    
    for idx, cls in enumerate(chosen_classes):
        sdir = os.path.join(dataset_root, cls, "train")
        files = [f for f in os.listdir(sdir) if f.endswith('.off') and not f.startswith('.')]
        chosen_f = np.random.choice(files)
        fpath = os.path.join(sdir, chosen_f)
        
        raw_v, raw_f = load_off_mesh(fpath)
        sampled = sample_point_cloud(raw_v, raw_f, n_points=1024)
        norm_pts, _, _ = normalize_point_cloud(sampled)
        
        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
        ax.scatter(norm_pts[:, 0], norm_pts[:, 1], norm_pts[:, 2], 
                   c=norm_pts[:, 2], cmap='coolwarm', s=4, alpha=0.85, edgecolors='none')
        set_equal_3d_axes(ax, norm_pts)
        
        x_span = raw_v[:, 0].max() - raw_v[:, 0].min()
        y_span = raw_v[:, 1].max() - raw_v[:, 1].min()
        z_span = raw_v[:, 2].max() - raw_v[:, 2].min()
        
        title = (f"{cls.upper()}\n"
                 f"V: {len(raw_v):,} | F: {len(raw_f):,}\n"
                 f"Δ: {x_span:.1f}×{y_span:.1f}×{z_span:.1f}")
        ax.set_title(title, fontsize=8, fontweight='bold', pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=22, azim=45)
        
    plt.suptitle(f"ModelNet40 Random Multi-Class Sample Gallery ({len(chosen_classes)} Distinct Classes)", fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved random samples gallery to: {save_path}")
    plt.close()

def visualize_comparison_grid(group_dict, group_title, save_path):
    """
    Visualizes comparison groups (e.g. Visually Similar vs Visually Distinct classes).
    """
    n_items = len(group_dict)
    fig = plt.figure(figsize=(4 * n_items, 4.5), dpi=200)
    
    for idx, (label, fpath) in enumerate(group_dict.items()):
        raw_v, raw_f = load_off_mesh(fpath)
        sampled = sample_point_cloud(raw_v, raw_f, n_points=1024)
        norm_pts, _, _ = normalize_point_cloud(sampled)
        
        ax = fig.add_subplot(1, n_items, idx + 1, projection='3d')
        ax.scatter(norm_pts[:, 0], norm_pts[:, 1], norm_pts[:, 2], 
                   c=norm_pts[:, 2], cmap='viridis', s=5, alpha=0.9, edgecolors='none')
        set_equal_3d_axes(ax, norm_pts)
        
        x_span = raw_v[:, 0].max() - raw_v[:, 0].min()
        y_span = raw_v[:, 1].max() - raw_v[:, 1].min()
        z_span = raw_v[:, 2].max() - raw_v[:, 2].min()
        
        ax.set_title(f"{label}\n(V: {len(raw_v):,} | F: {len(raw_f):,})\nΔ: {x_span:.1f}×{y_span:.1f}×{z_span:.1f}", 
                     fontsize=9, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=20, azim=45)
        
    plt.suptitle(group_title, fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved comparison gallery to: {save_path}")
    plt.close()

def main():
    np.random.seed(42)
    dataset_root = r"c:\DLproj\dataset\ModelNet40"
    vis_dir = r"c:\DLproj\visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 1: 3D POINT CLOUD VISUALIZATION PIPELINE")
    print("=" * 80)
    
    # 1. Single Object Visualization Demonstration
    sample_ap = os.path.join(dataset_root, "airplane", "train", "airplane_0001.off")
    visualize_object(sample_ap, num_points=1024, 
                     save_path=os.path.join(vis_dir, "single_object_airplane_demo.png"))
    
    sample_chair = os.path.join(dataset_root, "chair", "train", "chair_0001.off")
    visualize_object(sample_chair, num_points=1024, 
                     save_path=os.path.join(vis_dir, "single_object_chair_demo.png"))
    
    # 2. 10 Random Samples Across 10 Different Classes
    visualize_random_samples(num_samples=10, dataset_root=dataset_root, 
                             save_path=os.path.join(vis_dir, "10_random_classes_gallery.png"))
    
    # 3. Visually Similar Classes Comparison Group (Fine-grained geometric ambiguity)
    # E.g. Dresser vs Night Stand vs Desk vs Bookshelf; Chair vs Stool; Bed vs Sofa
    similar_group_1 = {
        "Dresser": os.path.join(dataset_root, "dresser", "train", "dresser_0001.off"),
        "Night Stand": os.path.join(dataset_root, "night_stand", "train", "night_stand_0001.off"),
        "Desk": os.path.join(dataset_root, "desk", "train", "desk_0001.off"),
        "Chair": os.path.join(dataset_root, "chair", "train", "chair_0001.off"),
        "Stool": os.path.join(dataset_root, "stool", "train", "stool_0001.off")
    }
    visualize_comparison_grid(similar_group_1, 
                              "Visually Similar / Fine-Grained Ambiguous Classes (Dresser vs Night Stand, Chair vs Stool)", 
                              os.path.join(vis_dir, "visually_similar_classes_comparison.png"))
    
    # 4. Visually Distinct Classes Comparison Group (Broad topological contrast)
    distinct_group = {
        "Airplane": os.path.join(dataset_root, "airplane", "train", "airplane_0001.off"),
        "Guitar": os.path.join(dataset_root, "guitar", "train", "guitar_0001.off"),
        "Person": os.path.join(dataset_root, "person", "train", "person_0001.off"),
        "Toilet": os.path.join(dataset_root, "toilet", "train", "toilet_0001.off"),
        "Stairs": os.path.join(dataset_root, "stairs", "train", "stairs_0001.off")
    }
    visualize_comparison_grid(distinct_group, 
                              "Visually Distinct / High Geometric Variance Classes", 
                              os.path.join(vis_dir, "visually_distinct_classes_comparison.png"))
    
    print("\nAll visualizations generated successfully!")

if __name__ == "__main__":
    main()
