import os
import glob
import json
import hashlib
import numpy as np

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

def load_off_vertices_fast(file_path, max_v=5000):
    """Loads vertices for geometric hashing."""
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
    n_read = min(n_verts, max_v)
    verts = []
    for i in range(start_v, start_v + n_read):
        p = lines[i].split()
        verts.append([float(p[0]), float(p[1]), float(p[2])])
    return np.array(verts, dtype=np.float32), n_verts

def compute_geometric_fingerprint(verts):
    """
    Computes a rotation- and translation-invariant geometric fingerprint.
    1. Centering at mean
    2. Covariance matrix eigenvalues (sorted)
    3. Radius moments (mean, std, max)
    4. Rounded sorted eigenvalues + moments form hash string
    """
    if len(verts) < 4:
        return "too_small"
    c = np.mean(verts, axis=0)
    v_centered = verts - c
    scale = np.max(np.sqrt(np.sum(v_centered ** 2, axis=-1))) + 1e-8
    v_norm = v_centered / scale
    
    # Covariance
    cov = np.cov(v_norm, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals) # 3 sorted eigenvalues
    
    # Distance distribution moments
    dists = np.linalg.norm(v_norm, axis=1)
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    
    # Quantize to 3 decimals to catch near-identical geometries
    fp = f"{evals[0]:.3f}_{evals[1]:.3f}_{evals[2]:.3f}_{mean_d:.3f}_{std_d:.3f}_{len(verts)}"
    return fp

def main():
    dataset_root = r"c:\DLproj\dataset\ModelNet40"
    
    print("=" * 80)
    print("PHASE 1: COMPREHENSIVE DATA QUALITY & DATA LEAKAGE AUDIT")
    print("=" * 80)
    
    class_dirs = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')])
    
    all_files = []
    for cls in class_dirs:
        for split in ['train', 'test']:
            sdir = os.path.join(dataset_root, cls, split)
            if os.path.exists(sdir):
                for fname in os.listdir(sdir):
                    if fname.endswith('.off') and not fname.startswith('.'):
                        all_files.append((cls, split, fname, os.path.join(sdir, fname)))
                        
    print(f"Auditing {len(all_files):,} files across {len(class_dirs)} classes...")
    
    # 1. Exact File Hash (MD5 of exact byte stream)
    md5_dict = {}
    exact_train_dups = []
    exact_test_dups = []
    exact_cross_split_leakage = []
    
    # 2. Filename check
    filename_dict = {}
    filename_cross_cls = []
    
    # 3. Geometric Fingerprinting for Near-Identical Geometries
    geo_dict = {}
    geo_cross_split_dups = []
    geo_internal_dups = []
    
    print("\n[Step 1/3] Computing Exact MD5 Hashes and Filename Collisions...")
    for cls, split, fname, fpath in all_files:
        # Check filename
        if fname in filename_dict:
            filename_dict[fname].append((cls, split, fpath))
        else:
            filename_dict[fname] = [(cls, split, fpath)]
            
        # Check MD5
        with open(fpath, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
            
        if h in md5_dict:
            p_cls, p_split, p_fname, p_fpath = md5_dict[h]
            if p_split == split:
                if split == 'train':
                    exact_train_dups.append((p_fpath, fpath, p_cls, cls))
                else:
                    exact_test_dups.append((p_fpath, fpath, p_cls, cls))
            else:
                exact_cross_split_leakage.append((p_fpath, fpath, p_cls, cls, p_split, split))
        else:
            md5_dict[h] = (cls, split, fname, fpath)
            
    print(f" - Exact Duplicate Files in Training Set: {len(exact_train_dups)}")
    print(f" - Exact Duplicate Files in Test Set: {len(exact_test_dups)}")
    print(f" - Exact Cross-Split Leakage (Identical MD5 across Train & Test): {len(exact_cross_split_leakage)}")
    
    print("\n[Step 2/3] Performing Geometric Fingerprint & Invariant Shape Audit...")
    # Sample/audit all files for geometric fingerprints
    for idx, (cls, split, fname, fpath) in enumerate(all_files):
        try:
            verts, nv = load_off_vertices_fast(fpath, max_v=3000)
            fp = compute_geometric_fingerprint(verts)
            if fp != "too_small":
                if fp in geo_dict:
                    p_cls, p_split, p_fname, p_fpath = geo_dict[fp]
                    # Check if not already captured as exact MD5 duplicate
                    if p_fpath != fpath:
                        if p_split != split:
                            geo_cross_split_dups.append((p_fpath, fpath, p_cls, cls, p_split, split, fp))
                        else:
                            geo_internal_dups.append((p_fpath, fpath, p_cls, cls, p_split, fp))
                else:
                    geo_dict[fp] = (cls, split, fname, fpath)
        except Exception as e:
            pass
            
    print(f" - Total Geometric Fingerprint Clusters: {len(geo_dict)}")
    print(f" - Suspicious / Near-Identical Geometric Pairs across Train & Test: {len(geo_cross_split_dups)}")
    print(f" - Suspicious / Near-Identical Geometric Pairs within same split: {len(geo_internal_dups)}")
    
    # Detailed examples of cross-split leakage
    print("\n--- Detailed Cross-Split Leakage Breakdown ---")
    cross_same_class = [x for x in exact_cross_split_leakage if x[2] == x[3]]
    cross_diff_class = [x for x in exact_cross_split_leakage if x[2] != x[3]]
    
    print(f" 1. Exact Same-Class Leakage (Identical CAD model in both train & test of same category): {len(cross_same_class)}")
    for p1, p2, c1, c2, s1, s2 in cross_same_class[:5]:
        print(f"    • [{c1}] {os.path.basename(p1)} ({s1}) <===> {os.path.basename(p2)} ({s2})")
        
    print(f" 2. Cross-Class Ambiguity / Leakage (Identical CAD model labeled under two different categories across train/test): {len(cross_diff_class)}")
    for p1, p2, c1, c2, s1, s2 in cross_diff_class[:10]:
        print(f"    • [{c1}] {os.path.basename(p1)} ({s1}) <===> [{c2}] {os.path.basename(p2)} ({s2})")
        
    report = {
        "total_files_audited": len(all_files),
        "exact_duplicates": {
            "train_internal": len(exact_train_dups),
            "test_internal": len(exact_test_dups),
            "cross_split_leakage_total": len(exact_cross_split_leakage),
            "same_class_leakage": len(cross_same_class),
            "cross_class_leakage": len(cross_diff_class),
            "cross_split_pairs": [
                {"train_file": x[0] if x[4]=='train' else x[1], 
                 "test_file": x[1] if x[4]=='train' else x[0], 
                 "train_class": x[2] if x[4]=='train' else x[3], 
                 "test_class": x[3] if x[4]=='train' else x[2]}
                for x in exact_cross_split_leakage
            ]
        },
        "geometric_fingerprint_duplicates": {
            "cross_split_suspicious_count": len(geo_cross_split_dups),
            "internal_suspicious_count": len(geo_internal_dups),
            "samples": [
                {"file1": x[0], "file2": x[1], "cls1": x[2], "cls2": x[3], "split1": x[4], "split2": x[5], "fp": x[6]}
                for x in geo_cross_split_dups[:10]
            ]
        }
    }
    
    with open(r"c:\DLproj\leakage_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nLeakage audit successfully saved to leakage_audit_report.json")

if __name__ == "__main__":
    main()
