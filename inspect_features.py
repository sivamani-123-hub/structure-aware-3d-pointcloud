import os
import json
import numpy as np

def inspect_raw_off_structure(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]
        
    l0 = raw_lines[0]
    if l0.startswith('OFF'):
        rem = l0[3:].strip()
        if rem:
            parts = rem.split()
            n_verts = int(parts[0])
            n_faces = int(parts[1])
            start_v = 1
        else:
            parts = raw_lines[1].split()
            n_verts = int(parts[0])
            n_faces = int(parts[1])
            start_v = 2
    else:
        parts = l0.split()
        n_verts = int(parts[0])
        n_faces = int(parts[1])
        start_v = 1
        
    sample_vertex_lines = raw_lines[start_v : start_v + min(10, n_verts)]
    vertex_token_counts = [len(l.split()) for l in sample_vertex_lines]
    
    sample_face_lines = raw_lines[start_v + n_verts : start_v + n_verts + min(10, n_faces)]
    face_token_counts = [len(l.split()) for l in sample_face_lines]
    
    return {
        "file": file_path,
        "n_verts": n_verts,
        "n_faces": n_faces,
        "sample_vertex_lines": sample_vertex_lines,
        "vertex_token_counts": list(set(vertex_token_counts)),
        "sample_face_lines": sample_face_lines,
        "face_token_counts": list(set(face_token_counts))
    }

def main():
    dataset_root = r"c:\DLproj\dataset\ModelNet40"
    
    print("=" * 80)
    print("FEATURE ATTRIBUTE AUDIT ACROSS MODELNET40 RAW FILES")
    print("=" * 80)
    
    class_dirs = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith('.')])
    
    checked_files = 0
    header_magics = set()
    has_noff = 0
    has_coff = 0
    
    sample_inspections = []
    
    for cls in class_dirs:
        for split in ['train', 'test']:
            sdir = os.path.join(dataset_root, cls, split)
            if os.path.exists(sdir):
                files = [f for f in os.listdir(sdir) if f.endswith('.off') and not f.startswith('.')]
                for idx, fname in enumerate(files):
                    fpath = os.path.join(sdir, fname)
                    checked_files += 1
                    
                    if idx == 0 and split == 'train':
                        info = inspect_raw_off_structure(fpath)
                        sample_inspections.append(info)
                    
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        l1 = f.readline().strip()
                        header_magics.add(l1.split()[0] if l1 else "")
                        if "NOFF" in l1: has_noff += 1
                        if "COFF" in l1: has_coff += 1
                        
    print(f"Total files audited: {checked_files}")
    print(f"Header Magics Encountered: {header_magics}")
    print(f"Files with Normal headers (NOFF): {has_noff}")
    print(f"Files with Color headers (COFF): {has_coff}")
    
    print("\nDetailed breakdown of 5 representative sample files:")
    for s in sample_inspections[:5]:
        cls_name = os.path.basename(os.path.dirname(os.path.dirname(s['file'])))
        print(f"\nClass: [{cls_name.upper()}] - {os.path.basename(s['file'])}")
        print(f"  Vertex Count: {s['n_verts']:,}, Face Count: {s['n_faces']:,}")
        print(f"  Tokens per Vertex line: {s['vertex_token_counts']} (strictly 3 floats: X, Y, Z)")
        print(f"  Sample Vertex Lines:")
        for vl in s['sample_vertex_lines'][:3]:
            print(f"    {vl}")
        print(f"  Sample Face Lines:")
        for fl in s['sample_face_lines'][:3]:
            print(f"    {fl}")
            
    with open(r"c:\DLproj\feature_inspection_results.json", "w") as f:
        json.dump(sample_inspections, f, indent=2)
    print("\nFeature audit complete! Results saved to feature_inspection_results.json")

if __name__ == "__main__":
    main()
