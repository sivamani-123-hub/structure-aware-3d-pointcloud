"""
Robust 3D Point Cloud and Mesh Parsers for Streamlit UI.

Supports:
- .xyz / .txt: Raw ASCII 3D coordinates.
- .obj: Wavefront OBJ vertex elements ('v x y z').
- .off: Object File Format headers and vertices.
- .ply: Stanford polygon file (ASCII header and vertex coordinates).
- .npy: Serialized numpy arrays [N, 3].
"""

import io
import os
from typing import Tuple, Optional
import numpy as np


def parse_xyz(file_bytes_or_str) -> np.ndarray:
    """Parses raw ASCII XYZ lines."""
    if isinstance(file_bytes_or_str, bytes):
        lines = file_bytes_or_str.decode('utf-8', errors='ignore').splitlines()
    else:
        lines = file_bytes_or_str.splitlines()

    points = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if len(points) == 0:
        raise ValueError("No valid 3D coordinate lines found in XYZ file.")
    return np.array(points, dtype=np.float32)


def parse_obj(file_bytes_or_str) -> np.ndarray:
    """Parses Wavefront OBJ vertex declarations ('v x y z')."""
    if isinstance(file_bytes_or_str, bytes):
        lines = file_bytes_or_str.decode('utf-8', errors='ignore').splitlines()
    else:
        lines = file_bytes_or_str.splitlines()

    vertices = []
    for line in lines:
        line = line.strip()
        if line.startswith('v '):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    continue

    if len(vertices) == 0:
        raise ValueError("No vertex declarations ('v x y z') found in OBJ file.")
    return np.array(vertices, dtype=np.float32)


def parse_off(file_bytes_or_str) -> np.ndarray:
    """Parses Object File Format (.off) vertices."""
    if isinstance(file_bytes_or_str, bytes):
        lines = file_bytes_or_str.decode('utf-8', errors='ignore').splitlines()
    else:
        lines = file_bytes_or_str.splitlines()

    # Filter blank lines and comments
    filtered = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
    if not filtered:
        raise ValueError("Empty OFF file.")

    first_line = filtered[0]
    if first_line.startswith('OFF'):
        rem = first_line[3:].strip()
        if rem:
            parts = rem.split()
            curr_idx = 1
        else:
            parts = filtered[1].split()
            curr_idx = 2
    else:
        parts = first_line.split()
        curr_idx = 1

    n_verts = int(parts[0])
    vertices = []
    for i in range(curr_idx, min(curr_idx + n_verts, len(filtered))):
        p = filtered[i].split()
        if len(p) >= 3:
            try:
                vertices.append([float(p[0]), float(p[1]), float(p[2])])
            except ValueError:
                continue

    if len(vertices) == 0:
        raise ValueError("No valid vertices found in OFF file.")
    return np.array(vertices, dtype=np.float32)


def parse_ply(file_bytes_or_str) -> np.ndarray:
    """Parses Stanford PLY files (ASCII vertex section)."""
    if isinstance(file_bytes_or_str, bytes):
        content = file_bytes_or_str.decode('utf-8', errors='ignore')
    else:
        content = file_bytes_or_str

    lines = content.splitlines()
    header_end = -1
    num_verts = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('element vertex'):
            parts = line.split()
            num_verts = int(parts[2])
        elif line.startswith('end_header'):
            header_end = i
            break

    if header_end == -1:
        raise ValueError("Invalid PLY file: 'end_header' marker not found.")

    vertices = []
    for i in range(header_end + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(vertices) == num_verts and num_verts > 0:
                    break
            except ValueError:
                continue

    if len(vertices) == 0:
        raise ValueError("No vertex coordinates found after PLY header.")
    return np.array(vertices, dtype=np.float32)


def load_point_cloud_from_upload(uploaded_file) -> Tuple[np.ndarray, str]:
    """
    Dispatcher taking a Streamlit UploadedFile object and returning [N, 3] points.
    """
    filename = uploaded_file.name.lower()
    bytes_data = uploaded_file.getvalue()

    if filename.endswith(('.xyz', '.txt')):
        pts = parse_xyz(bytes_data)
        fmt = "XYZ/ASCII"
    elif filename.endswith('.obj'):
        pts = parse_obj(bytes_data)
        fmt = "Wavefront OBJ"
    elif filename.endswith('.off'):
        pts = parse_off(bytes_data)
        fmt = "Object File Format (OFF)"
    elif filename.endswith('.ply'):
        pts = parse_ply(bytes_data)
        fmt = "Stanford PLY"
    elif filename.endswith('.npy'):
        pts = np.load(io.BytesIO(bytes_data))
        if pts.ndim == 3:
            pts = pts[0]
        fmt = "NumPy Array (.npy)"
    else:
        # Fallback to XYZ attempt
        try:
            pts = parse_xyz(bytes_data)
            fmt = "Generic Delimited Text"
        except Exception:
            raise ValueError(f"Unsupported file format '{filename}'. Supported: .ply, .obj, .off, .xyz, .txt, .npy")

    # Health Checks
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Invalid coordinate dimensions: expected [N, 3], got {pts.shape}")

    if np.isnan(pts).any() or np.isinf(pts).any():
        raise ValueError("Point cloud contains invalid NaN or Infinite values.")

    if len(pts) < 16:
        raise ValueError(f"Point cloud has only {len(pts)} points; minimum 16 required.")

    return pts.astype(np.float32), fmt

