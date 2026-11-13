import numpy as np
import torch

class PointCloudTransforms:
    """
    Modular transformation and augmentation pipeline for 3D point clouds.
    """
    @staticmethod
    def sample_surface_mesh(vertices, faces, num_points=1024, seed=None):
        """
        Area-weighted triangle mesh surface sampling via barycentric coordinates.
        """
        if seed is not None:
            np.random.seed(seed)
            
        if len(faces) == 0:
            idx = np.random.choice(len(vertices), num_points, replace=(len(vertices) < num_points))
            return vertices[idx]

        tri_verts = []
        for f in faces:
            if len(f) >= 3:
                for j in range(1, len(f) - 1):
                    tri_verts.append([f[0], f[j], f[j + 1]])
        tri_verts = np.array(tri_verts, dtype=np.int32)

        v0, v1, v2 = vertices[tri_verts[:, 0]], vertices[tri_verts[:, 1]], vertices[tri_verts[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        total_area = np.sum(areas)

        if total_area == 0 or np.isnan(total_area):
            idx = np.random.choice(len(vertices), num_points, replace=(len(vertices) < num_points))
            return vertices[idx]

        probs = areas / total_area
        chosen = np.random.choice(len(tri_verts), size=num_points, p=probs)

        # Barycentric coordinates
        u = np.random.uniform(0, 1, (num_points, 1))
        v = np.random.uniform(0, 1, (num_points, 1))
        mask = (u + v) > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - u - v

        sampled = (u * vertices[tri_verts[chosen, 0]] + 
                   v * vertices[tri_verts[chosen, 1]] + 
                   w * vertices[tri_verts[chosen, 2]])
        return sampled.astype(np.float32)

    @staticmethod
    def normalize_instance(points):
        """
        Per-instance Zero-Centering and Unit-Sphere Normalization.
        Guarantees zero data leakage across splits.
        """
        centroid = np.mean(points, axis=0)
        points = points - centroid
        furthest_distance = np.max(np.sqrt(np.sum(points ** 2, axis=-1))) + 1e-8
        points = points / furthest_distance
        return points.astype(np.float32)

    @staticmethod
    def random_rotation_z(points):
        """Random rotation around gravity (Z) axis in [0, 2pi]."""
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ], dtype=np.float32)
        return np.dot(points, rot_matrix)

    @staticmethod
    def random_anisotropic_scale(points, low=0.8, high=1.25):
        """Random anisotropic scaling along independent axes."""
        scale = np.random.uniform(low, high, size=(1, 3)).astype(np.float32)
        return points * scale

    @staticmethod
    def random_translation(points, shift_range=0.02):
        """Small translation perturbation."""
        shift = np.random.uniform(-shift_range, shift_range, size=(1, 3)).astype(np.float32)
        return points + shift

    @staticmethod
    def random_jitter(points, sigma=0.01, clip=0.05):
        """Coordinate Gaussian noise."""
        noise = np.clip(np.random.normal(0, sigma, points.shape), -clip, clip).astype(np.float32)
        return points + noise

    @staticmethod
    def random_point_dropout(points, max_dropout_ratio=0.15):
        """Simulates sensor occlusion by dropping random points and duplicating."""
        dropout_ratio = np.random.uniform(0.0, max_dropout_ratio)
        keep_indices = np.where(np.random.rand(len(points)) >= dropout_ratio)[0]
        if len(keep_indices) > 0 and len(keep_indices) < len(points):
            fill_indices = np.random.choice(keep_indices, len(points) - len(keep_indices), replace=True)
            points = np.concatenate([points[keep_indices], points[fill_indices]], axis=0)
        return points
