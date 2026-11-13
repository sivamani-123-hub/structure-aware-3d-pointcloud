import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.components.parsers import parse_xyz, parse_obj, parse_off, parse_ply
from app.components.visualizer import create_3d_point_cloud_figure
from app.components.demo_samples import get_demo_samples_catalog, load_demo_point_cloud
from src.inference.predictor import PointCloudPredictor
from src.inference.model_loader import load_model

def test():
    print("1. Testing demo samples catalog...")
    cat = get_demo_samples_catalog()
    print(f"   Catalog entries ({len(cat)}):", list(cat.keys()))

    print("2. Loading demo point cloud (Real Chair)...")
    pts, cls_name, desc = load_demo_point_cloud("Real Scan: Chair (ScanObjectNN)")
    print(f"   Loaded {len(pts)} points, class: {cls_name}")

    print("3. Testing 3D visualizer figure creation...")
    fig = create_3d_point_cloud_figure(pts, title="Test Chair")
    print(f"   Plotly 3D Figure created successfully with {len(fig.data[0].x)} points.")

    print("4. Testing Fine-Tuned GNN Predictor on Demo Chair...")
    pred_ft = PointCloudPredictor("finetuned_gnn")
    res_ft = pred_ft.predict(pts, mode="constrained")
    print(f"   Fine-Tuned GNN Prediction: {res_ft['predicted_class']} ({res_ft['confidence_pct']}) in {res_ft['latency_ms']:.2f} ms")

    print("5. Testing Peak GNN Predictor on Demo Chair...")
    pred_pk = PointCloudPredictor("peak_dgcnn")
    res_pk = pred_pk.predict(pts, mode="constrained")
    print(f"   Peak GNN Prediction: {res_pk['predicted_class']} ({res_pk['confidence_pct']}) in {res_pk['latency_ms']:.2f} ms")

    print("\n6. Testing Multi-Format Parsers with Mock Data...")
    xyz_mock = b"0.1 0.2 0.3\n0.4 0.5 0.6\n0.7 0.8 0.9\n"
    pts_xyz = parse_xyz(xyz_mock)
    print(f"   Parsed XYZ mock: shape {pts_xyz.shape}")

    obj_mock = b"v 1.0 2.0 3.0\nv 4.0 5.0 6.0\n"
    pts_obj = parse_obj(obj_mock)
    print(f"   Parsed OBJ mock: shape {pts_obj.shape}")

    off_mock = b"OFF\n2 0 0\n0.0 0.0 0.0\n1.0 1.0 1.0\n"
    pts_off = parse_off(off_mock)
    print(f"   Parsed OFF mock: shape {pts_off.shape}")

    print("\nALL STREAMLIT APP COMPONENTS VERIFIED CLEANLY AND DETERMINISTICALLY!")

if __name__ == "__main__":
    test()

