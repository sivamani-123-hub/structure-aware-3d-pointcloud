"""
Structure-Aware Cross-Domain 3D Point Cloud Classification MVP.

Interactive Streamlit Application showcasing:
1. Interactive 3D point cloud visualization (Plotly WebGL).
2. Multi-format point cloud upload (.ply, .obj, .off, .xyz, .npy).
3. Real-time inference across PointNet, DGCNN, Peak GNN, and Fine-Tuned GNN.
4. Comprehensive synthetic-to-real evaluation dashboard and confusion matrices.
5. Domain-shift geometry analysis.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# DLproj paths
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from app.components.parsers import load_point_cloud_from_upload
from app.components.visualizer import create_3d_point_cloud_figure
from app.components.demo_samples import get_demo_samples_catalog, load_demo_point_cloud
from src.inference.predictor import PointCloudPredictor
from src.inference.model_loader import get_device, MODEL_REGISTRY
from src.datasets.class_mapping import COMMON_CLASS_NAMES, MODELNET40_CLASSES

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Structure-Aware 3D Point Cloud AI",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .pred-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- CACHED MODEL LOADERS ---
@st.cache_resource(show_spinner="Loading 3D Deep Learning Model...")
def get_cached_predictor(model_key: str):
    dev = get_device("auto")
    return PointCloudPredictor(model_name=model_key, device=dev)


# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/isometric/100/cube.png", width=60)
st.sidebar.title("⚙️ Model Controls")

model_choice = st.sidebar.selectbox(
    "Active Architecture:",
    options=[
        "Peak Regularized GNN (k=40, ModelNet40)",
        "Fine-Tuned Domain-Adapted GNN (ScanObjectNN)",
        "Structure-Aware GNN (k=20, ModelNet40)",
        "DGCNN Baseline (k=20, ModelNet40)",
        "PointNet Baseline (ModelNet40)"
    ],
    index=0
)

# Map UI name to registry key
MODEL_MAP = {
    "Peak Regularized GNN (k=40, ModelNet40)": "peak_dgcnn",
    "Fine-Tuned Domain-Adapted GNN (ScanObjectNN)": "finetuned_gnn",
    "Structure-Aware GNN (k=20, ModelNet40)": "gnn",
    "DGCNN Baseline (k=20, ModelNet40)": "dgcnn",
    "PointNet Baseline (ModelNet40)": "pointnet"
}
selected_model_key = MODEL_MAP[model_choice]

eval_mode = st.sidebar.radio(
    "Evaluation Domain Space:",
    options=[
        "Auto-Adaptive (Smart Domain Detection)",
        "Open-World (40 ModelNet40 Classes)",
        "Constrained (10 Common Real-World Classes)"
    ],
    index=0,
    help="Auto-Adaptive automatically evaluates ScanObjectNN scans in the 10 common classes and ModelNet40 CAD models (e.g. Guitar, Airplane) across all 40 classes."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🌌 3D Simulation Settings")
point_size = st.sidebar.slider("Particle Size", min_value=1, max_value=6, value=2)
colorscale_choice = st.sidebar.selectbox("Color Palette", ["Viridis", "Plasma", "Turbo", "RdBu", "Cividis"], index=0)
particle_options = [1024, 2048, 4096, 8192, 12000]
display_points_limit = st.sidebar.select_slider(
    "Simulation Particle Count",
    options=particle_options,
    value=12000,
    format_func=lambda x: f"{x:,} particles" if x < 12000 else "12k particles (High-Density Simulation)"
)
enable_particle_sim = st.sidebar.toggle(
    "✨ 12k Particle Simulation View",
    value=True,
    help="Synthesizes continuous surface manifold particles up to 12,000 points via k-NN surface geodesics."
)

st.sidebar.markdown("---")
dev_type = "CUDA GPU (RTX 4050)" if torch.cuda.is_available() else "CPU Fallback"
st.sidebar.caption(f"**Execution Hardware:** {dev_type}")
st.sidebar.caption("**Project:** VIT-AP University CSE Dept")


# --- MAIN HEADER ---
st.markdown('<div class="main-title">🧊 Structure-Aware Cross-Domain 3D Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Domain Generalization & Real-World Adaptation Benchmarking (ModelNet40 &rarr; ScanObjectNN)</div>', unsafe_allow_html=True)

# Executive Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-val">74.62%</div><div class="metric-lbl">ModelNet40 In-Domain OA</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-val">9.84%</div><div class="metric-lbl">ScanObjectNN Zero-Shot OA</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-val">83.33%</div><div class="metric-lbl">Zero-Shot Chair Accuracy</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-val" style="color:#16A34A;">62.18%</div><div class="metric-lbl">Fine-Tuned Target OA (+52.3%)</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- NAVIGATION TABS ---
tab_predict, tab_benchmark, tab_domain_shift, tab_about = st.tabs([
    "🔮 Interactive 3D Classifier",
    "📊 Benchmark Dashboard",
    "🌐 Domain-Shift Explorer",
    "📖 Research & Methodology"
])


# ==============================================================================
# TAB 1: INTERACTIVE 3D CLASSIFIER
# ==============================================================================
with tab_predict:
    st.subheader("3D Point Cloud Input & Real-Time Classification")

    input_mode = st.radio(
        "Choose Input Method:",
        options=["Pre-loaded Benchmark Samples (Instant 1-Click)", "Upload Your Own 3D Point Cloud"],
        horizontal=True
    )

    pts_data = None
    input_source_desc = ""
    true_label_hint = None

    if input_mode == "Pre-loaded Benchmark Samples (Instant 1-Click)":
        catalog = get_demo_samples_catalog()
        sample_key = st.selectbox("Select Sample 3D Object:", options=list(catalog.keys()), index=0)
        pts_data, true_label_hint, input_source_desc = load_demo_point_cloud(sample_key)
        st.info(f"**Sample Details:** {input_source_desc} | **Ground Truth Category:** `{true_label_hint}`")

    else:
        uploaded_file = st.file_uploader(
            "Upload 3D point cloud or mesh file (.ply, .obj, .off, .xyz, .txt, .npy):",
            type=["ply", "obj", "off", "xyz", "txt", "npy"]
        )
        if uploaded_file is not None:
            try:
                pts_data, fmt_name = load_point_cloud_from_upload(uploaded_file)
                st.success(f"Successfully parsed **{uploaded_file.name}** ({fmt_name}) with **{len(pts_data):,}** 3D points.")
            except Exception as e:
                st.error(f"Failed to parse uploaded file: {e}")

    if pts_data is not None:
        c_left, c_right = st.columns([1.1, 0.9])

        with c_left:
            st.markdown("#### 🌌 3D Simulation View (12k Particles)")
            st.caption("Interactive WebGL 3D particle simulation. Rotate, pan, zoom, and inspect surface curvature in real-time.")
            target_pts = 12000 if enable_particle_sim else display_points_limit
            fig_3d = create_3d_point_cloud_figure(
                points=pts_data,
                title=f"3D Object: {true_label_hint.title() if true_label_hint else 'Custom Input'}",
                colorscale=colorscale_choice,
                point_size=point_size,
                max_display_points=target_pts,
                simulate_particles=enable_particle_sim
            )
            st.plotly_chart(fig_3d, width="stretch")

        with c_right:
            st.markdown("#### Classification Prediction")
            predictor = get_cached_predictor(selected_model_key)

            # Determine effective domain space
            if "Auto" in eval_mode:
                if true_label_hint and true_label_hint.lower() not in [c.lower() for c in COMMON_CLASS_NAMES]:
                    effective_mode = "unconstrained"
                elif "modelnet" in input_source_desc.lower():
                    effective_mode = "unconstrained"
                else:
                    effective_mode = "constrained"
            elif "Constrained" in eval_mode:
                effective_mode = "constrained"
            else:
                effective_mode = "unconstrained"

            # Contextual notice if user manually selected Constrained mode for an open-world category
            if effective_mode == "constrained" and true_label_hint and true_label_hint.lower() not in [c.lower() for c in COMMON_CLASS_NAMES]:
                st.info(f"ℹ️ **Domain Notice:** `{true_label_hint.title()}` is an open-world ModelNet40 category not present in the 10 constrained ScanObjectNN target classes. In 'Constrained' mode, predictions are mapped to the 10 target classes. Switch to **'Auto-Adaptive'** or **'Open-World (40 Classes)'** in the sidebar to recognize `{true_label_hint.title()}`.")

            # Run inference
            with st.spinner("Running 3D forward pass..."):
                res = predictor.predict(pts_data, mode=effective_mode, top_k=3)

            pred_class = res["predicted_class"].title()
            conf_pct = res["confidence_pct"]
            latency = res["latency_ms"]

            # Prediction Card
            is_match = (true_label_hint is not None and true_label_hint.lower() in pred_class.lower())
            card_border = "#22C55E" if is_match else "#3B82F6"
            bg_color = "#F0FDF4" if is_match else "#EFF6FF"
            
            st.markdown(f"""
            <div style="background-color:{bg_color}; border-left: 6px solid {card_border}; padding: 14px 18px; border-radius: 8px; margin-bottom: 15px;">
                <div style="font-size:0.85rem; font-weight:700; color:#6B7280; text-transform:uppercase;">Predicted Category</div>
                <div style="font-size:1.8rem; font-weight:800; color:#111827;">{pred_class}</div>
                <div style="font-size:0.95rem; font-weight:600; color:#1F2937;">Model Confidence: <b>{conf_pct}</b> &nbsp;|&nbsp; Latency: <b>{latency:.2f} ms</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Top-3 Predictions Bar Chart
            st.markdown("**Top-3 Confidence Distribution:**")
            top_df = pd.DataFrame(res["top_predictions"])
            top_df["probability_pct"] = top_df["probability"] * 100
            
            fig_bar = px.bar(
                top_df,
                x="probability_pct",
                y="class_name",
                orientation='h',
                text="confidence_pct",
                color="probability_pct",
                color_continuous_scale="Blues",
                labels={"probability_pct": "Confidence (%)", "class_name": "Category"}
            )
            fig_bar.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=0, r=0, t=10, b=10),
                height=160,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_bar, width="stretch")

        # Multi-Model Comparison Card
        st.markdown("---")
        st.markdown("#### ⚡ Cross-Model Comparative Inference")
        st.caption("Compare how PointNet, DGCNN Baseline, Peak GNN, and the Fine-Tuned GNN classify this exact 3D point cloud:")

        if st.button("Run Multi-Model Comparison on This Object"):
            with st.spinner("Evaluating all 4 architectures..."):
                models_to_run = ["pointnet", "dgcnn", "peak_dgcnn", "finetuned_gnn"]
                labels_map = {
                    "pointnet": "PointNet Baseline",
                    "dgcnn": "DGCNN Baseline (k=20)",
                    "peak_dgcnn": "Peak GNN (k=40)",
                    "finetuned_gnn": "Fine-Tuned GNN (Adapted)"
                }
                comp_rows = []
                for mk in models_to_run:
                    p_engine = get_cached_predictor(mk)
                    r = p_engine.predict(pts_data, mode=effective_mode, top_k=1)
                    comp_rows.append({
                        "Architecture": labels_map[mk],
                        "Predicted Class": r["predicted_class"].title(),
                        "Confidence": r["confidence_pct"],
                        "Latency (ms)": f"{r['latency_ms']:.2f} ms",
                        "Correct?": "✅ Correct" if (true_label_hint and true_label_hint.lower() in r["predicted_class"].lower()) else "❌ Differs"
                    })
                comp_df = pd.DataFrame(comp_rows)
                st.dataframe(comp_df, width="stretch", hide_index=True)


# ==============================================================================
# TAB 2: BENCHMARK DASHBOARD
# ==============================================================================
with tab_benchmark:
    st.subheader("Synthetic-to-Real Domain Generalization Benchmark")
    st.markdown("Comprehensive performance comparison between PointNet, DGCNN, and Structure-Aware GNN across ModelNet40 and ScanObjectNN.")

    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        st.markdown("#### 1. Zero-Shot Model Comparison")
        st.image("visualizations/phase7/zero_shot_model_comparison.png", caption="Zero-Shot Test Performance on ScanObjectNN (Untouched Test Partition, 386 Scans)", width="stretch")
    with b_col2:
        st.markdown("#### 2. Domain Adaptation Net Improvement")
        st.image("visualizations/phase7/zeroshot_vs_finetuned_comparison.png", caption="Zero-Shot vs. Fine-Tuned GNN (+52.33% Accuracy Gain on Real Scans)", width="stretch")

    st.markdown("---")
    st.markdown("#### Per-Class Accuracy Comparison (10 Common Real-World Classes)")

    csv_path = os.path.join(WORKSPACE_ROOT, "results", "phase7", "zero_shot_per_class_comparison.csv")
    if os.path.exists(csv_path):
        df_per_class = pd.read_csv(csv_path)
        st.dataframe(df_per_class, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔍 Interactive Confusion Matrix Viewer")
    cm_choice = st.selectbox(
        "Select Model Confusion Matrix to Inspect:",
        options=[
            "Fine-Tuned Domain-Adapted GNN (62.18% OA)",
            "Peak Regularized GNN (Zero-Shot, 9.84% OA)",
            "DGCNN Baseline (Zero-Shot, 6.48% OA)",
            "PointNet Baseline (Zero-Shot, 7.77% OA)"
        ]
    )

    cm_file_map = {
        "Fine-Tuned Domain-Adapted GNN (62.18% OA)": "visualizations/phase7/finetuned_gnn_confusion_matrix.png",
        "Peak Regularized GNN (Zero-Shot, 9.84% OA)": "visualizations/phase7/peak_dgcnn_zeroshot_confusion_matrix.png",
        "DGCNN Baseline (Zero-Shot, 6.48% OA)": "visualizations/phase7/dgcnn_zeroshot_confusion_matrix.png",
        "PointNet Baseline (Zero-Shot, 7.77% OA)": "visualizations/phase7/pointnet_zeroshot_confusion_matrix.png"
    }
    st.image(cm_file_map[cm_choice], caption=cm_choice, width="stretch")


# ==============================================================================
# TAB 3: DOMAIN-SHIFT & GEOMETRY EXPLORER
# ==============================================================================
with tab_domain_shift:
    st.subheader("Statistical & Geometric Domain-Shift Analysis")
    st.markdown("""
    When deep learning models trained on synthetic CAD shapes are evaluated on real LiDAR scans, they encounter three core challenges:
    1. **Missing Occluded Geometry:** Real scans only capture visible surfaces.
    2. **Coordinate Noise:** High-frequency Gaussian sensor jitter.
    3. **Non-Uniform Density:** Laser rays cluster closer to the sensor.
    """)

    st.markdown("#### Source (CAD) vs. Target (Real Scan) Multi-View Comparison")
    st.image("visualizations/phase7/source_vs_target_comparison.png", caption="ModelNet40 CAD vs. ScanObjectNN Real Scans", width="stretch")

    st.markdown("---")
    ds_col1, ds_col2 = st.columns(2)
    with ds_col1:
        st.markdown("#### Statistical Curvature & Density Shift")
        st.image("visualizations/phase7/domain_shift_geometry_distribution.png", caption="Probability Density of Surface Variation & Point Spacing", width="stretch")
    with ds_col2:
        st.markdown("#### Diagnostic Error & Success Cases")
        st.image("visualizations/phase7/error_diagnostics_gallery.png", caption="PointNet vs. GNN Case Studies", width="stretch")


# ==============================================================================
# TAB 4: RESEARCH & METHODOLOGY
# ==============================================================================
with tab_about:
    st.subheader("Structure-Aware Domain Generalization Methodology")
    st.markdown("""
    ### 1. Research Overview
    - **Title:** Structure-Aware Domain Generalization for 3D Point Cloud Classification using Graph Neural Networks
    - **Authors:** R. Upendra, B. J. V. Sivamani, B. Sasi, K. Pavan
    - **Affiliation:** Department of Computer Science and Engineering, VIT-AP University, Amaravati, India
    
    ### 2. EdgeConv Formulation
    Standard point-wise models (e.g., PointNet) process coordinates independently via shared MLPs:
    $$\\mathbf{z} = \\max_{i=1,\\dots,N} h_\\Theta(\\mathbf{p}_i)$$
    This fails to capture surface curvature and relative spatial configurations between neighboring points.
    
    The **EdgeConv** operator dynamically constructs a $k$-nearest neighbor ($k$-NN) graph in latent feature space:
    $$\\mathbf{e}_{ij} = h_\\Theta(\\mathbf{x}_i, \\mathbf{x}_j - \\mathbf{x}_i)$$
    incorporating both central coordinates $\\mathbf{x}_i$ and relative displacement $(\\mathbf{x}_j - \\mathbf{x}_i)$.
    
    ### 3. Key Findings & Contributions
    1. **Structural Invariance:** On articulated shapes (*chair*), the Structure-Aware GNN achieves **83.33% zero-shot accuracy**, outperforming PointNet (20.00%) by +63.33%.
    2. **Domain Adaptation Recovery:** Fine-tuning on target training representations recovers planar categories (*table*, *desk*, *cabinet*), boosting overall target accuracy from **9.84% to 62.18%**.
    """)

