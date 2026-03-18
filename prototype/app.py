"""
Heart SPECT Segmentation — Streamlit Prototype
Left Ventricle Segmentation using 3D U-Net on Cardiac SPECT Imaging
"""

import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_model
from utils import predict_volume, save_mask_as_nifti, TARGET_SHAPE

# --- Config ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "DICOM")

st.set_page_config(
    page_title="Heart SPECT Segmentation",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Load model once and cache it."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None, device
    model, info = load_model(CHECKPOINT_PATH, device)
    return model, info, device


def render_slice(img_slice, mask_slice=None, prob_slice=None, title="",
                 cmap_img='gray', alpha=0.4):
    """Render a 2D slice with optional mask overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img_slice.T, cmap=cmap_img, origin='lower', aspect='equal')

    if mask_slice is not None:
        mask_rgba = np.zeros((*mask_slice.T.shape, 4))
        mask_rgba[mask_slice.T > 0] = [1, 0.2, 0.2, alpha]
        ax.imshow(mask_rgba, origin='lower', aspect='equal')

    if prob_slice is not None:
        ax.imshow(prob_slice.T, cmap='hot', origin='lower', aspect='equal',
                  alpha=0.5, vmin=0, vmax=1)

    ax.set_title(title, fontsize=11, fontweight='bold', color='white')
    ax.axis('off')
    fig.patch.set_facecolor('#0e1117')
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-title">🫀 Heart SPECT Segmentation</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Left Ventricle Segmentation using 3D U-Net on Cardiac SPECT Imaging</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    model, model_info, device = get_model()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        if model is not None:
            st.success(f"Model loaded on **{device}**")
            if model_info:
                st.caption(f"Epoch: {model_info.get('epoch', 'N/A')} | "
                          f"Val Dice: {model_info.get('val_dice', 'N/A'):.4f}")
        else:
            st.error("Model checkpoint not found!")
            st.caption(f"Expected: `{CHECKPOINT_PATH}`")

        st.markdown("---")
        threshold = st.slider("Segmentation Threshold", 0.1, 0.9, 0.5, 0.05)
        overlay_alpha = st.slider("Overlay Opacity", 0.1, 0.8, 0.4, 0.05)

        st.markdown("---")
        st.markdown("### 📂 Input Source")
        input_mode = st.radio("", ["Upload DICOM", "Use Sample Data"],
                              label_visibility="collapsed")

    # Get DICOM file
    dicom_path = None

    if input_mode == "Upload DICOM":
        uploaded = st.file_uploader(
            "Upload a DICOM file (.dcm)",
            type=["dcm"],
            help="Upload a cardiac SPECT DICOM file for segmentation"
        )
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
            tmp.write(uploaded.read())
            tmp.close()
            dicom_path = tmp.name

    else:
        if os.path.exists(SAMPLE_DIR):
            samples = sorted([f for f in os.listdir(SAMPLE_DIR) if f.endswith('.dcm')])
            if samples:
                selected = st.selectbox("Select sample DICOM:", samples)
                dicom_path = os.path.join(SAMPLE_DIR, selected)
            else:
                st.warning("No .dcm files found in sample directory.")
        else:
            st.warning(f"Sample directory not found: `{SAMPLE_DIR}`")

    # Run inference
    if dicom_path and model is not None:
        with st.spinner("Running segmentation..."):
            try:
                img, pred_bin, prob_map = predict_volume(
                    dicom_path, model, device,
                    TARGET_SHAPE, threshold
                )
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return

        # --- Metrics ---
        voxel_count = int(pred_bin.sum())
        total_voxels = pred_bin.size
        mask_ratio = (voxel_count / total_voxels) * 100
        mean_confidence = float(prob_map[pred_bin > 0].mean()) if voxel_count > 0 else 0.0
        max_confidence = float(prob_map.max())

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{voxel_count:,}</div>
                <div class="metric-label">Voxels Detected</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mask_ratio:.2f}%</div>
                <div class="metric-label">Ventricle Ratio</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mean_confidence:.3f}</div>
                <div class="metric-label">Mean Confidence</div>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{max_confidence:.3f}</div>
                <div class="metric-label">Peak Confidence</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # --- Visualization Tabs ---
        tab1, tab2, tab3 = st.tabs(["🔬 Multi-Plane Viewer", "🎯 Segmentation Overlay", "🌡️ Probability Map"])

        with tab1:
            st.subheader("Multi-Plane Viewer")
            c1, c2, c3 = st.columns(3)

            with c1:
                ax_idx = st.slider("Axial (X)", 0, img.shape[0]-1, img.shape[0]//2, key="ax")
                fig = render_slice(img[ax_idx, :, :], pred_bin[ax_idx, :, :],
                                   title=f"Axial — slice {ax_idx}", alpha=overlay_alpha)
                st.pyplot(fig)
                plt.close(fig)

            with c2:
                cor_idx = st.slider("Coronal (Y)", 0, img.shape[1]-1, img.shape[1]//2, key="cor")
                fig = render_slice(img[:, cor_idx, :], pred_bin[:, cor_idx, :],
                                   title=f"Coronal — slice {cor_idx}", alpha=overlay_alpha)
                st.pyplot(fig)
                plt.close(fig)

            with c3:
                sag_idx = st.slider("Sagittal (Z)", 0, img.shape[2]-1, img.shape[2]//2, key="sag")
                fig = render_slice(img[:, :, sag_idx], pred_bin[:, :, sag_idx],
                                   title=f"Sagittal — slice {sag_idx}", alpha=overlay_alpha)
                st.pyplot(fig)
                plt.close(fig)

        with tab2:
            st.subheader("Segmentation Result")
            c1, c2, c3 = st.columns(3)

            mid = [s // 2 for s in img.shape]

            with c1:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                fig.patch.set_facecolor('#0e1117')
                axes[0].imshow(img[mid[0], :, :].T, cmap='gray', origin='lower')
                axes[0].set_title("SPECT", color='white', fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(pred_bin[mid[0], :, :].T, cmap='inferno', origin='lower')
                axes[1].set_title("Segmentation", color='white', fontweight='bold')
                axes[1].axis('off')
                plt.suptitle("Axial View", color='white', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with c2:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                fig.patch.set_facecolor('#0e1117')
                axes[0].imshow(img[:, mid[1], :].T, cmap='gray', origin='lower')
                axes[0].set_title("SPECT", color='white', fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(pred_bin[:, mid[1], :].T, cmap='inferno', origin='lower')
                axes[1].set_title("Segmentation", color='white', fontweight='bold')
                axes[1].axis('off')
                plt.suptitle("Coronal View", color='white', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with c3:
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                fig.patch.set_facecolor('#0e1117')
                axes[0].imshow(img[:, :, mid[2]].T, cmap='gray', origin='lower')
                axes[0].set_title("SPECT", color='white', fontweight='bold')
                axes[0].axis('off')
                axes[1].imshow(pred_bin[:, :, mid[2]].T, cmap='inferno', origin='lower')
                axes[1].set_title("Segmentation", color='white', fontweight='bold')
                axes[1].axis('off')
                plt.suptitle("Sagittal View", color='white', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with tab3:
            st.subheader("Probability Map (Model Confidence)")
            c1, c2, c3 = st.columns(3)

            with c1:
                prob_idx = st.slider("Axial (X)", 0, img.shape[0]-1, img.shape[0]//2, key="prob_ax")
                fig = render_slice(img[prob_idx, :, :], prob_slice=prob_map[prob_idx, :, :],
                                   title=f"Probability — axial {prob_idx}")
                st.pyplot(fig)
                plt.close(fig)

            with c2:
                prob_idx2 = st.slider("Coronal (Y)", 0, img.shape[1]-1, img.shape[1]//2, key="prob_cor")
                fig = render_slice(img[:, prob_idx2, :], prob_slice=prob_map[:, prob_idx2, :],
                                   title=f"Probability — coronal {prob_idx2}")
                st.pyplot(fig)
                plt.close(fig)

            with c3:
                prob_idx3 = st.slider("Sagittal (Z)", 0, img.shape[2]-1, img.shape[2]//2, key="prob_sag")
                fig = render_slice(img[:, :, prob_idx3], prob_slice=prob_map[:, :, prob_idx3],
                                   title=f"Probability — sagittal {prob_idx3}")
                st.pyplot(fig)
                plt.close(fig)

        # --- Download ---
        st.markdown("---")
        st.subheader("📥 Export Results")

        col1, col2 = st.columns(2)
        with col1:
            tmp_mask = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
            save_mask_as_nifti(pred_bin, tmp_mask.name)
            with open(tmp_mask.name, 'rb') as f:
                st.download_button(
                    label="Download Segmentation Mask (.nii.gz)",
                    data=f.read(),
                    file_name="predicted_mask.nii.gz",
                    mime="application/gzip"
                )

        with col2:
            tmp_prob = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
            save_mask_as_nifti(prob_map, tmp_prob.name)
            with open(tmp_prob.name, 'rb') as f:
                st.download_button(
                    label="Download Probability Map (.nii.gz)",
                    data=f.read(),
                    file_name="probability_map.nii.gz",
                    mime="application/gzip"
                )

        # Safety disclaimer
        st.markdown("---")
        st.caption("⚠️ **Disclaimer:** This tool is a research prototype for academic purposes only. "
                   "It is NOT intended for clinical diagnosis. Always consult qualified medical professionals "
                   "for diagnostic decisions.")

    elif model is None:
        st.info("👆 Please ensure the model checkpoint exists at the expected path.")
    else:
        st.info("👆 Upload a DICOM file or select sample data to begin segmentation.")


if __name__ == '__main__':
    main()
