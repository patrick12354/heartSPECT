"""
Heart SPECT Segmentation — Streamlit Prototype & Landing Page
Left Ventricle Segmentation using 3D U-Net on Cardiac SPECT Imaging
"""

import os
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_model
from utils import predict_volume, save_mask_as_nifti, TARGET_SHAPE

# --- Config ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pth")
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")

st.set_page_config(
    page_title="CorVision Prototype",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Global Typography & Layout */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Header/Footer for Landing Page feel */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Animations */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes subtleFloat {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .reveal {
        opacity: 0;
        transform: translateY(30px);
        transition: all 0.8s cubic-bezier(0.2, 0.8, 0.2, 1);
    }
    .reveal.visible {
        opacity: 1;
        transform: translateY(0);
    }
    
    .delay-1 { transition-delay: 0.1s; }
    .delay-2 { transition-delay: 0.2s; }
    .delay-3 { transition-delay: 0.3s; }

    /* Hero Section (Google/Apple Minimalist Style) */
    .hero-container {
        padding: 5rem 1rem 4rem;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .hero-label {
        background: rgba(255,255,255,0.1);
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.85rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 2rem;
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .hero-title {
        font-size: 4.8rem !important;
        font-weight: 800 !important;
        background: linear-gradient(270deg, #A0C4FF, #BDB2FF, #FFC6FF);
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        line-height: 1.1;
        letter-spacing: -2px;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #A0AAB2;
        max-width: 700px;
        margin: 0 auto 3rem;
        line-height: 1.7;
        font-weight: 300;
    }

    /* Modern Glass Cards (Sleek) */
    .glass-card {
        background: rgba(20, 20, 25, 0.4);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 2.5rem 2rem;
        height: 100%;
        transition: all 0.4s ease;
        text-align: left;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    .glass-card:hover {
        transform: translateY(-8px);
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }
    .glass-card:hover::before {
        opacity: 1;
    }
    .card-icon {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        display: inline-block;
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 16px;
    }
    .card-title {
        color: #FFFFFF;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .card-text {
        color: #8A94A6;
        line-height: 1.6;
        font-size: 1rem;
    }

    /* Metric Badges Minimalist */
    .metric-showcase {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 5rem;
        flex-wrap: wrap;
    }
    .metric-badge {
        text-align: center;
    }
    .metric-badge h2 {
        font-size: 4rem;
        margin: 0;
        color: #fff;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .metric-badge p {
        color: #6C7A89;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }

    /* CTA Button Override (Google Style) */
    div[data-testid="stButton"] button {
        background: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 30px;
        padding: 0.6rem 2.5rem;
        font-size: 1.1rem !important;
        font-weight: 600;
        border: none !important;
        transition: all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1);
        box-shadow: 0 8px 20px rgba(255,255,255,0.1) !important;
    }
    div[data-testid="stButton"] button:hover {
        transform: scale(1.03) translateY(-2px);
        box-shadow: 0 15px 30px rgba(255,255,255,0.2) !important;
        background: #f0f0f0 !important;
    }

    /* =============================================
       MOBILE RESPONSIVE (max-width: 768px)
       ============================================= */
    @media (max-width: 768px) {

        /* Hero Title — shrink massive 4.8rem down to readable */
        .hero-title {
            font-size: 2.6rem !important;
            letter-spacing: -1px;
            line-height: 1.2;
        }

        /* Hero Section — center everything on small screens */
        .hero-container {
            align-items: center !important;
            text-align: center !important;
            padding: 2rem 1rem 2rem;
        }

        /* Hero text - increase readability padding */
        .hero-subtitle {
            font-size: 1rem;
            margin: 0 auto 2rem;
        }

        /* Metric badges — stack vertically instead of side by side */
        .metric-showcase {
            flex-direction: column;
            align-items: center;
            gap: 1.5rem;
            margin-top: 2.5rem;
        }

        .metric-badge h2 {
            font-size: 2.8rem;
        }

        /* Glass cards — reduce padding so they don't overflow */
        .glass-card {
            padding: 1.5rem 1.2rem;
            border-radius: 18px;
            margin-bottom: 0.5rem;
        }

        /* Card icons — make slightly smaller */
        .card-icon {
            font-size: 2rem;
            padding: 0.7rem;
        }

        /* Sidebar — make it not take too much space */
        section[data-testid="stSidebar"] {
            min-width: 240px !important;
            max-width: 240px !important;
        }

        /* Streamlit main container padding */
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    /* Extra small phones (max-width: 480px) */
    @media (max-width: 480px) {
        .hero-title {
            font-size: 2rem !important;
            letter-spacing: -0.5px;
        }

        .metric-badge h2 {
            font-size: 2.2rem;
        }

        .metric-badge p {
            font-size: 0.8rem;
        }

        .card-title {
            font-size: 1.1rem;
        }

        .card-text {
            font-size: 0.9rem;
        }

        div[data-testid="stButton"] button {
            font-size: 0.95rem !important;
            padding: 0.5rem 1.5rem;
        }
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


def show_landing_page():
    """Landing page view for product pitching."""
    
    # --- HERO SECTION ---
    # Load transparent circular logo
    try:
        import base64
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_transparent.png"), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{encoded_string}" style="width: 70px; height: 70px; border-radius: 50%; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4); margin-right: 15px;" />'
    except Exception:
        logo_html = ""

    st.markdown(f"""
        <div class="hero-container reveal visible">
            <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 25px;">
                {logo_html}
                <div class="hero-label" style="margin-bottom: 0;">CV | MEDICAL AI ENGINE</div>
            </div>
            <h1 class="hero-title">CorVision</h1>
            <p class="hero-subtitle">Meningkatkan kualitas diagnosis kardiovaskular dengan segmentasi volumetrik 3D U-Net mutakhir otomatis pada jaringan pencitraan Myocardial Perfusion SPECT.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # --- CTA BUTTON (Moved up for better UX) ---
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown('<div class="reveal visible delay-1" style="display:flex; justify-content:center; margin-bottom: 5rem;">', unsafe_allow_html=True)
        if st.button("Jalankan Prototype AI", use_container_width=True, type="primary"):
            st.session_state.page = "app"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- FEATURES / CLINICAL IMPACT (Cards) ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
            <div class="glass-card reveal delay-1">
                <div class="card-icon">⚡</div>
                <div class="card-title">Analisis Real-time</div>
                <div class="card-text">
                    Menggantikan berjam-jam delineasi manual dokter. Model memproses volume organ utuh dalam hitungan detik, mempercepat siklus konfirmasi klinis di rumah sakit.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
            <div class="glass-card reveal delay-2">
                <div class="card-icon">🎯</div>
                <div class="card-title">Bebas Fatal Fatigue</div>
                <div class="card-text">
                    Menghilangkan bias dan inter-observer variabilitas akibat kelelahan tenaga kesehatan. Parameter ventrikel yang diekstrak selalu konsisten dan dapat direproduksi kapan saja.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
            <div class="glass-card reveal delay-3">
                <div class="card-icon">🧠</div>
                <div class="card-title">Topologis Spasial 3D</div>
                <div class="card-text">
                    Tidak seperti AI analisis foto biasa (2D), arsitektur U-Net 3D ini menjaga integritas anatomi XYZ sepenuhnya, membersihkan <i>background noise</i> organ tetangga secara cerdas.
                </div>
            </div>
        """, unsafe_allow_html=True)


    # --- METRICS SHOWCASE ---
    st.markdown("""
        <div class="metric-showcase reveal delay-3">
            <div class="metric-badge">
                <h2>91.4%</h2>
                <p>Dice Score Validasi</p>
            </div>
            <div class="metric-badge">
                <h2>>5.6M</h2>
                <p>Parameter Cerdas</p>
            </div>
            <div class="metric-badge">
                <h2>0.99</h2>
                <p>Uji Akurasi ROC-AUC</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # --- CLINICAL WORKFLOW SECTION ---
    st.markdown('<h2 class="hero-title reveal delay-1" style="font-size: 2.2rem; text-align: center; margin-top: 3rem; margin-bottom: 2rem;">Alur Kerja Klinis</h2>', unsafe_allow_html=True)
    
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("""
            <div class="glass-card reveal delay-1" style="text-align: center; padding: 20px;">
                <h1 style="color: #64ffda; margin: 0; font-size: 3rem;">1</h1>
                <h4 style="color: white; margin-top: 10px;">Akuisisi Data</h4>
                <p style="font-size: 0.85rem; color: #a0aec0;">Scan SPECT myokardial pasien dikumpulkan dalam format DICOM 3D standar dari Rumah Sakit.</p>
            </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
            <div class="glass-card reveal delay-2" style="text-align: center; padding: 20px;">
                <h1 style="color: #64ffda; margin: 0; font-size: 3rem;">2</h1>
                <h4 style="color: white; margin-top: 10px;">Pre-processing</h4>
                <p style="font-size: 0.85rem; color: #a0aec0;">CorVision secara cerdas menormalisasi intensitas voxel dan melakukan resample resolusi secara otomatis.</p>
            </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
            <div class="glass-card reveal delay-3" style="text-align: center; padding: 20px;">
                <h1 style="color: #64ffda; margin: 0; font-size: 3rem;">3</h1>
                <h4 style="color: white; margin-top: 10px;">Inference AI</h4>
                <p style="font-size: 0.85rem; color: #a0aec0;">Model U-Net mengekstraksi fitur spasial secara real-time untuk memisahkan anatomi ventrikel kiri.</p>
            </div>
        """, unsafe_allow_html=True)
    with w4:
        st.markdown("""
            <div class="glass-card reveal delay-4" style="text-align: center; padding: 20px;">
                <h1 style="color: #64ffda; margin: 0; font-size: 3rem;">4</h1>
                <h4 style="color: white; margin-top: 10px;">Verifikasi Medis</h4>
                <p style="font-size: 0.85rem; color: #a0aec0;">Dokter spesialis meninjau hasil mask 3D melalui visualizer volumetrik pada dasbor ini.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- METHODOLOGY SECTION ---
    st.markdown('<h2 class="hero-title reveal delay-1" style="font-size: 2.2rem; text-align: center; margin-bottom: 2rem;">Arsitektur & Metodologi</h2>', unsafe_allow_html=True)
    
    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown("""
            <div class="glass-card reveal delay-2" style="padding: 30px; height: 100%;">
                <h3 style="color: white; margin-bottom: 15px; font-size: 1.3rem;">🧠 Model 3D U-Net Deep Learning</h3>
                <p style="color: #a0aec0; line-height: 1.6;">
                    Inti dari CorVision adalah arsitektur <strong>Fully Convolutional Network (3D U-Net)</strong> kustom. Tidak seperti model klasifikasi gambar biasa, jaringan komputer ini membaca seluruh matriks kubus (Voxel) sekaligus, mengidentifikasi pola kepadatan radioisotop dalam 3 dimensi.
                </p>
                <ul style="color: #a0aec0; line-height: 1.6; margin-top: 15px;">
                    <li><strong>Encoder-Decoder:</strong> Menangkap fitur tekstur halus sekaligus konteks organ tetangga.</li>
                    <li><strong>Skip-Connections:</strong> Memulihkan batas visual ventrikel yang tajam dan persisi.</li>
                    <li><strong>5.6 Juta Parameter:</strong> Bobot terlatih pada dataset medis spesifik rumah sakit.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown("""
            <div class="glass-card reveal delay-3" style="padding: 30px; height: 100%;">
                <h3 style="color: white; margin-bottom: 15px; font-size: 1.3rem;">⚖️ Penanganan Ketidakseimbangan Ekstrim</h3>
                <p style="color: #a0aec0; line-height: 1.6;">
                    Salah satu tantangan terbesar medis adalah ventrikel kiri hanya mencakup kurang dari <strong>1% total volume gambar SPECT</strong> (ekstrim <i>class imbalance</i>).
                </p>
                <p style="color: #a0aec0; line-height: 1.6;">
                    CorVision mengatasinya dengan menggunakan <strong>Kombinasi Loss Function</strong> cerdas selama fase pelatihan: <code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">0.5 * Dice Loss + 0.5 * Binary Cross Entropy</code>. Metode ini secara paksa mensinergikan sensitivitas pelacakan batas piksel dan akurasi global voxels.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    # CTA BOTTOM
    col_bot1, col_bot2, col_bot3 = st.columns([1, 1, 1])
    with col_bot2:
        st.markdown('<div class="reveal visible delay-1" style="display:flex; justify-content:center; margin-bottom: 5rem;">', unsafe_allow_html=True)
        if st.button("Coba Analisis DICOM Sekarang", use_container_width=True, type="secondary"):
            st.session_state.page = "app"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Inject Custom JS at the BOTTOM to prevent layout shifts ---
    components.html(
        """
        <script>
        const doc = window.parent.document;
        
        // Fix horizontal scrolling and overflow
        doc.body.style.overflowX = 'hidden';
        doc.documentElement.style.overflowX = 'hidden';
        
        // 1. Sleek Background Glow (Apple/Google Style Soft Ambient Blob)
        if (!doc.getElementById("ambient-glow")) {
            const bg = doc.createElement("div");
            bg.id = "ambient-glow";
            bg.style.cssText = `
                position: fixed; top: -20vh; right: -10vw;
                width: 70vw; height: 70vw;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(160, 196, 255, 0.08) 0%, transparent 70%);
                filter: blur(100px);
                z-index: -1;
                pointer-events: none;
                transition: transform 3s ease-out;
            `;
            doc.body.insertBefore(bg, doc.body.firstChild);

            // Subtle parallax effect on mouse move for the background
            doc.addEventListener("mousemove", (e) => {
                const moveX = (e.clientX / window.innerWidth - 0.5) * -50;
                const moveY = (e.clientY / window.innerHeight - 0.5) * -50;
                bg.style.transform = `translate(${moveX}px, ${moveY}px)`;
            });
        }

        // 2. Minimalist Cursor Follower
        if (!doc.getElementById("cursor-ring")) {
            const ring = doc.createElement("div");
            ring.id = "cursor-ring";
            ring.style.cssText = `
                position: fixed; width: 40px; height: 40px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                pointer-events: none; z-index: 9999;
                transform: translate(-50%, -50%);
                transition: width 0.3s, height 0.3s, background 0.3s, border 0.3s;
            `;
            doc.body.appendChild(ring);

            let mouseX = window.innerWidth / 2;
            let mouseY = window.innerHeight / 2;
            let currentX = mouseX;
            let currentY = mouseY;

            doc.addEventListener("mousemove", (e) => {
                mouseX = e.clientX;
                mouseY = e.clientY;
            });

            // Smooth interpolation
            const animateCursor = () => {
                currentX += (mouseX - currentX) * 0.2;
                currentY += (mouseY - currentY) * 0.2;
                ring.style.left = `${currentX}px`;
                ring.style.top = `${currentY}px`;
                requestAnimationFrame(animateCursor);
            };
            animateCursor();
            
            // Interaction effects (Hover snaps/expands)
            doc.addEventListener("mouseover", (e) => {
                if(e.target.closest("button") || e.target.closest(".glass-card")) {
                    ring.style.width = "70px";
                    ring.style.height = "70px";
                    ring.style.background = "rgba(255,255,255,0.05)";
                    ring.style.border = "1px solid rgba(255,255,255,0.6)";
                }
            });
            doc.addEventListener("mouseout", (e) => {
                if(e.target.closest("button") || e.target.closest(".glass-card")) {
                    ring.style.width = "40px";
                    ring.style.height = "40px";
                    ring.style.background = "transparent";
                    ring.style.border = "1px solid rgba(255, 255, 255, 0.3)";
                }
            });
        }
        
        // 3. Proper Scroll Reveal Observer
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if(entry.isIntersecting) {
                    entry.target.classList.add('visible');
                }
            });
        }, { threshold: 0.1 });

        // Give Streamlit a moment to render DOM blocks
        setTimeout(() => {
            doc.querySelectorAll('.reveal').forEach(el => {
                observer.observe(el);
            });
        }, 500);
        </script>
        """, height=0
    )



def show_app_page():
    """Main application view for segmentation interactive UI."""
    components.html("""
        <script>
            let attempts = 0;
            const scrollInterval = setInterval(function() {
                window.parent.scrollTo(0, 0);
                attempts++;
                if (attempts > 10) clearInterval(scrollInterval);
            }, 50);
        </script>
    """, height=0)
    # Navigation Back Button
    if st.sidebar.button("← Back to Landing Page"):
        st.session_state.page = "landing"
        st.rerun()

    st.sidebar.markdown("---")
    
    # Header
    st.markdown('<h2 class="main-title" style="font-size:2rem;">👁️ CorVision Engine</h2>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Run segmentation interactively</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    model, model_info, device = get_model()

    # Sidebar Tools
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
        input_mode = st.radio("", ["Use Sample Data", "Upload DICOM"],
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
                selected = st.selectbox("Select sample DICOM to test:", samples)
                dicom_path = os.path.join(SAMPLE_DIR, selected)
            else:
                st.warning("No .dcm files found in sample directory.")
        else:
            st.warning(f"Sample directory not found: `{SAMPLE_DIR}`")

    # Run inference
    if dicom_path and model is not None:
        with st.spinner("Running deep learning segmentation..."):
            try:
                img, pred_bin, prob_map = predict_volume(
                    dicom_path, model, device,
                    TARGET_SHAPE, threshold
                )
            except Exception as e:
                st.error(f"Inference failed: {e}")
                return
            finally:
                # Clean up uploaded temporary DICOM
                if input_mode == "Upload DICOM" and dicom_path and os.path.exists(dicom_path):
                    try:
                        os.remove(dicom_path)
                    except Exception:
                        pass

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
                <div class="metric-label">L. Ventricle Ratio</div>
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
        tab1, tab2, tab3 = st.tabs(["🔬 Multi-Plane Viewer", "🎯 Segmentation Result", "🌡️ Probability Heatmap"])

        with tab1:
            st.subheader("Interactive Multi-Plane Viewer")
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
            st.subheader("Automated Segmentation Outline")
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
            st.subheader("Model Validation (Probability Map)")
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
        st.subheader("📥 Export & API")

        col1, col2 = st.columns(2)
        with col1:
            tmp_mask = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
            save_mask_as_nifti(pred_bin, tmp_mask.name)
            with open(tmp_mask.name, 'rb') as f:
                mask_bytes = f.read()
            try:
                os.remove(tmp_mask.name)
            except Exception:
                pass
                
            st.download_button(
                label="Download Segmentation Mask (.nii.gz)",
                data=mask_bytes,
                file_name="predicted_mask.nii.gz",
                mime="application/gzip"
            )

        with col2:
            tmp_prob = tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz")
            save_mask_as_nifti(prob_map, tmp_prob.name)
            with open(tmp_prob.name, 'rb') as f:
                prob_bytes = f.read()
            try:
                os.remove(tmp_prob.name)
            except Exception:
                pass
                
            st.download_button(
                label="Download Probability Map (.nii.gz)",
                data=prob_bytes,
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


def main():
    # Initialize State Control
    if 'page' not in st.session_state:
        st.session_state.page = "landing"

    # Routing
    if st.session_state.page == "landing":
        show_landing_page()
    elif st.session_state.page == "app":
        show_app_page()

if __name__ == '__main__':
    main()
