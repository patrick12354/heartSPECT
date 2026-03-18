nse# 🫀 Heart SPECT Segmentation — Prototype

[![Try Live Prototype](https://img.shields.io/badge/TRY_LIVE_PROTOTYPE-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://heartspect.streamlit.app/)

Web app prototype untuk segmentasi ventrikel kiri pada cardiac SPECT imaging menggunakan 3D U-Net.

## Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r prototype/requirements.txt
```

> **Note:** Untuk PyTorch dengan dukungan CUDA/GPU, install dari [pytorch.org](https://pytorch.org/get-started/locally/) sesuai konfigurasi PC:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Jalankan Aplikasi

**Cara Paling Mudah (Windows):**
Cukup *double-click* (klik ganda) file **`run_prototype.bat`** yang ada di folder luar (`spect/`). Aplikasi akan berjalan dan otomatis terbuka di browser Anda.

**Cara Manual via Terminal:**
```bash
streamlit run prototype/app.py
```

Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`.

### 3. Cara Pakai

1. Pilih **"Upload DICOM"** untuk upload file `.dcm` baru, atau **"Use Sample Data"** untuk memilih dari data yang sudah ada
2. Segmentasi akan berjalan otomatis
3. Lihat hasil di 3 tab: **Multi-Plane Viewer**, **Segmentation Overlay**, dan **Probability Map**
4. Download hasil prediksi sebagai file `.nii.gz`

## Fitur

| Fitur | Deskripsi |
|-------|-----------|
| **Upload DICOM** | Drag & drop file .dcm |
| **Sample Data** | Pilih dari data yang sudah ada |
| **Multi-Plane Viewer** | Axial, Coronal, Sagittal dengan slider |
| **Segmentation Overlay** | Visualisasi mask di atas gambar SPECT |
| **Probability Map** | Heatmap confidence model |
| **Metrics Panel** | Voxel count, ratio, confidence |
| **Export** | Download mask & probability map (.nii.gz) |
| **Adjustable Threshold** | Ubah threshold segmentasi (default 0.5) |

## Struktur File

```
prototype/
├── app.py              ← Main Streamlit app
├── model.py            ← UNet3D architecture
├── utils.py            ← Preprocessing & inference
├── requirements.txt    ← Dependencies
└── README.md           ← File ini
```

## Prerequisite

- Python 3.8+
- Model checkpoint (`models/best_model.pth`) harus sudah ada
- **Dataset Penuh:** Jika ingin mencoba lebih dari 5 sampel data bawaan, kamu bisa mengunduh full dataset DICOM di sini: [PhysioNet: Myocardial Perfusion SPECT (1.0.0)](https://physionet.org/content/myocardial-perfusion-spect/get-zip/1.0.0/). Ekstrak ke folder `data/raw/DICOM/` untuk menggunakannya di luar sampel.
