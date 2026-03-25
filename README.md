# IDSC 2026 - Team Dawnbringer

> *"Mathematics and Hope in Healthcare"*

*(For Indonesian readers / Bagi pembaca berbahasa Indonesia, silakan gulir ke bawah untuk versi Bahasa Indonesia.)*

---

## English Version

### What Is This Repository?

This repository contains the working submission materials of **Dawnbringer** for **IDSC 2026**. The work is built as a cross-modal proof of concept around a single thesis:

> **AI can provide hope in healthcare when it is trained rigorously on real biomedical data and designed to support earlier screening, better triage, and more consistent clinical decision support.**

This project is built as a broader proof for the competition: that carefully designed AI systems can extract meaningful clinical signals from noisy, heterogeneous, real-world biomedical datasets.

This direction aligns with the IDSC 2026 theme, **Mathematics and Hope in Healthcare**, and with the competition emphasis on official biomedical datasets, methodological rigor, interpretability, and healthcare relevance.

---

### Competition Context

IDSC 2026 asks participants to work with official datasets and build computational solutions that are accurate, reproducible, clinically meaningful, and capable of delivering real hope through screening and decision support.

This repository is organized as a portfolio of experiments across several medical modalities:
- cardiac imaging
- electrocardiography
- EEG / brain-computer interface
- ophthalmic imaging

Together, these folders support one broader submission narrative: **AI can remain clinically useful even when medical data is noisy, difficult, and heterogeneous, as long as the modeling and evaluation are rigorous.**

---

### Projects in This Repository

| Folder | Medical Domain | Task | Key Technology |
|:---|:---|:---|:---|
| [`SPECT/`](./SPECT/) | **Cardiology / Nuclear Imaging** | Left ventricle segmentation from Myocardial Perfusion SPECT scans | 3D U-Net (PyTorch) |
| [`bigP3BCI/`](./bigP3BCI/) | **Neurology / BCI** | Motor imagery EEG signal classification for Brain-Computer Interface | CNN / Deep Learning on EEG |
| [`Brugada/`](./Brugada/) | **Cardiology / Electrophysiology** | Brugada syndrome detection from 12-lead ECG recordings | XGBoost + 1D CNN + Quality-Aware ResNet1D |
| [`Hillel Yaffe Glaucoma/`](./Hillel%20Yaffe%20Glaucoma/) | **Ophthalmology** | Glaucoma detection from retinal fundus images | CNN / Transfer Learning |

---

### Why These Projects Exist

Biomedical data in the real world is rarely clean. ECG can be noisy. EEG can be unstable. Retinal images vary in quality. Cardiac imaging can be small-scale, imbalanced, and difficult to annotate.

That is exactly why these projects matter.

Each folder explores the same larger belief from a different angle:

#### 1. SPECT: CorVision
This project focuses on **myocardial perfusion SPECT**, where the task is to segment the left ventricle from 3D medical images. It demonstrates that AI can automate a difficult imaging workflow that normally requires time and expert attention. In practical terms, this can support faster cardiac analysis and more efficient downstream assessment.

#### 2. bigP3BCI
This project focuses on **EEG-based brain-computer interface signals**, especially in settings where patients may have severe motor limitations. The goal is to show that AI can still recover useful intention-related signals from difficult EEG data, opening the door to assistive communication systems that remain meaningful even under challenging physiological conditions.

#### 3. Brugada
This project focuses on **12-lead ECG classification for Brugada syndrome**, a clinically important condition associated with ventricular arrhythmia and sudden cardiac death. Here, the proof point is that AI can help detect subtle but high-stakes cardiac patterns and support earlier risk-oriented review of ECG recordings.

#### 4. Hillel Yaffe Glaucoma
This project focuses on **fundus image analysis for glaucoma detection**. It supports the idea that AI can expand access to early screening in ophthalmology, especially in settings where specialist availability is limited and delayed diagnosis carries permanent consequences.

---

### The Core Proof We Are Building

Taken together, these projects are meant to support four claims:

1. **AI can learn from official biomedical datasets across very different modalities.**
2. **AI can remain useful even when the underlying data is noisy, heterogeneous, or quality-sensitive.**
3. **AI becomes far more credible when paired with strong validation, calibration, interpretability, and clinically relevant metrics.**
4. **AI can provide practical hope in healthcare by helping clinicians screen earlier, review faster, and prioritize high-risk cases more effectively.**

This repository should therefore be read as **Dawnbringer's proof-of-concept submission space for IDSC 2026**. The folders are individual technical studies, but together they build one larger argument.

---

### Disclaimer

> **All projects in this repository are strictly for academic and research purposes.** None of the models, prototypes, or tools presented here are intended for clinical diagnosis or medical treatment decisions without the oversight of qualified medical professionals.

---
---

## Versi Bahasa Indonesia

### Apa Itu Repositori Ini?

Repositori ini berisi materi kerja submission milik **Dawnbringer** untuk **IDSC 2026**. Pekerjaan ini dibangun sebagai proof of concept lintas modalitas dengan satu tesis utama:

> **AI dapat membawa harapan dalam layanan kesehatan jika dilatih secara ketat pada data biomedis nyata dan dirancang untuk membantu skrining dini, triase yang lebih baik, serta dukungan keputusan klinis yang lebih konsisten.**

Proyek ini dibangun sebagai pembuktian yang lebih besar untuk kompetisi ini: bahwa sistem AI yang dirancang dengan baik dapat mengekstrak sinyal klinis yang bermakna dari dataset biomedis yang noisy, heterogen, dan menantang.

Arah ini selaras dengan tema IDSC 2026, yaitu **Mathematics and Hope in Healthcare**, serta dengan penekanan kompetisi pada penggunaan dataset resmi, rigor metodologis, interpretabilitas, dan relevansi klinis.

---

### Konteks Kompetisi

IDSC 2026 meminta peserta untuk bekerja dengan dataset resmi dan membangun solusi komputasional yang akurat, reproducible, bermakna secara klinis, serta mampu memberi harapan nyata melalui skrining dan decision support.

Repositori ini disusun sebagai portofolio eksperimen pada beberapa modalitas medis:
- pencitraan jantung
- elektrokardiografi
- EEG / brain-computer interface
- pencitraan oftalmologi

Seluruh folder ini mendukung satu narasi submission yang lebih besar: **AI tetap bisa berguna secara klinis walaupun data medis bersifat noisy, sulit, dan heterogen, selama pemodelan dan evaluasinya dilakukan dengan rigor yang kuat.**

---

### Proyek dalam Repositori Ini

| Folder | Domain Medis | Tugas | Teknologi Utama |
|:---|:---|:---|:---|
| [`SPECT/`](./SPECT/) | **Kardiologi / Pencitraan Nuklir** | Segmentasi ventrikel kiri dari citra Myocardial Perfusion SPECT | 3D U-Net (PyTorch) |
| [`bigP3BCI/`](./bigP3BCI/) | **Neurologi / BCI** | Klasifikasi sinyal EEG Motor Imagery untuk Brain-Computer Interface | CNN / Deep Learning pada EEG |
| [`Brugada/`](./Brugada/) | **Kardiologi / Elektrofisiologi** | Deteksi sindrom Brugada dari rekaman ECG 12-lead | XGBoost + 1D CNN + Quality-Aware ResNet1D |
| [`Hillel Yaffe Glaucoma/`](./Hillel%20Yaffe%20Glaucoma/) | **Oftalmologi** | Deteksi glaukoma dari citra fundus retina | CNN / Transfer Learning |

---

### Mengapa Proyek-Proyek Ini Dibuat

Data biomedis di dunia nyata hampir tidak pernah benar-benar bersih. ECG bisa noisy. EEG bisa tidak stabil. Citra retina bisa bervariasi kualitasnya. Pencitraan jantung bisa berukuran kecil, tidak seimbang, dan sulit dianotasi.

Justru karena itu proyek-proyek ini penting.

Setiap folder mengeksplorasi keyakinan besar yang sama dari sudut yang berbeda:

#### 1. SPECT: CorVision
Proyek ini berfokus pada **myocardial perfusion SPECT**, dengan tugas segmentasi ventrikel kiri dari citra medis 3D. Proyek ini menunjukkan bahwa AI dapat membantu mengotomatisasi workflow pencitraan yang sulit dan biasanya membutuhkan waktu serta perhatian ahli. Secara praktis, ini dapat mendukung analisis jantung yang lebih cepat dan evaluasi lanjutan yang lebih efisien.

#### 2. bigP3BCI
Proyek ini berfokus pada **sinyal EEG untuk brain-computer interface**, terutama pada konteks pasien dengan keterbatasan motorik berat. Tujuannya adalah menunjukkan bahwa AI tetap dapat menangkap sinyal yang berkaitan dengan niat dari data EEG yang sulit, sehingga membuka jalan menuju sistem komunikasi bantu yang tetap bermakna meskipun kondisi fisiologis pasien menantang.

#### 3. Brugada
Proyek ini berfokus pada **klasifikasi ECG 12-lead untuk sindrom Brugada**, yaitu kondisi klinis penting yang berkaitan dengan aritmia ventrikel dan sudden cardiac death. Di sini, pembuktiannya adalah bahwa AI dapat membantu mendeteksi pola jantung yang halus namun berisiko tinggi, serta mendukung peninjauan ECG berbasis risiko secara lebih dini.

#### 4. Hillel Yaffe Glaucoma
Proyek ini berfokus pada **analisis citra fundus untuk deteksi glaukoma**. Proyek ini mendukung gagasan bahwa AI dapat memperluas akses skrining dini di bidang oftalmologi, terutama pada situasi ketika ketersediaan dokter spesialis terbatas dan keterlambatan diagnosis membawa dampak permanen.

---

### Inti Pembuktian yang Sedang Dibangun

Jika dilihat bersama, proyek-proyek ini dimaksudkan untuk mendukung empat klaim:

1. **AI dapat belajar dari dataset biomedis resmi pada modalitas yang sangat berbeda.**
2. **AI tetap dapat berguna walaupun data dasarnya noisy, heterogen, atau sangat sensitif terhadap kualitas.**
3. **AI menjadi jauh lebih kredibel jika dipasangkan dengan validasi yang kuat, kalibrasi, interpretabilitas, dan metrik yang relevan secara klinis.**
4. **AI dapat membawa harapan praktis dalam layanan kesehatan dengan membantu klinisi melakukan skrining lebih dini, meninjau hasil lebih cepat, dan memprioritaskan kasus berisiko tinggi dengan lebih efektif.**

Karena itu, repositori ini sebaiknya dibaca sebagai **ruang proof-of-concept submission Dawnbringer untuk IDSC 2026**. Folder-folder di dalamnya adalah studi teknis individual, tetapi bersama-sama membangun satu argumen yang lebih besar.

---

### Sangkalan

> **Semua proyek dalam repositori ini murni untuk keperluan akademis dan penelitian.** Tidak ada model, prototipe, atau alat di repositori ini yang ditujukan untuk diagnosis klinis atau keputusan pengobatan tanpa pengawasan tenaga medis profesional yang berkualifikasi.
