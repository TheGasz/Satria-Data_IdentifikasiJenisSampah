# Identifikasi Jenis Sampah — Big Data Challenge Satria Data 2026

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Last Commit](https://img.shields.io/github/last-commit/TheGasz/Satria-Data_IdentifikasiJenisSampah)](https://github.com/TheGasz/Satria-Data_IdentifikasiJenisSampah)
[![F1 Score](https://img.shields.io/badge/F1_Score-0.9951-brightgreen.svg)]()

Solusi untuk kompetisi klasifikasi sampah tiga kelas: **Recyclable**, **Organic**, dan **Electronic**. F1 Score terbaik yang berhasil diraih: **0.9962** pada test set kompetisi.

---

## Ide Utama

Backbone SigLIP 2 dibekukan sepenuhnya — tidak ada fine-tuning. Yang dilatih hanya MLP head kecil di atasnya. Karena satu epoch training head cuma butuh hitungan detik (bukan satu jam seperti partial fine-tuning backbone), pencarian hyperparameter lewat Optuna bisa jalan lebih banyak trial dalam waktu yang sama.

Trik utama yang bikin skor naik signifikan:

1. **Patch-Max Pooling** — selain `pooler_output` (representasi global dari MAP head SigLIP2), diambil juga hasil *max pooling* di seluruh token patch mentah (`last_hidden_state`). Kalau ada satu area kecil di gambar yang jelas-jelas recyclable tapi dikelilingi background organik, max pooling tetap "menangkap" sinyal itu — tidak ketutup rata-rata seperti pooled biasa.

2. **Concat embedding SigLIP v1 + v2** — embedding dari SigLIP v1 (1152 dim, pooled) digabung dengan embedding SigLIP v2 patchmax (2304 dim, concat pooled+patch-max), menghasilkan vektor fitur 3456 dim per gambar.

3. **Label audit** — label train dibersihkan dulu dari noise (hasil audit cosine-distance + Tomek links via `label_audit_rekap.csv`) sebelum masuk training. Embedding tidak perlu diekstrak ulang, yang berubah cuma label dan bobot per barisnya.

---

## Arsitektur

```
Gambar
  └─► SigLIP 2 (google/siglip2-so400m-patch14-384, FROZEN)
        ├─► pooler_output (1152 dim, attention-pooled global)     ─┐
        └─► last_hidden_state → max pooling (1152 dim, patch-max) ─┘ concat → 2304 dim (v2)
                                                                    +
                              SigLIP v1 pooled (1152 dim, dari cache sesi sebelumnya)
                                                                    ↓
                                                    Concat total: 3456 dim
                                                                    ↓
                               MLPProbeClassifier (Two-Stage MLP)
                               ├─ Linear(3456 → hidden_dim)
                               ├─ RMSNorm + SiLU + Dropout
                               ├─ Linear(hidden_dim → hidden_dim2)
                               ├─ RMSNorm + SiLU + Dropout
                               └─ Linear(hidden_dim2 → 3 kelas)
```

Hyperparameter `hidden_dim`, `hidden_dim2`, `dropout`, learning rate, dan bobot loss dicari lewat **Optuna**. Semua training di sini hanya menyentuh MLP head, bukan backbone.

---

## Training Details

| Komponen | Detail |
|---|---|
| Backbone | `google/siglip2-so400m-patch14-384` (frozen) |
| Embedding | SigLIP v1 (1152) + SigLIP v2 patchmax (2304) = **3456 dim** |
| Augmentasi Train | N_AUG=2 (RandomResizedCrop, HFlip, ColorJitter, GaussianBlur, Rotation) |
| Sampler | `WeightedRandomSampler` — Recyclable 2.0×, Electronic 1.0×, Organic 0.7× |
| Loss | Focal Loss dengan bobot manual per kelas |
| Validasi | GroupKFold 5-fold (foto asli dan augmentasinya selalu satu fold) |
| TTA | N_TTA=5 (1 clean + 4 augmentasi ringan) |
| Ensemble | Blend prediksi K-Fold + model full-data |
| Seed | 40 (reproducible) |

---

## Hasil Visual EDA (Exploratory Data Analysis)

Bukti visual distribusi dan pemisahan kelas dari hasil eksplorasi panjang. 

### t-SNE Embedding
Visualisasi pemisahan fitur embedding (SigLIP) dari ketiga kelas dalam representasi 2D.
![t-SNE Embedding](EDA/figures/eda_tsne_embedding.png)

### Distribusi Warna (Hue) per Kelas
Karakteristik warna hue yang menunjukkan kecenderungan warna tertentu untuk kelas organik vs. recyclable/electronic.
![Distribusi Hue](EDA/figures/eda_histogram_hue_per_kelas.png)

---

## Struktur Direktori

```
├── train/                          # Gambar training
├── test/                           # Gambar testing
├── Preprocessing_Data/
│   ├── train_manifest.csv
│   └── val_manifest.csv
├── Train_model/
│   └── Best_model/
│       └── 9962(gtasli)/
│           └── siglip_patchmax_clean.ipynb   ← ENTRY POINT UTAMA
├── Inference/
│   └── prediksi_full_test.py
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

**Persyaratan:** GPU dengan VRAM minimal 8GB (Google Colab T4/V100/A100 sudah cukup).

```bash
pip install -r requirements.txt
```

1. Buka **`Train_model/Best_model/9962(gtasli)/siglip_patchmax_clean.ipynb`**.
2. Sesuaikan `ROOT_DIR` di cell konfigurasi awal jika struktur folder berbeda.
   - Di Google Colab: arahkan ke path Google Drive kamu.
   - Di lokal: cukup ubah ke path absolut folder ini.
3. Pastikan folder `train/`, `test/`, dan manifest CSV sudah berada di path yang benar.
4. Jalankan notebook dari atas ke bawah (**Run All**). Urutan eksekusi:
   - Load konfigurasi & setup seed
   - Ekstraksi embedding SigLIP v1 dan v2 dengan caching (kalau cache sudah ada, step ini langsung skip)
   - Concat embedding + label audit
   - Pencarian hyperparameter Optuna
   - Training K-Fold + full-data
   - Evaluasi dan prediksi test set dengan TTA

> Cache embedding disimpan ke `EDA/cache_embedding_siglip/` (v1) dan `EDA/cache_embedding_siglip2_patchmax/` (v2). Kalau runtime Colab putus di tengah ekstraksi, ada mekanisme checkpoint per 2000 gambar jadi tidak perlu mulai dari nol.

---

*Catatan: model ini adalah hasil iterasi dari beberapa versi notebook sebelumnya yang sudah dihapus histori eksperimennya. Versi bersih ini (v9) adalah yang dipakai untuk submission final.*
