# Identifikasi Jenis Sampah - Big Data Challenge Satria Data 2026

Repository ini merupakan solusi untuk kompetisi identifikasi jenis sampah, yang membedakan gambar sampah menjadi tiga kelas: `Recyclable`, `Organic`, dan `Electronic`.

Model terbaik kami menggunakan arsitektur **SigLIP 2 (Patch-Max Pooling)** yang kemudian dipasangkan dengan head **Two-Stage MLP** kustom, mencapai F1 Score yang sangat tinggi pada validation/test set. Pipeline kode dirancang agar *plug-and-play* dan siap dieksekusi dari awal (bisa berjalan di Google Colab maupun PC lokal).

## Struktur Direktori Utama
Pastikan Anda meletakkan dataset kompetisi di *root directory* (atau sesuaikan variabel `ROOT_DIR` pada notebook jika Anda menyimpannya di tempat lain).

```
├── train/                  # Kumpulan gambar training
├── test/                   # Kumpulan gambar testing
├── Preprocessing_Data/
│   ├── train_manifest.csv  # Manifest file untuk training
│   └── val_manifest.csv    # Manifest file untuk validasi
├── Reproduce_Best_Model_SigLIP.ipynb   <-- (ENTRY POINT UTAMA)
├── requirements.txt        # Library yang dibutuhkan
└── ...
```

## Persyaratan (Requirements)
Disarankan menjalankan script/notebook pada environment dengan dukungan GPU (seperti Google Colab T4/V100/A100 atau PC lokal dengan VRAM minimal 8GB).

Untuk meng-install semua dependencies:
```bash
pip install -r requirements.txt
```

## Cara Menjalankan (Reproduce Solution)

1. Buka notebook **`Reproduce_Best_Model_SigLIP.ipynb`**.
2. Secara bawaan, notebook ini sudah kami ubah untuk mendeteksi secara otomatis apakah ia dijalankan di **Google Colab** atau di **Lokal**.
   - Jika di lokal, `ROOT_DIR` akan otomatis menunjuk ke `.`.
   - Pastikan *folder dataset* (`train/` dan `test/`) serta *manifest* (`Preprocessing_Data/train_manifest.csv`) berada di jalur yang benar.
3. Jalankan notebook dari atas ke bawah (**Run All**). Notebook tersebut sudah mencakup:
   - Pembacaan konfigurasi awal.
   - Ekstraksi *embedding* SigLIP secara batch dengan augmentasi.
   - Pencarian *hyperparameter* (Opsional jika ingin membaca hasil Optuna atau menjalankan ulang).
   - *Training* pada Two-Stage MLP dengan kustomisasi *Focal Loss* dan *Weighted Sampler*.
   - Evaluasi dengan K-Fold ensemble dan Test-Time Augmentation (TTA).

## Mengenai Model Architecture (End-to-End CNN)
Jika Anda tertarik dengan pendekatan konvensional menggunakan *backbone* CNN murni (seperti `tf_efficientnetv2_m` atau `convnext_base`), Anda dapat melihat modul pendukungnya di dalam file:
- `Model_Architecture/model_architecture_lengkap.py`

File di atas berisi definisi kustom model, *loss function*, serta *data loader* yang siap di-*import* pada skrip *training* terpisah.

---
> *Repository ini sudah dibersihkan (*clean-up*) dan dikonfigurasi untuk memudahkan replikasi tanpa perlu mengubah path secara manual secara ekstensif.*
