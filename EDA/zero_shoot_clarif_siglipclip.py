"""
Zero Shot Classification memakai CLIP dan SigLIP
Big Data Challenge Satria Data 2026

Script ini menjalankan zero shot classification yang sesungguhnya, yaitu
mencocokkan gambar dengan prompt teks tiap kelas lewat CLIP dan SigLIP,
bukan sekadar melihat posisi embedding di ruang t-SNE.

Evaluasi dilakukan pada dua kelompok yang sama seperti eksperimen
sebelumnya, yaitu 68 kasus Recyclable yang salah diprediksi Organic oleh
model utama, dan 1823 kasus Organic yang sudah benar diprediksi Organic
oleh model utama. Dampak terhadap Macro F1 tiga kelas dihitung secara
penuh lewat confusion matrix, bukan cuma angka rescued dan flip mentah,
supaya kesimpulannya tidak menyesatkan seperti kejadian pada meta model
sebelumnya.
"""

import os
import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ============================================================
# Konfigurasi Path
# ============================================================
# Script ini diasumsikan berada di folder EDA, sama seperti
# eda_clip_siglip_vs_resnet.py, sehingga BASE_DIR mengarah ke folder
# root project (BDC 2026/), bukan folder EDA itu sendiri.
#
# __file__ cuma tersedia kalau kode dijalankan sebagai file .py murni.
# Kalau dijalankan di sel Jupyter Notebook, __file__ tidak ada, jadi
# perlu fallback memakai direktori kerja saat ini.

try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path(os.path.abspath("")).resolve()

BASE_DIR = current_dir.parent
OUTPUT_DIR = current_dir

print(f"Direktori script atau notebook sekarang, {current_dir}")
print(f"BASE_DIR terdeteksi, {BASE_DIR}")

# Sesuaikan baris berikut kalau lokasi file validation_postprocess_results.csv
# kamu ternyata berbeda. Berdasarkan pola project kamu, kemungkinan besar file
# ini berada di folder Inference, sama seperti audit_organic_misclassification.csv
# dan file file hasil meta_model_koreksi sebelumnya.
VALIDATION_POSTPROCESS_CSV = BASE_DIR / "Feature Engineering" / "validation_postprocess_results.csv"

VAL_MANIFEST_PATH = BASE_DIR / "Preprocessing_Data" / "val_manifest.csv"
KOLOM_PATH_MANIFEST = "file_path"

OUTPUT_CSV = OUTPUT_DIR / "zero_shot_clip_siglip_hasil.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device {DEVICE}")

# ============================================================
# Prompt Teks Tiap Kelas
# ============================================================
# Beberapa variasi kalimat per kelas digabung lewat rata rata embedding
# teksnya (prompt ensembling), supaya hasil lebih stabil dibanding cuma
# memakai satu kalimat tunggal.

PROMPT_PER_KELAS = {
    "Recyclable": [
        "a photo of recyclable plastic or glass waste",
        "a photo of a plastic bottle, can, or piece of cardboard",
        "a photo of recyclable packaging material",
    ],
    "Electronic": [
        "a photo of electronic waste",
        "a photo of a broken phone, laptop, or cable",
        "a photo of discarded electronic devices",
    ],
    "Organic": [
        "a photo of organic biodegradable waste",
        "a photo of fruit, vegetable, or food scraps",
        "a photo of leaves or plant material",
    ],
}

DAFTAR_KELAS = list(PROMPT_PER_KELAS.keys())


def ekstrak_tensor_dari_output(out):
    """
    Beberapa versi library transformers mengembalikan tensor polos dari
    get_text_features dan get_image_features, versi lain mengembalikan
    objek BaseModelOutputWithPooling. Fungsi ini menangani keduanya agar
    script tetap berjalan di versi transformers manapun.
    """
    if torch.is_tensor(out):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0, :]
    raise TypeError(f"Tidak tahu cara mengambil tensor dari tipe, {type(out)}")


# ============================================================
# Memuat Model CLIP dan SigLIP
# ============================================================

def muat_model_clip():
    from transformers import CLIPModel, CLIPProcessor
    print("Memuat CLIP")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model.to(DEVICE)
    return model, processor


def muat_model_siglip():
    from transformers import SiglipModel, SiglipProcessor
    print("Memuat SigLIP")
    processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
    model.eval()
    model.to(DEVICE)
    return model, processor


def hitung_embedding_teks_per_kelas(model, processor):
    """
    Menghitung embedding teks untuk tiap kelas, dirata ratakan dari
    beberapa variasi kalimat (prompt ensembling), lalu dinormalisasi.
    """
    embedding_kelas = {}

    with torch.no_grad():
        for kelas, daftar_prompt in PROMPT_PER_KELAS.items():
            inputs = processor(text=daftar_prompt, return_tensors="pt", padding=True).to(DEVICE)
            fitur_teks = ekstrak_tensor_dari_output(model.get_text_features(**inputs))
            fitur_teks = fitur_teks / fitur_teks.norm(dim=-1, keepdim=True)
            fitur_rata = fitur_teks.mean(dim=0)
            fitur_rata = fitur_rata / fitur_rata.norm()
            embedding_kelas[kelas] = fitur_rata

    return embedding_kelas


def prediksi_zero_shot_satu_gambar(model, processor, embedding_kelas, img_path):
    """
    Menjalankan zero shot classification satu gambar, mengembalikan
    label prediksi beserta skor kemiripan (cosine similarity setelah
    softmax) terhadap tiap kelas.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error membaca gambar {img_path}, pesan {e}")
        return None, {kelas: 0.0 for kelas in DAFTAR_KELAS}

    with torch.no_grad():
        inputs = processor(images=[img], return_tensors="pt").to(DEVICE)
        fitur_gambar = ekstrak_tensor_dari_output(model.get_image_features(**inputs))
        fitur_gambar = fitur_gambar / fitur_gambar.norm(dim=-1, keepdim=True)

        skor_per_kelas = {}
        for kelas, fitur_teks in embedding_kelas.items():
            skor_per_kelas[kelas] = (fitur_gambar[0] @ fitur_teks).item()

        logits = torch.tensor([skor_per_kelas[k] for k in DAFTAR_KELAS])
        probs = torch.softmax(logits * 100, dim=0)
        probs_dict = {k: probs[i].item() for i, k in enumerate(DAFTAR_KELAS)}

    label_prediksi = max(probs_dict, key=probs_dict.get)
    return label_prediksi, probs_dict


# ============================================================
# Membangun Peta Nama File ke Path dan Mengambil Kelompok Target
# ============================================================

def bangun_peta_nama_file_ke_path(manifest_path, kolom_path):
    df_manifest = pd.read_csv(manifest_path)
    peta = {}
    for path_lengkap in df_manifest[kolom_path]:
        nama_file = os.path.basename(path_lengkap)
        peta[nama_file] = path_lengkap
    print(f"Peta nama file ke path selesai dibangun, total {len(peta)} entri")
    return peta


def ambil_kelompok_kasus_target(validation_csv):
    df_audit = pd.read_csv(validation_csv)

    kelompok_salah = df_audit[
        (df_audit["label"] == "Recyclable") & (df_audit["pred_label"] == "Organic")
    ].copy()
    kelompok_salah["kelompok"] = "recyclable_salah_jadi_organic"

    kelompok_benar = df_audit[
        (df_audit["label"] == "Organic") & (df_audit["pred_label"] == "Organic")
    ].copy()
    kelompok_benar["kelompok"] = "organic_sudah_benar"

    print(f"Jumlah kasus Recyclable salah diprediksi Organic, total {len(kelompok_salah)}")
    print(f"Jumlah kasus Organic sudah benar diprediksi Organic, total {len(kelompok_benar)}")

    return pd.concat([kelompok_salah, kelompok_benar], ignore_index=True)


# ============================================================
# Menjalankan Zero Shot untuk Satu Model pada Seluruh Kasus Target
# ============================================================

def jalankan_zero_shot_untuk_model(nama_model, model, processor, df_target, peta_path):
    embedding_kelas = hitung_embedding_teks_per_kelas(model, processor)

    hasil = []
    for _, baris in tqdm(df_target.iterrows(), total=len(df_target), desc=f"Zero shot {nama_model}"):
        nama_file = baris["nama_file"]
        path_lengkap = peta_path.get(nama_file)
        if path_lengkap is None:
            continue

        label_pred, probs = prediksi_zero_shot_satu_gambar(model, processor, embedding_kelas, path_lengkap)

        hasil.append({
            "nama_file": nama_file,
            "label_asli": baris["label"],
            "kelompok": baris["kelompok"],
            "pred_asli_model_utama": baris["pred_label"],
            f"pred_{nama_model}": label_pred,
            f"prob_recyclable_{nama_model}": round(probs["Recyclable"], 4),
            f"prob_electronic_{nama_model}": round(probs["Electronic"], 4),
            f"prob_organic_{nama_model}": round(probs["Organic"], 4),
        })

    return pd.DataFrame(hasil)


# ============================================================
# Menghitung Dampak Macro F1 Sebenarnya
# ============================================================

def macro_f1_dari_confmat(recyclable, electronic, organic):
    def f1(tp, fn, fp):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return f
    return (f1(*recyclable) + f1(*electronic) + f1(*organic)) / 3


def hitung_dampak_macro_f1(df_hasil, kolom_prediksi):
    """
    Menghitung Macro F1 sebenarnya kalau prediksi model utama pada kedua
    kelompok target diganti dengan prediksi zero shot, memakai baseline
    confusion matrix penuh dari data validasi kamu sebelumnya.
    """
    kelompok_salah = df_hasil[df_hasil["kelompok"] == "recyclable_salah_jadi_organic"]
    kelompok_benar = df_hasil[df_hasil["kelompok"] == "organic_sudah_benar"]

    rescued = (kelompok_salah[kolom_prediksi] == "Recyclable").sum()
    false_flip = (kelompok_benar[kolom_prediksi] != "Organic").sum()

    tp_rec = 1421 + rescued
    fn_rec = 9 + (68 - rescued)
    fp_rec = 6 + 57 + false_flip
    rec = (tp_rec, fn_rec, fp_rec)

    tp_org = 1823 - false_flip
    fn_org = (57 + false_flip) + 1
    fp_org = (68 - rescued) + 7
    org = (tp_org, fn_org, fp_org)

    ele = (576, 6 + 7, 9 + 1)

    macro_f1 = macro_f1_dari_confmat(rec, ele, org)
    return rescued, false_flip, macro_f1


# ============================================================
# Alur Utama
# ============================================================

def main():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        print(f"VRAM terpakai sebelum mulai, {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")

    peta_path = bangun_peta_nama_file_ke_path(VAL_MANIFEST_PATH, KOLOM_PATH_MANIFEST)
    df_target = ambil_kelompok_kasus_target(VALIDATION_POSTPROCESS_CSV)

    baseline_macro_f1 = macro_f1_dari_confmat(
        (1421, 9 + 68, 6 + 57), (576, 6 + 7, 9 + 1), (1823, 57 + 1, 68 + 7)
    )
    print(f"\nMacro F1 baseline tanpa koreksi apapun, nilai {baseline_macro_f1:.4f}")

    df_gabungan = df_target.copy()

    print("\nMenjalankan zero shot classification memakai CLIP")
    model_clip, processor_clip = muat_model_clip()
    df_clip = jalankan_zero_shot_untuk_model("clip", model_clip, processor_clip, df_target, peta_path)
    df_gabungan = pd.merge(df_gabungan, df_clip.drop(columns=["label_asli", "kelompok", "pred_asli_model_utama"]), on="nama_file", how="left")

    print("Membebaskan VRAM CLIP sebelum memuat SigLIP")
    del model_clip
    del processor_clip
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print("\nMenjalankan zero shot classification memakai SigLIP")
    model_siglip, processor_siglip = muat_model_siglip()
    df_siglip = jalankan_zero_shot_untuk_model("siglip", model_siglip, processor_siglip, df_target, peta_path)
    df_gabungan = pd.merge(df_gabungan, df_siglip.drop(columns=["label_asli", "kelompok", "pred_asli_model_utama"]), on="nama_file", how="left")

    print("Membebaskan VRAM SigLIP")
    del model_siglip
    del processor_siglip
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    df_gabungan.to_csv(OUTPUT_CSV, index=False)
    print(f"\nHasil lengkap disimpan pada {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print("RINGKASAN DAMPAK MACRO F1 SEBENARNYA")
    print("=" * 60)
    print(f"Baseline (tanpa koreksi apapun), Macro F1 {baseline_macro_f1:.4f}")

    for nama_model, kolom in [("CLIP", "pred_clip"), ("SigLIP", "pred_siglip")]:
        rescued, false_flip, macro_f1 = hitung_dampak_macro_f1(df_gabungan, kolom)
        delta = macro_f1 - baseline_macro_f1
        status = "LEBIH BAIK dari baseline" if delta > 0 else "masih di bawah baseline"
        print(f"\n{nama_model}, kalau dipakai menggantikan prediksi model utama pada dua kelompok target")
        print(f"  Recyclable terselamatkan, jumlah {rescued} dari 68")
        print(f"  Organic yang jadi salah, jumlah {false_flip} dari 1823")
        print(f"  Macro F1 hasil, nilai {macro_f1:.4f}, delta {delta:+.4f}, {status}")


if __name__ == "__main__":
    main()