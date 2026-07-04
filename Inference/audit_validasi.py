import os
import sys
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Mencegah crash memori di Windows
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

sys.path.append(os.path.abspath("../Model_Architecture"))
from model_architecture_lengkap import SampahClassifier, dapatkan_transform_val, UKURAN_INPUT

ROOT = os.path.abspath("..")
VAL_MANIFEST_PATH = os.path.join(ROOT, "Preprocessing_Data", "val_manifest.csv")
EDA_CSV_PATH = os.path.join(ROOT, "EDA", "eda_deteksi_full_frame.csv")
MODEL_PATH = os.path.join(ROOT, "Train_model", "Best_model", "best_model_final.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX_TO_LABEL = {0: "Recyclable", 1: "Electronic", 2: "Organic"}

print("Memuat dataset validasi & data EDA...")
val_df = pd.read_csv(VAL_MANIFEST_PATH)
eda_df = pd.read_csv(EDA_CSV_PATH)

print("Memuat model...")
model = SampahClassifier(num_classes=3, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

transform_val = dapatkan_transform_val(UKURAN_INPUT)

def proses_gambar(img_path):
    try:
        with Image.open(img_path) as img:
            lebar, tinggi = img.size
            MAX_SISI = 2048 
            if max(lebar, tinggi) > MAX_SISI:
                rasio = MAX_SISI / max(lebar, tinggi)
                ukuran_baru = (int(lebar * rasio), int(tinggi * rasio))
                img = img.resize(ukuran_baru, Image.BILINEAR)
            img_rgb = img.convert("RGB")
        img_tensor = transform_val(img_rgb)
        return img_tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error membaca gambar {img_path}: {e}")
        return torch.zeros(1, 3, UKURAN_INPUT, UKURAN_INPUT)

hasil_audit = []

print(f"Melakukan inferensi pada {len(val_df)} data validasi...")
with torch.no_grad():
    torch.cuda.empty_cache()
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df)):
        fpath = row['file_path']
        label_asli = row['label']
        nama_file = os.path.basename(fpath)
        
        img_tensor = proses_gambar(fpath).to(DEVICE)
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        
        pred_idx = torch.argmax(probs).item()
        pred_label = IDX_TO_LABEL[pred_idx]
        confidence = probs[pred_idx].item()
        
        # Simpan jika prediksinya salah dan diprediksi sebagai Organic dengan confidence > 0.85
        if pred_label == "Organic" and label_asli != "Organic" and confidence > 0.85:
            hasil_audit.append({
                "nama_file": nama_file,
                "label_asli": label_asli,
                "prediksi": pred_label,
                "confidence": confidence,
                "prob_organic": probs[2].item(),
                "prob_recyclable": probs[0].item(),
                "prob_electronic": probs[1].item()
            })

audit_df = pd.DataFrame(hasil_audit)
print(f"\nDitemukan {len(audit_df)} misklasifikasi 'Organic' dengan confidence tinggi (>0.85).")

if len(audit_df) > 0:
    # Gabungkan dengan data EDA
    merged_df = audit_df.merge(eda_df, on="nama_file", how="left")
    
    # Kriteria "full frame dan warna cerah beragam":
    # kategori == 'full_frame' DAN (skor_variansi tinggi ATAU skor_entropi tinggi)
    kriteria_pola = merged_df[
        (merged_df['kategori'] == 'full_frame') & 
        ((merged_df['skor_variansi'] > 0.5) | (merged_df['skor_entropi'] > 0.7))
    ]
    
    proporsi = len(kriteria_pola) / len(merged_df) * 100
    
    print("\n=== HASIL AUDIT ===")
    print(f"Total gambar bukan Organic yang keliru diprediksi Organic (Conf > 0.85) : {len(merged_df)} gambar")
    print(f"Berapa yang sesuai pola 'Full-Frame & Warna Beragam'?                 : {len(kriteria_pola)} gambar")
    print(f"Proporsi error akibat kelemahan pola ini                                : {proporsi:.2f} %")
    
    output_audit_csv = "audit_organic_misclassification.csv"
    merged_df.to_csv(output_audit_csv, index=False)
    print(f"\nData detail pola disimpan ke: {output_audit_csv}")
else:
    print("Tidak ditemukan misklasifikasi Organic dengan confidence tinggi di set validasi.")
