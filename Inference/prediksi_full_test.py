import os
import sys
import glob
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
TEST_DIR = os.path.join(ROOT, "test")
MODEL_PATH = os.path.join(ROOT, "Train_model", "Best_model", "best_model_final.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IDX_TO_LABEL = {0: "Recyclable", 1: "Electronic", 2: "Organic"}

print("Memuat arsitektur model...")
model = SampahClassifier(num_classes=3, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.to(DEVICE)
model.eval()

transform_val = dapatkan_transform_val(UKURAN_INPUT)

def proses_gambar_test(img_path):
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

test_files = glob.glob(os.path.join(TEST_DIR, "*.jpg"))
# Sortir numerik
test_files = sorted(test_files, key=lambda x: int(os.path.basename(x).split('.')[0]))

hasil_prediksi = []

print(f"Melakukan inferensi pada seluruh {len(test_files)} data test...")
with torch.no_grad():
    torch.cuda.empty_cache()
    for fpath in tqdm(test_files, desc="Inference"):
        filename = os.path.basename(fpath)
        img_tensor = proses_gambar_test(fpath).to(DEVICE)
        
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        
        pred_idx = torch.argmax(probs).item()
        pred_label = IDX_TO_LABEL[pred_idx]
        
        hasil_prediksi.append({
            "File": filename,
            "Prob_Recyclable": round(probs[0].item(), 4),
            "Prob_Electronic": round(probs[1].item(), 4),
            "Prob_Organic": round(probs[2].item(), 4),
            "Prediksi": pred_label
        })

df_hasil = pd.DataFrame(hasil_prediksi)
print("\nInference selesai!\n")

output_csv = "prediksi_full_test.csv"
df_hasil.to_csv(output_csv, index=False)
print(f"Hasil prediksi {len(test_files)} data telah disimpan ke {output_csv}")

print("\nDistribusi Prediksi Test Set:")
print(df_hasil['Prediksi'].value_counts())
