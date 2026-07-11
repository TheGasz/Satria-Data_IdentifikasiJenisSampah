"""
Pipeline Training Penuh
Big Data Challenge Satria Data 2026
Script ini menjalankan training penuh mulai dari Stage 1 (backbone
dibekukan) sampai Stage 2 (backbone dibuka penuh), hingga menghasilkan
satu file model terbaik berdasarkan Macro F1 Score pada data validasi.
Komponen arsitektur, loss, optimizer, dataset, dan EarlyStopping diimpor
dari file model_architecture_lengkap.py, jadi pastikan kedua file berada
pada folder yang sama.
Penyesuaian khusus untuk GPU dengan VRAM terbatas (RTX 3050, 4 GB) sudah
diterapkan di sini, yaitu batch size kecil dikombinasikan dengan gradient
accumulation, mixed precision training, dan gradient clipping.

Perubahan dari versi sebelumnya:
  - Focal Loss menggantikan WeightedCrossEntropyLoss (dari model_architecture_lengkap)
  - Mixup diaktifkan di Stage 2 untuk paksa model belajar konsep, bukan menghafal tekstur
  - Augmentasi per-kelas: Recyclable mendapat transform paling agresif (dari model_architecture_lengkap)
  - Backbone bisa dikonfigurasi via konstanta BACKBONE_NAME di model_architecture_lengkap
"""
import os
import sys
# Wajib: cegah CUDA memory fragmentation crash (ExitCode 3221225477) di Windows
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.append(os.path.abspath("../../Model_Architecture"))
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
from model_architecture_lengkap import (
    SampahClassifier,
    hitung_class_counts_dari_manifest,
    create_loss_function,
    setup_optimizer_scheduler_stage1,
    setup_optimizer_scheduler_stage2,
    buat_dataloader,
    EarlyStopping,
    muat_bobot_terbaik,
    DEVICE,
    LABEL_MAP,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    KOLOM_LABEL,
    BEST_MODEL_STAGE1_PATH,
    BEST_MODEL_STAGE2_PATH,
    EPOCHS_STAGE1,
    EPOCHS_STAGE2,
    PATIENCE_EARLY_STOPPING,
    GRADIENT_CLIP_NORM,
)
# Override path manifest agar mengarah ke folder yang benar
TRAIN_MANIFEST_PATH = os.path.abspath("../../Preprocessing_Data/train_manifest.csv")
VAL_MANIFEST_PATH = os.path.abspath("../../Preprocessing_Data/val_manifest.csv")
print(f"Setup selesai. Device: {DEVICE}")
print(f"Train Manifest path: {TRAIN_MANIFEST_PATH}")
print(f"Val Manifest path: {VAL_MANIFEST_PATH}")
print(f"Train Manifest exists: {os.path.exists(TRAIN_MANIFEST_PATH)}")
print(f"Val Manifest exists: {os.path.exists(VAL_MANIFEST_PATH)}")


# ============================================================
# Konfigurasi GPU — batch size besar karena VRAM RTX 3050 masih lega
# ============================================================
# VRAM terpakai sebelumnya hanya 101MB dari 4000MB → bottleneck ada di CPU,
# bukan GPU. Naikkan batch size untuk bikin GPU selalu sibuk.
BATCH_SIZE_AKTUAL = 32     # dari 8 → GPU utilization naik drastis
GRADIENT_ACCUMULATION_STEPS = 1  # tidak perlu akumulasi, batch sudah besar
NUM_WORKERS = 0            # parallel data loading, CPU prefetch saat GPU sibuk
PIN_MEMORY = False          # transfer CPU→GPU lebih cepat via pinned memory
# Mixup hanya aktif di Stage 2 (backbone terbuka penuh).
# Stage 1 tidak pakai Mixup karena head belum konvergen dan
# label campuran Mixup bisa membingungkan gradient di awal training.
MIXUP_ALPHA = 0.2          # Distribusi Beta; 0.2 adalah nilai standar yang stabil
USE_MIXUP_STAGE2 = True    # Set False untuk menonaktifkan Mixup di Stage 2
# Path simpan model final di folder yang sama dengan notebook ini
NOTEBOOK_DIR = os.path.abspath("./")
BEST_MODEL_FINAL_PATH = os.path.join(NOTEBOOK_DIR, "best_model_final.pth")
print(f"Batch size aktual       : {BATCH_SIZE_AKTUAL}")
print(f"Gradient accumulation   : {GRADIENT_ACCUMULATION_STEPS} steps (tidak aktif)")
print(f"Efektif setara batch    : {BATCH_SIZE_AKTUAL * GRADIENT_ACCUMULATION_STEPS}")
print(f"Num workers             : {NUM_WORKERS}")
print(f"Pin memory              : {PIN_MEMORY}")
print(f"Mixup Stage 2           : {'aktif (alpha=' + str(MIXUP_ALPHA) + ')' if USE_MIXUP_STAGE2 else 'nonaktif'}")
print(f"Model final disimpan di : {BEST_MODEL_FINAL_PATH}")


# ============================================================
# Fungsi Mixup
# ============================================================

def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """
    Mencampur dua batch gambar dan label secara proporsional sesuai
    lambda yang diambil dari distribusi Beta(alpha, alpha).

    Mixup memaksa model belajar representasi linear antar kelas,
    sehingga tidak bergantung pada tekstur atau warna spesifik satu kelas.
    Ini sangat berguna untuk mengatasi over-reliance pada ciri visual
    kelas Recyclable yang seringkali ambigu (mirip Electronic/Organic).

    Mengembalikan:
      x_mix     : gambar campuran
      y_a, y_b  : label asli dari dua sampel yang dicampur
      lam       : bobot campuran (0-1)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    x_mix = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]

    return x_mix, y_a, y_b, lam


def mixup_criterion(criterion, output, y_a, y_b, lam):
    """
    Menghitung loss Mixup sebagai kombinasi linear dari dua loss,
    sesuai proporsi lambda yang dipakai saat mencampur gambar.
    Kompatibel dengan FocalLoss maupun CrossEntropyLoss biasa.
    """
    return lam * criterion(output, y_a) + (1 - lam) * criterion(output, y_b)


# ============================================================
# Fungsi Training Satu Epoch dengan Gradient Accumulation
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler,
                     gradient_clip_norm=GRADIENT_CLIP_NORM,
                     accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                     use_mixup=False, mixup_alpha=MIXUP_ALPHA):
    """
    Menjalankan satu epoch training memakai mixed precision dan gradient
    accumulation. Dengan batch size aktual kecil (misalnya 8) dan
    accumulation_steps 4, efeknya setara dengan batch size 32 dari sisi
    stabilitas gradien, tanpa membutuhkan VRAM sebesar batch size 32.

    Jika use_mixup=True, setiap batch dicampur menggunakan Mixup sebelum
    diumpankan ke model. Ini direkomendasikan hanya untuk Stage 2.
    """
    model.train()
    total_loss = 0.0
    jumlah_batch = 0
    use_amp = (DEVICE.type == "cuda")
    optimizer.zero_grad()
    for idx_batch, (gambar, label) in enumerate(loader):
        gambar = gambar.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        if use_mixup:
            gambar, label_a, label_b, lam = mixup_data(gambar, label, alpha=mixup_alpha)

        with torch.amp.autocast(DEVICE.type, enabled=use_amp):
            output = model(gambar)
            if use_mixup:
                loss = mixup_criterion(criterion, output, label_a, label_b, lam)
            else:
                loss = criterion(output, label)
            loss_dibagi = loss / accumulation_steps

        scaler.scale(loss_dibagi).backward()
        langkah_terakhir = (idx_batch + 1) == len(loader)
        if (idx_batch + 1) % accumulation_steps == 0 or langkah_terakhir:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss.item()
        jumlah_batch += 1
        # Progress setiap 10 batch agar tidak terkesan stuck
        if (idx_batch + 1) % 10 == 0 or langkah_terakhir:
            vram_mb = torch.cuda.memory_allocated(0) / 1024**2 if DEVICE.type == "cuda" else 0
            print(f"  [{idx_batch+1:4d}/{len(loader)}] loss={loss.item():.4f}  VRAM={vram_mb:.0f}MB",
                  flush=True)
    rata_loss = total_loss / jumlah_batch
    return rata_loss

# ============================================================
# Fungsi Validasi Satu Epoch
# ============================================================
def validasi_satu_epoch(model, loader, criterion):
    """
    Menjalankan validasi satu epoch, mengembalikan rata rata loss serta
    Macro F1 Score yang menjadi metrik utama kompetisi ini.
    Validasi tidak memakai Mixup karena kita ingin evaluasi murni
    performa model terhadap sampel asli tanpa campuran.
    """
    model.eval()
    total_loss = 0.0
    jumlah_batch = 0
    use_amp = (DEVICE.type == "cuda")
    seluruh_prediksi = []
    seluruh_label = []
    with torch.no_grad():
        for gambar, label in loader:
            gambar = gambar.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            with torch.amp.autocast(DEVICE.type, enabled=use_amp):
                output = model(gambar)
                loss = criterion(output, label)
            total_loss += loss.item()
            jumlah_batch += 1
            prediksi = torch.argmax(output, dim=1)
            seluruh_prediksi.extend(prediksi.cpu().numpy().tolist())
            seluruh_label.extend(label.cpu().numpy().tolist())
    rata_loss = total_loss / jumlah_batch
    macro_f1 = f1_score(seluruh_label, seluruh_prediksi, average="macro")
    return rata_loss, macro_f1, seluruh_label, seluruh_prediksi

def plot_training_history(history, stage_name, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title(f'{stage_name} - Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Macro F1
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_macro_f1'], label='Val Macro F1', color='green', marker='o')
    plt.title(f'{stage_name} - Macro F1 per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot training {stage_name} disimpan di: {save_path}")

print("Fungsi train_one_epoch, validasi_satu_epoch, dan plot_training_history siap.")


# ============================================================
# Alur Stage 1 — backbone dibekukan, tanpa Mixup
# ============================================================
def jalankan_stage1(model, loader_train, loader_val, class_counts):
    print("\nMemulai Stage 1 — backbone dibekukan, hanya melatih head")
    print("Mixup TIDAK dipakai di Stage 1 agar head bisa konvergen lebih cepat.")
    model.freeze_backbone()
    criterion = create_loss_function(class_counts)
    optimizer, scheduler = setup_optimizer_scheduler_stage1(model)
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_amp)
    early_stopping = EarlyStopping(patience=PATIENCE_EARLY_STOPPING, mode="max")
    
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
    for epoch in range(1, EPOCHS_STAGE1 + 1):
        waktu_mulai = time.time()
        print(f"\n--- Stage 1 Epoch {epoch}/{EPOCHS_STAGE1} ---")
        # Stage 1 tanpa Mixup
        rata_loss_train = train_one_epoch(
            model, loader_train, criterion, optimizer, scaler,
            use_mixup=False
        )
        rata_loss_val, macro_f1_val, _, _ = validasi_satu_epoch(model, loader_val, criterion)
        scheduler.step()
        
        history['train_loss'].append(rata_loss_train)
        history['val_loss'].append(rata_loss_val)
        history['val_macro_f1'].append(macro_f1_val)
        
        durasi = time.time() - waktu_mulai
        print(
            f"HASIL  loss_train={rata_loss_train:.4f}  loss_val={rata_loss_val:.4f}  "
            f"macro_f1={macro_f1_val:.4f}  waktu={durasi:.1f}s"
        )
        early_stopping(macro_f1_val, model, BEST_MODEL_STAGE1_PATH)
        if early_stopping.early_stop:
            print("Stage 1 dihentikan lebih awal")
            break
    model = muat_bobot_terbaik(model, BEST_MODEL_STAGE1_PATH)
    return model, history

# ============================================================
# Alur Stage 2 — backbone dibuka penuh, Mixup aktif
# ============================================================
def jalankan_stage2(model, loader_train, loader_val, class_counts):
    print("\nMemulai Stage 2 — backbone dibuka penuh")
    if USE_MIXUP_STAGE2:
        print(f"Mixup AKTIF di Stage 2 dengan alpha={MIXUP_ALPHA}")
    model.unfreeze_backbone()
    criterion = create_loss_function(class_counts)
    optimizer, scheduler = setup_optimizer_scheduler_stage2(model)
    use_amp = (DEVICE.type == "cuda")
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_amp)
    early_stopping = EarlyStopping(patience=PATIENCE_EARLY_STOPPING, mode="max")
    
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}
    for epoch in range(1, EPOCHS_STAGE2 + 1):
        waktu_mulai = time.time()
        print(f"\n--- Stage 2 Epoch {epoch}/{EPOCHS_STAGE2} ---")
        # Stage 2 dengan Mixup (jika diaktifkan)
        rata_loss_train = train_one_epoch(
            model, loader_train, criterion, optimizer, scaler,
            use_mixup=USE_MIXUP_STAGE2, mixup_alpha=MIXUP_ALPHA
        )
        rata_loss_val, macro_f1_val, _, _ = validasi_satu_epoch(model, loader_val, criterion)
        scheduler.step()
        
        history['train_loss'].append(rata_loss_train)
        history['val_loss'].append(rata_loss_val)
        history['val_macro_f1'].append(macro_f1_val)
        
        durasi = time.time() - waktu_mulai
        print(
            f"HASIL  loss_train={rata_loss_train:.4f}  loss_val={rata_loss_val:.4f}  "
            f"macro_f1={macro_f1_val:.4f}  waktu={durasi:.1f}s"
        )
        early_stopping(macro_f1_val, model, BEST_MODEL_STAGE2_PATH)
        if early_stopping.early_stop:
            print("Stage 2 dihentikan lebih awal")
            break
    model = muat_bobot_terbaik(model, BEST_MODEL_STAGE2_PATH)
    return model, criterion, history
print("Fungsi jalankan_stage1 dan jalankan_stage2 siap.")


# ============================================================
# Alur Utama — Jalankan cell ini untuk memulai training
# ============================================================
def main():
    torch.cuda.empty_cache()  # Bersihkan VRAM sebelum mulai
    print(f"Menggunakan device      : {DEVICE}")
    print(f"Batch size aktual       : {BATCH_SIZE_AKTUAL}")
    print(f"Gradient accumulation   : {GRADIENT_ACCUMULATION_STEPS} steps")
    print(f"Efektif setara batch    : {BATCH_SIZE_AKTUAL * GRADIENT_ACCUMULATION_STEPS}")
    print()
    class_counts = hitung_class_counts_dari_manifest(TRAIN_MANIFEST_PATH, KOLOM_LABEL, LABEL_MAP)
    print("\nMemuat DataLoader...")
    loader_train, loader_val = buat_dataloader(
        TRAIN_MANIFEST_PATH, VAL_MANIFEST_PATH, LABEL_MAP,
        batch_size=BATCH_SIZE_AKTUAL,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    print("\nMemuat model EfficientNetV2-S (pretrained)...")
    model = SampahClassifier(num_classes=3, pretrained=True)
    model.to(DEVICE)
    vram_mb = torch.cuda.memory_allocated(0) / 1024**2
    print(f"Model siap. VRAM terpakai: {vram_mb:.1f} MB")
    print()
    
    # ---- Stage 1 — tanpa Mixup ----
    model, history_stage1 = jalankan_stage1(model, loader_train, loader_val, class_counts)
    plot_training_history(history_stage1, "Stage 1", os.path.join(NOTEBOOK_DIR, "plot_stage1.png"))
    
    # ---- Stage 2 — dengan Mixup + Focal Loss + Recyclable augmentation ----
    model, criterion_terakhir, history_stage2 = jalankan_stage2(model, loader_train, loader_val, class_counts)
    plot_training_history(history_stage2, "Stage 2", os.path.join(NOTEBOOK_DIR, "plot_stage2.png"))

    # ---- Simpan model final ----
    torch.save(model.state_dict(), BEST_MODEL_FINAL_PATH)
    print(f"\nModel terbaik akhir disimpan pada {BEST_MODEL_FINAL_PATH}")
    
    # ---- Evaluasi akhir ----
    _, macro_f1_final, label_final, prediksi_final = validasi_satu_epoch(model, loader_val, criterion_terakhir)
    nama_kelas = sorted(LABEL_MAP, key=lambda x: LABEL_MAP[x])
    print(f"\nMacro F1 Score akhir: {macro_f1_final:.4f}")
    print("\nClassification Report:")
    print(classification_report(label_final, prediksi_final, target_names=nama_kelas))
    print("\nTraining selesai!")
    print(f"  Stage 1 terbaik : {BEST_MODEL_STAGE1_PATH}")
    print(f"  Stage 2 terbaik : {BEST_MODEL_STAGE2_PATH}")
    print(f"  Model final     : {BEST_MODEL_FINAL_PATH}")

    # --- PENTING UNTUK WINDOWS ---
    # Membersihkan memory GPU dan object secara eksplisit untuk mencegah 
    # crash "Unhandled exception caught in c10/util/AbortHandler.h" saat Python exit
    del model
    del criterion_terakhir
    del loader_train
    del loader_val
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()



