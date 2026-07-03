"""
Model Architecture: EfficientNetV2 S
Big Data Challenge Satria Data 2026

File ini hanya berisi definisi komponen arsitektur dan pendukungnya,
tanpa fungsi training satu epoch, tanpa fungsi validasi, dan tanpa alur
eksekusi training. Fungsi training serta pemanggilannya akan dibuat pada
notebook terpisah yang mengimpor komponen dari file ini.

Bagian pertama, arsitektur model dengan backbone EfficientNetV2 S dari
timm, lengkap dengan fungsi freeze dan unfreeze untuk fine tuning dua
tahap.

Bagian kedua, penghitungan class weight otomatis dari file manifest, bukan
angka tetap.

Bagian ketiga, fungsi loss dengan class weight dan label smoothing.

Bagian keempat, optimizer AdamW dengan dukungan differential learning
rate untuk tahap kedua fine tuning, serta scheduler cosine annealing.

Bagian kelima, dataset dan dataloader sederhana berbasis manifest CSV.

Bagian keenam, EarlyStopping berbasis Macro F1 Score dengan penyimpanan
dan pemuatan kembali bobot model terbaik.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device {DEVICE}")

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.path.abspath("")

BASE_DIR = os.path.dirname(current_dir)

MANIFEST_PATH = os.path.join(BASE_DIR, "Preprocessing_Data", "train_manifest.csv")
KOLOM_PATH = "file_path"
KOLOM_LABEL = "label"

LABEL_MAP = {
    "Recyclable": 0,
    "Electronic": 1,
    "Organic": 2,
}

UKURAN_INPUT = 256
BATCH_SIZE = 32
NUM_WORKERS = 0

BEST_MODEL_STAGE1_PATH = os.path.join(BASE_DIR, "Train_model", "Best_model", "best_model_stage1.pth")
BEST_MODEL_STAGE2_PATH = os.path.join(BASE_DIR, "Train_model", "Best_model", "best_model_stage2.pth")

EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 20
LR_STAGE1 = 1e-3
LR_STAGE2_HEAD = 1e-4
LR_STAGE2_BACKBONE = 1e-5

LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_NORM = 1.0
PATIENCE_EARLY_STOPPING = 7


# ============================================================
# Bagian 1, Arsitektur Model
# ============================================================

class SampahClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(SampahClassifier, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=pretrained, num_classes=0
        )
        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out

    def freeze_backbone(self):
        """
        Stage 1 fine tuning. Membekukan seluruh parameter backbone,
        hanya melatih bagian classifier head yang baru, agar bobot
        pretrained tidak rusak akibat inisialisasi acak pada head baru.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("Backbone dibekukan. Hanya melatih classifier head.")

    def unfreeze_backbone(self):
        """
        Stage 2 fine tuning. Membuka seluruh parameter backbone untuk
        dilatih penuh, biasanya memakai learning rate jauh lebih kecil.
        """
        for param in self.parameters():
            param.requires_grad = True
        print("Backbone dibuka penuh. Melatih seluruh model.")

    def dapatkan_parameter_differential_lr(self, lr_backbone, lr_head):
        """
        Mengelompokkan parameter menjadi dua grup dengan learning rate
        berbeda. Backbone memakai learning rate lebih kecil karena sudah
        punya fitur umum dari pretraining, sementara head memakai
        learning rate lebih besar karena perlu penyesuaian lebih besar
        untuk domain baru.
        """
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": self.classifier.parameters(), "lr": lr_head},
        ]


# ============================================================
# Bagian 2, Penghitungan Class Weight Otomatis
# ============================================================

def hitung_class_counts_dari_manifest(manifest_path, kolom_label, label_map):
    """
    Menghitung jumlah gambar tiap kelas langsung dari file manifest,
    supaya tidak bergantung pada angka yang ditulis manual.
    """
    df_manifest = pd.read_csv(manifest_path)
    jumlah_per_label = df_manifest[kolom_label].value_counts()

    class_counts = []
    for label_nama in sorted(label_map, key=lambda x: label_map[x]):
        jumlah = int(jumlah_per_label.get(label_nama, 0))
        class_counts.append(jumlah)
        print(f"Kelas {label_nama}, jumlah gambar {jumlah}")

    return class_counts


# ============================================================
# Bagian 3, Fungsi Loss dengan Class Weight dan Label Smoothing
# ============================================================

def create_loss_function(class_counts, label_smoothing=LABEL_SMOOTHING):
    """
    Membuat Weighted CrossEntropyLoss dengan tambahan label smoothing.
    Label smoothing membantu model tidak terlalu percaya diri berlebihan
    pada sampel yang secara visual ambigu, seperti kasus baterai yang
    mirip kaleng pada dataset ini.
    """
    total_samples = sum(class_counts)
    num_classes = len(class_counts)

    weights = [total_samples / (num_classes * count) for count in class_counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print(f"Class weight yang digunakan {weights_tensor.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=label_smoothing)
    return criterion


# ============================================================
# Bagian 4, Optimizer dan Scheduler
# ============================================================

def setup_optimizer_scheduler_stage1(model, base_lr=LR_STAGE1, epochs=EPOCHS_STAGE1):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=base_lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    return optimizer, scheduler


def setup_optimizer_scheduler_stage2(model, lr_backbone=LR_STAGE2_BACKBONE,
                                      lr_head=LR_STAGE2_HEAD, epochs=EPOCHS_STAGE2):
    """
    Memakai differential learning rate, backbone dilatih dengan learning
    rate lebih kecil dibanding classifier head, karena backbone sudah
    punya fitur umum yang tidak perlu berubah drastis.
    """
    grup_parameter = model.dapatkan_parameter_differential_lr(lr_backbone, lr_head)
    optimizer = optim.AdamW(grup_parameter, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    return optimizer, scheduler


# ============================================================
# Bagian 5, Dataset dan DataLoader
# ============================================================

class SampahDataset(Dataset):
    """
    Dataset sederhana berbasis file manifest CSV berisi kolom path
    gambar dan label kelasnya. Augmentasi berat sudah dilakukan secara
    offline pada tahap preprocessing, jadi transform di sini hanya
    melakukan resize, normalisasi, dan augmentasi ringan tambahan.
    """

    def __init__(self, dataframe, label_map, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        baris = self.dataframe.iloc[idx]
        path_gambar = baris[KOLOM_PATH]
        label_nama = baris[KOLOM_LABEL]
        label_idx = self.label_map[label_nama]

        try:
            with Image.open(path_gambar) as img:
                # Downsample dulu jika gambar sangat besar sebelum convert RGB
                # untuk mencegah RAM crash akibat gambar Electronic yg bisa 18MB+
                lebar, tinggi = img.size
                MAX_SISI = 2048
                if max(lebar, tinggi) > MAX_SISI:
                    rasio = MAX_SISI / max(lebar, tinggi)
                    ukuran_baru = (int(lebar * rasio), int(tinggi * rasio))
                    img = img.resize(ukuran_baru, Image.BILINEAR)

                img_rgb = img.convert("RGB")

            if self.transform:
                img_tensor = self.transform(img_rgb)
            else:
                img_tensor = transforms.ToTensor()(img_rgb)

        except Exception:
            # Fallback: kembalikan tensor nol jika gambar gagal dibaca
            img_tensor = torch.zeros(3, UKURAN_INPUT, UKURAN_INPUT)

        return img_tensor, label_idx


def dapatkan_transform_train(ukuran_input=UKURAN_INPUT):
    return transforms.Compose([
        transforms.Resize((ukuran_input, ukuran_input)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def dapatkan_transform_val(ukuran_input=UKURAN_INPUT):
    return transforms.Compose([
        transforms.Resize((ukuran_input, ukuran_input)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def buat_dataloader(manifest_path, label_map, batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS, rasio_val=0.15, seed=42):
    """
    Membaca manifest, membagi data menjadi train dan validasi dengan
    stratifikasi sederhana per kelas, lalu membungkusnya ke DataLoader.
    """
    df_manifest = pd.read_csv(manifest_path)

    df_train_list = []
    df_val_list = []

    rng = np.random.default_rng(seed)

    for label_nama in df_manifest[KOLOM_LABEL].unique():
        subset = df_manifest[df_manifest[KOLOM_LABEL] == label_nama]
        subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        jumlah_val = int(len(subset) * rasio_val)
        df_val_list.append(subset.iloc[:jumlah_val])
        df_train_list.append(subset.iloc[jumlah_val:])

    df_train = pd.concat(df_train_list).reset_index(drop=True)
    df_val = pd.concat(df_val_list).reset_index(drop=True)

    print(f"Jumlah data train {len(df_train)}, jumlah data validasi {len(df_val)}")

    dataset_train = SampahDataset(df_train, label_map, transform=dapatkan_transform_train())
    dataset_val = SampahDataset(df_val, label_map, transform=dapatkan_transform_val())

    loader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    loader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return loader_train, loader_val


# ============================================================
# Bagian 6, EarlyStopping dengan Pemuatan Bobot Terbaik
# ============================================================

class EarlyStopping:
    def __init__(self, patience=PATIENCE_EARLY_STOPPING, mode="max", delta=0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score, model, save_path):
        if self.best_score is None:
            self.best_score = current_score
            self.simpan_checkpoint(current_score, model, save_path)
        else:
            if self.mode == "max":
                lebih_baik = current_score > (self.best_score + self.delta)
            else:
                lebih_baik = current_score < (self.best_score - self.delta)

            if lebih_baik:
                self.best_score = current_score
                self.counter = 0
                self.simpan_checkpoint(current_score, model, save_path)
            else:
                self.counter += 1
                print(f"EarlyStopping counter {self.counter} dari {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
                    print("Early stopping diaktifkan")

    def simpan_checkpoint(self, score, model, save_path):
        print(f"Menemukan model terbaik dengan skor baru {score:.4f}. Menyimpan model")
        torch.save(model.state_dict(), save_path)


def muat_bobot_terbaik(model, path_bobot):
    """
    Memuat kembali bobot terbaik hasil EarlyStopping, wajib dijalankan
    sebelum memulai Stage 2, karena epoch terakhir belum tentu sama
    dengan epoch dengan skor terbaik.
    """
    if os.path.exists(path_bobot):
        model.load_state_dict(torch.load(path_bobot, map_location=DEVICE, weights_only=True))
        print(f"Bobot terbaik berhasil dimuat dari {path_bobot}")
    else:
        print(f"Peringatan, file bobot {path_bobot} tidak ditemukan, model tetap memakai bobot saat ini")

    return model


# ============================================================
# Catatan penggunaan
# ============================================================
# File ini hanya berisi definisi arsitektur, loss, optimizer,
# scheduler, dataset, dan EarlyStopping. Fungsi training satu epoch,
# validasi, serta alur eksekusi Stage 1 dan Stage 2 akan dibuat pada
# notebook terpisah yang mengimpor komponen komponen dari file ini.
#
# Contoh pemakaian pada notebook training nantinya.
#
# from model_architecture_lengkap import (
#     SampahClassifier, hitung_class_counts_dari_manifest,
#     create_loss_function, setup_optimizer_scheduler_stage1,
#     setup_optimizer_scheduler_stage2, buat_dataloader,
#     EarlyStopping, muat_bobot_terbaik, DEVICE, LABEL_MAP,
#     MANIFEST_PATH,
# )