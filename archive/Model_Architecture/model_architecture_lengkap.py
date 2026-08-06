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

Bagian ketiga, fungsi loss dengan class weight, label smoothing, dan
dukungan Focal Loss untuk fokus pada hard example secara otomatis.

Bagian keempat, optimizer AdamW dengan dukungan differential learning
rate untuk tahap kedua fine tuning, serta scheduler cosine annealing.

Bagian kelima, dataset dan dataloader dengan augmentasi per-kelas:
- transform biasa     : Organic & Electronic (ringan)
- hard_transform      : Recyclable full_frame hard example (sedang)
- recyclable_transform: seluruh Recyclable (paling agresif, tambah Erasing + Blur)

Bagian keenam, EarlyStopping berbasis Macro F1 Score dengan penyimpanan
dan pemuatan kembali bobot model terbaik.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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

TRAIN_MANIFEST_PATH = os.path.join(BASE_DIR, "Preprocessing_Data", "train_manifest.csv")
VAL_MANIFEST_PATH = os.path.join(BASE_DIR, "Preprocessing_Data", "val_manifest.csv")
KOLOM_PATH = "file_path"
KOLOM_LABEL = "label"

LABEL_MAP = {
    "Recyclable": 0,
    "Electronic": 1,
    "Organic": 2,
}

# Backbone bisa diganti ke "tf_efficientnetv2_m" atau "convnext_base"
# jika VRAM mencukupi (EfficientNetV2-M butuh sekitar 6 GB).
BACKBONE_NAME = "tf_efficientnetv2_m"

UKURAN_INPUT = 256
BATCH_SIZE = 32
NUM_WORKERS = 0

BEST_MODEL_STAGE1_PATH = os.path.join(BASE_DIR, "Train_model", "Best_model", "best_model_stage1.pth")
BEST_MODEL_STAGE2_PATH = os.path.join(BASE_DIR, "Train_model", "Best_model", "best_model_stage2.pth")
EDA_FULL_FRAME_PATH = os.path.join(BASE_DIR, "EDA", "eda_deteksi_full_frame.csv")

EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 20
LR_STAGE1 = 1e-3
LR_STAGE2_HEAD = 1e-4
LR_STAGE2_BACKBONE = 1e-5

LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0          # Focal Loss gamma, makin besar makin fokus hard example
USE_FOCAL_LOSS = False     # Set False untuk kembali ke WeightedCrossEntropy biasa
GRADIENT_CLIP_NORM = 1.0
PATIENCE_EARLY_STOPPING = 5
HARD_EXAMPLE_FRACTION = 0.15
HARD_EXAMPLE_OVERSAMPLE_FACTOR = 3.0
HARD_EXAMPLE_MIN_SCORE = 0.72


# ============================================================
# Bagian 1, Arsitektur Model
# ============================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Menormalkan skala aktivasi lewat root mean square saja.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class SampahClassifier(nn.Module):
    def __init__(
        self, num_classes=3, pretrained=True, backbone_name=BACKBONE_NAME,
        hidden_dim=512, dropout=0.3, activation="silu", norm_type="rmsnorm",
        n_layers=2, hidden_dim2=256
    ):
        super(SampahClassifier, self).__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        num_features = self.backbone.num_features

        def buat_aktivasi():
            if activation == "gelu":
                return nn.GELU()
            elif activation == "silu":
                return nn.SiLU()
            raise ValueError(f"activation '{activation}' tidak dikenali, pakai 'gelu' atau 'silu'")

        def buat_norm(dim):
            if norm_type == "rmsnorm":
                return RMSNorm(dim)
            elif norm_type == "none":
                return nn.Identity()
            raise ValueError(f"norm_type '{norm_type}' tidak dikenali, pakai 'none' atau 'rmsnorm'")

        if n_layers == 1:
            self.classifier = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                buat_norm(hidden_dim),
                buat_aktivasi(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        elif n_layers == 2:
            if hidden_dim2 is None:
                raise ValueError("hidden_dim2 wajib diisi kalau n_layers=2")
            self.classifier = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                buat_norm(hidden_dim),
                buat_aktivasi(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, hidden_dim2),
                buat_norm(hidden_dim2),
                buat_aktivasi(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim2, num_classes),
            )
        else:
            raise ValueError(f"n_layers={n_layers} tidak didukung, cuma 1 atau 2")

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
# Bagian 3, Focal Loss dan Fungsi Loss dengan Class Weight
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss untuk mengatasi masalah class imbalance dan hard example.
    Rumus: FL = -alpha_t * (1 - pt)^gamma * log(pt)

    gamma mengontrol seberapa kuat penalti berkurang untuk sampel yang
    sudah mudah diklasifikasikan. Makin besar gamma, makin model fokus
    ke sampel yang susah (hard example). Nilai gamma=2.0 adalah standar
    yang direkomendasikan pada paper aslinya (Lin et al., 2017).

    label_smoothing diterapkan sebelum hitung Focal Loss agar model tidak
    terlalu percaya diri pada sampel yang secara visual ambigu.
    """

    def __init__(self, weight=None, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING,
                 reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input_logits, target):
        num_classes = input_logits.size(1)

        # Label smoothing manual sebelum masuk ke cross entropy
        with torch.no_grad():
            smooth_target = torch.full_like(
                input_logits, self.label_smoothing / (num_classes - 1)
            )
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.label_smoothing)

        log_prob = F.log_softmax(input_logits, dim=1)
        prob = log_prob.exp()

        # Ambil probabilitas kelas yang benar (pt) untuk setiap sampel
        pt = (prob * smooth_target).sum(dim=1)
        focal_weight = (1.0 - pt) ** self.gamma

        # Cross entropy terhadap label yang sudah di-smooth
        ce_loss = -(smooth_target * log_prob).sum(dim=1)

        # Terapkan class weight jika ada
        if self.weight is not None:
            class_weight_per_sample = (self.weight.unsqueeze(0) * smooth_target).sum(dim=1)
            ce_loss = class_weight_per_sample * ce_loss

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def create_loss_function(class_counts, label_smoothing=LABEL_SMOOTHING,
                         use_focal=USE_FOCAL_LOSS, gamma=FOCAL_GAMMA):
    """
    Membuat loss function dengan class weight dari distribusi data manifest.

    Jika use_focal=True, menggunakan FocalLoss yang secara otomatis memberi
    bobot lebih besar ke sampel yang susah diklasifikasikan (hard example),
    cocok untuk Recyclable yang sering missclassified.

    Jika use_focal=False, kembali ke Weighted CrossEntropyLoss biasa dengan
    label smoothing sebagai fallback.
    """
    total_samples = sum(class_counts)
    num_classes = len(class_counts)

    weights = [total_samples / (num_classes * count) for count in class_counts]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    print(f"Class weight yang digunakan {weights_tensor.cpu().numpy()}")

    if use_focal:
        print(f"Menggunakan FocalLoss dengan gamma={gamma}, label_smoothing={label_smoothing}")
        criterion = FocalLoss(
            weight=weights_tensor,
            gamma=gamma,
            label_smoothing=label_smoothing
        )
    else:
        print(f"Menggunakan WeightedCrossEntropyLoss dengan label_smoothing={label_smoothing}")
        criterion = nn.CrossEntropyLoss(
            weight=weights_tensor,
            label_smoothing=label_smoothing
        )

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

def dapatkan_hard_examples(train_manifest_path, eda_full_frame_path, label_nama="Recyclable",
                           hard_fraction=HARD_EXAMPLE_FRACTION,
                           min_score=HARD_EXAMPLE_MIN_SCORE):
    """
    Memilih sampel Recyclable yang masuk kategori full_frame dan punya
    skor warna/tekstur yang tinggi sebagai hard example untuk fase
    fine-tuning lanjutan.
    """
    if not os.path.exists(train_manifest_path):
        print(f"Manifest train tidak ditemukan: {train_manifest_path}")
        return [], {"selected": 0, "reason": "manifest_missing"}

    if not os.path.exists(eda_full_frame_path):
        print(f"File EDA full frame tidak ditemukan: {eda_full_frame_path}")
        return [], {"selected": 0, "reason": "eda_missing"}

    df_manifest = pd.read_csv(train_manifest_path)
    df_eda = pd.read_csv(eda_full_frame_path)

    if "nama_file" not in df_eda.columns:
        print("Kolom nama_file tidak ditemukan pada file EDA, hard example mining dilewati")
        return [], {"selected": 0, "reason": "eda_columns_missing"}

    df_eda = df_eda.copy()
    df_eda["nama_file"] = df_eda["nama_file"].astype(str)
    df_eda["kategori"] = df_eda["kategori"].astype(str).str.lower()
    df_eda["label"] = df_eda["label"].astype(str)

    hard_candidates = df_eda[
        (df_eda["label"].str.lower() == label_nama.lower()) &
        (df_eda["kategori"] == "full_frame")
    ].copy()

    if hard_candidates.empty:
        print("Tidak ada hard example Recyclable full_frame yang ditemukan di EDA")
        return [], {"selected": 0, "reason": "no_candidates"}

    if {"skor_variansi", "skor_entropi", "skor_akhir"}.issubset(hard_candidates.columns):
        hard_candidates["combined_score"] = (
            0.4 * hard_candidates["skor_variansi"] +
            0.35 * hard_candidates["skor_entropi"] +
            0.25 * hard_candidates["skor_akhir"]
        )
    else:
        hard_candidates["combined_score"] = hard_candidates.get("skor_akhir", 0.0)

    hard_candidates = hard_candidates[
        hard_candidates["combined_score"] >= min_score
    ].copy()

    if hard_candidates.empty:
        hard_candidates = df_eda[
            (df_eda["label"].str.lower() == label_nama.lower()) &
            (df_eda["kategori"] == "full_frame")
        ].copy()

    hard_candidates = hard_candidates.sort_values("combined_score", ascending=False)
    max_selected = max(1, int(len(hard_candidates) * hard_fraction))
    hard_candidates = hard_candidates.head(max_selected)

    hard_names = set(hard_candidates["nama_file"].tolist())
    hard_indices = []
    for idx, path_gambar in enumerate(df_manifest[KOLOM_PATH].astype(str)):
        if os.path.basename(path_gambar) in hard_names:
            hard_indices.append(idx)

    print(
        f"Hard example mining selesai: {len(hard_indices)} sampel Recyclable full_frame dipilih "
        f"dari {len(df_manifest)} data train"
    )

    return hard_indices, {
        "selected": len(hard_indices),
        "fraction": hard_fraction,
        "min_score": min_score,
        "reason": "ok",
    }


class SampahDataset(Dataset):
    """
    Dataset berbasis file manifest CSV dengan augmentasi per-kelas.

    Augmentasi berat sudah dilakukan secara offline pada tahap preprocessing,
    sehingga transform di sini bersifat online (ringan sampai sedang) sesuai
    tingkat kesulitan masing-masing kelas:

      transform             → Organic & Electronic (ringan)
      hard_transform        → Recyclable full_frame hard example (sedang)
      recyclable_transform  → seluruh Recyclable (paling agresif)

    Prioritas transform: hard_transform > recyclable_transform > transform
    Hard example Recyclable full_frame mendapat hard_transform (sedang) bukan
    recyclable_transform (agresif) agar oversampling tidak terlalu kasar.
    """

    def __init__(self, dataframe, label_map, transform=None,
                 hard_indices=None, hard_transform=None,
                 recyclable_transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_map = label_map
        self.transform = transform
        self.hard_transform = hard_transform
        self.recyclable_transform = recyclable_transform
        self.hard_indices = set(hard_indices or [])
        # Buat lookup indeks → nama label untuk pemilihan transform per kelas
        self.recyclable_label = "Recyclable"

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

            # Pilih transform berdasarkan kelas dan apakah ini hard example:
            # Hard example Recyclable full_frame  → hard_transform (sedang)
            # Recyclable biasa                    → recyclable_transform (agresif)
            # Organic / Electronic                → transform (ringan)
            if idx in self.hard_indices and self.hard_transform is not None:
                transform_to_use = self.hard_transform
            elif label_nama == self.recyclable_label and self.recyclable_transform is not None:
                transform_to_use = self.recyclable_transform
            else:
                transform_to_use = self.transform

            if transform_to_use:
                img_tensor = transform_to_use(img_rgb)
            else:
                img_tensor = transforms.ToTensor()(img_rgb)

        except Exception:
            # Fallback: kembalikan tensor nol jika gambar gagal dibaca
            img_tensor = torch.zeros(3, UKURAN_INPUT, UKURAN_INPUT)

        return img_tensor, label_idx


class RandomChannelShuffle:
    """
    Mengacak urutan channel R, G, B secara acak sehingga warna objek
    berubah sangat dramatis tanpa mengubah tekstur, bentuk, atau struktur
    gambar sama sekali.

    Contoh efek:
      R,G,B → B,R,G : kaleng perak jadi terlihat kehijauan
      R,G,B → G,B,R : botol transparan jadi terlihat kemerahan

    Ini memaksa model tidak memakai warna sebagai fitur utama Recyclable,
    sehingga kalau ketemu kaleng berwarna-warni atau botol kuning di
    data nyata, model tetap mengenalinya dari bentuk dan tekstur.
    """
    def __init__(self, p=0.35):
        self.p = p

    def __call__(self, img):
        if np.random.random() < self.p:
            arr = np.array(img)
            idx = np.random.permutation(3)
            arr = arr[:, :, idx]
            return Image.fromarray(arr.astype(np.uint8))
        return img


def dapatkan_transform_train(ukuran_input=UKURAN_INPUT, mode="normal"):
    """
    Mengembalikan transform augmentasi sesuai mode:

    mode="normal"      → Organic & Electronic, augmentasi ringan
    mode="hard"        → Recyclable full_frame hard example, augmentasi sedang
    mode="recyclable"  → Seluruh Recyclable, augmentasi paling agresif
                         Termasuk:
                         - RandomChannelShuffle: acak urutan channel RGB agar
                           warna berubah sangat dramatis (metalik → merah/kuning)
                         - hue=0.5: rotasi warna penuh di color wheel
                         - RandomSolarize: efek warna tidak natural
                         Tujuannya: model belajar dari bentuk/tekstur, bukan warna
    """
    if mode == "recyclable":
        return transforms.Compose([
            transforms.Resize((ukuran_input, ukuran_input)),
            transforms.RandomResizedCrop(
                (ukuran_input, ukuran_input), scale=(0.7, 1.0), ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            # Hue dikembalikan ke level aman (0.1) agar warna tidak hancur.
            # ChannelShuffle & Solarize dihapus karena terbukti merusak performa.
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # RandomErasing dikurangi drastis p-nya agar tidak terlalu banyak merusak fitur
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])

    if mode == "hard":
        return transforms.Compose([
            transforms.Resize((ukuran_input, ukuran_input)),
            transforms.RandomResizedCrop(
                (ukuran_input, ukuran_input), scale=(0.7, 1.0), ratio=(0.8, 1.25)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.25, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # mode="normal" — default untuk Organic & Electronic
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


def buat_dataloader(train_manifest_path, val_manifest_path, label_map, batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKERS, hard_indices=None, use_hard_sampler=False,
                     hard_sample_weight=HARD_EXAMPLE_OVERSAMPLE_FACTOR,
                     pin_memory=True):
    """
    Membaca manifest train dan validasi secara terpisah, lalu membungkusnya ke DataLoader.

    Dataset train menggunakan tiga level augmentasi:
      - recyclable_transform untuk seluruh Recyclable (paling agresif)
      - hard_transform untuk Recyclable full_frame yang terpilih sebagai hard example
      - transform biasa untuk Organic & Electronic (ringan)

    Ketika hard_indices tersedia, DataLoader dapat memakai WeightedRandomSampler untuk
    oversampling sampel Recyclable full_frame yang dipilih sebagai hard example.

    pin_memory=True mempercepat transfer tensor dari CPU ke GPU melalui pinned memory.
    Aktifkan hanya jika DEVICE adalah CUDA; otomatis diatur aman oleh DEVICE.type check.
    """
    _pin = pin_memory and (DEVICE.type == "cuda")

    df_train = pd.read_csv(train_manifest_path)
    df_val = pd.read_csv(val_manifest_path)

    print(f"Jumlah data train {len(df_train)}, jumlah data validasi {len(df_val)}")

    dataset_train = SampahDataset(
        df_train,
        label_map,
        transform=dapatkan_transform_train(mode="normal"),
        hard_indices=hard_indices,
        hard_transform=dapatkan_transform_train(mode="hard"),
        recyclable_transform=dapatkan_transform_train(mode="recyclable"),
    )
    dataset_val = SampahDataset(df_val, label_map, transform=dapatkan_transform_val())

    if use_hard_sampler and hard_indices:
        weights = torch.ones(len(df_train), dtype=torch.double)
        hard_indices_tensor = list(hard_indices)
        weights[hard_indices_tensor] = hard_sample_weight
        sampler = WeightedRandomSampler(weights, num_samples=len(df_train), replacement=True)
        loader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=False, sampler=sampler,
            num_workers=num_workers, pin_memory=_pin, persistent_workers=(num_workers > 0)
        )
    else:
        loader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=_pin, persistent_workers=(num_workers > 0)
        )

    loader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin, persistent_workers=(num_workers > 0)
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
#     TRAIN_MANIFEST_PATH, VAL_MANIFEST_PATH,
#     FocalLoss, USE_FOCAL_LOSS, BACKBONE_NAME,
# )