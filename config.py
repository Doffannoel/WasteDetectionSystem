"""
config.py - Konfigurasi terpusat untuk proyek Waste Detection
Ubah nilai di sini untuk menyesuaikan dengan environment kamu.
"""

import os
from pathlib import Path


def _auto_device() -> str:
    """
    Deteksi device otomatis.
    Prioritas: GPU CUDA (jika tersedia), fallback ke CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cuda_visible and cuda_visible != "-1":
        return "0"

    return "cpu"


AUTO_DEVICE = _auto_device()

# ---------------- ROOT PROJECT ----------------
ROOT_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = ROOT_DIR / "data"
DATASET_DIR= ROOT_DIR / "datasets"
MODEL_DIR  = ROOT_DIR / "models"
RUNS_DIR   = ROOT_DIR / "runs"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Pastikan semua folder ada
for d in [DATA_DIR, DATASET_DIR, MODEL_DIR, RUNS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------- KELAS SAMPAH (6 kelas utama) ----------------
# Ini adalah kelas final setelah penyederhanaan dari dataset TACO + Roboflow
CLASS_NAMES = [
    "plastik",          # 0 - botol plastik, gelas plastik, sedotan
    "kertas_kardus",    # 1 - kertas, kardus, kotak
    "logam",            # 2 - kaleng, logam
    "kaca",             # 3 - botol kaca, pecahan kaca
    "kantong_plastik",  # 4 - kantong plastik, sachet
    "sampah",           # 5 - sampah campuran / tidak teridentifikasi
]
NUM_CLASSES = len(CLASS_NAMES)

# Warna bounding box per kelas (BGR untuk OpenCV)
CLASS_COLORS = {
    "plastik":         (0,   165, 255),   # oranye
    "kertas_kardus":   (0,   255, 127),   # hijau muda
    "logam":           (180, 180, 0  ),   # cyan gelap
    "kaca":            (255, 100, 100),   # biru muda
    "kantong_plastik": (147, 20,  255),   # ungu
    "sampah":          (0,   0,   200),   # merah gelap
}

# ---------------- MODEL ----------------
# Pilih: "yolov8n.pt" atau "yolo11n.pt"
# YOLOv8n -> lebih mature, banyak referensi, cocok production demo
# YOLO11n -> arsitektur terbaru Ultralytics, sedikit lebih akurat, ekosistem berkembang
BASE_MODEL      = "yolov8s.pt"   # model sedikit lebih besar untuk akurasi lebih baik
TRAINED_MODEL   = MODEL_DIR / "best.pt"  # path model hasil training
DATASET_YAML    = DATASET_DIR / "waste_dataset.yaml"

# ---------------- TRAINING ----------------
TRAIN_CONFIG = {
    "epochs"       : 140,       # ditambah agar training punya waktu belajar lebih stabil
    "batch"        : 16,        # turunkan ke 8 jika RAM GPU < 4 GB
    "imgsz"        : 736,       # sedikit lebih besar untuk bantu detail objek
    "lr0"          : 0.01,      # learning rate awal
    "lrf"          : 0.01,      # final lr ratio
    "momentum"     : 0.937,
    "weight_decay" : 0.0005,
    "warmup_epochs": 3,
    "patience"     : 30,        # beri ruang lebih untuk model membaik
    "device"       : AUTO_DEVICE,
    "workers"      : 4,
    "project"      : str(RUNS_DIR),
    "name"         : "waste_detection",
    "exist_ok"     : True,
    "pretrained"   : True,
    "optimizer"    : "AdamW",
    "verbose"      : True,
    "seed"         : 42,
    "val"          : True,
    "save_period"  : 1,         # simpan checkpoint tiap epoch (buat aman saat runtime habis)
    # Augmentasi - penting untuk dataset sampah yang beragam
    "amp"          : False,
    "hsv_h"        : 0.015,
    "hsv_s"        : 0.7,
    "hsv_v"        : 0.4,
    "degrees"      : 10.0,
    "translate"    : 0.1,
    "scale"        : 0.5,
    "shear"        : 2.0,
    "flipud"       : 0.1,
    "fliplr"       : 0.5,
    "mosaic"       : 1.0,
    "mixup"        : 0.15,
    "copy_paste"   : 0.15,
}

# ---------------- INFERENCE ----------------
INFERENCE_CONFIG = {
    "conf"     : 0.50,   # threshold confidence; naikan untuk kurangi false positive
    "iou"      : 0.45,   # IoU threshold untuk NMS
    "imgsz"    : 640,
    "max_det"  : 50,     # max deteksi per frame
    "device"   : AUTO_DEVICE,  # ganti "0" jika ada GPU
    "verbose"  : False,
    # Filter tambahan agar objek aneh (misal muka) tidak terdeteksi
    "ignore_classes" : ["sampah"],  # jangan tampilkan kelas ini
    "min_area_ratio" : 0.005,       # terlalu kecil -> buang
    "max_area_ratio" : 0.60,        # terlalu besar -> buang
}

# ---------------- OUTPUT ----------------
SAVE_CSV    = True
SAVE_JSON   = True
OUTPUT_CSV  = OUTPUT_DIR / "detections.csv"
OUTPUT_JSON = OUTPUT_DIR / "detections.json"
OUTPUT_IMAGE_DIR = OUTPUT_DIR / "hasil_gambar_bbox"

# ---------------- DATASET SPLIT ----------------
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# ---------------- ROBOFLOW DATASET ----------------
# Dataset: "Garbage Classification" by Roboflow Universe
# URL: https://universe.roboflow.com/material-identification/garbage-classification-3
# Versi: 2  |  Format: YOLOv8
ROBOFLOW_WORKSPACE = "material-identification"
ROBOFLOW_PROJECT   = "garbage-classification-3"
ROBOFLOW_VERSION   = 2

# ---------------- TACO DATASET ----------------
# URL: http://tacodataset.org/
# GitHub: https://github.com/pedropro/TACO
TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TACO_IMAGES_DIR      = DATA_DIR / "taco" / "images"
TACO_ANNO_FILE       = DATA_DIR / "taco" / "annotations.json"
