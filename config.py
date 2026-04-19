"""
config.py â€” Konfigurasi terpusat untuk proyek Waste Detection
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

# â”€â”€â”€ ROOT PROJECT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = ROOT_DIR / "data"
DATASET_DIR= ROOT_DIR / "datasets"
MODEL_DIR  = ROOT_DIR / "models"
RUNS_DIR   = ROOT_DIR / "runs"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Pastikan semua folder ada
for d in [DATA_DIR, DATASET_DIR, MODEL_DIR, RUNS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# â”€â”€â”€ KELAS SAMPAH (6 kelas utama) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Ini adalah kelas final setelah penyederhanaan dari dataset TACO + Roboflow
CLASS_NAMES = [
    "plastic",          # 0 â€” botol plastik, gelas plastik, sedotan
    "paper_cardboard",  # 1 â€” kertas, kardus, kotak
    "metal",            # 2 â€” kaleng, logam
    "glass",            # 3 â€” botol kaca, pecahan kaca
    "plastic_bag",      # 4 â€” kantong plastik, sachet
    "trash",            # 5 â€” sampah campuran / tidak teridentifikasi
]
NUM_CLASSES = len(CLASS_NAMES)

# Warna bounding box per kelas (BGR untuk OpenCV)
CLASS_COLORS = {
    "plastic":         (0,   165, 255),   # oranye
    "paper_cardboard": (0,   255, 127),   # hijau muda
    "metal":           (180, 180, 0  ),   # cyan gelap
    "glass":           (255, 100, 100),   # biru muda
    "plastic_bag":     (147, 20,  255),   # ungu
    "trash":           (0,   0,   200),   # merah gelap
}

# â”€â”€â”€ MODEL â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pilih: "yolov8n.pt" atau "yolo11n.pt"
# YOLOv8n  â†’ lebih mature, banyak referensi, cocok production demo
# YOLO11n  â†’ arsitektur terbaru Ultralytics, sedikit lebih akurat, ekosistem berkembang
BASE_MODEL      = "yolov8n.pt"   # pre-trained COCO, akan di-fine-tune
TRAINED_MODEL   = MODEL_DIR / "best.pt"  # path model hasil training
DATASET_YAML    = DATASET_DIR / "waste_dataset.yaml"

# â”€â”€â”€ TRAINING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TRAIN_CONFIG = {
    "epochs"       : 150,        # cukup untuk fine-tune; naikkan ke 150 kalau data banyak
    "batch"        : 16,        # turunkan ke 8 jika RAM GPU < 4 GB
    "imgsz"        : 640,       # standar YOLO; turunkan ke 416 jika lambat
    "lr0"          : 0.02,      # learning rate awal
    "lrf"          : 0.01,      # final lr ratio
    "momentum"     : 0.9,
    "weight_decay" : 0.001,
    "warmup_epochs": 3,
    "patience"     : 20,        # early stopping
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
    # Augmentasi â€” penting untuk dataset sampah yang beragam
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
    "mixup"        : 0.1,
    "copy_paste"   : 0.1,
}

# â”€â”€â”€ INFERENCE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
INFERENCE_CONFIG = {
    "conf"              : 0.55,   # threshold confidence; turunkan ke 0.25 jika banyak miss
    "iou"               : 0.45,   # IoU threshold untuk NMS
    "imgsz"             : 640,
    "max_det"           : 20,     # max deteksi per frame
    "device"            : AUTO_DEVICE,  # ganti "0" jika ada GPU
    "verbose"           : False,
    # Filtering confidence rendah
    "low_conf_threshold": 0.5,    # jika conf < ini, objek dianggap "LAINNYA"
    "show_low_conf"     : True,    # tampilkan objek low-confidence sebagai "LAINNYA"
    "filter_low_conf"   : False,   # True = jangan tampilkan, False = tampilkan as LAINNYA
}

# Nama untuk objek dengan confidence rendah
UNKNOWN_CLASS_NAME = "LAINNYA"

# â”€â”€â”€ OUTPUT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SAVE_CSV    = True
SAVE_JSON   = True
OUTPUT_CSV  = OUTPUT_DIR / "detections.csv"
OUTPUT_JSON = OUTPUT_DIR / "detections.json"

# â”€â”€â”€ DATASET SPLIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TRAIN_RATIO = 0.75
VAL_RATIO   = 0.15
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# â”€â”€â”€ ROBOFLOW DATASET â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Dataset: "Garbage Classification" by Roboflow Universe
# URL: https://universe.roboflow.com/material-identification/garbage-classification-3
# Versi: 2  |  Format: YOLOv8
ROBOFLOW_WORKSPACE = "material-identification"
ROBOFLOW_PROJECT   = "garbage-classification-3"
ROBOFLOW_VERSION   = 2

# â”€â”€â”€ TACO DATASET â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# URL: http://tacodataset.org/
# GitHub: https://github.com/pedropro/TACO
TACO_ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TACO_IMAGES_DIR      = DATA_DIR / "taco" / "images"
TACO_ANNO_FILE       = DATA_DIR / "taco" / "annotations.json"

