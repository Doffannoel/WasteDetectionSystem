"""
utils.py — Fungsi utilitas bersama untuk proyek Waste Detection
"""

import csv
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import CLASS_COLORS, CLASS_NAMES, OUTPUT_CSV, OUTPUT_JSON, SAVE_CSV, SAVE_JSON

# ─── LOGGING ───────────────────────────────────────────────────────────────────
def setup_logger(name: str = "waste_detection", level: int = logging.INFO) -> logging.Logger:
    """Setup logger dengan format yang rapi."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = setup_logger()


# ─── VISUALISASI ───────────────────────────────────────────────────────────────
def draw_detections(
    frame: np.ndarray,
    boxes: List,
    class_names: List[str] = CLASS_NAMES,
    fps: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Gambar bounding box, label, dan confidence pada frame.
    
    Returns:
        frame       — frame dengan anotasi
        count_dict  — jumlah deteksi per kelas
    """
    count_dict: Dict[str, int] = {}
    frame = frame.copy()

    for box in boxes:
        # Ambil koordinat, confidence, dan class id
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])

        if cls_id >= len(class_names):
            continue

        cls_name = class_names[cls_id]
        color    = CLASS_COLORS.get(cls_name, (0, 255, 0))

        # Hitung jumlah per kelas
        count_dict[cls_name] = count_dict.get(cls_name, 0) + 1

        # Gambar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label    = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top_left     = (x1, max(y1 - th - 6, 0))
        bottom_right = (x1 + tw + 4, max(y1, th + 6))
        cv2.rectangle(frame, top_left, bottom_right, color, -1)

        # Teks label
        cv2.putText(
            frame, label,
            (x1 + 2, max(y1 - 4, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )

    # Overlay info — total deteksi & FPS
    total = sum(count_dict.values())
    _draw_overlay(frame, count_dict, total, fps)

    return frame, count_dict


def _draw_overlay(frame: np.ndarray, count_dict: Dict, total: int, fps: Optional[float]):
    """Gambar panel info di pojok kiri atas."""
    overlay = frame.copy()
    panel_h = 30 + (len(count_dict) + 2) * 22
    cv2.rectangle(overlay, (8, 8), (220, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = 28
    fps_text = f"FPS: {fps:.1f}" if fps else "FPS: --"
    cv2.putText(frame, fps_text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    y += 22
    cv2.putText(frame, f"Total: {total} obj", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 100), 1)
    y += 22

    for cls_name, count in sorted(count_dict.items(), key=lambda x: -x[1]):
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.putText(frame, f"  {cls_name}: {count}", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        y += 20


# ─── PENYIMPANAN HASIL DETEKSI ─────────────────────────────────────────────────
def save_detection_csv(
    source: str,
    count_dict: Dict[str, int],
    timestamp: Optional[str] = None,
    filepath: Path = OUTPUT_CSV,
):
    """Simpan hasil deteksi ke file CSV."""
    if not SAVE_CSV:
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    ts   = timestamp or datetime.now().isoformat()
    total = sum(count_dict.values())

    write_header = not filepath.exists()
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "source", "class_name", "count", "total_objects"
        ])
        if write_header:
            writer.writeheader()

        if count_dict:
            for cls_name, count in count_dict.items():
                writer.writerow({
                    "timestamp"    : ts,
                    "source"       : source,
                    "class_name"   : cls_name,
                    "count"        : count,
                    "total_objects": total,
                })
        else:
            # Frame kosong — tidak ada deteksi
            writer.writerow({
                "timestamp"    : ts,
                "source"       : source,
                "class_name"   : "none",
                "count"        : 0,
                "total_objects": 0,
            })


def save_detection_json(
    source: str,
    count_dict: Dict[str, int],
    boxes_raw: Optional[List] = None,
    timestamp: Optional[str] = None,
    filepath: Path = OUTPUT_JSON,
):
    """Simpan hasil deteksi ke file JSON (append ke array)."""
    if not SAVE_JSON:
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    ts    = timestamp or datetime.now().isoformat()
    total = sum(count_dict.values())

    # Buat entry baru
    entry = {
        "timestamp"    : ts,
        "source"       : source,
        "total_objects": total,
        "counts"       : count_dict,
        "detections"   : [],
    }

    # Tambahkan detail per box jika tersedia
    if boxes_raw:
        for box in boxes_raw:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            entry["detections"].append({
                "class_id"  : cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
                "confidence": round(conf, 4),
                "bbox"      : {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

    # Load existing JSON atau buat baru
    data: List = []
    if filepath.exists():
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = []

    data.append(entry)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── FPS COUNTER ───────────────────────────────────────────────────────────────
class FPSCounter:
    """Menghitung FPS secara rolling average."""

    def __init__(self, window: int = 30):
        self.window    = window
        self.timestamps: List[float] = []

    def tick(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ─── LABEL MAPPING TACO → CLASS NAMES ─────────────────────────────────────────
# Mapping dari label TACO asli (60+ kelas) ke 6 kelas yang disederhanakan
TACO_LABEL_MAP = {
    # PLASTIC
    "Plastic bottle"             : "plastic",
    "Plastic bottle cap"         : "plastic",
    "Plastic cup"                : "plastic",
    "Plastic lid"                : "plastic",
    "Plastic straw"              : "plastic",
    "Plastic utensils"           : "plastic",
    "Disposable plastic cup"     : "plastic",
    "Plastic film"               : "plastic_bag",
    "Single-use carrier bag"     : "plastic_bag",
    "Polypropylene bag"          : "plastic_bag",
    "Plastic bag & wrapper"      : "plastic_bag",
    "Six pack rings"             : "plastic",
    "Tupperware"                 : "plastic",
    "Other plastic"              : "plastic",
    "Other plastic bottle"       : "plastic",
    "Clear plastic bottle"       : "plastic",
    "Plastic gloves"             : "plastic",
    "Plastic pipe"               : "plastic",

    # PAPER / CARDBOARD
    "Paper"                      : "paper_cardboard",
    "Cardboard"                  : "paper_cardboard",
    "Paper bag"                  : "paper_cardboard",
    "Paper cup"                  : "paper_cardboard",
    "Wrapping paper"             : "paper_cardboard",
    "Paper straw"                : "paper_cardboard",
    "Newspaper"                  : "paper_cardboard",
    "Tissues"                    : "paper_cardboard",
    "Magazine paper"             : "paper_cardboard",

    # METAL
    "Metal"                      : "metal",
    "Drink can"                  : "metal",
    "Food can"                   : "metal",
    "Aluminium foil"             : "metal",
    "Metal bottle cap"           : "metal",
    "Metal lid"                  : "metal",
    "Pop tab"                    : "metal",
    "Scrap metal"                : "metal",
    "Aerosol"                    : "metal",

    # GLASS
    "Glass bottle"               : "glass",
    "Glass cup"                  : "glass",
    "Glass jar"                  : "glass",
    "Broken glass"               : "glass",
    "Other glass"                : "glass",

    # TRASH (umum / campuran)
    "Cigarette"                  : "trash",
    "Food waste"                 : "trash",
    "Rope & strings"             : "trash",
    "Shoe"                       : "trash",
    "Battery"                    : "trash",
    "Blister pack"               : "trash",
    "Other"                      : "trash",
    "Unlabeled litter"           : "trash",
    "Styrofoam piece"            : "trash",
    "Plastic film canister"      : "trash",
    "Foam cup"                   : "trash",
    "Foam food container"        : "trash",
    "Meal carton"                : "trash",
    "Pizza box"                  : "trash",
    "Drink carton"               : "trash",
}

# Mapping dari label Roboflow "Garbage Classification" ke kelas kita
ROBOFLOW_LABEL_MAP = {
    "cardboard"  : "paper_cardboard",
    "glass"      : "glass",
    "metal"      : "metal",
    "paper"      : "paper_cardboard",
    "plastic"    : "plastic",
    "trash"      : "trash",
    "organic"    : "trash",
    "battery"    : "trash",
    "clothes"    : "trash",
    "shoes"      : "trash",
    "white-glass": "glass",
    "brown-glass": "glass",
    "green-glass": "glass",
}


def map_label(label: str, source: str = "taco") -> Optional[str]:
    """
    Map label asli dataset ke kelas yang disederhanakan.
    
    Args:
        label  — nama label dari dataset
        source — "taco" atau "roboflow"
    
    Returns:
        Nama kelas yang disederhanakan, atau None jika tidak ditemukan
    """
    mapping = TACO_LABEL_MAP if source == "taco" else ROBOFLOW_LABEL_MAP
    # Coba exact match dulu
    if label in mapping:
        return mapping[label]
    # Coba case-insensitive
    label_lower = label.lower()
    for k, v in mapping.items():
        if k.lower() == label_lower:
            return v
    # Fallback: cek apakah ada kata kunci
    for keyword, cls in [
        ("plastic", "plastic"),
        ("bag", "plastic_bag"),
        ("paper", "paper_cardboard"),
        ("cardboard", "paper_cardboard"),
        ("metal", "metal"),
        ("can", "metal"),
        ("glass", "glass"),
    ]:
        if keyword in label_lower:
            return cls
    return "trash"  # default fallback


# ─── HELPER ────────────────────────────────────────────────────────────────────
def get_output_path(source: str, suffix: str = "") -> Path:
    """Buat path output unik berdasarkan timestamp."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = Path(source).stem if source not in ("0", "webcam") else "webcam"
    return OUTPUT_DIR / f"{name}_{ts}{suffix}"


# Buat direktori output saat import
from config import OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
