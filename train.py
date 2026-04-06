"""
train.py — Training pipeline untuk Waste Detection YOLO model

Fitur:
- Fine-tune YOLOv8n (atau YOLO11n) dari pre-trained COCO weights
- Konfigurasi lengkap dengan augmentasi
- Early stopping & resume training
- Evaluasi otomatis setelah training
- Simpan best model ke models/best.pt

Jalankan: python train.py
Jalankan dengan resume: python train.py --resume
"""

import argparse
import shutil
import sys
from pathlib import Path

from config import (
    BASE_MODEL, DATASET_YAML, MODEL_DIR,
    RUNS_DIR, TRAIN_CONFIG, TRAINED_MODEL,
)
from utils import logger


def check_prerequisites():
    """Cek apakah semua yang dibutuhkan tersedia."""
    errors = []

    # Cek ultralytics
    try:
        import ultralytics
        logger.info(f"✅ Ultralytics versi: {ultralytics.__version__}")
    except ImportError:
        errors.append("❌ ultralytics tidak terinstall. Jalankan: pip install ultralytics")

    # Cek dataset YAML
    if not DATASET_YAML.exists():
        errors.append(
            f"❌ Dataset YAML tidak ditemukan: {DATASET_YAML}\n"
            "   Jalankan dulu: python prepare_dataset.py"
        )

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)


def find_last_checkpoint() -> str | None:
    """Cari checkpoint terakhir untuk resume training."""
    train_dir = RUNS_DIR / "waste_detection"
    last_pt   = train_dir / "weights" / "last.pt"
    if last_pt.exists():
        logger.info(f"🔄 Checkpoint ditemukan: {last_pt}")
        return str(last_pt)
    return None


def train(resume: bool = False):
    """
    Jalankan training YOLO.
    
    Args:
        resume — True untuk lanjutkan training dari checkpoint terakhir
    """
    from ultralytics import YOLO

    # Tentukan model yang dipakai
    if resume:
        checkpoint = find_last_checkpoint()
        if checkpoint:
            model_path = checkpoint
            logger.info(f"🔄 Resume training dari: {model_path}")
        else:
            logger.warning("⚠️  Tidak ada checkpoint ditemukan, mulai dari awal.")
            model_path = BASE_MODEL
            resume     = False
    else:
        model_path = BASE_MODEL

    # Load model
    logger.info(f"📦 Load model: {model_path}")
    model = YOLO(model_path)

    # Tampilkan info arsitektur
    logger.info(f"   Arsitektur: {model.info(verbose=False)}")

    # ── TRAINING ──
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Memulai training...")
    logger.info(f"   Dataset  : {DATASET_YAML}")
    logger.info(f"   Epoch    : {TRAIN_CONFIG['epochs']}")
    logger.info(f"   Batch    : {TRAIN_CONFIG['batch']}")
    logger.info(f"   Image sz : {TRAIN_CONFIG['imgsz']}")
    logger.info(f"   Device   : {TRAIN_CONFIG['device']}")
    logger.info("=" * 60 + "\n")

    results = model.train(
        data    = str(DATASET_YAML),
        resume  = resume,
        **TRAIN_CONFIG,
    )

    # ── SALIN BEST MODEL ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_weights = RUNS_DIR / "waste_detection" / "weights" / "best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, TRAINED_MODEL)
        logger.info(f"\n✅ Best model disalin ke: {TRAINED_MODEL}")
    else:
        logger.warning("⚠️  best.pt tidak ditemukan.")

    return results


def evaluate(model_path: str | None = None):
    """
    Evaluasi model pada test set.
    Menampilkan mAP50, mAP50-95, precision, recall.
    """
    from ultralytics import YOLO

    path = model_path or str(TRAINED_MODEL)
    if not Path(path).exists():
        logger.error(f"❌ Model tidak ditemukan: {path}")
        return

    logger.info(f"\n📊 Evaluasi model: {path}")
    model   = YOLO(path)
    metrics = model.val(
        data   = str(DATASET_YAML),
        split  = "test",
        imgsz  = TRAIN_CONFIG["imgsz"],
        device = TRAIN_CONFIG["device"],
        verbose= True,
        plots  = True,   # confusion matrix, PR curve, dsb
        save_json= True,
    )

    # Ringkasan metrik
    logger.info("\n" + "─" * 40)
    logger.info("📈 Hasil Evaluasi:")
    logger.info(f"   mAP50     : {metrics.box.map50:.4f}")
    logger.info(f"   mAP50-95  : {metrics.box.map:.4f}")
    logger.info(f"   Precision : {metrics.box.mp:.4f}")
    logger.info(f"   Recall    : {metrics.box.mr:.4f}")
    logger.info("─" * 40)

    # Metrik per kelas
    logger.info("\n📊 Metrik per kelas:")
    from config import CLASS_NAMES
    if hasattr(metrics.box, "ap_class_index"):
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"class_{cls_idx}"
            ap50 = metrics.box.ap50[i] if hasattr(metrics.box, "ap50") else 0
            logger.info(f"   {cls_name:20s} AP50: {ap50:.4f}")

    # Saran jika performa rendah
    if metrics.box.map50 < 0.4:
        logger.warning(
            "\n⚠️  mAP50 < 0.4. Saran peningkatan:\n"
            "   1. Tambah data training (terutama kelas dengan AP rendah)\n"
            "   2. Naikkan epoch ke 150+\n"
            "   3. Gunakan YOLOv8s (lebih besar dari nano)\n"
            "   4. Cek kualitas label — apakah ada mislabel?\n"
            "   5. Turunkan confidence threshold di inference"
        )

    return metrics


def export_model(format: str = "onnx"):
    """
    Export model ke format lain untuk deployment.
    Opsi: onnx, tflite, coreml, tensorrt
    """
    from ultralytics import YOLO

    if not TRAINED_MODEL.exists():
        logger.error(f"❌ Trained model tidak ditemukan: {TRAINED_MODEL}")
        return

    logger.info(f"📦 Export model ke format: {format}")
    model     = YOLO(str(TRAINED_MODEL))
    export_path = model.export(
        format = format,
        imgsz  = TRAIN_CONFIG["imgsz"],
        dynamic= True,
        simplify= True,
    )
    logger.info(f"✅ Model di-export ke: {export_path}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Waste Detection — Training Script")
    parser.add_argument("--resume",  action="store_true", help="Resume training dari checkpoint terakhir")
    parser.add_argument("--eval",    action="store_true", help="Hanya evaluasi model (tidak training)")
    parser.add_argument("--export",  type=str, default=None, help="Export model (onnx/tflite/coreml)")
    parser.add_argument("--model",   type=str, default=None, help="Path model untuk evaluasi/export")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🗑️  Waste Detection — Training Pipeline")
    logger.info("=" * 60)

    check_prerequisites()

    if args.eval:
        evaluate(args.model)
    elif args.export:
        export_model(args.export)
    else:
        results = train(resume=args.resume)
        logger.info("\n🏁 Training selesai! Evaluasi pada test set...")
        evaluate()
        logger.info("\n✅ Semua selesai!")
        logger.info(f"   Model: {TRAINED_MODEL}")
        logger.info(f"   Runs : {RUNS_DIR / 'waste_detection'}")
        logger.info("\n🚀 Jalankan inference: python predict.py --source 0  (webcam)")


if __name__ == "__main__":
    main()
