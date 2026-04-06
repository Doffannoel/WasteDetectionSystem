"""
scripts/demo_quick_test.py — Quick test inference tanpa training

Script ini berguna untuk:
- Test apakah environment sudah benar
- Demo tanpa perlu training (pakai model YOLOv8n pre-trained COCO)
- Verifikasi visualisasi bounding box

Jalankan: python scripts/demo_quick_test.py
"""

import sys
from pathlib import Path

# Tambahkan root ke sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from config import CLASS_NAMES, CLASS_COLORS
from utils import draw_detections, FPSCounter, setup_logger, save_detection_csv, save_detection_json

logger = setup_logger("quick_test")


def test_with_sample_image():
    """
    Test inference menggunakan model pre-trained COCO.
    Karena COCO tidak punya kelas sampah, deteksi mungkin tidak optimal —
    ini hanya untuk memverifikasi bahwa pipeline berjalan.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("❌ ultralytics tidak terinstall: pip install ultralytics")
        return

    logger.info("📦 Load YOLOv8n pre-trained (COCO)...")
    model = YOLO("yolov8n.pt")   # otomatis download jika belum ada

    # Buat gambar dummy berwarna untuk test
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[:] = (60, 60, 60)   # background abu-abu
    cv2.putText(
        test_img,
        "WASTE DETECTION TEST",
        (80, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2, (255, 255, 255), 2
    )
    cv2.putText(
        test_img,
        "Pipeline OK - Train model for real detection",
        (20, 290),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (100, 255, 100), 1
    )

    logger.info("🔍 Jalankan inference pada gambar test...")
    results = model.predict(
        source  = test_img,
        conf    = 0.3,
        verbose = False,
    )

    boxes = results[0].boxes
    annotated, count_dict = draw_detections(test_img, boxes, class_names=CLASS_NAMES)

    logger.info(f"✅ Inference berhasil. Deteksi: {count_dict}")

    # Simpan gambar test
    output_path = Path("outputs/test_output.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    logger.info(f"💾 Hasil disimpan: {output_path}")

    return True


def test_webcam_single_frame():
    """Test buka webcam dan ambil 1 frame."""
    logger.info("📹 Test buka webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.warning("⚠️  Webcam tidak tersedia. Skip test webcam.")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret:
        logger.warning("⚠️  Gagal baca frame dari webcam.")
        return False

    logger.info(f"✅ Webcam berhasil — frame size: {frame.shape}")
    cv2.imwrite("outputs/webcam_test.jpg", frame)
    return True


def test_csv_json_output():
    """Test penyimpanan CSV dan JSON."""
    logger.info("💾 Test output CSV/JSON...")

    test_count = {"plastic": 2, "metal": 1, "trash": 3}

    save_detection_csv(source="test", count_dict=test_count)
    save_detection_json(source="test", count_dict=test_count)

    from config import OUTPUT_CSV, OUTPUT_JSON
    assert OUTPUT_CSV.exists(),  f"❌ CSV tidak dibuat: {OUTPUT_CSV}"
    assert OUTPUT_JSON.exists(), f"❌ JSON tidak dibuat: {OUTPUT_JSON}"

    logger.info(f"✅ CSV : {OUTPUT_CSV}")
    logger.info(f"✅ JSON: {OUTPUT_JSON}")
    return True


def run_all_tests():
    logger.info("=" * 55)
    logger.info("🧪 Quick Environment Test — Waste Detection")
    logger.info("=" * 55)

    results = {}

    # Test 1: Import
    logger.info("\n[1/4] Test imports...")
    try:
        import ultralytics, cv2, numpy, torch, yaml
        logger.info("✅ Semua library berhasil diimport")
        results["imports"] = True
    except ImportError as e:
        logger.error(f"❌ Import gagal: {e}")
        results["imports"] = False

    # Test 2: Inference
    logger.info("\n[2/4] Test inference...")
    results["inference"] = test_with_sample_image()

    # Test 3: Webcam
    logger.info("\n[3/4] Test webcam...")
    results["webcam"] = test_webcam_single_frame()

    # Test 4: Output
    logger.info("\n[4/4] Test output CSV/JSON...")
    results["output"] = test_csv_json_output()

    # Ringkasan
    logger.info("\n" + "─" * 40)
    logger.info("Ringkasan Test:")
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {name:15s}: {status}")

    all_pass = all(v for v in results.values() if v is not None)
    logger.info("─" * 40)
    if all_pass:
        logger.info("🎉 Semua test berhasil! Environment siap.")
        logger.info("   Langkah selanjutnya: python prepare_dataset.py")
    else:
        logger.warning("⚠️  Ada test yang gagal. Cek error di atas.")


if __name__ == "__main__":
    run_all_tests()
