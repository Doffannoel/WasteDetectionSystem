"""
predict.py — Inference realtime untuk Waste Detection

Mendukung:
  - Gambar        : python predict.py --source foto.jpg
  - Video file    : python predict.py --source video.mp4
  - Webcam        : python predict.py --source 0
  - RTSP / CCTV   : python predict.py --source rtsp://user:pass@ip:port/stream

Opsi tambahan:
  --no-save       : jangan simpan output video
  --no-csv        : jangan simpan ke CSV
  --no-json       : jangan simpan ke JSON
  --conf 0.4      : ubah confidence threshold
  --model path    : gunakan model custom
  --show          : tampilkan frame (default True kecuali headless)

Contoh:
  python predict.py --source 0 --conf 0.35
  python predict.py --source data/test_video.mp4 --conf 0.4
  python predict.py --source gambar.jpg
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from config import (
    CLASS_NAMES, INFERENCE_CONFIG, OUTPUT_DIR, TRAINED_MODEL,
    SAVE_CSV, SAVE_JSON,
)
from utils import (
    FPSCounter, draw_detections, get_output_path,
    logger, save_detection_csv, save_detection_json,
)


# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model(model_path: str | None = None):
    """Load YOLO model dari path yang ditentukan."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("❌ ultralytics tidak terinstall: pip install ultralytics")
        sys.exit(1)

    path = model_path or str(TRAINED_MODEL)

    # Kalau belum ada trained model, gunakan base model sebagai fallback
    if not Path(path).exists():
        logger.warning(f"⚠️  Model tidak ditemukan: {path}")
        logger.info("   Menggunakan YOLOv8n pre-trained (COCO) sebagai fallback.")
        logger.info("   Jalankan python train.py untuk training model custom.")
        path = "yolov8n.pt"

    logger.info(f"📦 Load model: {path}")
    model = YOLO(path)
    return model


# ─── INFERENCE IMAGE ───────────────────────────────────────────────────────────
def predict_image(
    model,
    image_path: str,
    save_output: bool = True,
    conf: float = INFERENCE_CONFIG["conf"],
    show: bool = True,
):
    """Inference pada satu gambar."""
    img_path = Path(image_path)
    if not img_path.exists():
        logger.error(f"❌ File tidak ditemukan: {image_path}")
        return

    logger.info(f"🖼️  Inference gambar: {image_path}")

    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.error(f"❌ Gagal membaca gambar: {image_path}")
        return

    # Jalankan prediksi
    results = model.predict(
        source  = frame,
        conf    = conf,
        iou     = INFERENCE_CONFIG["iou"],
        imgsz   = INFERENCE_CONFIG["imgsz"],
        max_det = INFERENCE_CONFIG["max_det"],
        device  = INFERENCE_CONFIG["device"],
        verbose = False,
    )

    boxes      = results[0].boxes
    annotated, count_dict = draw_detections(frame, boxes)

    # Simpan hasil
    ts = datetime.now().isoformat()
    save_detection_csv(source=str(img_path), count_dict=count_dict, timestamp=ts)
    save_detection_json(source=str(img_path), count_dict=count_dict, boxes_raw=boxes, timestamp=ts)

    # Simpan gambar output
    if save_output:
        out_path = get_output_path(str(img_path), suffix=".jpg")
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"✅ Hasil disimpan: {out_path}")

    # Tampilkan
    if show:
        cv2.imshow("Waste Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Ringkasan
    logger.info(f"📊 Terdeteksi: {count_dict}")
    return count_dict


# ─── INFERENCE VIDEO / WEBCAM ──────────────────────────────────────────────────
def predict_stream(
    model,
    source: str,
    save_output: bool = True,
    conf: float = INFERENCE_CONFIG["conf"],
    show: bool = True,
    save_csv: bool = SAVE_CSV,
    save_json: bool = SAVE_JSON,
    log_interval: int = 30,   # simpan ke CSV/JSON setiap N frame
):
    """
    Inference realtime pada video file, webcam, atau RTSP stream.
    
    Args:
        source       — "0" untuk webcam, path file, atau rtsp://...
        log_interval — simpan hasil ke CSV/JSON setiap N frame
    """
    # Buka sumber video
    if source == "0" or source.isdigit():
        src = int(source)
        logger.info(f"📹 Membuka webcam {src}...")
    else:
        src = source
        logger.info(f"📹 Membuka video: {source}")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"❌ Gagal membuka sumber video: {source}")
        if source == "0":
            logger.info("   Pastikan webcam terhubung dan tidak dipakai aplikasi lain.")
        return

    # Info video
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    logger.info(f"   Resolusi: {width}x{height} @ {fps_in:.0f}fps")

    # VideoWriter untuk simpan output
    writer = None
    if save_output:
        out_path = get_output_path(source, suffix=".mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps_in, (width, height))
        logger.info(f"💾 Output video: {out_path}")

    fps_counter = FPSCounter(window=30)
    frame_count = 0
    total_detections = 0

    logger.info("▶️  Inference dimulai. Tekan 'q' untuk berhenti.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("   Video selesai atau stream terputus.")
                break

            frame_count  += 1
            ts            = datetime.now().isoformat()

            # ── Inference ──
            results = model.predict(
                source  = frame,
                conf    = conf,
                iou     = INFERENCE_CONFIG["iou"],
                imgsz   = INFERENCE_CONFIG["imgsz"],
                max_det = INFERENCE_CONFIG["max_det"],
                device  = INFERENCE_CONFIG["device"],
                verbose = False,
                stream  = False,
            )

            boxes               = results[0].boxes
            fps                 = fps_counter.tick()
            annotated, count_dict = draw_detections(frame, boxes, fps=fps)
            total_obj           = sum(count_dict.values())
            total_detections   += total_obj

            # ── Simpan ke CSV/JSON setiap N frame ──
            if frame_count % log_interval == 0:
                if save_csv:
                    save_detection_csv(source=source, count_dict=count_dict, timestamp=ts)
                if save_json:
                    save_detection_json(
                        source=source, count_dict=count_dict,
                        boxes_raw=boxes, timestamp=ts
                    )

            # ── Tulis ke output video ──
            if writer:
                writer.write(annotated)

            # ── Tampilkan frame ──
            if show:
                cv2.imshow("Waste Detection — tekan Q untuk keluar", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q atau ESC
                    logger.info("⏹️  Dihentikan oleh user.")
                    break
                # Pause dengan spasi
                elif key == ord(" "):
                    cv2.waitKey(0)

    except KeyboardInterrupt:
        logger.info("\n⏹️  Dihentikan (Ctrl+C).")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Statistik akhir
    logger.info("\n" + "─" * 40)
    logger.info("📊 Statistik Inference:")
    logger.info(f"   Total frame    : {frame_count}")
    logger.info(f"   Total deteksi  : {total_detections}")
    logger.info(f"   Rata-rata FPS  : {fps:.1f}")
    logger.info("─" * 40)

    from config import OUTPUT_CSV, OUTPUT_JSON
    if save_csv:
        logger.info(f"   CSV disimpan ke: {OUTPUT_CSV}")
    if save_json:
        logger.info(f"   JSON disimpan ke: {OUTPUT_JSON}")


# ─── INFERENCE BATCH ───────────────────────────────────────────────────────────
def predict_folder(
    model,
    folder_path: str,
    conf: float = INFERENCE_CONFIG["conf"],
    save_output: bool = True,
):
    """Inference batch pada semua gambar dalam folder."""
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"❌ Folder tidak ditemukan: {folder_path}")
        return

    images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
    if not images:
        logger.warning(f"⚠️  Tidak ada gambar ditemukan di: {folder_path}")
        return

    logger.info(f"📁 Batch inference: {len(images)} gambar")
    out_dir = get_output_path(str(folder))
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images, 1):
        logger.info(f"  [{i}/{len(images)}] {img_path.name}")
        predict_image(model, str(img_path), save_output=save_output, show=False)

    logger.info(f"✅ Batch selesai. Output: {out_dir}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Waste Detection — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python predict.py --source 0                     # webcam
  python predict.py --source video.mp4             # video file
  python predict.py --source foto.jpg              # satu gambar
  python predict.py --source folder/gambar/        # batch folder
  python predict.py --source rtsp://ip:port/stream # CCTV

  # Tambahan opsi
  python predict.py --source 0 --conf 0.3 --no-save
  python predict.py --source 0 --model models/custom.pt
        """
    )

    parser.add_argument("--source",   type=str,   default="0",
                        help="Sumber input: 0=webcam, path file/folder, atau rtsp://...")
    parser.add_argument("--conf",     type=float, default=INFERENCE_CONFIG["conf"],
                        help=f"Confidence threshold (default: {INFERENCE_CONFIG['conf']})")
    parser.add_argument("--model",    type=str,   default=None,
                        help="Path ke file model .pt")
    parser.add_argument("--no-save",  action="store_true",
                        help="Jangan simpan output video/gambar")
    parser.add_argument("--no-show",  action="store_true",
                        help="Jangan tampilkan preview (headless mode)")
    parser.add_argument("--no-csv",   action="store_true",
                        help="Jangan simpan ke CSV")
    parser.add_argument("--no-json",  action="store_true",
                        help="Jangan simpan ke JSON")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🗑️  Waste Detection — Inference")
    logger.info("=" * 60)

    # Load model
    model = load_model(args.model)

    source      = args.source
    save_output = not args.no_save
    show        = not args.no_show
    save_csv    = not args.no_csv
    save_json   = not args.no_json

    # Tentukan mode inference
    src_path = Path(source)

    if src_path.is_dir():
        # Batch folder
        predict_folder(model, source, conf=args.conf, save_output=save_output)

    elif src_path.is_file() and src_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        # Gambar tunggal
        predict_image(model, source, save_output=save_output, conf=args.conf, show=show)

    else:
        # Video, webcam, atau RTSP
        predict_stream(
            model       = model,
            source      = source,
            save_output = save_output,
            conf        = args.conf,
            show        = show,
            save_csv    = save_csv,
            save_json   = save_json,
        )


if __name__ == "__main__":
    main()
