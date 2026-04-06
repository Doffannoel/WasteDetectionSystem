"""
main.py — Entry point interaktif untuk Waste Detection System

Menu:
  1. Persiapan dataset
  2. Training model
  3. Evaluasi model
  4. Inference webcam
  5. Inference video file
  6. Inference gambar
  7. Lihat statistik output
  8. Keluar

Jalankan: python main.py
"""

import sys
from pathlib import Path

from config import OUTPUT_CSV, OUTPUT_JSON, TRAINED_MODEL, DATASET_YAML
from utils import logger


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║         🗑️  WASTE DETECTION SYSTEM v1.0              ║
║         Realtime Garbage Detection with YOLO          ║
╚══════════════════════════════════════════════════════╝
""")


def print_menu():
    trained = "✅" if TRAINED_MODEL.exists() else "❌ (belum training)"
    dataset = "✅" if DATASET_YAML.exists()  else "❌ (belum siap)"

    print(f"""
Status:
  Dataset YAML : {dataset}
  Trained Model: {trained}

Menu:
  1. Persiapan Dataset (TACO + Roboflow)
  2. Training Model
  3. Evaluasi Model
  4. Inference — Webcam (realtime)
  5. Inference — Video File
  6. Inference — Gambar
  7. Statistik Hasil Deteksi
  0. Keluar
""")


def run_dataset_prep():
    print("\n📦 Memulai persiapan dataset...\n")
    from prepare_dataset import main
    main()


def run_training():
    resume = input("Resume dari checkpoint terakhir? (y/N): ").strip().lower() == "y"
    print("\n🚀 Memulai training...\n")
    from train import check_prerequisites, train, evaluate
    check_prerequisites()
    train(resume=resume)
    print("\n📊 Evaluasi model...")
    evaluate()


def run_evaluation():
    model_path = input(f"Path model (Enter untuk default {TRAINED_MODEL}): ").strip()
    if not model_path:
        model_path = None
    from train import evaluate
    evaluate(model_path)


def run_webcam():
    conf = input("Confidence threshold (Enter untuk 0.35): ").strip()
    conf = float(conf) if conf else 0.35
    print(f"\n📹 Membuka webcam... (confidence={conf})\n")
    from predict import load_model, predict_stream
    model = load_model()
    predict_stream(model, source="0", conf=conf, show=True)


def run_video():
    path = input("Path video file: ").strip()
    if not path:
        print("❌ Path kosong.")
        return
    conf = input("Confidence threshold (Enter untuk 0.35): ").strip()
    conf = float(conf) if conf else 0.35
    from predict import load_model, predict_stream
    model = load_model()
    predict_stream(model, source=path, conf=conf, show=True)


def run_image():
    path = input("Path gambar: ").strip()
    if not path:
        print("❌ Path kosong.")
        return
    conf = input("Confidence threshold (Enter untuk 0.35): ").strip()
    conf = float(conf) if conf else 0.35
    from predict import load_model, predict_image
    model = load_model()
    predict_image(model, path, conf=conf, show=True)


def show_stats():
    """Tampilkan ringkasan statistik dari file output."""
    print("\n📊 Statistik Hasil Deteksi\n")

    if OUTPUT_CSV.exists():
        import csv
        from collections import defaultdict

        class_totals: dict = defaultdict(int)
        row_count = 0

        with open(OUTPUT_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cls  = row.get("class_name", "unknown")
                cnt  = int(row.get("count", 0))
                class_totals[cls] += cnt
                row_count += 1

        print(f"  File CSV  : {OUTPUT_CSV}")
        print(f"  Total log : {row_count} entries\n")
        print("  Deteksi per kelas:")
        for cls, total in sorted(class_totals.items(), key=lambda x: -x[1]):
            bar = "█" * min(total // 5, 40)
            print(f"    {cls:20s}: {total:5d}  {bar}")
    else:
        print(f"  Belum ada data CSV di {OUTPUT_CSV}")

    if OUTPUT_JSON.exists():
        import json
        with open(OUTPUT_JSON) as f:
            data = json.load(f)
        print(f"\n  File JSON : {OUTPUT_JSON}")
        print(f"  Total entri: {len(data)}")
    else:
        print(f"\n  Belum ada data JSON di {OUTPUT_JSON}")


def main():
    print_banner()

    while True:
        print_menu()
        choice = input("Pilih menu (0-7): ").strip()

        if choice == "1":
            run_dataset_prep()
        elif choice == "2":
            run_training()
        elif choice == "3":
            run_evaluation()
        elif choice == "4":
            run_webcam()
        elif choice == "5":
            run_video()
        elif choice == "6":
            run_image()
        elif choice == "7":
            show_stats()
        elif choice == "0":
            print("\n👋 Sampai jumpa!\n")
            sys.exit(0)
        else:
            print("⚠️  Pilihan tidak valid.")

        input("\nTekan Enter untuk kembali ke menu...")


if __name__ == "__main__":
    main()
