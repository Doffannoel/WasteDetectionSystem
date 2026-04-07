"""
prepare_dataset.py â€” Persiapan dan penggabungan dataset TACO + Roboflow

Langkah yang dilakukan:
1. Download dataset TACO (annotations + images)
2. Download dataset Roboflow via roboflow SDK
3. Konversi TACO COCO format â†’ YOLO format
4. Normalisasi label ke 6 kelas utama
5. Gabungkan kedua dataset
6. Split train/val/test
7. Buat file waste_dataset.yaml

Jalankan: python prepare_dataset.py
"""

import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    CLASS_NAMES, DATASET_DIR, DATA_DIR,
    DATASET_YAML, RANDOM_SEED, ROBOFLOW_PROJECT,
    ROBOFLOW_VERSION, ROBOFLOW_WORKSPACE,
    TACO_ANNO_FILE, TACO_IMAGES_DIR,
    TRAIN_RATIO, VAL_RATIO,
)
from utils import ROBOFLOW_LABEL_MAP, TACO_LABEL_MAP, logger, map_label


def _find_roboflow_dataset_root(search_root: Path) -> Optional[Path]:
    """
    Cari root dataset Roboflow YOLO (harus ada data.yaml dan folder split).
    """
    if not search_root.exists():
        return None

    yaml_candidates = list(search_root.glob("**/data.yaml")) + list(search_root.glob("**/*.yaml"))
    for yaml_file in yaml_candidates:
        root = yaml_file.parent
        has_train = (root / "train" / "images").exists()
        has_valid_or_val = (root / "valid" / "images").exists() or (root / "val" / "images").exists()
        if has_train and has_valid_or_val:
            return root
    return None


def _download_roboflow_zip_direct(api_key: str, output_dir: Path) -> Optional[Path]:
    """
    Fallback download langsung ke API Roboflow (tanpa SDK parser).
    Mengembalikan root dataset YOLO jika berhasil.
    """
    import urllib.request

    rf_dir = output_dir / "roboflow_raw"
    rf_dir.mkdir(parents=True, exist_ok=True)

    for fmt in ["yolov8", "yolov5"]:
        api_url = (
            f"https://api.roboflow.com/dataset/"
            f"{ROBOFLOW_WORKSPACE}/{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}/download/{fmt}"
            f"?api_key={api_key}"
        )
        tmp_path = rf_dir / f"roboflow_{fmt}.bin"
        zip_path = rf_dir / f"roboflow_{fmt}.zip"
        try:
            logger.info(f"  Fallback direct API: {api_url}")
            urllib.request.urlretrieve(api_url, tmp_path)

            # Jika response langsung ZIP
            if zipfile.is_zipfile(tmp_path):
                if zip_path.exists():
                    zip_path.unlink()
                tmp_path.rename(zip_path)
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(rf_dir)
                root = _find_roboflow_dataset_root(rf_dir)
                if root:
                    return root
                continue

            # Jika response JSON/link, cari URL zip lalu unduh lagi
            text = tmp_path.read_text(encoding="utf-8", errors="ignore")
            zip_urls = re.findall(r"https?://[^\\s\"']+\\.zip[^\\s\"']*", text)
            for zurl in zip_urls:
                try:
                    urllib.request.urlretrieve(zurl, zip_path)
                    if zipfile.is_zipfile(zip_path):
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            zf.extractall(rf_dir)
                        root = _find_roboflow_dataset_root(rf_dir)
                        if root:
                            return root
                except Exception as e:
                    logger.warning(f"  Fallback URL zip gagal: {e}")
        except Exception as e:
            logger.warning(f"  Fallback direct API gagal ({fmt}): {e}")

    return None


# â”€â”€â”€ TACO DOWNLOAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def download_taco(max_images: int = 1500):
    """
    Download dataset TACO menggunakan script resmi dari GitHub.
    Limit ke max_images agar tidak terlalu besar untuk demo.
    """
    taco_dir = DATA_DIR / "taco"
    taco_dir.mkdir(parents=True, exist_ok=True)

    anno_file = taco_dir / "annotations.json"

    import urllib.request
    if anno_file.exists():
        logger.info("âœ… TACO annotations sudah ada, skip download annotations.")
    else:
        logger.info("ðŸ“¥ Download TACO annotations...")
        url = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
        try:
            urllib.request.urlretrieve(url, anno_file)
            logger.info(f"âœ… TACO annotations disimpan ke {anno_file}")
        except Exception as e:
            logger.error(f"âŒ Gagal download TACO annotations: {e}")
            logger.info("â„¹ï¸  Download manual: https://github.com/pedropro/TACO")
            return None

    # Download images menggunakan script TACO
    img_dir = taco_dir / "images"
    img_dir.mkdir(exist_ok=True)

    if (img_dir / "batch_1").exists():
        logger.info("âœ… TACO images sudah ada.")
        return str(anno_file)

    logger.info("ðŸ“¥ Downloading TACO images (ini bisa memakan waktu ~5-15 menit)...")
    download_script = taco_dir / "download_dataset.py"

    # Download script dari GitHub (beberapa URL fallback, karena struktur repo bisa berubah)
    script_urls = [
        "https://raw.githubusercontent.com/pedropro/TACO/master/data/download_dataset.py",
        "https://raw.githubusercontent.com/pedropro/TACO/main/data/download_dataset.py",
        "https://raw.githubusercontent.com/pedropro/TACO/master/download_dataset.py",
        "https://raw.githubusercontent.com/pedropro/TACO/main/download_dataset.py",
    ]
    script_downloaded = False
    for script_url in script_urls:
        try:
            urllib.request.urlretrieve(script_url, download_script)
            script_downloaded = True
            logger.info(f"✅ Download script TACO berhasil dari: {script_url}")
            break
        except Exception as e:
            logger.warning(f"⚠️ Gagal download script TACO dari {script_url}: {e}")

    if not script_downloaded:
        logger.warning(
            "⚠️ Semua URL script TACO gagal diunduh. "
            "Lanjut tanpa TACO images (pipeline tetap berjalan untuk sumber lain)."
        )
        return str(anno_file)

    try:
        result = subprocess.run(
            [sys.executable, str(download_script),
             "--dataset_path", str(taco_dir),
             "--round", "all"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            logger.info("âœ… TACO images berhasil didownload.")
        else:
            logger.warning(f"âš ï¸ TACO image download: {result.stderr[:300]}")
    except Exception as e:
        logger.error(f"âŒ Error download images: {e}")

    return str(anno_file)


# â”€â”€â”€ TACO COCO â†’ YOLO â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def convert_taco_to_yolo(
    anno_file: str,
    image_dir: Path,
    output_dir: Path,
) -> int:
    """
    Konversi TACO (format COCO) ke format YOLO.
    
    Format YOLO per baris: <class_id> <cx> <cy> <w> <h>
    Semua nilai relatif terhadap ukuran gambar (0.0 - 1.0).
    
    Returns: jumlah gambar yang berhasil dikonversi
    """
    logger.info("ðŸ”„ Konversi TACO â†’ YOLO format...")

    with open(anno_file) as f:
        data = json.load(f)

    # Build mapping: image_id â†’ info gambar
    images_map: Dict[int, dict] = {img["id"]: img for img in data["images"]}

    # Build mapping: category_id â†’ nama label â†’ kelas kita
    cat_map: Dict[int, Optional[str]] = {}
    for cat in data["categories"]:
        mapped = map_label(cat["name"], source="taco")
        cat_map[cat["id"]] = mapped
        if mapped:
            logger.debug(f"  TACO '{cat['name']}' â†’ '{mapped}'")

    # Kelompokkan annotations per image
    img_annotations: Dict[int, List[dict]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        img_annotations.setdefault(img_id, []).append(ann)

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    imgs_dir   = output_dir / "images"
    labels_dir.mkdir(exist_ok=True)
    imgs_dir.mkdir(exist_ok=True)

    converted = 0
    skipped   = 0

    for img_id, img_info in images_map.items():
        img_filename = img_info["file_name"]  # e.g. "batch_1/000001.jpg"
        img_path     = image_dir / img_filename

        if not img_path.exists():
            skipped += 1
            continue

        annotations = img_annotations.get(img_id, [])
        if not annotations:
            skipped += 1
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        yolo_lines = []
        for ann in annotations:
            cat_id    = ann["category_id"]
            cls_name  = cat_map.get(cat_id)
            if cls_name is None:
                continue

            cls_id = CLASS_NAMES.index(cls_name)

            # COCO bbox: [x, y, width, height] dalam pixel
            x, y, w, h = ann["bbox"]

            # Konversi ke YOLO: cx, cy, w, h relatif
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h

            # Validasi nilai
            if not all(0.0 <= v <= 1.0 for v in [cx, cy, nw, nh]):
                continue
            if nw < 0.005 or nh < 0.005:   # skip bbox terlalu kecil
                continue

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not yolo_lines:
            continue

        # Salin gambar
        safe_name = img_filename.replace("/", "_").replace("\\", "_")
        dest_img  = imgs_dir / f"taco_{safe_name}"
        shutil.copy2(img_path, dest_img)

        # Simpan label
        label_file = labels_dir / f"taco_{Path(safe_name).stem}.txt"
        label_file.write_text("\n".join(yolo_lines))

        converted += 1

    logger.info(f"âœ… TACO: {converted} gambar dikonversi, {skipped} dilewati.")
    return converted


# â”€â”€â”€ ROBOFLOW DOWNLOAD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def download_roboflow_dataset(output_dir: Path) -> Optional[Path]:
    """
    Download dataset dari Roboflow menggunakan roboflow SDK.
    Dataset: Garbage Classification by Roboflow Universe
    """
    rf_dir = output_dir / "roboflow_raw"

    existing_root = _find_roboflow_dataset_root(rf_dir)
    if existing_root:
        logger.info(f"âœ… Roboflow dataset sudah ada: {existing_root}")
        return existing_root

    # Catatan: beberapa versi roboflow SDK bisa skip download jika folder location
    # sudah ada. Jadi jangan pre-create folder target.
    if rf_dir.exists() and not any(rf_dir.iterdir()):
        try:
            rf_dir.rmdir()
            logger.info(f"  Folder kosong dihapus agar download fresh: {rf_dir}")
        except Exception:
            pass

    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("âŒ roboflow tidak terinstall. Jalankan: pip install roboflow")
        return None

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        logger.warning(
            "âš ï¸  ROBOFLOW_API_KEY tidak ditemukan di environment.\n"
            "   Set dengan: export ROBOFLOW_API_KEY=your_key\n"
            "   Atau download manual dari: https://universe.roboflow.com\n"
            "   Pilih format 'YOLOv8' saat download."
        )
        # Coba download public dataset tanpa API key
        api_key = "YOUR_KEY_HERE"

    try:
        logger.info(f"ðŸ“¥ Download Roboflow dataset: {ROBOFLOW_PROJECT}...")
        rf      = Roboflow(api_key=api_key)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)

        dataset = None
        last_err = None
        # Coba beberapa format export untuk kompatibilitas SDK/dataset.
        for fmt in ["yolov8", "yolov5"]:
            try:
                logger.info(f"  Mencoba export format: {fmt} (dengan location)")
                dataset = project.version(ROBOFLOW_VERSION).download(fmt, location=str(rf_dir))
                break
            except Exception as e:
                last_err = e
                logger.warning(f"  Gagal export {fmt} (dengan location): {e}")

        # Fallback: beberapa versi SDK lebih stabil tanpa parameter location.
        if dataset is None:
            for fmt in ["yolov8", "yolov5"]:
                try:
                    logger.info(f"  Mencoba export format: {fmt} (tanpa location)")
                    dataset = project.version(ROBOFLOW_VERSION).download(fmt)
                    break
                except Exception as e:
                    last_err = e
                    logger.warning(f"  Gagal export {fmt} (tanpa location): {e}")

        if dataset is None:
            raise RuntimeError(f"Semua percobaan export Roboflow gagal. Last error: {last_err}")

        # Beberapa versi SDK menyimpan dataset di subfolder/lokasi lain.
        dataset_location = Path(getattr(dataset, "location", str(rf_dir)))
        logger.info(f"  Roboflow dataset.location: {dataset_location}")

        # Jika SDK memberi ZIP, ekstrak dulu ke rf_dir.
        zip_candidates: List[Path] = []
        if dataset_location.is_file() and dataset_location.suffix.lower() == ".zip":
            zip_candidates.append(dataset_location)
        zip_candidates.extend(list(rf_dir.glob("**/*.zip")))
        zip_candidates.extend(list(output_dir.glob("**/*.zip")))

        for zip_path in zip_candidates:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(rf_dir)
                logger.info(f"âœ… ZIP diekstrak: {zip_path} -> {rf_dir}")
            except Exception as e:
                logger.warning(f"âš ï¸ Gagal ekstrak ZIP {zip_path}: {e}")

        search_roots = [dataset_location, rf_dir, output_dir, Path.cwd()]
        detected_root = None
        for root in search_roots:
            detected_root = _find_roboflow_dataset_root(root)
            if detected_root:
                break

        if detected_root:
            logger.info(f"âœ… Roboflow dataset terdeteksi di: {detected_root}")
            return detected_root

        # Jika call awal "sukses" tapi hasil tetap kosong, coba ulang tanpa location.
        logger.warning("âš ï¸ Hasil download kosong, mencoba ulang tanpa parameter location...")
        retry_dataset = None
        for fmt in ["yolov8", "yolov5"]:
            try:
                logger.info(f"  Retry export format: {fmt} (tanpa location)")
                retry_dataset = project.version(ROBOFLOW_VERSION).download(fmt)
                break
            except Exception as e:
                logger.warning(f"  Retry gagal {fmt} (tanpa location): {e}")

        if retry_dataset is not None:
            retry_location = Path(getattr(retry_dataset, "location", str(Path.cwd())))
            logger.info(f"  Retry dataset.location: {retry_location}")
            retry_roots = [retry_location, Path.cwd(), output_dir, rf_dir]
            for root in retry_roots:
                detected_root = _find_roboflow_dataset_root(root)
                if detected_root:
                    logger.info(f"âœ… Roboflow dataset terdeteksi setelah retry di: {detected_root}")
                    return detected_root

        # Fallback terakhir: download langsung via API ZIP endpoint.
        if api_key and api_key != "YOUR_KEY_HERE":
            direct_root = _download_roboflow_zip_direct(api_key, output_dir)
            if direct_root:
                logger.info(f"âœ… Roboflow dataset terdeteksi via direct API di: {direct_root}")
                return direct_root

        logger.warning(
            "âš ï¸ Download Roboflow selesai, tapi struktur YOLO tidak terdeteksi "
            f"di roots: {[str(r) for r in search_roots]}."
        )
        for root in search_roots:
            try:
                if root.exists():
                    sample_files = [str(p) for p in list(root.glob("**/*"))[:10]]
                    logger.info(f"  Debug isi {root}: {sample_files}")
            except Exception:
                pass
        return dataset_location
    except Exception as e:
        logger.error(f"âŒ Gagal download Roboflow dataset: {e}")
        logger.info(
            "â„¹ï¸  Alternatif: download manual dari\n"
            "   https://universe.roboflow.com/material-identification/garbage-classification-3\n"
            "   Pilih Export â†’ YOLOv8 format â†’ Download ZIP\n"
            "   Ekstrak ke: data/roboflow_raw/"
        )
        return None


def import_roboflow_yolo(
    roboflow_dir: Path,
    output_dir: Path,
) -> int:
    """
    Import dataset Roboflow yang sudah dalam format YOLO.
    Re-map label ke kelas kita jika perlu.
    
    Returns: jumlah gambar yang berhasil diimport
    """
    logger.info("ðŸ”„ Import Roboflow dataset...")
    roboflow_root = _find_roboflow_dataset_root(roboflow_dir) or roboflow_dir

    # Cari data.yaml untuk mendapatkan mapping kelas asli
    yaml_files = list(roboflow_root.glob("**/data.yaml")) + list(roboflow_root.glob("**/*.yaml"))
    if not yaml_files:
        logger.error(f"âŒ Tidak ditemukan YAML di Roboflow dataset: {roboflow_root}")
        return 0

    import yaml
    with open(yaml_files[0]) as f:
        rf_config = yaml.safe_load(f)

    # Kelas asli Roboflow
    rf_classes: List[str] = rf_config.get("names", [])
    if isinstance(rf_classes, dict):
        rf_classes = [rf_classes[i] for i in sorted(rf_classes.keys())]

    logger.info(f"  Kelas Roboflow asli: {rf_classes}")

    # Build mapping: rf_class_id â†’ our_class_id
    rf_to_ours: Dict[int, Optional[int]] = {}
    for i, name in enumerate(rf_classes):
        mapped = map_label(name, source="roboflow")
        if mapped and mapped in CLASS_NAMES:
            rf_to_ours[i] = CLASS_NAMES.index(mapped)
        else:
            rf_to_ours[i] = CLASS_NAMES.index("trash")  # fallback

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = output_dir / "labels"
    imgs_dir   = output_dir / "images"
    labels_dir.mkdir(exist_ok=True)
    imgs_dir.mkdir(exist_ok=True)

    imported = 0

    for split in ["train", "valid", "val", "test"]:
        split_img_dir   = roboflow_root / split / "images"
        split_label_dir = roboflow_root / split / "labels"

        if not split_img_dir.exists():
            continue

        for img_path in split_img_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            label_path = split_label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            # Re-map label
            new_lines = []
            for line in label_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                rf_cls_id = int(parts[0])
                our_cls_id = rf_to_ours.get(rf_cls_id)
                if our_cls_id is None:
                    continue
                new_lines.append(f"{our_cls_id} {' '.join(parts[1:])}")

            if not new_lines:
                continue

            # Salin gambar dan label
            dest_img   = imgs_dir / f"rf_{img_path.name}"
            dest_label = labels_dir / f"rf_{img_path.stem}.txt"
            shutil.copy2(img_path, dest_img)
            dest_label.write_text("\n".join(new_lines))
            imported += 1

    logger.info(f"âœ… Roboflow: {imported} gambar diimport.")
    return imported


# â”€â”€â”€ GABUNGKAN DAN SPLIT â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def merge_and_split(source_dirs: List[Path], output_dir: Path):
    """
    Gabungkan semua dataset, lalu split ke train/val/test.
    Pastikan distribusi kelas seimbang dengan stratified sampling.
    """
    logger.info("ðŸ”€ Menggabungkan dan split dataset...")

    # Kumpulkan semua pasangan (image, label)
    all_pairs: List[Tuple[Path, Path]] = []

    for src_dir in source_dirs:
        imgs_dir   = src_dir / "images"
        labels_dir = src_dir / "labels"
        if not imgs_dir.exists():
            continue
        for img_path in imgs_dir.iterdir():
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                all_pairs.append((img_path, label_path))

    if not all_pairs:
        logger.error("âŒ Tidak ada data untuk digabungkan!")
        return

    logger.info(f"  Total gambar: {len(all_pairs)}")

    # Stratified split berdasarkan kelas mayoritas per gambar
    class_buckets: Dict[int, List[Tuple[Path, Path]]] = {i: [] for i in range(len(CLASS_NAMES))}
    for img_p, lbl_p in all_pairs:
        lines = lbl_p.read_text().strip().split("\n")
        class_ids = [int(l.split()[0]) for l in lines if l.strip()]
        if class_ids:
            majority = max(set(class_ids), key=class_ids.count)
            class_buckets[majority].append((img_p, lbl_p))

    # Analisis distribusi
    logger.info("  Distribusi kelas:")
    for cls_id, bucket in class_buckets.items():
        logger.info(f"    {CLASS_NAMES[cls_id]:20s}: {len(bucket):4d} gambar")

    # Split
    train_pairs, val_pairs, test_pairs = [], [], []
    random.seed(RANDOM_SEED)

    for cls_id, bucket in class_buckets.items():
        if not bucket:
            continue
        random.shuffle(bucket)
        n      = len(bucket)
        n_val  = max(1, int(n * VAL_RATIO))
        n_test = max(1, int(n * (1 - TRAIN_RATIO - VAL_RATIO)))
        n_train = n - n_val - n_test

        train_pairs.extend(bucket[:n_train])
        val_pairs.extend(bucket[n_train:n_train + n_val])
        test_pairs.extend(bucket[n_train + n_val:])

    logger.info(f"  Split â†’ train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    # Salin ke folder tujuan
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        dest_img   = output_dir / split_name / "images"
        dest_label = output_dir / split_name / "labels"
        dest_img.mkdir(parents=True, exist_ok=True)
        dest_label.mkdir(parents=True, exist_ok=True)

        for img_p, lbl_p in pairs:
            shutil.copy2(img_p, dest_img / img_p.name)
            shutil.copy2(lbl_p, dest_label / lbl_p.name)

    logger.info("âœ… Dataset berhasil digabungkan dan displit.")


# â”€â”€â”€ BUAT YAML â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def create_dataset_yaml(dataset_dir: Path, yaml_path: Path):
    """Buat file YAML konfigurasi dataset untuk YOLO training."""
    content = f"""# waste_dataset.yaml â€” Konfigurasi dataset untuk YOLO training
# Di-generate otomatis oleh prepare_dataset.py

path: {dataset_dir.resolve()}

train: train/images
val:   val/images
test:  test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}

# Catatan kelas:
# 0 - plastic        : botol plastik, gelas plastik, wadah plastik
# 1 - paper_cardboard: kertas, kardus, kotak
# 2 - metal          : kaleng, logam, foil
# 3 - glass          : botol kaca, pecahan kaca
# 4 - plastic_bag    : kantong plastik, sachet, bungkus
# 5 - trash          : sampah campuran, tidak teridentifikasi
"""
    yaml_path.write_text(content)
    logger.info(f"âœ… Dataset YAML dibuat: {yaml_path}")


# â”€â”€â”€ ANALISIS DISTRIBUSI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def analyze_dataset(dataset_dir: Path):
    """Analisis dan tampilkan statistik distribusi dataset."""
    logger.info("\nðŸ“Š Analisis Dataset:")
    total_images  = 0
    total_objects = 0
    class_counts  = {name: 0 for name in CLASS_NAMES}

    for split in ["train", "val", "test"]:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue

        split_images  = 0
        split_objects = 0

        for label_file in labels_dir.glob("*.txt"):
            lines = [l.strip() for l in label_file.read_text().split("\n") if l.strip()]
            if not lines:
                continue
            split_images += 1
            for line in lines:
                cls_id = int(line.split()[0])
                if 0 <= cls_id < len(CLASS_NAMES):
                    class_counts[CLASS_NAMES[cls_id]] += 1
                    split_objects += 1

        logger.info(f"  {split:6s}: {split_images:4d} gambar, {split_objects:5d} objek")
        total_images  += split_images
        total_objects += split_objects

    logger.info(f"\n  Total: {total_images} gambar, {total_objects} objek")
    logger.info("\n  Distribusi per kelas:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        bar = "â–ˆ" * min(count // 10, 40)
        logger.info(f"  {cls_name:20s}: {count:5d} {bar}")

    # Cek imbalance
    counts = list(class_counts.values())
    if counts and max(counts) > 0 and min(counts) < max(counts) * 0.1:
        logger.warning(
            "\n  âš ï¸  Dataset tidak seimbang. Saran:\n"
            "     - Gunakan class_weights di training\n"
            "     - Tambahkan augmentasi mosaic lebih agresif\n"
            "     - Pertimbangkan oversampling kelas minor\n"
            "     - Atau gabungkan kelas yang sangat sedikit ke 'trash'"
        )


# â”€â”€â”€ MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    """Jalankan pipeline persiapan dataset lengkap."""
    logger.info("=" * 60)
    logger.info("ðŸ—‘ï¸  Waste Detection â€” Persiapan Dataset")
    logger.info("=" * 60)

    merged_dir = DATASET_DIR / "merged"
    final_dir  = DATASET_DIR / "final"

    # Direktori untuk data mentah tiap sumber
    taco_yolo_dir = DATA_DIR / "taco_yolo"
    rf_yolo_dir   = DATA_DIR / "roboflow_yolo"

    # â”€â”€ Step 1: TACO â”€â”€
    logger.info("\n[1/5] Proses TACO dataset...")
    anno_file = download_taco(max_images=1500)
    if anno_file and Path(anno_file).exists():
        convert_taco_to_yolo(
            anno_file   = anno_file,
            image_dir   = DATA_DIR / "taco",
            output_dir  = taco_yolo_dir,
        )
    else:
        logger.warning("âš ï¸  TACO tidak tersedia, lanjut tanpa TACO.")

    # â”€â”€ Step 2: Roboflow â”€â”€
    logger.info("\n[2/5] Proses Roboflow dataset...")
    rf_raw = download_roboflow_dataset(DATA_DIR)
    if rf_raw and rf_raw.exists():
        import_roboflow_yolo(rf_raw, rf_yolo_dir)
    else:
        logger.warning("âš ï¸  Roboflow tidak tersedia, lanjut tanpa Roboflow.")

    # â”€â”€ Step 3: Gabungkan â”€â”€
    logger.info("\n[3/5] Menggabungkan dataset...")
    source_dirs = [d for d in [taco_yolo_dir, rf_yolo_dir] if d.exists()]
    if not source_dirs:
        logger.error("âŒ Tidak ada dataset yang berhasil dipersiapkan!")
        logger.info(
            "\nðŸ’¡ Saran: Download manual:\n"
            "   TACO   : https://github.com/pedropro/TACO\n"
            "   Roboflow: https://universe.roboflow.com/material-identification/garbage-classification-3\n"
            "   Lalu jalankan ulang script ini."
        )
        sys.exit(1)

    merge_and_split(source_dirs, final_dir)

    # â”€â”€ Step 4: Buat YAML â”€â”€
    logger.info("\n[4/5] Membuat dataset YAML...")
    create_dataset_yaml(final_dir, DATASET_YAML)

    # â”€â”€ Step 5: Analisis â”€â”€
    logger.info("\n[5/5] Analisis dataset...")
    analyze_dataset(final_dir)

    logger.info("\n" + "=" * 60)
    logger.info("âœ… Persiapan dataset selesai!")
    logger.info(f"   Dataset YAML: {DATASET_YAML}")
    logger.info(f"   Jalankan training: python train.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()




