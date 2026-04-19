# Waste Detection System

Sistem deteksi sampah realtime berbasis YOLO untuk klasifikasi material sampah dari gambar, video, webcam, dan stream kamera. Proyek ini memakai gabungan dataset TACO dan Roboflow, lalu memetakan label ke kelas akhir berbahasa Indonesia.

## Deskripsi Singkat

| Atribut | Detail |
|---|---|
| Model dasar | YOLOv8s |
| Dataset | TACO + Roboflow Garbage Classification |
| Kelas akhir | `plastik`, `kertas_kardus`, `logam`, `kaca`, `kantong_plastik`, `sampah` |
| Input | Gambar, video, webcam, RTSP/CCTV |
| Output | Bounding box realtime, CSV, JSON, video hasil, gambar hasil bbox |
| Folder model | `models/` |
| Folder output | `outputs/` dan `outputs/hasil_gambar_bbox/` |

## Struktur Folder

```text
WasteDetectionSystem/
|-- main.py
|-- train.py
|-- predict.py
|-- prepare_dataset.py
|-- config.py
|-- utils.py
|-- requirements.txt
|-- README.md
|
|-- data/
|   |-- taco/
|   |-- taco_yolo/
|   |-- roboflow_raw/
|   `-- roboflow_yolo/
|
|-- datasets/
|   |-- final/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   `-- waste_dataset.yaml
|
|-- models/
|   |-- best.pt
|   `-- best_v2.pt
|
|-- runs/
|   `-- waste_detection/
|
`-- outputs/
    |-- detections.csv
    |-- detections.json
    |-- *.mp4
    `-- hasil_gambar_bbox/
        `-- sample_test/
```

## Kelas Deteksi

Model akhir memakai 6 kelas berikut.

| ID | Kelas |
|---|---|
| 0 | `plastik` |
| 1 | `kertas_kardus` |
| 2 | `logam` |
| 3 | `kaca` |
| 4 | `kantong_plastik` |
| 5 | `sampah` |

## Dataset

### TACO
- Sumber: `https://github.com/pedropro/TACO`
- Format asli: COCO JSON
- Dipakai sebagai sumber anotasi material sampah di lingkungan nyata

### Roboflow Garbage Classification
- Sumber: `https://universe.roboflow.com/material-identification/garbage-classification-3`
- Format: YOLO
- Dipakai sebagai sumber utama untuk menambah jumlah data dan variasi objek

### Mapping Label

Contoh pemetaan label ke kelas final:

| Label Asli | Kelas Final |
|---|---|
| `plastic`, `Plastic bottle` | `plastik` |
| `paper`, `cardboard` | `kertas_kardus` |
| `metal`, `Drink can` | `logam` |
| `glass`, `Glass bottle` | `kaca` |
| `Plastic bag & wrapper` | `kantong_plastik` |
| `trash`, `organic`, `Other` | `sampah` |

## Instalasi

### Lokal

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!rm -rf WasteDetectionSystem
!git clone URL_REPO_KAMU WasteDetectionSystem
%cd /content/WasteDetectionSystem
!pip install -r requirements.txt
!pip install ultralytics roboflow opencv-python-headless pillow pyyaml
```

## Persiapan Dataset

### Otomatis dengan Roboflow API Key

```bash
set ROBOFLOW_API_KEY=api_key_anda
python prepare_dataset.py
```

Di Colab:

```python
import os
os.environ["ROBOFLOW_API_KEY"] = "api_key_anda"
!python prepare_dataset.py
```

### Output Dataset

Kalau sukses, folder berikut akan terbentuk:

```text
datasets/final/train/images
datasets/final/train/labels
datasets/final/val/images
datasets/final/val/labels
datasets/final/test/images
datasets/final/test/labels
```

## Training

### Training dari model dasar

```bash
python train.py
```

Mode ini akan:
- memakai `models/best_v2.pt` jika sudah ada
- kalau belum ada, memakai `models/best.pt`
- kalau keduanya belum ada, fallback ke `BASE_MODEL` di `config.py`

### Resume checkpoint lama

```bash
python train.py --resume
```

Catatan:
- `--resume` mengikuti state checkpoint `last.pt`
- total epoch akan mengikuti checkpoint lama, bukan angka epoch baru di `config.py`
- gunakan mode ini hanya jika Anda memang ingin melanjutkan run lama apa adanya

### Evaluasi saja

```bash
python train.py --eval --model models/best_v2.pt
```

### Export model

```bash
python train.py --export onnx
```

## Evaluasi dan Confusion Matrix

Untuk menampilkan confusion matrix dan plot evaluasi:

```bash
python train.py --eval --model models/best_v2.pt
```

Hasil evaluasi biasanya tersimpan di:

```text
runs/detect/val/
```

File penting yang biasanya muncul:
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `F1_curve.png`
- `PR_curve.png`
- `P_curve.png`
- `R_curve.png`

## Inference

### Webcam

```bash
python predict.py --model models/best_v2.pt --source 0
```

### Gambar

```bash
python predict.py --model models/best_v2.pt --source contoh.jpg
```

### Video

```bash
python predict.py --model models/best_v2.pt --source video.mp4
```

### Folder gambar

```bash
python predict.py --model models/best_v2.pt --source folder_gambar
```

### RTSP / CCTV / Kamera HP

```bash
python predict.py --model models/best_v2.pt --source "rtsp://user:pass@ip:port/stream"
```

Atau jika memakai DroidCam / IP camera:

```bash
python predict.py --model models/best_v2.pt --source "http://IP_KAMERA/video"
```

## Output Hasil

### Realtime
- bounding box tampil langsung di window preview
- label tampil dalam Bahasa Indonesia
- confidence tampil sebagai `Akurasi`

### File hasil

Output tersimpan ke:

```text
outputs/
outputs/hasil_gambar_bbox/
outputs/hasil_gambar_bbox/sample_test/
```

Jenis file:
- `detections.csv`
- `detections.json`
- `*.mp4` hasil video inference
- `*.jpg` hasil gambar dengan bounding box

### Contoh output gambar setelah training

Setelah `python train.py` selesai, script akan otomatis menyimpan beberapa contoh hasil prediksi test set ke:

```text
outputs/hasil_gambar_bbox/sample_test/
```

Folder ini cocok untuk bukti visual laporan atau presentasi.

## Konfigurasi Penting

File utama konfigurasi ada di [config.py](./config.py).

Yang paling sering diubah:

```python
BASE_MODEL = "yolov8s.pt"
TRAINED_MODEL = MODEL_DIR / "best_v2.pt"
```

```python
TRAIN_CONFIG = {
    "epochs": 40,
    "batch": 16,
    "imgsz": 736,
    "save_period": 1,
}
```

```python
INFERENCE_CONFIG = {
    "conf": 0.50,
    "ignore_classes": ["sampah"],
}
```

## Saran Alur Kerja

Untuk Colab yang sering putus, alur paling aman:

1. Jalankan `prepare_dataset.py`
2. Jalankan `python train.py`
3. Backup `runs/`, `models/`, dan `datasets/` ke Drive
4. Kalau pindah akun Colab, restore semua folder itu
5. Gunakan `--resume` hanya jika benar-benar ingin melanjutkan run checkpoint lama
6. Untuk fine-tuning tambahan, lebih aman pakai `python train.py` biasa dari model `best_v2.pt`

## Backup ke Google Drive

```python
!mkdir -p /content/drive/MyDrive/waste_backup/datasets
!mkdir -p /content/drive/MyDrive/waste_backup/runs
!mkdir -p /content/drive/MyDrive/waste_backup/models

!rsync -a /content/WasteDetectionSystem/datasets/ /content/drive/MyDrive/waste_backup/datasets/
!rsync -a /content/WasteDetectionSystem/runs/ /content/drive/MyDrive/waste_backup/runs/
!rsync -a /content/WasteDetectionSystem/models/ /content/drive/MyDrive/waste_backup/models/
```

## Troubleshooting

| Masalah | Penyebab umum | Solusi |
|---|---|---|
| `datasets/final` tidak ada | dataset gagal diprepare | jalankan lagi `python prepare_dataset.py` |
| Roboflow 401 | API key tidak valid | set `ROBOFLOW_API_KEY` yang benar |
| path YAML jadi Windows path di Colab | YAML dibuat di lokal | tulis ulang `datasets/waste_dataset.yaml` dengan path `/content/...` |
| resume tetap ikut epoch lama | checkpoint menyimpan total epoch lama | jangan pakai `--resume`, pakai `python train.py` biasa |
| webcam tidak terbuka | index kamera salah | coba `--source 1` atau `--source 2` |
| inference wajah terdeteksi sampah | false positive | naikkan `conf`, pakai filter kelas dan box area |

## Catatan

- `best.pt` bisa dipakai sebagai model hasil training lama
- `best_v2.pt` dipakai untuk model hasil training lanjutan
- saat training non-resume, epoch di log memang mulai lagi dari 1, tetapi bobot model tetap berasal dari model hasil training sebelumnya

## Penutup

Proyek ini cocok untuk demo deteksi sampah material berbasis computer vision, eksperimen fine-tuning YOLO, dan bahan presentasi karena sudah mendukung:
- label Bahasa Indonesia
- confusion matrix evaluasi
- output visual bounding box
- workflow lokal maupun Google Colab
