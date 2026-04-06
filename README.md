# 🗑️ Waste Detection System

Sistem deteksi sampah realtime berbasis **YOLOv8n** menggunakan dataset **TACO** + **Roboflow Garbage Classification**. Dibuat untuk kebutuhan demo, tugas akhir, dan prototype smart waste monitoring.

---

## 📋 Deskripsi Proyek

| Atribut       | Detail                                         |
|---------------|------------------------------------------------|
| Model         | YOLOv8n (bisa diganti YOLO11n)                 |
| Dataset       | TACO + Roboflow Garbage Classification          |
| Kelas         | 6 kelas (plastic, paper_cardboard, metal, glass, plastic_bag, trash) |
| Input         | Gambar, video, webcam, RTSP/CCTV               |
| Output        | Video anotasi + CSV + JSON                     |
| Target device | Laptop CPU / GPU opsional                      |

---

## 🏗️ Struktur Folder

```
waste-detection/
├── main.py               # Entry point interaktif (menu)
├── train.py              # Pipeline training & evaluasi
├── predict.py            # Inference (gambar/video/webcam/RTSP)
├── prepare_dataset.py    # Download + konversi + merge dataset
├── config.py             # Konfigurasi terpusat
├── utils.py              # Fungsi utilitas & visualisasi
├── requirements.txt
├── README.md
│
├── data/
│   ├── taco/             # Raw TACO dataset
│   ├── taco_yolo/        # TACO setelah konversi ke YOLO format
│   └── roboflow_yolo/    # Roboflow setelah konversi
│
├── datasets/
│   ├── final/            # Dataset gabungan (train/val/test)
│   └── waste_dataset.yaml
│
├── models/
│   └── best.pt           # Trained model terbaik
│
├── runs/
│   └── waste_detection/  # Log training, weights, plots
│
└── outputs/
    ├── detections.csv
    ├── detections.json
    └── *.mp4             # Video hasil inference
```

---

## 📦 Dataset yang Digunakan

### 1. TACO Dataset (Primary)
- **Sumber**: [tacodataset.org](http://tacodataset.org/) / [GitHub](https://github.com/pedropro/TACO)
- **Isi**: 1500+ foto sampah di lingkungan nyata (jalanan, pantai, taman)
- **Format asli**: COCO JSON
- **Anotasi**: 60+ kelas — disederhanakan menjadi 6 kelas

### 2. Roboflow Garbage Classification (Supplementary)
- **Sumber**: [Roboflow Universe](https://universe.roboflow.com/material-identification/garbage-classification-3)
- **Isi**: Gambar sampah berlabel dengan variasi yang tinggi
- **Format**: YOLOv8 (langsung kompatibel)
- **Kelas**: cardboard, glass, metal, paper, plastic, trash

### Mapping Kelas

| Kelas Final       | Dari TACO                          | Dari Roboflow     |
|-------------------|------------------------------------|-------------------|
| `plastic`         | Plastic bottle, cup, straw, dsb    | plastic           |
| `paper_cardboard` | Paper, Cardboard, Paper bag        | paper, cardboard  |
| `metal`           | Drink can, Food can, Aluminium foil| metal             |
| `glass`           | Glass bottle, Broken glass         | glass             |
| `plastic_bag`     | Single-use carrier bag, Film       | —                 |
| `trash`           | Cigarette, Food waste, Unlabeled   | trash, organic    |

---

## 🚀 Cara Memulai

### 1. Install Dependencies

```bash
# Clone atau ekstrak project
cd waste-detection

# Buat virtual environment (disarankan)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# atau
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Persiapan Dataset

#### Opsi A: Download Otomatis (butuh API key Roboflow)
```bash
# Set Roboflow API key (gratis, daftar di roboflow.com)
export ROBOFLOW_API_KEY=your_api_key_here   # Linux/Mac
# atau
set ROBOFLOW_API_KEY=your_api_key_here      # Windows

# Jalankan script persiapan
python prepare_dataset.py
```

#### Opsi B: Download Manual
1. **TACO**: 
   - Kunjungi https://github.com/pedropro/TACO
   - Download `annotations.json`
   - Download images menggunakan `data/download_dataset.py --round all`
   - Taruh di `data/taco/`

2. **Roboflow**:
   - Kunjungi https://universe.roboflow.com/material-identification/garbage-classification-3
   - Klik **Export Dataset** → Format: **YOLOv8** → Download ZIP
   - Ekstrak ke `data/roboflow_raw/`

3. Jalankan ulang:
   ```bash
   python prepare_dataset.py
   ```

### 3. Training Model

```bash
# Training standard (80 epoch)
python train.py

# Resume jika training terputus
python train.py --resume

# Hanya evaluasi (setelah ada model)
python train.py --eval

# Dengan monitoring GPU
watch -n 1 nvidia-smi   # di terminal lain (Linux)
```

> **Estimasi waktu training:**
> - CPU (laptop biasa): ~4-8 jam untuk 80 epoch
> - GPU RTX 3060: ~30-45 menit
> - Google Colab (T4): ~1-2 jam

### 4. Inference

#### Webcam Realtime
```bash
python predict.py --source 0
```

#### Video File
```bash
python predict.py --source data/test_video.mp4
```

#### Gambar
```bash
python predict.py --source foto_sampah.jpg
```

#### CCTV / RTSP
```bash
python predict.py --source "rtsp://admin:password@192.168.1.100:554/stream"
```

#### Opsi Tambahan
```bash
# Ubah confidence threshold (lebih rendah = lebih banyak deteksi)
python predict.py --source 0 --conf 0.25

# Headless (tanpa preview, untuk server)
python predict.py --source video.mp4 --no-show

# Jangan simpan output
python predict.py --source 0 --no-save --no-csv --no-json
```

### 5. Menu Interaktif

```bash
python main.py
```

---

## 📊 Cara Melihat Output

### File CSV (`outputs/detections.csv`)
```
timestamp,source,class_name,count,total_objects
2024-01-15T14:30:22,0,plastic,3,5
2024-01-15T14:30:22,0,metal,2,5
```

### File JSON (`outputs/detections.json`)
```json
[
  {
    "timestamp": "2024-01-15T14:30:22",
    "source": "0",
    "total_objects": 5,
    "counts": {"plastic": 3, "metal": 2},
    "detections": [
      {
        "class_id": 0,
        "class_name": "plastic",
        "confidence": 0.8712,
        "bbox": {"x1": 120, "y1": 80, "x2": 340, "y2": 290}
      }
    ]
  }
]
```

### Analisis dengan Pandas
```python
import pandas as pd

df = pd.read_csv("outputs/detections.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Total per kelas
print(df.groupby("class_name")["count"].sum())

# Trend per waktu
df.set_index("timestamp").resample("1min")["total_objects"].mean().plot()
```

---

## ⚙️ Cara Modifikasi Kelas

### Ubah daftar kelas
Di `config.py`:
```python
CLASS_NAMES = [
    "plastic",
    "paper_cardboard",
    "metal",
    "glass",
    "plastic_bag",
    "trash",
    # Tambah kelas baru di sini
]
```

### Update mapping label
Di `utils.py`, tambahkan ke `TACO_LABEL_MAP` atau `ROBOFLOW_LABEL_MAP`:
```python
TACO_LABEL_MAP = {
    "Plastic bottle": "plastic",
    "My New Label"  : "kelas_baru",  # tambahkan mapping
    ...
}
```

Setelah ubah kelas, jalankan ulang `prepare_dataset.py` dan `train.py`.

---

## 🔧 Troubleshooting

| Masalah                        | Solusi                                                        |
|-------------------------------|---------------------------------------------------------------|
| FPS rendah di laptop           | Kurangi `imgsz` ke 416 di `config.py`                        |
| Banyak false positive          | Naikkan `conf` ke 0.5 di `INFERENCE_CONFIG`                  |
| Banyak miss detection          | Turunkan `conf` ke 0.2                                        |
| OOM (out of memory) training   | Turunkan `batch` ke 8                                         |
| Training lambat di CPU         | Normal. Gunakan Google Colab untuk training, lalu download model |
| Webcam tidak terbuka           | Coba `--source 1` atau `--source 2`                          |
| CUDA error                     | Set `device: "cpu"` di `INFERENCE_CONFIG`                    |

---

## 📈 Cara Upgrade Proyek

### 1. Gunakan model yang lebih besar
```python
# Di config.py, ubah:
BASE_MODEL = "yolov8s.pt"  # small — lebih akurat, sedikit lebih lambat
# atau
BASE_MODEL = "yolov8m.pt"  # medium — untuk server/cloud
```

### 2. Tambah dataset lebih banyak
- [Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html) — kelas "Tin can", "Bottle", dsb
- [MJU-Waste](https://github.com/realwecan/mju-waste)
- Foto sendiri dengan LabelImg atau Roboflow Annotate

### 3. Deploy ke web/mobile
```bash
# Export ke ONNX
python train.py --export onnx

# Gunakan untuk inference tanpa GPU
onnxruntime inference: model.onnx
```

### 4. Integrasi MQTT / IoT
```python
import paho.mqtt.client as mqtt
# Publish count_dict ke broker MQTT setiap frame
```

---

## ⚠️ Keterbatasan & Catatan

1. **Pencahayaan buruk** — Model kurang akurat di kondisi malam atau backlight. Saran: tambahkan data augmentasi dengan brightness variation.

2. **Occlusion** — Sampah yang saling tumpang tindih sulit terdeteksi. Saran: aktifkan `copy_paste` augmentation.

3. **Objek kecil** — Puntung rokok atau sachet kecil sering terlewat. Saran: naikkan resolusi ke 1280 atau gunakan SAHI (Slicing Aided Hyper Inference).

4. **Dataset mismatch** — TACO diambil di luar negeri, gambar mungkin berbeda dengan sampah lokal Indonesia. Saran: tambahkan foto lokal.

5. **Class imbalance** — Beberapa kelas seperti `glass` mungkin jumlah datanya sedikit. Saran: aktifkan `class_weights` atau augment agresif kelas minor.

---

## 📝 Lisensi

MIT License. Dataset TACO dan Roboflow memiliki lisensi masing-masing — cek website resmi sebelum penggunaan komersial.

---

*Dibuat untuk keperluan edukasi, demo, dan prototype.*
*"Code is Solitude" 🧘*
