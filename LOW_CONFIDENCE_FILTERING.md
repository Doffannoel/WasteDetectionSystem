# 🔍 Panduan Filtering Objek Low-Confidence

## Masalah
Ketika ada objek yang terdeteksi tapi dengan confidence rendah (misal prediksi tidak yakin), Anda ingin:
- **Opsi A**: Jangan tampilkan sama sekali (filter strict)
- **Opsi B**: Tampilkan tapi dengan label "LAINNYA" (lebih fleksibel)

## ✅ Solusi yang Sudah Diimplementasikan

Ada 3 parameter baru di `INFERENCE_CONFIG` dalam `config.py`:

### Parameter 1: `low_conf_threshold` (0.0 - 1.0)
**Default**: `0.5`

Confidence di bawah nilai ini dianggap "rendah".
- `0.35` = sangat permisif (hampir semua ditampilkan)
- `0.5` = moderate (hanya deteksi dengan 50% confidence keatas yang dianggap baik)
- `0.7` = strict (hanya deteksi dengan 70%+ yang ditampilkan)

### Parameter 2: `show_low_conf` (True/False)
**Default**: `True`

Jika `True` dan `filter_low_conf` = `False`:
- Objek dengan confidence rendah ditampilkan dengan label **"LAINNYA"**
- Warna bounding box = abu-abu

### Parameter 3: `filter_low_conf` (True/False)
**Default**: `False`

Jika `True`:
- Objek dengan confidence rendah **tidak ditampilkan sama sekali**
- Lebih clean tapi bisa miss deteksi

---

## 🔧 Cara Menggunakan

### Mode 1: Tampilkan Low-Conf sebagai "LAINNYA" (REKOMENDASI)
```python
INFERENCE_CONFIG = {
    ...
    "low_conf_threshold": 0.5,     # threshold 50%
    "show_low_conf"     : True,    # tampilkan low-conf
    "filter_low_conf"   : False,   # jangan filter (tampilkan sebagai LAINNYA)
}
```

**Hasil**: Jika ada objek dengan confidence < 50%, tampilkan sebagai "LAINNYA" (warna abu-abu).

---

### Mode 2: Filter Strict (Jangan Tampilkan Low-Conf)
```python
INFERENCE_CONFIG = {
    ...
    "low_conf_threshold": 0.6,     # threshold 60%
    "show_low_conf"     : False,   # jangan tampilkan
    "filter_low_conf"   : True,    # filter (tidak tampilkan)
}
```

**Hasil**: Objek dengan confidence < 60% **tidak ditampilkan**.

---

### Mode 3: Permisif (Tampilkan Semua)
```python
INFERENCE_CONFIG = {
    ...
    "low_conf_threshold": 0.0,     # threshold 0% (semua ditampilkan)
    "show_low_conf"     : True,
    "filter_low_conf"   : False,
}
```

**Hasil**: Semua deteksi ditampilkan (default behavior).

---

## 📊 Perbandingan Mode

| Mode | Setting | Hasil | Kapan Digunakan |
|------|---------|-------|-----------------|
| **A: Tampilkan LAINNYA** | `low_conf=0.5, show=T, filter=F` | Low-conf → "LAINNYA" (abu-abu) | Debugging, lihat apa yang missed |
| **B: Filter Strict** | `low_conf=0.6, show=F, filter=T` | Low-conf → tidak ditampilkan | Production, hanya deteksi confident |
| **C: Permisif** | `low_conf=0.0, show=T, filter=F` | Semua ditampilkan | Testing, lihat semua prediksi |

---

## 🚀 Contoh Penggunaan

### Coba Mode A (Recommended untuk sekarang):
```bash
# Edit config.py
```
```python
INFERENCE_CONFIG = {
    ...
    "conf"              : 0.35,
    "iou"               : 0.45,
    "imgsz"             : 640,
    "max_det"           : 50,
    "device"            : AUTO_DEVICE,
    "verbose"           : False,
    "low_conf_threshold": 0.50,    # 👈 Threshold 50%
    "show_low_conf"     : True,    # 👈 Tampilkan
    "filter_low_conf"   : False,   # 👈 Sebagai LAINNYA
}
```

Lalu jalankan:
```bash
python predict.py --source 0
```

### Testing untuk setiap mode:
```bash
# Mode: Tampilkan LAINNYA
python predict.py --source 0

# Mode: Filter Strict (kalau sudah ubah config)
# ... ubah filter_low_conf=True terlebih dahulu
python predict.py --source 0
```

---

## 📈 Rekomendasi Tuning

### Jika banyak **False Positive** (deteksi salah):
```python
"low_conf_threshold": 0.6,      # naikkan ke 60%
"filter_low_conf"   : True,     # aktifkan filter
```

### Jika banyak **False Negative** (missed detections):
```python
"low_conf_threshold": 0.3,      # turunkan ke 30%
"show_low_conf"     : True,     # tampilkan LAINNYA
"filter_low_conf"   : False,    # jangan filter
```

### Balance (default):
```python
"low_conf_threshold": 0.5,      # moderate threshold
"show_low_conf"     : True,     # tampilkan
"filter_low_conf"   : False,    # sebagai LAINNYA
```

---

## 💡 Tips

1. **Jalankan predict.py berulang kali dengan threshold berbeda** untuk lihat perubahan:
   - Buka `config.py`
   - Ubah `low_conf_threshold` (0.3, 0.5, 0.7)
   - Jalankan `python predict.py --source 0`

2. **Perhatikan output**:
   - Berapa % objek yang labeled "LAINNYA"?
   - Apakah ada false positives yang berkurang?

3. **Jika masih ada masalah**:
   - Tingkatkan **epoch** di training (lihat `PENINGKATAN_METRIK.md`)
   - Gunakan model yang lebih besar (yolov8s, bukan nano)

---

## 🎯 Kesimpulan

Dengan parameter baru ini, Anda bisa:
- ✅ **Filter objek confidence rendah** sebagai "LAINNYA"
- ✅ **Tidak perlu ubah train.py** — semua di config.py
- ✅ **Fleksibel**: bisa tampilkan, filter, atau permisif
- ✅ **Mudah tuning**: cukup ubah 3 parameter

**Mulai dari Mode A (tampilkan LAINNYA) dan sesuaikan berdasarkan kebutuhan!**
