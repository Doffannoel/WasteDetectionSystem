"""
predict.py – Inference realtime untuk Waste Detection

Mendukung:
  - Gambar        : python predict.py --source foto.jpg
  - Video file    : python predict.py --source video.mp4
  - Webcam        : python predict.py               <- GUI pemilih kamera otomatis
  - Webcam manual : python predict.py --source 0
  - RTSP / CCTV   : python predict.py --source rtsp://user:pass@ip:port/stream

Opsi tambahan:
  --no-save       : jangan simpan output video
  --no-csv        : jangan simpan ke CSV
  --no-json       : jangan simpan ke JSON
  --conf 0.4      : ubah confidence threshold
  --model path    : gunakan model custom
  --no-show       : headless mode (tanpa preview)

Contoh:
  python predict.py                        # GUI pemilih kamera
  python predict.py --source 0 --conf 0.35
  python predict.py --source data/test_video.mp4 --conf 0.4
  python predict.py --source gambar.jpg
"""

import argparse
import sys
import time
import threading
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


# --- AMBIL NAMA KAMERA (WINDOWS) ---------------------------------------------
def get_camera_names_windows() -> dict:
    """
    Ambil nama kamera sungguhan di Windows via pygrabber.
    Mengembalikan dict {index: "Nama Kamera"}.
    Fallback ke nama generik jika pygrabber tidak tersedia.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph   = FilterGraph()
        devices = graph.get_input_devices()
        return {i: name for i, name in enumerate(devices)}
    except Exception:
        return {}


# --- CAMERA SCANNER ----------------------------------------------------------
def scan_cameras(max_index: int = 10) -> list:
    """
    Scan kamera yang tersedia dan ambil nama aslinya (Windows).
    Mengembalikan list dict dengan info lengkap tiap kamera.
    """
    logger.info("Mencari kamera yang tersedia...")

    cam_names = get_camera_names_windows()
    available = []

    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        ret, _ = cap.read()
        if not ret:
            cap.release()
            continue

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        raw_name = cam_names.get(i, f"Camera {i}")
        display  = f"{raw_name}  ({width}x{height} @ {int(fps)}fps)"

        available.append({
            "index":   i,
            "name":    raw_name,
            "display": display,
            "width":   width,
            "height":  height,
            "fps":     fps,
        })
        logger.info(f"  OK [{i}] {raw_name}  {width}x{height} @ {int(fps)}fps")

    if not available:
        logger.warning("Tidak ada kamera ditemukan.")

    return available


# --- CAMERA SELECTOR + INFERENCE GUI -----------------------------------------
def show_camera_gui(
    cameras: list,
    model,
    conf: float,
    save_output: bool,
    save_csv: bool,
    save_json: bool,
    log_interval: int = 30,
):
    """
    GUI satu-window:
      - Kiri  : daftar kamera (nama asli dari Windows)
      - Kanan : preview live -> setelah Mulai berubah jadi feed inference
      - Footer: tombol Mulai / Stop / Refresh
    Tidak membuka window OpenCV terpisah.
    """
    try:
        import tkinter as tk
        from PIL import Image, ImageTk
    except ImportError:
        logger.error("Butuh Pillow: pip install Pillow")
        sys.exit(1)

    # Warna & font
    BG       = "#0f0f0f"
    BG2      = "#1a1a1a"
    BG3      = "#141414"
    ACCENT   = "#00ff88"
    ACCENT2  = "#00cc66"
    TEXT     = "#e8e8e8"
    TEXT_DIM = "#555555"
    BORDER   = "#2a2a2a"
    DANGER   = "#ff4444"

    F_TITLE = ("Courier New", 13, "bold")
    F_LABEL = ("Courier New", 10)
    F_BTN   = ("Courier New", 11, "bold")
    F_SMALL = ("Courier New",  9)

    MIN_PREV_W, MIN_PREV_H = 560, 315   # ukuran minimum preview area
    LEFT_W = 240                         # lebar panel kiri (tetap)

    # State bersama antar thread
    state = {
        "selected_cam": cameras[0] if cameras else None,
        "mode":         "preview",   # "preview" | "inference"
        "active":       True,
        "cap":          None,
        "writer":       None,
        "frame_count":  0,
        "total_det":    0,
        "fps":          0.0,
        "canvas_w":     MIN_PREV_W,
        "canvas_h":     MIN_PREV_H,
    }

    root = tk.Tk()
    root.title("Waste Detection")
    root.resizable(True, True)   # izinkan resize bebas
    root.configure(bg=BG)
    root.minsize(LEFT_W + MIN_PREV_W + 60, MIN_PREV_H + 120)  # batas minimum

    # ── Header ──
    hdr = tk.Frame(root, bg=BG, pady=14)
    hdr.pack(fill="x", padx=20)

    tk.Label(hdr, text="  WASTE DETECTION SYSTEM",
             font=F_TITLE, fg=ACCENT, bg=BG).pack(side="left")

    status_dot = tk.Label(hdr, text="  IDLE",
                          font=F_SMALL, fg=TEXT_DIM, bg=BG)
    status_dot.pack(side="right")

    tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=20)

    # ── Body: gunakan grid agar bisa stretch ──
    body = tk.Frame(root, bg=BG)
    body.pack(padx=20, pady=14, fill="both", expand=True)
    body.grid_columnconfigure(1, weight=1)   # kolom kanan ikut melebar
    body.grid_rowconfigure(0, weight=1)      # baris ikut memanjang

    # Panel kiri - daftar kamera (lebar tetap, tinggi ikut)
    left = tk.Frame(body, bg=BG2,
                    highlightthickness=1, highlightbackground=BORDER,
                    width=LEFT_W)
    left.grid(row=0, column=0, sticky="ns", padx=(0, 14))
    left.grid_propagate(False)

    tk.Label(left, text="  PILIH KAMERA",
             font=F_SMALL, fg=TEXT_DIM, bg=BG2,
             anchor="w", pady=8).pack(fill="x", padx=8)
    tk.Frame(left, bg=BORDER, height=1).pack(fill="x")

    btn_refs = []   # list of (tk.Button, cam_dict)

    def _highlight(selected_cam):
        for btn, cam in btn_refs:
            if cam["index"] == selected_cam["index"]:
                btn.configure(bg=ACCENT, fg="#000000")
            else:
                btn.configure(bg=BG2, fg=TEXT)

    def on_cam_click(cam):
        if state["mode"] == "inference":
            return
        state["selected_cam"] = cam
        _highlight(cam)
        info_var.set(cam["display"])

    for cam in cameras:
        is_first = cam["index"] == cameras[0]["index"]
        btn = tk.Button(
            left,
            text=f"  {cam['name']}",
            font=F_LABEL,
            fg="#000000" if is_first else TEXT,
            bg=ACCENT    if is_first else BG2,
            activebackground=ACCENT2, activeforeground="#000000",
            relief="flat", bd=0,
            anchor="w", padx=12, pady=9,
            cursor="hand2",
            wraplength=210, justify="left",
            command=lambda c=cam: on_cam_click(c),
        )
        btn.pack(fill="x", pady=1)
        btn_refs.append((btn, cam))

    if not cameras:
        tk.Label(left, text="  Tidak ada kamera\n  ditemukan",
                 font=F_LABEL, fg=DANGER, bg=BG2, pady=20).pack()

    # Tombol Refresh
    tk.Frame(left, bg=BORDER, height=1).pack(fill="x", pady=(8, 0))

    def on_rescan():
        if state["mode"] == "inference":
            return
        _stop_loop()
        time.sleep(0.15)
        root.destroy()
        new_cams = scan_cameras()
        if not new_cams:
            logger.error("Tidak ada kamera ditemukan setelah refresh.")
            sys.exit(1)
        show_camera_gui(new_cams, model, conf, save_output, save_csv, save_json, log_interval)

    tk.Button(
        left, text="Refresh Kamera",
        font=F_SMALL, fg=TEXT_DIM, bg=BG2,
        activebackground=BORDER, activeforeground=TEXT,
        relief="flat", bd=0, pady=8, cursor="hand2",
        command=on_rescan,
    ).pack(fill="x")

    # Panel kanan - preview / inference (ikut stretch saat resize)
    right = tk.Frame(body, bg=BG)
    right.grid(row=0, column=1, sticky="nsew")
    right.grid_rowconfigure(0, weight=1)
    right.grid_columnconfigure(0, weight=1)

    # Pakai tk.Canvas agar bisa bind <Configure> untuk resize dinamis
    canvas_frame = tk.Canvas(right, bg="#000000",
                             highlightthickness=1, highlightbackground=BORDER)
    canvas_frame.grid(row=0, column=0, sticky="nsew")

    img_label = tk.Label(canvas_frame, bg="#000000")
    _img_window = canvas_frame.create_window(0, 0, anchor="nw", window=img_label)

    no_signal_lbl = tk.Label(
        canvas_frame, text="[ NO SIGNAL ]",
        font=("Courier New", 14, "bold"),
        fg=TEXT_DIM, bg="#000000",
    )
    _nosig_window = canvas_frame.create_window(0, 0, anchor="center", window=no_signal_lbl)

    # Stats overlay pojok kiri atas
    stats_frame = tk.Frame(canvas_frame, bg="#000000")
    _stats_window = canvas_frame.create_window(0, 0, anchor="nw", window=stats_frame)

    fps_var = tk.StringVar(value="")
    det_var = tk.StringVar(value="")
    stat_fps = tk.Label(stats_frame, textvariable=fps_var,
                        font=("Courier New", 10, "bold"),
                        fg=ACCENT, bg="#000000", padx=6, pady=2)
    stat_det = tk.Label(stats_frame, textvariable=det_var,
                        font=("Courier New", 10),
                        fg=TEXT, bg="#000000", padx=6, pady=2)

    def on_canvas_resize(event):
        """Saat window diresize, sesuaikan ukuran & posisi elemen di canvas."""
        w, h = event.width, event.height
        state["canvas_w"] = w
        state["canvas_h"] = h
        canvas_frame.itemconfig(_img_window, width=w, height=h)
        canvas_frame.coords(_nosig_window, w // 2, h // 2)
        # stats tetap di pojok kiri atas (koordinat 0,0 sudah benar)

    canvas_frame.bind("<Configure>", on_canvas_resize)

    init_display = cameras[0]["display"] if cameras else ""
    info_var = tk.StringVar(value=init_display)
    tk.Label(right, textvariable=info_var,
             font=F_SMALL, fg=TEXT_DIM, bg=BG, pady=5).grid(row=1, column=0)

    # ── Footer ──
    tk.Frame(root, bg=BORDER, height=1).pack(fill="x", padx=20)
    footer = tk.Frame(root, bg=BG, pady=12)
    footer.pack(fill="x", padx=20)

    tk.Label(footer, text="Tekan  Q / ESC  untuk menghentikan inference",
             font=F_SMALL, fg=TEXT_DIM, bg=BG).pack(side="left")

    btn_start = tk.Button(footer)
    btn_stop  = tk.Button(footer)

    # ── Loop thread ──
    _loop_thread = [None]

    def _stop_loop():
        state["active"] = False
        time.sleep(0.08)
        if state["cap"]:
            try:
                state["cap"].release()
            except Exception:
                pass
            state["cap"] = None
        if state["writer"]:
            try:
                state["writer"].release()
            except Exception:
                pass
            state["writer"] = None

    def _render(frame_bgr):
        """Resize frame sesuai ukuran canvas aktual lalu tampilkan."""
        w = max(state["canvas_w"], 1)
        h = max(state["canvas_h"], 1)
        frame_rgb = cv2.cvtColor(
            cv2.resize(frame_bgr, (w, h)),
            cv2.COLOR_BGR2RGB,
        )
        img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
        img_label.configure(image=img)
        img_label.image = img

    def loop_preview(cam_idx):
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        state["cap"] = cap
        if not cap.isOpened():
            return
        no_signal_lbl.lower()
        while state["active"] and state["mode"] == "preview":
            ret, frame = cap.read()
            if not ret:
                break
            _render(frame)
            time.sleep(0.033)
        cap.release()
        state["cap"] = None

    def loop_inference(cam_idx):
        fps_counter = FPSCounter(window=30)
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        state["cap"] = cap
        if not cap.isOpened():
            logger.error(f"Gagal membuka kamera {cam_idx}")
            return

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30

        if save_output:
            out_path = get_output_path(f"webcam_{cam_idx}", suffix=".mp4")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                fps_in, (width, height)
            )
            state["writer"] = writer
            logger.info(f"Output video: {out_path}")

        state["frame_count"] = 0
        state["total_det"]   = 0
        no_signal_lbl.lower()

        while state["active"] and state["mode"] == "inference":
            ret, frame = cap.read()
            if not ret:
                break

            state["frame_count"] += 1
            ts = datetime.now().isoformat()

            results = model.predict(
                source=frame, conf=conf,
                iou=INFERENCE_CONFIG["iou"],
                imgsz=INFERENCE_CONFIG["imgsz"],
                max_det=INFERENCE_CONFIG["max_det"],
                device=INFERENCE_CONFIG["device"],
                verbose=False, stream=False,
            )

            boxes                 = results[0].boxes
            fps_val               = fps_counter.tick()
            state["fps"]          = fps_val
            annotated, count_dict = draw_detections(frame, boxes, fps=fps_val)
            state["total_det"]   += sum(count_dict.values())

            # Update stats overlay
            fps_var.set(f"FPS: {fps_val:.1f}")
            det_var.set(f"Frame: {state['frame_count']}  |  Total deteksi: {state['total_det']}")

            if state["frame_count"] % log_interval == 0:
                if save_csv:
                    save_detection_csv(source=str(cam_idx),
                                       count_dict=count_dict, timestamp=ts)
                if save_json:
                    save_detection_json(source=str(cam_idx),
                                        count_dict=count_dict,
                                        boxes_raw=boxes, timestamp=ts)

            if state["writer"]:
                state["writer"].write(annotated)

            _render(annotated)

        cap.release()
        state["cap"] = None
        if state["writer"]:
            state["writer"].release()
            state["writer"] = None

        logger.info(f"Selesai - Frame: {state['frame_count']} | "
                    f"Deteksi: {state['total_det']} | FPS: {state['fps']:.1f}")

    def _start_thread(target_fn, *args):
        t = threading.Thread(target=target_fn, args=args, daemon=True)
        _loop_thread[0] = t
        t.start()

    # Mulai preview kamera pertama otomatis
    if cameras:
        state["active"] = True
        state["mode"]   = "preview"
        _start_thread(loop_preview, cameras[0]["index"])

    # ── Kontrol UI ──
    def set_ui_inference(running: bool):
        if running:
            btn_start.configure(state="disabled", bg=BG3, fg=TEXT_DIM)
            btn_stop.configure(state="normal",   bg=DANGER, fg="#ffffff")
            status_dot.configure(text="  RUNNING", fg=ACCENT)
            stat_fps.pack(anchor="w")
            stat_det.pack(anchor="w")
            for btn, _ in btn_refs:
                btn.configure(state="disabled", cursor="arrow")
        else:
            btn_start.configure(state="normal", bg=ACCENT, fg="#000000")
            btn_stop.configure(state="disabled", bg=BG2, fg=TEXT_DIM)
            status_dot.configure(text="  IDLE", fg=TEXT_DIM)
            stat_fps.pack_forget()
            stat_det.pack_forget()
            fps_var.set("")
            det_var.set("")
            for btn, _ in btn_refs:
                btn.configure(state="normal", cursor="hand2")

    def on_start():
        cam = state["selected_cam"]
        if cam is None:
            return
        _stop_loop()
        time.sleep(0.1)
        state["active"] = True
        state["mode"]   = "inference"
        set_ui_inference(True)
        info_var.set(f"[REC]  {cam['name']}")
        logger.info(f"Mulai inference: {cam['name']} (index {cam['index']})")
        _start_thread(loop_inference, cam["index"])

    def on_stop():
        _stop_loop()
        time.sleep(0.15)
        state["active"] = True
        state["mode"]   = "preview"
        cam = state["selected_cam"]
        set_ui_inference(False)
        info_var.set(cam["display"] if cam else "")
        logger.info("Inference dihentikan.")
        if cam:
            _start_thread(loop_preview, cam["index"])

    btn_stop.configure(
        text="Stop",
        font=F_BTN, fg=TEXT_DIM, bg=BG2,
        activebackground="#cc3333", activeforeground="#ffffff",
        relief="flat", bd=0, padx=18, pady=10,
        cursor="hand2", state="disabled",
        command=on_stop,
    )
    btn_stop.pack(side="right", padx=(6, 0))

    btn_start.configure(
        text="Mulai Inferensi",
        font=F_BTN, fg="#000000", bg=ACCENT,
        activebackground=ACCENT2, activeforeground="#000000",
        relief="flat", bd=0, padx=20, pady=10,
        cursor="hand2",
        state="normal" if cameras else "disabled",
        command=on_start,
    )
    btn_start.pack(side="right")

    # Keyboard shortcut Q / ESC
    def on_key(event):
        if event.keysym in ("q", "Q", "Escape"):
            if state["mode"] == "inference":
                on_stop()

    root.bind("<Key>", on_key)

    def on_close():
        state["active"] = False
        _stop_loop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    # Ukuran awal window, tengahkan di layar
    root.update_idletasks()
    init_w = LEFT_W + MIN_PREV_W + 60
    init_h = MIN_PREV_H + 160
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x  = (sw - init_w) // 2
    y  = (sh - init_h) // 2
    root.geometry(f"{init_w}x{init_h}+{x}+{y}")

    root.mainloop()


# --- LOAD MODEL --------------------------------------------------------------
def load_model(model_path=None):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics tidak terinstall: pip install ultralytics")
        sys.exit(1)

    path = model_path or str(TRAINED_MODEL)
    if not Path(path).exists():
        logger.warning(f"Model tidak ditemukan: {path}")
        logger.info("Menggunakan YOLOv8n pre-trained (COCO) sebagai fallback.")
        path = "yolov8n.pt"

    logger.info(f"Load model: {path}")
    return YOLO(path)


# --- INFERENCE IMAGE ---------------------------------------------------------
def predict_image(model, image_path, save_output=True,
                  conf=INFERENCE_CONFIG["conf"], show=True):
    img_path = Path(image_path)
    if not img_path.exists():
        logger.error(f"File tidak ditemukan: {image_path}")
        return

    logger.info(f"Inference gambar: {image_path}")
    frame = cv2.imread(str(img_path))
    if frame is None:
        logger.error(f"Gagal membaca gambar: {image_path}")
        return

    results = model.predict(
        source=frame, conf=conf,
        iou=INFERENCE_CONFIG["iou"], imgsz=INFERENCE_CONFIG["imgsz"],
        max_det=INFERENCE_CONFIG["max_det"], device=INFERENCE_CONFIG["device"],
        verbose=False,
    )
    boxes = results[0].boxes
    annotated, count_dict = draw_detections(frame, boxes)

    ts = datetime.now().isoformat()
    save_detection_csv(source=str(img_path), count_dict=count_dict, timestamp=ts)
    save_detection_json(source=str(img_path), count_dict=count_dict,
                        boxes_raw=boxes, timestamp=ts)

    if save_output:
        out_path = get_output_path(str(img_path), suffix=".jpg")
        cv2.imwrite(str(out_path), annotated)
        logger.info(f"Hasil disimpan: {out_path}")

    if show:
        cv2.imshow("Waste Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logger.info(f"Terdeteksi: {count_dict}")
    return count_dict


# --- INFERENCE VIDEO / RTSP --------------------------------------------------
def predict_stream(model, source, save_output=True,
                   conf=INFERENCE_CONFIG["conf"], show=True,
                   save_csv=SAVE_CSV, save_json=SAVE_JSON, log_interval=30):
    if isinstance(source, str) and (source == "0" or source.isdigit()):
        src = int(source)
    elif isinstance(source, int):
        src = source
    else:
        src = source

    logger.info(f"Membuka: {src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error(f"Gagal membuka: {source}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    logger.info(f"Resolusi: {width}x{height} @ {fps_in:.0f}fps")

    writer = None
    if save_output:
        out_path = get_output_path(str(source), suffix=".mp4")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height)
        )
        logger.info(f"Output video: {out_path}")

    fps_counter = FPSCounter(window=30)
    frame_count = total_det = 0
    fps = 0.0
    win = "Waste Detection - tekan Q untuk keluar"

    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    logger.info("Inference dimulai. Tekan 'q' untuk berhenti.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            ts = datetime.now().isoformat()

            results = model.predict(
                source=frame, conf=conf,
                iou=INFERENCE_CONFIG["iou"], imgsz=INFERENCE_CONFIG["imgsz"],
                max_det=INFERENCE_CONFIG["max_det"], device=INFERENCE_CONFIG["device"],
                verbose=False, stream=False,
            )
            boxes = results[0].boxes
            fps   = fps_counter.tick()
            annotated, count_dict = draw_detections(frame, boxes, fps=fps)
            total_det += sum(count_dict.values())

            if frame_count % log_interval == 0:
                if save_csv:
                    save_detection_csv(source=str(source),
                                       count_dict=count_dict, timestamp=ts)
                if save_json:
                    save_detection_json(source=str(source), count_dict=count_dict,
                                        boxes_raw=boxes, timestamp=ts)
            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow(win, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    cv2.waitKey(0)

    except KeyboardInterrupt:
        logger.info("Dihentikan (Ctrl+C).")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    logger.info(f"Frame: {frame_count} | Deteksi: {total_det} | FPS: {fps:.1f}")


# --- INFERENCE BATCH ---------------------------------------------------------
def predict_folder(model, folder_path,
                   conf=INFERENCE_CONFIG["conf"], save_output=True):
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder tidak ditemukan: {folder_path}")
        return

    images = [*folder.glob("*.jpg"), *folder.glob("*.png"), *folder.glob("*.jpeg")]
    if not images:
        logger.warning(f"Tidak ada gambar di: {folder_path}")
        return

    logger.info(f"Batch inference: {len(images)} gambar")
    for i, img_path in enumerate(images, 1):
        logger.info(f"  [{i}/{len(images)}] {img_path.name}")
        predict_image(model, str(img_path), save_output=save_output, show=False)
    logger.info("Batch selesai.")


# --- MAIN --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Waste Detection - Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh:
  python predict.py                        # GUI pemilih kamera (default)
  python predict.py --source 0             # webcam index 0 langsung
  python predict.py --source video.mp4
  python predict.py --source foto.jpg
  python predict.py --source folder/
  python predict.py --source rtsp://ip:port/stream
        """
    )
    parser.add_argument("--source",  type=str,   default=None,
                        help="Input: index kamera, path file/folder, atau rtsp://... "
                             "(kosong = GUI pemilih kamera)")
    parser.add_argument("--conf",    type=float, default=INFERENCE_CONFIG["conf"])
    parser.add_argument("--model",   type=str,   default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-csv",  action="store_true")
    parser.add_argument("--no-json", action="store_true")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Waste Detection - Inference")
    logger.info("=" * 60)

    save_output = not args.no_save
    show        = not args.no_show
    save_csv_f  = not args.no_csv
    save_json_f = not args.no_json
    source      = args.source

    # Mode GUI (tanpa --source)
    if source is None:
        model   = load_model(args.model)
        cameras = scan_cameras()
        if not cameras:
            logger.error("Tidak ada kamera ditemukan.")
            logger.info("Sambungkan webcam atau gunakan: python predict.py --source <path>")
            sys.exit(1)

        show_camera_gui(
            cameras     = cameras,
            model       = model,
            conf        = args.conf,
            save_output = save_output,
            save_csv    = save_csv_f,
            save_json   = save_json_f,
        )
        return

    # Mode CLI (dengan --source)
    model    = load_model(args.model)
    src_path = Path(source)

    if src_path.is_dir():
        predict_folder(model, source, conf=args.conf, save_output=save_output)

    elif src_path.is_file() and src_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        predict_image(model, source, save_output=save_output,
                      conf=args.conf, show=show)
    else:
        predict_stream(
            model       = model,
            source      = source,
            save_output = save_output,
            conf        = args.conf,
            show        = show,
            save_csv    = save_csv_f,
            save_json   = save_json_f,
        )


if __name__ == "__main__":
    main()