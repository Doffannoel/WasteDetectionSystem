"""
scripts/export_report.py — Generate laporan statistik dari hasil deteksi

Menghasilkan:
- Ringkasan per kelas (count, %)
- Plot distribusi deteksi
- Timeline deteksi per menit
- Export ke CSV summary

Jalankan: python scripts/export_report.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from pathlib import Path

from config import OUTPUT_CSV, OUTPUT_JSON, OUTPUT_DIR
from utils import logger


def generate_report():
    """Generate laporan lengkap dari deteksi yang sudah tersimpan."""
    logger.info("📊 Generate laporan deteksi...\n")

    # ─── Load data JSON ───────────────────────────────────────
    if not OUTPUT_JSON.exists():
        logger.error(f"❌ File JSON tidak ditemukan: {OUTPUT_JSON}")
        logger.info("   Jalankan inference dulu: python predict.py --source 0")
        return

    with open(OUTPUT_JSON) as f:
        records = json.load(f)

    if not records:
        logger.warning("⚠️  Tidak ada data deteksi.")
        return

    logger.info(f"Total entri: {len(records)}")

    # ─── Statistik per kelas ──────────────────────────────────
    from collections import defaultdict
    class_totals: dict = defaultdict(int)
    total_objects = 0

    for record in records:
        for cls, cnt in record.get("counts", {}).items():
            class_totals[cls] += cnt
            total_objects += cnt

    logger.info("\n📦 Distribusi Deteksi per Kelas:")
    logger.info(f"{'Kelas':<20} {'Count':>7}  {'%':>6}")
    logger.info("─" * 40)
    for cls, cnt in sorted(class_totals.items(), key=lambda x: -x[1]):
        pct = cnt / total_objects * 100 if total_objects > 0 else 0
        bar = "▓" * int(pct / 2)
        logger.info(f"  {cls:<18} {cnt:>7}  {pct:>5.1f}%  {bar}")
    logger.info("─" * 40)
    logger.info(f"  {'TOTAL':<18} {total_objects:>7}")

    # ─── Timeline ─────────────────────────────────────────────
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame([
            {
                "timestamp": r["timestamp"],
                "total"    : r["total_objects"],
                **{f"cls_{k}": v for k, v in r.get("counts", {}).items()}
            }
            for r in records
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").dropna(subset=["timestamp"])

        if len(df) > 5:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # Plot 1: Total objek per waktu
            axes[0].plot(df["timestamp"], df["total"], color="#f97316", linewidth=1.5)
            axes[0].set_title("Total Objek Terdeteksi per Waktu", fontsize=13)
            axes[0].set_ylabel("Jumlah Objek")
            axes[0].grid(alpha=0.3)

            # Plot 2: Pie chart distribusi kelas
            ax2 = axes[1]
            labels = list(class_totals.keys())
            sizes  = [class_totals[l] for l in labels]
            colors = ["#f97316", "#22c55e", "#94a3b8", "#3b82f6", "#a855f7", "#ef4444"]
            ax2.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)])
            ax2.set_title("Distribusi Kelas Sampah", fontsize=13)

            plt.tight_layout()
            report_path = OUTPUT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
            plt.savefig(report_path, dpi=120, bbox_inches="tight")
            plt.close()
            logger.info(f"\n✅ Plot disimpan: {report_path}")

        # Export summary CSV
        summary_path = OUTPUT_DIR / "summary.csv"
        summary_data = [
            {"class_name": cls, "total_detections": cnt,
             "percentage": f"{cnt/total_objects*100:.1f}%" if total_objects > 0 else "0%"}
            for cls, cnt in sorted(class_totals.items(), key=lambda x: -x[1])
        ]
        pd.DataFrame(summary_data).to_csv(summary_path, index=False)
        logger.info(f"✅ Summary CSV: {summary_path}")

    except ImportError:
        logger.warning("⚠️  pandas/matplotlib tidak tersedia. Install: pip install pandas matplotlib")

    logger.info("\n✅ Laporan selesai!")


if __name__ == "__main__":
    generate_report()
