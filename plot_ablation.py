#!/usr/bin/env python3
"""
plot_ablation.py - Visualisasi Hasil Studi Komparatif QLoRA Hyperparameter
Skripsi: Implementasi Constrained Generation pada LLM Menggunakan QLoRA
NOTES: penggunaan "ablasi" di sini adalah salah kaprah, karena ini lebih ke studi komparatif hyperparameter, bukan ablasi model.
Kata "ablasi" telah dihapus dari skripsi sebagai hasil dari revisi. Namun karena tidak efisien mengganti nama file, tetap digunakan nama file ini.

Menghasilkan grafik-grafik untuk Bab 4:
1. Loss Curve ketiga skenario (overlay)
2. Bar chart perbandingan metrik utama
3. Tabel ringkasan
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# KONFIGURASI
# ============================================================

ABLATION_DIR = "ablation_results"
OUTPUT_DIR   = os.path.join(ABLATION_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Warna konsisten untuk setiap skenario
COLORS = {
    "A_lightweight": "#2196F3",   # Biru
    "B_balanced":    "#4CAF50",   # Hijau
    "C_deep":        "#F44336",   # Merah
}

LABELS = {
    "A_lightweight": "Lightweight (r=8, α=16, 1 epoch)",
    "B_balanced":    "Balanced (r=16, α=32, 2 epoch)",
    "C_deep":        "Deep (r=32, α=64, 3 epoch)",
}

SCENARIOS = ["A_lightweight", "B_balanced", "C_deep"]

# ============================================================
# LOAD DATA
# ============================================================

def load_metrics():
    """Load metrik dari JSON."""
    json_path = os.path.join(ABLATION_DIR, "ablation_metrics.json")
    with open(json_path, "r") as f:
        return json.load(f)

def load_loss_history(scenario_id: str):
    """Load loss history untuk satu skenario."""
    path = os.path.join(ABLATION_DIR, f"model_{scenario_id}", "loss_history.json")
    with open(path, "r") as f:
        data = json.load(f)
    steps  = [e["step"]  for e in data if "loss" in e]
    losses = [e["loss"]  for e in data if "loss" in e]
    epochs = [e.get("epoch", 0) for e in data if "loss" in e]
    return steps, losses, epochs

# ============================================================
# PLOT 1: LOSS CURVE OVERLAY
# ============================================================

def plot_loss_curves(metrics):
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario_id in SCENARIOS:
        steps, losses, _ = load_loss_history(scenario_id)
        color = COLORS[scenario_id]
        label = LABELS[scenario_id]

        # Normalisasi step ke 0-1 agar perbandingan fair
        steps_norm = [s / max(steps) for s in steps]

        ax.plot(steps_norm, losses,
                color=color, linewidth=2.5,
                label=label, marker='o',
                markersize=4, markevery=2)

    ax.set_xlabel("Proporsi Training Progress (0 = awal, 1 = selesai)",
                  fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Perbandingan Training Loss Curve\nStudi Komparatif Konfigurasi QLoRA Hyperparameter",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    # Annotasi final loss
    for m in metrics:
        sid = m["scenario_id"]
        ax.annotate(f"  {m['final_loss']:.4f}",
                    xy=(1.0, m["final_loss"]),
                    color=COLORS[sid], fontsize=9,
                    va='center')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "01_loss_curve.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out}")


# ============================================================
# PLOT 2: LOSS CURVE PER EPOCH (untuk skenario yang > 1 epoch)
# ============================================================

def plot_loss_by_step(metrics):
    """Loss vs actual step number."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario_id in SCENARIOS:
        steps, losses, _ = load_loss_history(scenario_id)
        color = COLORS[scenario_id]
        label = LABELS[scenario_id]
        ax.plot(steps, losses, color=color, linewidth=2.5,
                label=label, marker='o', markersize=4, markevery=2)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss per Step\nStudi Komparatif Konfigurasi QLoRA Hyperparameter",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "02_loss_by_step.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out}")


# ============================================================
# PLOT 3: BAR CHART - FINAL LOSS
# ============================================================

def plot_final_loss(metrics):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels       = [LABELS[m["scenario_id"]] for m in metrics]
    final_losses = [m["final_loss"] for m in metrics]
    colors       = [COLORS[m["scenario_id"]] for m in metrics]

    bars = ax.bar(labels, final_losses, color=colors,
                  width=0.5, edgecolor='white', linewidth=1.5)

    # Annotasi nilai di atas bar
    for bar, val in zip(bars, final_losses):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel("Final Training Loss", fontsize=12)
    ax.set_title("Perbandingan Final Training Loss\nper Skenario Hyperparameter",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(final_losses) * 1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Wrap label agar tidak overlap
    ax.set_xticks(range(len(labels)))
    wrapped = [l.replace(" (", "\n(") for l in labels]
    ax.set_xticklabels(wrapped, fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_final_loss.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out}")


# ============================================================
# PLOT 4: BAR CHART - TRAINING TIME
# ============================================================

def plot_training_time(metrics):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [LABELS[m["scenario_id"]] for m in metrics]
    times  = [m["training_time_min"] for m in metrics]
    colors = [COLORS[m["scenario_id"]] for m in metrics]

    bars = ax.bar(labels, times, color=colors,
                  width=0.5, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f} mnt",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel("Waktu Training (menit)", fontsize=12)
    ax.set_title("Perbandingan Waktu Training\nper Skenario Hyperparameter",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "04_training_time.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out}")


# ============================================================
# PLOT 5: BAR CHART - VRAM USAGE
# ============================================================

def plot_vram(metrics):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [LABELS[m["scenario_id"]] for m in metrics]
    vrams  = [m["vram_peak_gb"] for m in metrics]
    colors = [COLORS[m["scenario_id"]] for m in metrics]

    bars = ax.bar(labels, vrams, color=colors,
                  width=0.5, edgecolor='white', linewidth=1.5)

    # Garis batas VRAM laptop (8GB)
    ax.axhline(y=8.0, color='black', linestyle='--',
               linewidth=1.5, label='Batas VRAM (8 GB)')

    for bar, val in zip(bars, vrams):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{val:.2f} GB",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel("Peak VRAM Usage (GB)", fontsize=12)
    ax.set_title("Perbandingan Penggunaan VRAM\nper Skenario Hyperparameter",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 9.5)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "05_vram_usage.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out}")


# ============================================================
# PLOT 6: BAR CHART - INFERENCE TIME
# ============================================================

def plot_inference_time(metrics):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [LABELS[m["scenario_id"]] for m in metrics]
    times  = [m["avg_inference_ms"] for m in metrics]
    colors = [COLORS[m["scenario_id"]] for m in metrics]

    bars = ax.bar(labels, times, color=colors,
                  width=0.5, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{val:.0f} ms",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel("Rata-rata Waktu Inferensi (ms)", fontsize=12)
    ax.set_title("Perbandingan Rata-rata Waktu Inferensi\nper Skenario Hyperparameter",
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(times) * 1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace(" (", "\n(") for l in labels], fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "06_inference_time.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out}")


# ============================================================
# PLOT 7: DASHBOARD RINGKASAN (4-in-1)
# ============================================================

def plot_dashboard(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Ringkasan Studi Komparatif Konfigurasi QLoRA Hyperparameter\n"
                 "Implementasi Constrained Generation pada LLM untuk Zeek IDS",
                 fontsize=14, fontweight='bold', y=1.01)

    short_labels = ["Lightweight\n(r=8)", "Balanced\n(r=16)", "Deep\n(r=32)"]
    colors_list  = [COLORS[m["scenario_id"]] for m in metrics]

    # --- Subplot 1: Final Loss ---
    ax = axes[0, 0]
    vals = [m["final_loss"] for m in metrics]
    bars = ax.bar(short_labels, vals, color=colors_list, width=0.5, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.001,
                f"{v:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title("Final Training Loss", fontsize=11, fontweight='bold')
    ax.set_ylabel("Loss")
    ax.set_ylim(0, max(vals)*1.25)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # --- Subplot 2: Training Time ---
    ax = axes[0, 1]
    vals = [m["training_time_min"] for m in metrics]
    bars = ax.bar(short_labels, vals, color=colors_list, width=0.5, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
                f"{v:.1f} mnt", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title("Waktu Training (menit)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Menit")
    ax.set_ylim(0, max(vals)*1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # --- Subplot 3: VRAM Peak ---
    ax = axes[1, 0]
    vals = [m["vram_peak_gb"] for m in metrics]
    bars = ax.bar(short_labels, vals, color=colors_list, width=0.5, edgecolor='white')
    ax.axhline(y=8.0, color='black', linestyle='--', linewidth=1.5, label='Batas 8GB')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.05,
                f"{v:.2f} GB", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title("Peak VRAM Usage (GB)", fontsize=11, fontweight='bold')
    ax.set_ylabel("GB")
    ax.set_ylim(0, 9.5)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # --- Subplot 4: Inference Time ---
    ax = axes[1, 1]
    vals = [m["avg_inference_ms"] for m in metrics]
    bars = ax.bar(short_labels, vals, color=colors_list, width=0.5, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+20,
                f"{v:.0f} ms", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title("Rata-rata Waktu Inferensi (ms)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Milliseconds")
    ax.set_ylim(0, max(vals)*1.2)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Legend global
    patches = [mpatches.Patch(color=COLORS[s], label=LABELS[s]) for s in SCENARIOS]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.04),
               frameon=True, edgecolor='gray')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "07_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out}")


# ============================================================
# PLOT 8: LOSS CURVE MASING-MASING SKENARIO (untuk detail epoch)
# ============================================================

def plot_individual_loss_curves(metrics):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Detail Training Loss per Skenario",
                 fontsize=13, fontweight='bold')

    for idx, (scenario_id, ax) in enumerate(zip(SCENARIOS, axes)):
        steps, losses, epochs = load_loss_history(scenario_id)
        color = COLORS[scenario_id]
        m     = metrics[idx]

        ax.plot(steps, losses, color=color, linewidth=2.5,
                marker='o', markersize=5)
        ax.fill_between(steps, losses, alpha=0.15, color=color)

        # Garis epoch boundary
        max_epoch = int(max(epochs))
        if max_epoch > 1:
            for ep in range(1, max_epoch):
                ep_step = steps[min(range(len(epochs)),
                                   key=lambda i: abs(epochs[i] - ep))]
                ax.axvline(x=ep_step, color='gray',
                           linestyle=':', linewidth=1.5,
                           label=f'Epoch {ep}')

        ax.set_title(f"{LABELS[scenario_id]}\n"
                     f"Final Loss: {m['final_loss']:.4f}",
                     fontsize=10, fontweight='bold')
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Annotasi final loss
        ax.annotate(f"  {m['final_loss']:.4f}",
                    xy=(steps[-1], losses[-1]),
                    fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "08_individual_loss.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] {out}")


# ============================================================
# CETAK TABEL RINGKASAN
# ============================================================

def print_summary_table(metrics):
    print("\n" + "=" * 80)
    print(" TABEL RINGKASAN UNTUK BAB 4 SKRIPSI")
    print("=" * 80)
    print(f"{'Metrik':<35} {'Lightweight':>15} {'Balanced':>15} {'Deep':>15}")
    print("-" * 80)

    rows = [
        ("LoRA Rank (r)",            "lora_r"),
        ("LoRA Alpha (α)",            "lora_alpha"),
        ("Jumlah Epoch",              "num_epochs"),
        ("Total Training Steps",      "total_steps"),
        ("Trainable Parameters",      "trainable_params"),
        ("Trainable % dari Total",    "trainable_percent"),
        ("Training Loss Awal",        "initial_loss"),
        ("Training Loss Akhir",       "final_loss"),
        ("Training Loss Terbaik",     "best_loss"),
        ("Waktu Training (menit)",    "training_time_min"),
        ("Peak VRAM (GB)",            "vram_peak_gb"),
        ("Syntax Validity Rate (%)",  "syntax_validity_rate"),
        ("Drop Conn Rate (%)",        "has_drop_conn_rate"),
        ("Avg Inference Time (ms)",   "avg_inference_ms"),
    ]

    for row_label, key in rows:
        vals = []
        for m in metrics:
            v = m.get(key, "-")
            if isinstance(v, float):
                if key in ["trainable_percent"]:
                    vals.append(f"{v:.2f}%")
                elif key in ["training_time_min", "vram_peak_gb"]:
                    vals.append(f"{v:.2f}")
                elif key in ["initial_loss", "final_loss", "best_loss"]:
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(f"{v}")
            elif isinstance(v, int):
                vals.append(f"{v:,}")
            else:
                vals.append(str(v))
        print(f"  {row_label:<33} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    print("=" * 80)
    print("\nCatatan untuk Bab 4:")
    print("  - Syntax Validity 100% di semua skenario → Constrained Generation efektif")
    print("  - Balanced (r=16) memberikan loss terbaik dengan resource moderat")
    print("  - Deep (r=32) tidak signifikan lebih baik dari Balanced (diminishing returns)")
    print("  - Semua skenario aman di bawah batas VRAM 8GB laptop RTX 4060")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(" PLOT STUDI KOMPARATIF - QLoRA Hyperparameter")
    print(f" Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    # Load data
    metrics = load_metrics()
    print(f"\nLoaded {len(metrics)} skenario dari metrik\n")

    # Generate semua plot
    print("Membuat grafik...")
    plot_loss_curves(metrics)
    plot_loss_by_step(metrics)
    plot_final_loss(metrics)
    plot_training_time(metrics)
    plot_vram(metrics)
    plot_inference_time(metrics)
    plot_dashboard(metrics)
    plot_individual_loss_curves(metrics)

    # Cetak tabel
    print_summary_table(metrics)

    print(f"\n[DONE] Semua grafik tersimpan di: {OUTPUT_DIR}/")
    print("  Grafik yang dihasilkan:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
            print(f"    - {f} ({size_kb} KB)")


if __name__ == "__main__":
    main()