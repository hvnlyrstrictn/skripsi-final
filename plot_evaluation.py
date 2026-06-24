#!/usr/bin/env python3
"""
plot_evaluation.py - Visualisasi Hasil Evaluasi Komparatif 3 Model
Skripsi: Implementasi Constrained Generation pada LLM Menggunakan QLoRA
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ============================================================
# KONFIGURASI
# ============================================================

EVAL_DIR   = "evaluation_results"
OUTPUT_DIR = os.path.join(EVAL_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "A_lightweight": "#2196F3",
    "B_balanced":    "#4CAF50",
    "C_deep":        "#F44336",
}

MODEL_LABELS = {
    "A_lightweight": "Lightweight\n(r=8, 1 epoch)",
    "B_balanced":    "Balanced\n(r=16, 2 epoch)",
    "C_deep":        "Deep\n(r=32, 3 epoch)",
}

SCENARIO_LABELS = {
    "port_scan":  "Port Scan\n(Zero-Day)",
    "sqli":       "SQL Injection\n(Known)",
    "bruteforce": "Brute Force\n(Known)",
    "dos":        "DoS Flood\n(Known)",
    "botnet":     "Botnet C2\n(Asprox)",
    "malware":    "Malware\n(Emotet)",
    "benign":     "Benign\n(FP Test)",
}

MODELS    = ["A_lightweight", "B_balanced", "C_deep"]
SCENARIOS = ["port_scan", "sqli", "bruteforce", "dos", "botnet", "malware", "benign"]
# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    path = os.path.join(EVAL_DIR, "eval_merged_iterasi2.json")
    with open(path) as f:
        data = json.load(f)
    # Index: results[(model_id, scenario)] = entry
    index = {}
    for entry in data:
        index[(entry["model_id"], entry["scenario"])] = entry
    return index

# ============================================================
# PLOT 1: DETECTION RATE PER SKENARIO (grouped bar)
# ============================================================

def plot_detection_rate(idx):
    attack_scenarios = ["port_scan", "sqli", "bruteforce", "dos", "botnet", "malware"]
    x     = np.arange(len(attack_scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model_id in enumerate(MODELS):
        vals = []
        for sc in attack_scenarios:
            entry = idx.get((model_id, sc))
            vals.append(entry["detection_rate"] if entry and entry["detection_rate"] is not None else 0)

        bars = ax.bar(x + i * width, vals, width,
                      label=MODEL_LABELS[model_id].replace("\n", " "),
                      color=COLORS[model_id], edgecolor="white", linewidth=1.2)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in attack_scenarios], fontsize=10)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_title("Detection Rate per Skenario Serangan\nKomparasi 3 Model QLoRA",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "01_detection_rate.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")

# ============================================================
# PLOT 2: SYNTAX VALIDITY RATE (grouped bar)
# ============================================================

def plot_syntax_validity(idx):
    x     = np.arange(len(SCENARIOS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_id in enumerate(MODELS):
        vals = []
        for sc in SCENARIOS:
            entry = idx.get((model_id, sc))
            vals.append(entry["syntax_validity_rate"] if entry else 0)

        bars = ax.bar(x + i * width, vals, width,
                      label=MODEL_LABELS[model_id].replace("\n", " "),
                      color=COLORS[model_id], edgecolor="white", linewidth=1.2)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=10)
    ax.set_ylabel("Syntax Validity Rate (%)", fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_title("Syntax Validity Rate per Skenario\nKomparasi 3 Model QLoRA",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "02_syntax_validity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")

# ============================================================
# PLOT 3: INFERENCE TIME (grouped bar dengan error bar)
# ============================================================

def plot_inference_time(idx):
    x     = np.arange(len(SCENARIOS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_id in enumerate(MODELS):
        means = []
        stds  = []
        for sc in SCENARIOS:
            entry = idx.get((model_id, sc))
            means.append(entry["avg_inference_ms"] if entry else 0)
            stds.append(entry["std_inference_ms"] if entry else 0)

        bars = ax.bar(x + i * width, means, width,
                      label=MODEL_LABELS[model_id].replace("\n", " "),
                      color=COLORS[model_id], edgecolor="white", linewidth=1.2,
                      yerr=stds, capsize=4, error_kw={"linewidth": 1.5})

        for bar, v in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.1 + 50,
                    f"{v:.0f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=10)
    ax.set_ylabel("Rata-rata Inference Time (ms)", fontsize=12)
    ax.set_title("Inference Time per Skenario\nKomparasi 3 Model QLoRA (dengan Std Dev)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_inference_time.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")

# ============================================================
# PLOT 4: IP ACCURACY RATE
# ============================================================

def plot_ip_accuracy(idx):
    attack_scenarios = ["port_scan", "sqli", "bruteforce", "dos", "botnet", "malware"]
    x     = np.arange(len(attack_scenarios))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model_id in enumerate(MODELS):
        vals = []
        for sc in attack_scenarios:
            entry = idx.get((model_id, sc))
            vals.append(entry["ip_accuracy_rate"] if entry else 0)

        bars = ax.bar(x + i * width, vals, width,
                      label=MODEL_LABELS[model_id].replace("\n", " "),
                      color=COLORS[model_id], edgecolor="white", linewidth=1.2)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in attack_scenarios], fontsize=10)
    ax.set_ylabel("IP Extraction Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_title("Akurasi Ekstraksi IP Penyerang per Skenario\nKomparasi 3 Model QLoRA",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "04_ip_accuracy.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")

# ============================================================
# PLOT 5: DASHBOARD 2x2 (Detection, Syntax, IP Acc, Inference)
# ============================================================

def plot_dashboard(idx):
    attack_scenarios = ["port_scan", "sqli", "bruteforce", "dos", "botnet", "malware"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dashboard Evaluasi Komparatif 3 Model QLoRA\n"
                 "Implementasi Constrained Generation untuk Zeek IDS",
                 fontsize=13, fontweight="bold", y=1.01)

    x     = np.arange(len(attack_scenarios))
    width = 0.25
    xlabels = [SCENARIO_LABELS[s] for s in attack_scenarios]

    metrics = [
        ("Detection Rate (%)",       "detection_rate",      axes[0, 0], False),
        ("Syntax Validity Rate (%)",  "syntax_validity_rate", axes[0, 1], False),
        ("IP Extraction Acc (%)",     "ip_accuracy_rate",    axes[1, 0], False),
        ("Avg Inference Time (ms)",   "avg_inference_ms",    axes[1, 1], True),
    ]

    for metric_label, key, ax, is_time in metrics:
        for i, model_id in enumerate(MODELS):
            vals = []
            for sc in attack_scenarios:
                entry = idx.get((model_id, sc))
                v = entry.get(key) if entry else None
                vals.append(v if v is not None else 0)

            bars = ax.bar(x + i * width, vals, width,
                          color=COLORS[model_id], edgecolor="white",
                          linewidth=1.0,
                          label=MODEL_LABELS[model_id].replace("\n", " "))

            for bar, v in zip(bars, vals):
                label = f"{v:.0f}" + ("" if is_time else "%")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (50 if is_time else 1.5),
                        label, ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + width)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel(metric_label, fontsize=10)
        if not is_time:
            ax.set_ylim(0, 120)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        ax.set_title(metric_label, fontsize=10, fontweight="bold")

    patches = [mpatches.Patch(color=COLORS[m],
               label=MODEL_LABELS[m].replace("\n", " ")) for m in MODELS]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.03),
               frameon=True, edgecolor="gray")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "05_dashboard_eval.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")

# ============================================================
# PLOT 6: FALSE POSITIVE RATE (benign only)
# ============================================================

def plot_false_positive(idx):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = [MODEL_LABELS[m].replace("\n", " ") for m in MODELS]
    vals   = []
    for m in MODELS:
        entry = idx.get((m, "benign"))
        vals.append(entry["false_positive_rate"] if entry and
                    entry["false_positive_rate"] is not None else 0)

    colors = [COLORS[m] for m in MODELS]
    bars   = ax.bar(labels, vals, color=colors, width=0.5,
                    edgecolor="white", linewidth=1.5)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{v:.0f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

    ax.set_ylabel("False Positive Rate (%)", fontsize=12)
    ax.set_ylim(0, 120)
    ax.set_title("False Positive Rate pada Traffic Normal (Benign)\n"
                 "Komparasi 3 Model QLoRA",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="green", linestyle="--",
               linewidth=2, label="Target FPR = 0%")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "06_false_positive_rate.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {out}")

# ============================================================
# CETAK TABEL RINGKASAN UNTUK BAB 4
# ============================================================

def print_summary_table(idx):
    print("\n" + "=" * 90)
    print(" TABEL RINGKASAN EVALUASI UNTUK BAB 4")
    print("=" * 90)
    print(f"{'Skenario':<14} {'Metrik':<22} {'Lightweight (r=8)':>18} "
          f"{'Balanced (r=16)':>16} {'Deep (r=32)':>12}")
    print("-" * 90)

    for sc in SCENARIOS:
        is_attack = sc != "benign"
        rows = []

        if is_attack:
            rows.append(("Detection Rate (%)", "detection_rate"))
        else:
            rows.append(("False Positive Rate (%)", "false_positive_rate"))

        rows += [
            ("Syntax Validity (%)",  "syntax_validity_rate"),
            ("IP Accuracy (%)",       "ip_accuracy_rate"),
            ("Avg Inference (ms)",    "avg_inference_ms"),
            ("Std Dev (ms)",          "std_inference_ms"),
        ]

        print(f"\n  [{SCENARIO_LABELS[sc].replace(chr(10), ' ')}]")
        for metric_label, key in rows:
            vals = []
            for m in MODELS:
                entry = idx.get((m, sc))
                v = entry.get(key) if entry else None
                if v is None:
                    vals.append("  -")
                elif key in ["avg_inference_ms", "std_inference_ms"]:
                    vals.append(f"{v:>6.0f} ms")
                else:
                    vals.append(f"{v:>6.1f}%")
            print(f"    {metric_label:<22} {vals[0]:>18} {vals[1]:>16} {vals[2]:>12}")

    print("\n" + "=" * 90)

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print(" PLOT EVALUASI KOMPARATIF - QLoRA Zeek IDS")
    print(f" Output dir: {OUTPUT_DIR}")
    print("=" * 60)

    idx = load_data()
    print(f"\nLoaded {len(idx)} kombinasi model × skenario\n")

    print("Membuat grafik...")
    plot_detection_rate(idx)
    plot_syntax_validity(idx)
    plot_inference_time(idx)
    plot_ip_accuracy(idx)
    plot_dashboard(idx)
    plot_false_positive(idx)

    print_summary_table(idx)

    print(f"\n[DONE] Semua grafik tersimpan di: {OUTPUT_DIR}/")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
            print(f"  - {f} ({size_kb} KB)")

if __name__ == "__main__":
    main()