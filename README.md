# QLoRA Fine-Tuning LLM untuk Otomasi Skrip Mitigasi Zeek

Repositori ini berisi kode untuk skripsi S1 Teknik Komputer, Fakultas Teknik, Universitas Indonesia:

> **"Perancangan dan Evaluasi Sistem IDS dengan Komponen Generatif LLM Berbasis QLoRA untuk Otomasi Skrip Mitigasi Zeek"**

**Penulis:** Nakita Rahma Dinanti (2206059401)  
**Pembimbing:** Yan Maraden, S.T., M.T., M.Sc.  
**Program Studi:** Teknik Komputer, FTUI

---

## ⚠️ Catatan Terminologi

Selama pengembangan, eksperimen perbandingan konfigurasi hyperparameter QLoRA ini disebut sebagai **"ablation study"** dalam kode dan nama folder (misalnya `train_ablation.py`, `ablation_results/`). Istilah ini dipertahankan di repositori ini untuk konsistensi dengan kode yang berjalan.

Namun secara teknis, istilah tersebut kurang tepat karena eksperimen ini tidak mempelajari efek penghapusan komponen individual, melainkan **membandingkan tiga konfigurasi hyperparameter (rank, alpha, epoch) secara bersamaan**. Dalam dokumentasi skripsi, istilah yang digunakan adalah **"Studi Komparatif Konfigurasi Hyperparameter QLoRA"**.

| Di kode / repositori | Di skripsi |
|---|---|
| ablation study | studi komparatif konfigurasi hyperparameter QLoRA |
| `model_A_lightweight` | Model Lightweight |
| `model_B_balanced` | Model Balanced |
| `model_C_deep` | Model Deep |
| iterasi 1 / iterasi 2 | eksperimen pertama / eksperimen kedua |

---

## Struktur Repositori

```
skripsi-final/
├── README.md
├── train_ablation.py          # QLoRA fine-tuning (3 konfigurasi)
├── evaluate_models.py         # Evaluasi komparatif 3 model
├── ids_engine_v2.py           # IDS engine (inferensi live)
├── plot_ablation.py           # Visualisasi hasil ablation study
├── plot_evaluation.py         # Visualisasi hasil evaluasi skenario
│
├── dataset/
│   ├── build_dataset_final.py # Pipeline konstruksi dataset JSONL
│   └── mass_zeek.sh           # Batch processing PCAP dengan Zeek
│
├── eval_logs/                 # Log Zeek per skenario pengujian
│   ├── benign/
│   ├── sqli/
│   ├── bruteforce/
│   ├── dos/
│   ├── port_scan/
│   ├── botnet/
│   └── malware/
│
├── evaluation_results/        # Hasil evaluasi (JSON + CSV)
│   └── eval_merged_iterasi2.json
│
├── framework.zeek             # Framework abstraksi Zeek
├── generated_rules.zeek       # Contoh output skrip mitigasi
│
└── scripts/
    ├── run_attacks.sh         # Script serangan (dijalankan di attacker)
    └── setup_victim.sh        # Setup environment victim
```

---

## Requirements

- Ubuntu 22.04 (diuji via WSL2 di Windows 11)
- Python 3.10
- CUDA 12.1
- GPU: minimum 8GB VRAM (diuji: NVIDIA GeForce RTX 4060 Laptop GPU)
- Zeek LTS

---

## Installation

```bash
# Buat environment conda
conda create -n ids-llm python=3.10
conda activate ids-llm

# Install PyTorch dengan CUDA
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies utama
pip install unsloth==2026.1.2
pip install transformers==4.57.3 peft trl bitsandbytes accelerate
```

---

## Step 1 — Persiapan Dataset

### Struktur Folder Dataset

Buat struktur berikut di luar direktori repositori ini (misalnya di `~/dataset_raw/`):
> **Seluruh sumber akuisisi data ada pada buku skripsi di Tabel 3.1. Kategori dan Sumber Akuisisi Data PCAP**

```
dataset_raw/
├── 01_benign/        ← rekam sendiri via wireshark (traffic normal)
├── 02_botnet/        ← Mirai.pcap, Asprox.pcap
├── 03_bruteforce/    ← bruteforce.pcap, dictionary.pcap
├── 04_dos/           ← SynFlood.pcap, GoldenEye.pcap
├── 05_malware/       ← Emotet.pcap
└── 06_web_attack/    ← sqli.pcap, wireshark_sqli_dns.pcap
```

### Ekstraksi Log Zeek dari PCAP

```bash
cd ~/dataset_raw
bash ~/skripsi-final/dataset/mass_zeek.sh
```

Script ini memproses setiap file PCAP secara independen di subdirektori sementara, lalu menggabungkan log menggunakan strategi *header-once append* untuk mengatasi ketidakkompatibilan format link-layer antar file PCAP.

### Konstruksi Dataset Instruksi Alpaca

```bash
cd ~/dataset_raw
python ~/skripsi-final/dataset/build_dataset_final.py
# Output: dataset_finetune_v3.jsonl (13.010 entri)
```

Dataset yang dihasilkan merupakan sintesis dari log jaringan nyata yang dipasangkan dengan ground truth skrip mitigasi yang dikonstruksi secara deterministik berbasis template dan diverifikasi menggunakan Zeek dry-run.

---

## Step 2 — Fine-Tuning Model

```bash
cd ~/skripsi-final
python train_ablation.py
```

Tiga model akan dihasilkan:

| Model | Rank (r) | Alpha (α) | Epoch | Output |
|---|---|---|---|---|
| Lightweight | 8 | 16 | 1 | `ablation_results/model_A_lightweight/` |
| Balanced | 16 | 32 | 2 | `ablation_results/model_B_balanced/` |
| Deep | 32 | 64 | 3 | `ablation_results/model_C_deep/` |

> **Alternatif:** Download model yang sudah ditraining dari HuggingFace (link menyusul setelah upload).

---

## Step 3 — Evaluasi Skenario

Log Zeek per skenario sudah tersedia di `eval_logs/` dan dapat langsung digunakan.

> **Catatan:** Untuk merekam traffic evaluasi selain botnet dan malware secara mandiri, gunakan topologi dua laptop yang terhubung via switch dengan IP victim `192.168.1.10` dan attacker `192.168.1.20`. Jalankan `scripts/run_attacks.sh` di attacker dan `scripts/setup_victim.sh` di victim terlebih dahulu.


```bash
# Evaluasi semua skenario (7 skenario × 3 model = 21 kombinasi)
python evaluate_models.py

# Evaluasi skenario tertentu
python evaluate_models.py --scenario sqli --repetitions 10
python evaluate_models.py --scenario port_scan --repetitions 10

# Daftar skenario yang tersedia:
# port_scan  — Port Scanning via Nmap (zero-day)
# sqli       — SQL Injection via HTTP
# bruteforce — TCP Connection Flood
# dos        — SYN Flood via hping3
# benign     — Traffic normal (false positive test)
# botnet     — Replay PCAP Asprox C2
# malware    — Replay PCAP Emotet
```

Output tersimpan di `evaluation_results/` dalam format CSV dan JSON.

---

## Step 4 — Generate Plot

```bash
python plot_ablation.py    # Grafik training loss, VRAM, waktu training
python plot_evaluation.py  # Grafik detection rate, FPR, SVR
```

---

## Reproducing Results

Hasil evaluasi yang dilaporkan dalam skripsi tersimpan di `evaluation_results/eval_merged_iterasi2.json`.

Expected results (eksperimen kedua, dataset 13.010 entri, benign 36.5%):

| Metrik | Nilai |
|---|---|
| Detection Rate | 100% (semua 7 skenario, semua model) |
| False Positive Rate | 0% (skenario benign, semua model) |
| Syntax Validity Rate | 100% (semua skenario, semua model) |
| Peak VRAM (Model Deep) | 4.66 GB dari 8 GB |
| Training time (Model Deep) | 202.0 menit |

> **Catatan:** Hasil evaluasi untuk skenario `botnet` dan `malware` menggunakan metode replay PCAP (bukan live traffic), sehingga dapat direproduksi langsung menggunakan log yang tersedia di `eval_logs/`.

---

## Topologi Pengujian

```
[Attacker Laptop]          [Switch]          [Victim Laptop]
 192.168.1.20     ←————————————————————→     192.168.1.10
 run_attacks.sh                               Zeek + evaluate_models.py
```

Untuk skenario botnet dan malware, replay PCAP dilakukan secara offline:

```bash
# Di victim laptop
mkdir -p /tmp/zeek_replay && cd /tmp/zeek_replay
zeek -C -r ~/dataset_raw/02_botnet/Asprox.pcap framework.zeek
```

---

## Lisensi

Kode dalam repositori ini dikembangkan untuk keperluan penelitian skripsi S1 dan bebas digunakan untuk keperluan akademis dengan menyertakan atribusi yang sesuai.
