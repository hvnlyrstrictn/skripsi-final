#!/usr/bin/env python3
"""
build_dataset_final.py - Full-Spectrum Dataset Builder
Skripsi: Implementasi Constrained Generation pada LLM Menggunakan QLoRA
         untuk Otomasi Aturan Mitigasi Zeek IDS

Membaca semua log Zeek (conn, http, dns, files, weird, ssl) dari
struktur folder dataset_raw/ dan menghasilkan dataset JSONL
dalam format Alpaca dengan constraint prompt yang konsisten
dengan ids_engine.py.
"""

import os
import json
import csv
import random
import ipaddress
import re
import subprocess
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================

ROOT_DIR       = "."           # Folder dataset_raw (jalankan dari sini)
OUTPUT_FILE    = "dataset_finetune_v3.jsonl"
RANDOM_SEED    = 42

# Sampling rate: 1 = ambil semua baris, N = ambil 1 dari setiap N baris
# Botnet biasanya punya ratusan ribu baris → perlu downsampling
FOLDER_SAMPLING = {
    "benign":    1,    # 1720 conn → ~172 baris -- UPDATE: disemuakan jadi "benign": 1
    "botnet":    150,   # 115883 conn → ~772 baris
    "dos":       500,   # 50k conn + 29k http + 63k weird → ~285 baris total
    "default":   1,     # bruteforce (168), malware (170), web_attack (35) → ambil semua
}

# Augmentasi: gandakan data untuk kategori yang sedikit
# (DoS, Web Attack biasanya hanya 30-50 baris)
AUGMENT_THRESHOLD = 500
AUGMENT_FACTOR_CONN  = 10   # Untuk conn.log attack yang sedikit
AUGMENT_FACTOR_HTTP  = 12   # Untuk http.log attack yang sedikit
AUGMENT_FACTOR_FILES = 15   # Untuk files.log yang sangat jarang
AUGMENT_FACTOR_WEIRD = 10   # Untuk weird.log
AUGMENT_FACTOR_DNS   = 8   # Untuk dns.log

# ============================================================
# PROMPT TEMPLATE (HARUS KONSISTEN DENGAN ids_engine.py)
# ============================================================
# PENTING: Prompt ini SAMA PERSIS dengan yang dipakai saat inference
# agar model terbiasa dengan constraint-nya sejak training.

SYSTEM_CONSTRAINT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
IMPORTANT:
1. Identify the potential attack type based on the log provided.
2. Generate a Zeek script using 'event new_connection' for network attacks or 'event http_request' for web attacks.
3. Use 'drop_connection(c)' to block the attacker IP.
4. Do not create custom Notice types."""

# ============================================================
# INSTRUKSI HEURISTIK PER TIPE LOG
# (Konsisten dengan ids_engine.py - tidak menyebutkan nama tool)
# ============================================================

def get_instruction(log_type: str, log_line: str, folder_name: str) -> str:
    """
    Menghasilkan instruksi heuristik berdasarkan tipe log dan konten.
    Python hanya memberikan KONTEKS GEJALA, AI yang mendiagnosis.
    Tidak ada penyebutan nama tool spesifik (Nmap, Hydra, dll).
    """
    log_lower = log_line.lower()
    folder_lower = folder_name.lower()

    if "benign" in folder_lower:
        return "Analisis log jaringan berikut. Traffic ini adalah aktivitas normal yang tidak mencurigakan. Hasilkan skrip Zeek yang mengizinkan koneksi tanpa pemblokiran."

    # === CONN.LOG ===
    if log_type == "conn":
        if "REJ" in log_line:
            return "Analisis log koneksi yang ditolak (status REJ) berikut. Identifikasi apakah ini indikasi pengintaian jaringan (Reconnaissance) dan buat rule Zeek untuk memblokir IP sumber."
        if "S0" in log_line:
            return "Analisis log koneksi TCP tidak lengkap (status S0) berikut. Identifikasi apakah ini indikasi serangan flood/DoS dan buat rule Zeek untuk memblokir IP sumber."
        if "OTH" in log_line or "RSTR" in log_line:
            return "Analisis log koneksi anomali (status OTH/RSTR) berikut. Identifikasi potensi ancaman dan buat rule Zeek untuk memblokir IP sumber jika diperlukan."
        if "botnet" in folder_lower:
            return "Analisis log koneksi berikut yang terindikasi sebagai komunikasi Command & Control (C2) Botnet. Buat rule Zeek untuk mengisolasi host terinfeksi."
        if "malware" in folder_lower:
            return "Analisis log koneksi berikut yang berkaitan dengan aktivitas malware. Buat rule Zeek untuk memblokir koneksi berbahaya ini."
        if "brute" in folder_lower:
            return "Analisis log koneksi berikut yang terindikasi sebagai percobaan login berulang (Brute Force). Buat rule Zeek untuk memblokir IP penyerang."
        if "dos" in folder_lower:
            return "Analisis log koneksi berikut yang terindikasi sebagai serangan Denial of Service (DoS). Buat rule Zeek untuk memitigasi serangan ini."
        return "Analisis log koneksi jaringan berikut untuk potensi ancaman dan buat rule Zeek yang sesuai."

    # === HTTP.LOG ===
    if log_type == "http":
        if "union" in log_lower or "select" in log_lower:
            return "Analisis log HTTP berikut yang mengandung indikator SQL Injection (SQLi). Buat rule Zeek pada layer aplikasi untuk memblokir payload berbahaya ini."
        if "script" in log_lower or "alert(" in log_lower or "%3c" in log_lower:
            return "Analisis log HTTP berikut yang mengandung indikator Cross-Site Scripting (XSS). Buat rule Zeek untuk memblokir request berbahaya ini."
        if "../" in log_line or "%2e%2e" in log_lower:
            return "Analisis log HTTP berikut yang mengandung indikator Directory Traversal. Buat rule Zeek untuk memblokir percobaan akses ilegal ini."
        return "Analisis log HTTP berikut untuk payload berbahaya dan buat rule Zeek untuk memblokir request mencurigakan dari IP sumber."

    # === DNS.LOG ===
    if log_type == "dns":
        return "Analisis log DNS berikut untuk indikasi komunikasi Command & Control atau DNS Tunneling. Buat rule Zeek untuk memblokir IP sumber jika mencurigakan."

    # === FILES.LOG ===
    if log_type == "files":
        return "Analisis log transfer file berikut untuk indikator malware atau executable berbahaya. Buat rule Zeek untuk memblokir IP sumber yang mentransfer file berbahaya."

    # === WEIRD.LOG ===
    if log_type == "weird":
        return "Analisis anomali protokol jaringan (Weird Log) berikut yang mengindikasikan teknik evasion atau paket malformed. Buat rule Zeek untuk memblokir IP sumber."

    # === SSL.LOG ===
    if log_type == "ssl":
        return "Analisis log SSL/TLS berikut untuk indikasi sertifikat tidak valid atau enkripsi mencurigakan yang berkaitan dengan C2 communication. Buat rule Zeek jika mencurigakan."

    return "Analisis log jaringan berikut untuk aktivitas mencurigakan dan buat rule Zeek mitigasi yang sesuai."


# ============================================================
# GENERATOR SCRIPT ZEEK (Ground Truth)
# ============================================================

def generate_zeek_script(log_type: str, src_ip: str, folder_name: str, log_line: str) -> str:
    """
    Menghasilkan script Zeek mitigasi sebagai ground truth.
    Format konsisten dengan output yang diharapkan dari model.
    """
    folder_lower = folder_name.lower()
    log_lower    = log_line.lower()

    # === BENIGN: Rule "Allow" ===
    # Ini nanti benerin lagi biar lebih jelas
    if "benign" in folder_lower:
        return "event new_connection(c: connection) { # Traffic normal (benign). Tidak ada tindakan blokir yang diperlukan. }"

    # === HTTP: Gunakan event http_request ===
    if log_type == "http":
        if "union" in log_lower or "select" in log_lower:
            return (f'event http_request(c: connection, method: string, original_URI: string, '
                    f'unescaped_URI: string, version: string) {{ '
                    f'if (c$id$orig_h == {src_ip}) {{ '
                    f'drop_connection(c); }}}}')
        return (f'event http_request(c: connection, method: string, original_URI: string, '
                f'unescaped_URI: string, version: string) {{ '
                f'if (c$id$orig_h == {src_ip}) {{ '
                f'NOTICE([$note=Suspicious_Activity, $msg=fmt("Suspicious HTTP from %s", c$id$orig_h)]); '
                f'drop_connection(c); }}}}')

    # === CONN: Botnet ===
    if "botnet" in folder_lower:
        return (f'event new_connection(c: connection) {{ '
                f'if (c$id$orig_h == {src_ip}) {{ '
                f'NOTICE([$note=Botnet_Detected, $msg=fmt("Botnet C2 beacon from %s", c$id$orig_h)]); '
                f'drop_connection(c); }}}}')

    # === CONN: Brute Force ===
    if "brute" in folder_lower:
        return (f'event new_connection(c: connection) {{ '
                f'if (c$id$orig_h == {src_ip}) {{ '
                f'NOTICE([$note=Anomaly_Detected, $msg=fmt("Brute force attempt from %s", c$id$orig_h)]); '
                f'drop_connection(c); }}}}')

    # === CONN: DoS ===
    if "dos" in folder_lower:
        return (f'event new_connection(c: connection) {{ '
                f'if (c$id$orig_h == {src_ip}) {{ '
                f'drop_connection(c); }}}}')

    # === CONN: Malware / Files ===
    if "malware" in folder_lower or log_type == "files":
        return (f'event new_connection(c: connection) {{ '
                f'if (c$id$orig_h == {src_ip}) {{ '
                f'NOTICE([$note=Malware_Detected, $msg=fmt("Malware IoC from %s", c$id$orig_h)]); '
                f'drop_connection(c); }}}}')

    # === WEIRD: Protocol anomaly ===
    if log_type == "weird":
        return (f'event new_connection(c: connection) {{ '
                f'if (c$id$orig_h == {src_ip}) {{ '
                f'NOTICE([$note=Anomaly_Detected, $msg=fmt("Protocol anomaly from %s", c$id$orig_h)]); '
                f'drop_connection(c); }}}}')

    # === DEFAULT ===
    return (f'event new_connection(c: connection) {{ '
            f'if (c$id$orig_h == {src_ip}) {{ '
            f'drop_connection(c); }}}}')


# ============================================================
# AUGMENTASI DATA
# ============================================================

def random_ip() -> str:
    """Generate IP publik acak (non-private) untuk augmentasi."""
    while True:
        ip = ipaddress.IPv4Address(random.randint(0, 2**32 - 1))
        if not (ip.is_private or ip.is_loopback or ip.is_multicast or ip.is_reserved):
            return str(ip)

def augment_entry(entry: dict, src_ip: str, factor: int) -> list:
    """
    Gandakan satu entri dataset dengan variasi IP acak.
    Ini adalah teknik Data Augmentation yang valid untuk
    meningkatkan generalisasi model.
    """
    augmented = []
    for _ in range(factor):
        fake_ip   = random_ip()
        fake_in   = entry["input"].replace(src_ip, fake_ip)
        fake_out  = entry["output"].replace(src_ip, fake_ip)
        augmented.append({
            "instruction": entry["instruction"],
            "input":       fake_in,
            "output":      fake_out
        })
    return augmented


# ============================================================
# PARSER LOG ZEEK
# ============================================================

def parse_zeek_log(file_path: str):
    """
    Generator: membaca log Zeek baris per baris, skip header.
    Menghasilkan (headers, row_dict) untuk setiap baris data.
    """
    headers = None
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("#fields"):
                    headers = line.replace("#fields\t", "").split("\t")
                elif line.startswith("#"):
                    continue
                elif headers and line.strip():
                    parts = line.split("\t")
                    if len(parts) == len(headers):
                        yield headers, dict(zip(headers, parts)), line
    except Exception as e:
        print(f"  [ERROR] Gagal membaca {file_path}: {e}")


def extract_src_ip(data: dict, log_type: str) -> str:
    """Ekstrak IP sumber dari berbagai format log Zeek."""
    candidates = ["id.orig_h", "id_orig_h", "orig_h", "src"]
    for key in candidates:
        val = data.get(key, "")
        if val and val != "-" and val != "(empty)":
            return val
    return ""


def build_log_input_string(data: dict, log_type: str, raw_line: str) -> str:
    """
    Buat string input yang representatif untuk LLM.
    Format: field-field paling informatif dari log.
    """
    if log_type == "conn":
        return (f"{data.get('ts','')} {data.get('uid','')} "
                f"{data.get('id.orig_h','')} {data.get('id.orig_p','')} "
                f"{data.get('id.resp_h','')} {data.get('id.resp_p','')} "
                f"{data.get('proto','')} {data.get('service','-')} "
                f"{data.get('conn_state','-')}")

    if log_type == "http":
        return (f"{data.get('ts','')} {data.get('uid','')} "
                f"{data.get('id.orig_h','')} -> {data.get('id.resp_h','')}:{data.get('id.resp_p','')} "
                f"{data.get('method','-')} {data.get('host','-')} "
                f"{data.get('uri','-')} status:{data.get('status_code','-')}")

    if log_type == "dns":
        return (f"{data.get('ts','')} {data.get('uid','')} "
                f"{data.get('id.orig_h','')} -> {data.get('id.resp_h','')} "
                f"query:{data.get('query','-')} rcode:{data.get('rcode_name','-')}")

    if log_type == "files":
        return (f"{data.get('ts','')} {data.get('fuid','')} "
                f"src:{data.get('tx_hosts','-')} dst:{data.get('rx_hosts','-')} "
                f"mime:{data.get('mime_type','-')} size:{data.get('total_bytes','-')}")

    if log_type == "weird":
        return (f"{data.get('ts','')} {data.get('uid','')} "
                f"{data.get('id.orig_h','-')} -> {data.get('id.resp_h','-')} "
                f"name:{data.get('name','-')} addl:{data.get('addl','-')}")

    # Default: gunakan raw line (untuk ssl, dll)
    return raw_line


# ============================================================
# FUNGSI SAMPLING
# ============================================================

def get_sample_rate(folder_name: str) -> int:
    folder_lower = folder_name.lower()
    for key, rate in FOLDER_SAMPLING.items():
        if key in folder_lower:
            return rate
    return FOLDER_SAMPLING["default"]


def should_augment(folder_name: str, log_type: str, raw_line_count: int) -> bool:
    """
    Augmentasi hanya jika:
    1. Bukan benign (tidak perlu variasi serangan)
    2. Jumlah baris asli setelah sampling < threshold
    """
    folder_lower = folder_name.lower()
    # Dos sudah banyak walaupun di sample, jangan augmentasi
    if "dos" in folder_lower:
        return False
    # Botnet banyak
    if "botnet" in folder_lower:
        return False
    if raw_line_count >= AUGMENT_THRESHOLD:
        return False
    return True


def get_augment_factor(log_type: str) -> int:
    factors = {
        "conn":  AUGMENT_FACTOR_CONN,
        "http":  AUGMENT_FACTOR_HTTP,
        "files": AUGMENT_FACTOR_FILES,
        "weird": AUGMENT_FACTOR_WEIRD,
        "dns":   AUGMENT_FACTOR_DNS,
    }
    return factors.get(log_type, 10)


# ============================================================
# MAIN BUILDER
# ============================================================

def main():
    random.seed(RANDOM_SEED)

    dataset  = []
    stats    = {}
    warnings = []

    print("=" * 60)
    print(" FULL-SPECTRUM DATASET BUILDER")
    print(f" Root Dir : {os.path.abspath(ROOT_DIR)}")
    print(f" Output   : {OUTPUT_FILE}")
    print("=" * 60)

    log_files = sorted(Path(ROOT_DIR).rglob("*.log"))

    if not log_files:
        print("[ERROR] Tidak ada file .log ditemukan!")
        print("  Pastikan kamu sudah menjalankan mass_zeek.sh terlebih dahulu.")
        return

    print(f"\nDitemukan {len(log_files)} file log. Memproses...\n")

    for log_path in log_files:
        filename    = log_path.name
        folder_name = log_path.parent.name
        parts       = filename.replace(".log", "").split("_", 1)

        if len(parts) < 1:
            continue

        log_type = parts[0]   # conn, http, dns, files, weird, ssl

        if log_type not in ["conn", "http", "dns", "files", "weird", "ssl"]:
            continue

        if folder_name not in stats:
            stats[folder_name] = {}
        if log_type not in stats[folder_name]:
            stats[folder_name][log_type] = 0

        sample_rate   = get_sample_rate(folder_name)
        raw_count_result = subprocess.run(
            ["grep", "-vc", "^#", str(log_path)],
            capture_output=True, text=True
        )
        raw_count = int(raw_count_result.stdout.strip()) if raw_count_result.returncode == 0 else 9999
        estimated_after_sampling = raw_count // sample_rate
        do_augment = should_augment(folder_name, log_type, estimated_after_sampling)
        augment_factor = get_augment_factor(log_type)
        line_num      = 0
        added_count   = 0

        print(f"  [{folder_name}] {filename} (rate=1/{sample_rate}, augment={do_augment})")

        for headers, data, raw_line in parse_zeek_log(str(log_path)):
            line_num += 1

            # Sampling
            if line_num % sample_rate != 0:
                continue

            # Ekstrak IP sumber
            src_ip = extract_src_ip(data, log_type)
            if not src_ip or src_ip == "-":
                continue

            # Buat string input untuk LLM
            log_input = build_log_input_string(data, log_type, raw_line)
            if not log_input.strip():
                continue

            # Buat instruksi (heuristik, tidak menyebut nama tool)
            instruction = get_instruction(log_type, raw_line, folder_name)

            # Buat ground truth script Zeek
            zeek_script = generate_zeek_script(log_type, src_ip, folder_name, raw_line)

            # Buat entry
            entry = {
                "instruction": instruction,
                "input":       log_input.strip(),
                "output":      zeek_script
            }

            dataset.append(entry)
            stats[folder_name][log_type] += 1
            added_count += 1

            # Augmentasi untuk data yang sedikit
            if do_augment:
                aug_entries = augment_entry(entry, src_ip, augment_factor)
                dataset.extend(aug_entries)
                stats[folder_name][log_type] += len(aug_entries)
                added_count += len(aug_entries)

        print(f"     -> {added_count} entri ditambahkan")

    print(f"\n{'=' * 60}")
    print(" STATISTIK AKHIR DATASET")
    print(f"{'=' * 60}")

    total = 0
    for folder, log_stats in sorted(stats.items()):
        folder_total = sum(log_stats.values())
        total += folder_total
        print(f"\n  [{folder}] Total: {folder_total}")
        for lt, count in sorted(log_stats.items()):
            print(f"    - {lt}.log : {count:,} entri")

    print(f"\n  GRAND TOTAL : {total:,} entri")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")

    print(f"\nMengacak dataset (seed={RANDOM_SEED})...")
    random.shuffle(dataset)

    print(f"Menyimpan ke {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Dataset berhasil disimpan: {OUTPUT_FILE}")
    print(f"       Total: {total:,} baris")
    print(f"\nVerifikasi 3 sampel acak:")
    for i in random.sample(range(min(len(dataset), 100)), 3):
        print(f"\n  --- Sampel #{i} ---")
        print(f"  Instruction: {dataset[i]['instruction'][:80]}...")
        print(f"  Input      : {dataset[i]['input'][:80]}...")
        print(f"  Output     : {dataset[i]['output'][:100]}...")


if __name__ == "__main__":
    main()