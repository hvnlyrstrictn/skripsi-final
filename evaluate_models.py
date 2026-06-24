#!/usr/bin/env python3
"""
evaluate_models.py - Evaluasi Komparatif 3 Model QLoRA
Skripsi: Implementasi Constrained Generation pada LLM Menggunakan QLoRA
         untuk Otomasi Aturan Mitigasi Zeek IDS

Skenario Testing (Topologi 2 Laptop via Switch):
  - Victim IP : 192.168.1.10 (laptop ini, menjalankan Zeek + script ini)
  - Attacker IP: 192.168.1.20 (laptop lain)

Alur:
  1. Zeek capture traffic dari interface fisik (eth0)
  2. Script ini membaca log yang dihasilkan Zeek
  3. Setiap baris log diproses oleh ketiga model bergantian
  4. Metrik dicatat ke CSV + JSON

Cara penggunaan:
  python evaluate_models.py --scenario <nama> --repetitions 10

Skenario yang tersedia:
  port_scan    - Nmap port scanning (Zero-Day)
  sqli         - SQL Injection via HTTP
  bruteforce   - TCP connection flood (Brute Force)
  dos          - SYN Flood (DoS)
  benign       - Traffic normal (False Positive test)
"""

import torch
import json
import csv
import time
import os
import re
import gc
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from unsloth import FastLanguageModel

# ============================================================
# KONFIGURASI
# ============================================================

MODELS = {
    "A_lightweight": {
        "path":  "ablation_results/model_A_lightweight",
        "label": "Lightweight (r=8, α=16, 1 epoch)",
        "r":     8,
        "alpha": 16,
    },
    "B_balanced": {
        "path":  "ablation_results/model_B_balanced",
        "label": "Balanced (r=16, α=32, 2 epoch)",
        "r":     16,
        "alpha": 32,
    },
    "C_deep": {
        "path":  "ablation_results/model_C_deep",
        "label": "Deep (r=32, α=64, 3 epoch)",
        "r":     32,
        "alpha": 64,
    },
}

# Direktori untuk hasil evaluasi
EVAL_OUTPUT_DIR  = "evaluation_results"
FRAMEWORK_FILE   = "framework.zeek"
ZEEK_LOG_DIR     = "eval_logs"          # Zeek akan menulis log di sini
ZEEK_INTERFACE   = "eth0"               # Ganti sesuai interface fisik laptop victim

# Prompt template (SAMA dengan training)
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
IMPORTANT:
1. Identify the potential attack type based on the log provided.
2. Generate a Zeek script using 'event new_connection' for network attacks or 'event http_request' for web attacks.
3. Use 'drop_connection(c)' to block the attacker IP.
4. Do not create custom Notice types.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

def get_instruction(log_line: str, log_type: str, is_benign: bool = False) -> str:
    if is_benign:
        return "Analisis log jaringan berikut. Traffic ini adalah aktivitas normal yang tidak mencurigakan. Hasilkan skrip Zeek yang mengizinkan koneksi tanpa pemblokiran."
    # if "REJ" in log_line or "RSTR" in log_line:
    #     return "Analisis log koneksi yang ditolak (REJ/RSTR). Identifikasi apakah ini Reconnaissance/Scanning dan buat rule blokir."
    # if "S0" in log_line:
    #     return "Analisis koneksi TCP tidak lengkap (S0). Identifikasi apakah ini DoS Flood dan buat rule blokir."
    # if log_type == "http":
    #     return "Analisis log HTTP ini. Identifikasi payload berbahaya (SQLi/XSS) dan buat rule blokir IP sumber."
    # if log_type == "conn":
    #     return "Analisis log koneksi jaringan ini untuk potensi ancaman dan buat rule Zeek yang sesuai."
    if log_type == "http":
        return "Analisis log HTTP berikut. Tentukan apakah request ini mencurigakan. Jika ya, buat skrip mitigasi Zeek yang sesuai."
    if log_type == "conn":
        return "Analisis log koneksi jaringan berikut. Tentukan apakah koneksi ini mencurigakan. Jika ya, buat skrip mitigasi Zeek yang sesuai."

    return "Analisis log jaringan ini untuk aktivitas mencurigakan dan buat rule Zeek mitigasi."

# ============================================================
# SKENARIO TESTING
# ============================================================

SCENARIOS = {
    "port_scan": {
        "label":       "Port Scanning (Zero-Day)",
        "description": "Nmap SYN scan ke victim. Tidak ada di training data.",
        "log_type":    "conn",
        "attacker_cmd": "nmap -sS -p 1-1000 192.168.1.10",
        "expected_event": "new_connection",
        "is_attack":   True,
        "zeek_logs":   ["conn.log"],
        "filter_fn":   lambda line: "REJ" in line or "S0" in line or "RSTR" in line,
    },
    "sqli": {
        "label":       "SQL Injection (Known Attack)",
        "description": "HTTP request dengan payload SQLi ke web server victim.",
        "log_type":    "http",
        "attacker_cmd": "curl -s 'http://192.168.1.10:8080/login?id=1%27UNION%20SELECT%201,2,3--'",
        "expected_event": "http_request",
        "is_attack":   True,
        "zeek_logs":   ["http.log"],
        "filter_fn":   lambda line: not line.startswith("#") and line.strip(),
    },
    "bruteforce": {
        "label":       "Brute Force (Known Attack)",
        "description": "TCP connection flood ke port 2222 victim.",
        "log_type":    "conn",
        "attacker_cmd": "for i in $(seq 1 20); do nc -z -w1 192.168.1.10 2222; done",
        "expected_event": "new_connection",
        "is_attack":   True,
        "zeek_logs":   ["conn.log"],
        "filter_fn":   lambda line: "REJ" in line or "SF" in line or "S0" in line,
    },
    "dos": {
        "label":       "DoS SYN Flood (Known Attack)",
        "description": "hping3 SYN flood ke victim port 80.",
        "log_type":    "conn",
        "attacker_cmd": "sudo hping3 -S -p 80 --flood 192.168.1.10",
        "expected_event": "new_connection",
        "is_attack":   True,
        "zeek_logs":   ["conn.log"],
        "filter_fn":   lambda line: "S0" in line or "REJ" in line,
    },
    "benign": {
        "label":       "Benign Traffic (False Positive Test)",
        "description": "HTTP request normal. Model TIDAK BOLEH memblokir.",
        "log_type":    "http",
        "attacker_cmd": "curl -s 'http://192.168.1.10:8080/index.html'",
        "expected_event": None,   # Tidak ada blokir yang diharapkan
        "is_attack":   False,
        "zeek_logs":   ["http.log", "conn.log"],
        "filter_fn":   lambda line: not line.startswith("#") and line.strip(),
    },
    "botnet": {
    "label":       "Botnet C2 Communication (Asprox)",
    "description": "Replay PCAP Asprox. Tidak dimasukkan ke training (training hanya Mirai).",
    "log_type":    "http",
    "attacker_cmd": "N/A - Replay PCAP Asprox.pcap via Zeek offline",
    "expected_event": "http_request",
    "is_attack":   True,
    "zeek_logs":   ["http.log", "conn.log"],
    "filter_fn":   lambda line: not line.startswith("#") and line.strip(),
    },
    "malware": {
        "label":       "Malware Communication (Emotet)",
        "description": "Replay PCAP Emotet. Dipisah dari data training untuk evaluasi.",
        "log_type":    "http",
        "attacker_cmd": "N/A - Replay PCAP Emotet.pcap via Zeek offline",
        "expected_event": "http_request",
        "is_attack":   True,
        "zeek_logs":   ["http.log", "conn.log"],
        "filter_fn":   lambda line: not line.startswith("#") and line.strip(),
    },
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_zeek_code(code: str) -> str:
    """Post-processing: bersihkan komentar, patch HTTP."""
    code_clean = re.sub(r'#[^}]*', '', code).strip()
    if "event http_request" in code_clean and "version: string" not in code_clean:
        code_clean = re.sub(
            r'event http_request\((.*?)\)',
            'event http_request(c: connection, method: string, '
            'original_URI: string, unescaped_URI: string, version: string)',
            code_clean
        )
    return code_clean


def validate_zeek_syntax(script: str) -> tuple[bool, str]:
    """Validasi sintaks Zeek. Return (is_valid, error_msg)."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".zeek", delete=False, dir="/tmp"
    ) as tf:
        tf.write(script + "\n")
        tmp_path = tf.name
    try:
        cmd = ["zeek", tmp_path]
        if os.path.exists(FRAMEWORK_FILE):
            cmd = ["zeek", FRAMEWORK_FILE, tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr.strip().split('\n')[0]
    except Exception as e:
        return False, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def extract_attacker_ip(log_line: str) -> str:
    """Ekstrak IP sumber dari baris log Zeek (kolom ke-3 tab-separated)."""
    parts = log_line.split('\t')
    if len(parts) > 2:
        ip = parts[2].strip()
        if ip and ip != '-':
            return ip
    # Fallback: regex IPv4
    ipv4 = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', log_line)
    return ipv4[0] if ipv4 else ""


def check_ip_in_script(script: str, expected_ip: str) -> bool:
    """Cek apakah IP yang benar ada di dalam script."""
    return expected_ip in script if expected_ip else False


def check_event_type(script: str, expected_event: str) -> bool:
    """Cek apakah event handler yang digunakan sudah benar."""
    if not expected_event:
        return True
    return f"event {expected_event}" in script


def check_drop_connection(script: str) -> bool:
    """Cek apakah ada perintah blokir di script."""
    return "drop_connection" in script


def read_zeek_log(log_path: str, filter_fn=None) -> list:
    """Baca log Zeek dan kembalikan daftar baris data (non-header)."""
    lines = []
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('#'):
                    continue
                if not line.strip():
                    continue
                if filter_fn is None or filter_fn(line):
                    lines.append(line)
    except FileNotFoundError:
        pass
    return lines


# ============================================================
# CORE EVALUATION FUNCTION
# ============================================================

def run_inference(model, tokenizer, log_line: str, log_type: str, is_benign: bool = False) -> dict:
    """
    Jalankan satu inference dan kembalikan hasil lengkap.
    """
    instruction = get_instruction(log_line, log_type, is_benign)
    prompt      = ALPACA_PROMPT.format(instruction, log_line, "")

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    t_start = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = 128,
            use_cache      = True,
            do_sample      = False,   # Greedy untuk konsistensi
        )
    t_end = time.time()

    generated  = tokenizer.batch_decode(output_ids)[0]
    raw_code   = generated.split("### Response:")[-1].replace("<|im_end|>", "").strip()
    clean_code = clean_zeek_code(raw_code)

    is_valid, err = validate_zeek_syntax(clean_code)
    attacker_ip   = extract_attacker_ip(log_line)
    has_correct_ip   = check_ip_in_script(clean_code, attacker_ip)
    has_correct_event = check_event_type(clean_code, None)  # Will check per scenario
    has_drop_conn = check_drop_connection(clean_code)

    return {
        "instruction":       instruction,
        "log_input":         log_line[:100],
        "generated_code":    clean_code,
        "attacker_ip":       attacker_ip,
        "has_drop_conn":     has_drop_conn,
        "syntax_valid":      is_valid,
        "syntax_error":      err,
        "ip_in_script":      has_correct_ip,
        "inference_ms":      round((t_end - t_start) * 1000, 2),
    }


def evaluate_scenario_with_model(
    model, tokenizer, scenario_key: str,
    log_lines: list, repetitions: int,
    attacker_ip: str, model_id: str
) -> dict:
    """
    Evaluasi satu skenario dengan satu model.
    Jalankan repetitions kali dan agregasi hasilnya.
    """
    scenario  = SCENARIOS[scenario_key]
    log_type  = scenario["log_type"]
    is_attack = scenario["is_attack"]
    exp_event = scenario["expected_event"]

    if not log_lines:
        print(f"    [WARN] Tidak ada log untuk skenario {scenario_key}")
        return None

    # Ambil max 3 baris log paling representatif
    sample_lines = log_lines[:min(3, len(log_lines))]

    all_results = []

    for rep in range(repetitions):
        for log_line in sample_lines:
            result = run_inference(model, tokenizer, log_line, log_type, is_benign=not is_attack)

            # Evaluasi khusus per skenario
            result["expected_event"]    = exp_event
            result["correct_event"]     = check_event_type(result["generated_code"], exp_event)
            result["is_attack_scenario"] = is_attack

            # True/False Positive logic
            model_says_attack = result["has_drop_conn"]
            if is_attack:
                result["tp"] = 1 if model_says_attack else 0
                result["fn"] = 0 if model_says_attack else 1
                result["fp"] = 0
                result["tn"] = 0
            else:
                result["tp"] = 0
                result["fn"] = 0
                result["fp"] = 1 if model_says_attack else 0
                result["tn"] = 0 if model_says_attack else 1

            all_results.append(result)

    # Agregasi
    n = len(all_results)
    aggregated = {
        "model_id":             model_id,
        "scenario":             scenario_key,
        "scenario_label":       scenario["label"],
        "is_attack":            is_attack,
        "total_inferences":     n,
        "repetitions":          repetitions,
        "log_samples_used":     len(sample_lines),

        # Detection metrics
        "tp_count":             sum(r["tp"] for r in all_results),
        "fn_count":             sum(r["fn"] for r in all_results),
        "fp_count":             sum(r["fp"] for r in all_results),
        "tn_count":             sum(r["tn"] for r in all_results),

        # Quality metrics
        "syntax_valid_count":   sum(r["syntax_valid"] for r in all_results),
        "syntax_validity_rate": round(sum(r["syntax_valid"] for r in all_results) / n * 100, 2),
        "ip_correct_count":     sum(r["ip_in_script"] for r in all_results),
        "ip_accuracy_rate":     round(sum(r["ip_in_script"] for r in all_results) / n * 100, 2),
        "correct_event_count":  sum(r["correct_event"] for r in all_results),
        "event_accuracy_rate":  round(sum(r["correct_event"] for r in all_results) / n * 100, 2),

        # Performance metrics
        "avg_inference_ms":     round(sum(r["inference_ms"] for r in all_results) / n, 2),
        "min_inference_ms":     round(min(r["inference_ms"] for r in all_results), 2),
        "max_inference_ms":     round(max(r["inference_ms"] for r in all_results), 2),
        "std_inference_ms":     round(
            (sum((r["inference_ms"] - sum(r2["inference_ms"] for r2 in all_results)/n)**2
                 for r in all_results) / n) ** 0.5, 2
        ),

        # Sample outputs
        "sample_outputs":       [r["generated_code"] for r in all_results[:3]],
    }

    # Derived rates
    attack_inferences = sum(1 for r in all_results if r["is_attack_scenario"])
    benign_inferences = n - attack_inferences

    if attack_inferences > 0:
        aggregated["detection_rate"] = round(
            aggregated["tp_count"] / attack_inferences * 100, 2)
    else:
        aggregated["detection_rate"] = None

    if benign_inferences > 0:
        aggregated["false_positive_rate"] = round(
            aggregated["fp_count"] / benign_inferences * 100, 2)
    else:
        aggregated["false_positive_rate"] = None

    return aggregated


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi komparatif 3 model QLoRA untuk Zeek IDS"
    )
    parser.add_argument(
        "--scenario", "-s",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Skenario yang akan dievaluasi (default: all)"
    )
    parser.add_argument(
        "--repetitions", "-r",
        type=int, default=10,
        help="Jumlah repetisi per skenario per model (default: 10)"
    )
    parser.add_argument(
        "--log-dir", "-l",
        default=ZEEK_LOG_DIR,
        help=f"Direktori log Zeek (default: {ZEEK_LOG_DIR})"
    )
    parser.add_argument(
        "--attacker-ip", "-a",
        default="192.168.1.20",
        help="IP attacker (default: 192.168.1.20)"
    )
    args = parser.parse_args()

    scenarios_to_run = (
        list(SCENARIOS.keys()) if args.scenario == "all"
        else [args.scenario]
    )

    os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print(" EVALUASI KOMPARATIF MODEL QLoRA - Zeek IDS")
    print(f" Timestamp  : {timestamp}")
    print(f" Skenario   : {scenarios_to_run}")
    print(f" Repetisi   : {args.repetitions}x per skenario per model")
    print(f" Log dir    : {args.log_dir}")
    print(f" Attacker IP: {args.attacker_ip}")
    print("=" * 65)

    # Inisialisasi output files
    csv_path  = os.path.join(EVAL_OUTPUT_DIR, f"eval_{timestamp}.csv")
    json_path = os.path.join(EVAL_OUTPUT_DIR, f"eval_{timestamp}.json")

    CSV_HEADERS = [
        "model_id", "model_label", "scenario", "scenario_label",
        "is_attack", "total_inferences", "repetitions",
        "tp_count", "fn_count", "fp_count", "tn_count",
        "detection_rate", "false_positive_rate",
        "syntax_validity_rate", "ip_accuracy_rate", "event_accuracy_rate",
        "avg_inference_ms", "min_inference_ms", "max_inference_ms", "std_inference_ms",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()

    all_results = []

    # ======================================================
    # Loop per MODEL
    # ======================================================
    for model_id, model_cfg in MODELS.items():
        print(f"\n{'=' * 65}")
        print(f" MODEL: {model_cfg['label']}")
        print(f"{'=' * 65}")

        # Load model
        print(f"\n[LOAD] Memuat model dari: {model_cfg['path']}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name    = model_cfg["path"],
            max_seq_length = 2048,
            dtype          = None,
            load_in_4bit   = True,
        )
        FastLanguageModel.for_inference(model)
        print(f"  Model siap.")

        # ======================================================
        # Loop per SKENARIO
        # ======================================================
        for scenario_key in scenarios_to_run:
            scenario = SCENARIOS[scenario_key]
            print(f"\n  [SCENARIO] {scenario['label']}")
            print(f"  Deskripsi : {scenario['description']}")
            print(f"  Log type  : {scenario['log_type']}")

            # Baca log yang tersedia
            log_lines = []
            for log_filename in scenario["zeek_logs"]:
                log_path = os.path.join(args.log_dir, log_filename)
                if os.path.exists(log_path):
                    lines = read_zeek_log(log_path, scenario["filter_fn"])
                    log_lines.extend(lines)
                    print(f"  Log file  : {log_path} ({len(lines)} baris relevan)")
                else:
                    print(f"  [WARN] Log tidak ditemukan: {log_path}")
                    print(f"         Pastikan Zeek sudah dijalankan untuk skenario ini.")

            if not log_lines:
                print(f"  [SKIP] Tidak ada log untuk skenario ini. Lewati.")
                continue

            # Jalankan evaluasi
            print(f"  Menjalankan {args.repetitions} repetisi × "
                  f"{min(3, len(log_lines))} log samples...")

            result = evaluate_scenario_with_model(
                model        = model,
                tokenizer    = tokenizer,
                scenario_key = scenario_key,
                log_lines    = log_lines,
                repetitions  = args.repetitions,
                attacker_ip  = args.attacker_ip,
                model_id     = model_id,
            )

            if result is None:
                continue

            result["model_label"] = model_cfg["label"]
            all_results.append(result)

            # Cetak ringkasan
            print(f"\n  [HASIL] {model_id} × {scenario_key}:")
            if result["is_attack"]:
                print(f"    Detection Rate     : {result['detection_rate']}%")
            else:
                print(f"    False Positive Rate: {result['false_positive_rate']}%")
            print(f"    Syntax Validity    : {result['syntax_validity_rate']}%")
            print(f"    IP Accuracy        : {result['ip_accuracy_rate']}%")
            print(f"    Event Accuracy     : {result['event_accuracy_rate']}%")
            print(f"    Avg Inference      : {result['avg_inference_ms']} ms "
                  f"(±{result['std_inference_ms']} ms)")

            # Tulis ke CSV
            csv_row = {k: result.get(k, "") for k in CSV_HEADERS}
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writerow(csv_row)

            # Tulis JSON (update setiap entry)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        # Cleanup GPU memory antar model
        print(f"\n[CLEANUP] Membersihkan memory setelah model {model_id}...")
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # ======================================================
    # RINGKASAN AKHIR
    # ======================================================
    print(f"\n{'=' * 65}")
    print(" RINGKASAN HASIL EVALUASI")
    print(f"{'=' * 65}")

    # Tabel per model × per skenario
    print(f"\n{'Model':<20} {'Skenario':<22} {'Det%':>6} {'FPR%':>6} "
          f"{'Syntax%':>8} {'IP%':>6} {'ms':>8}")
    print("-" * 78)

    for r in all_results:
        det_str = (f"{r['detection_rate']:.1f}" if r['detection_rate'] is not None
                   else "-")
        fpr_str = (f"{r['false_positive_rate']:.1f}" if r['false_positive_rate'] is not None
                   else "-")
        print(f"  {r['model_id']:<18} {r['scenario']:<22} "
              f"{det_str:>6} {fpr_str:>6} "
              f"{r['syntax_validity_rate']:>7.1f}% "
              f"{r['ip_accuracy_rate']:>5.1f}% "
              f"{r['avg_inference_ms']:>7.0f}")

    print(f"\n[DONE] Hasil tersimpan:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()