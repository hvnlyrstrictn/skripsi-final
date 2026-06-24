import time
import torch
import os
import subprocess
import re
import sys
from unsloth import FastLanguageModel

# --- KONFIGURASI ---
MODEL_PATH = "lora_model_real"
OUTPUT_RULE = "generated_rules.zeek"
FRAMEWORK_FILE = "framework.zeek"
TARGET_LOGS = ["conn.log", "http.log", "dns.log", "files.log", "weird.log"]

# --- LOAD MODEL ---
print("[LOADING AI MODEL...]")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)
print("[MODEL IS READY!]")

# --- PROMPT ---
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

IMPORTANT:
1. Identify the attack type based on the log provided.
2. Generate a Zeek script to block the source IP using 'drop_connection(c)'.
3. Do not create custom Notice types.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def get_heuristic_instruction(log_line, log_type):
    if log_type == "http":
        if "union" in log_line.lower() or "select" in log_line.lower() or "script" in log_line.lower():
            return "Analisis log HTTP ini. Terdeteksi pola injeksi (SQLi/XSS). Buat rule blokir IP."
        return "Analisis log HTTP ini untuk payload berbahaya."

    if log_type == "conn":
        if "REJ" in log_line or "RST" in log_line or "SH" in log_line:
            return "Analisis koneksi gagal (REJ/RST/SH). Indikasi Port Scanning atau Probing. Buat rule blokir."
        if "S0" in log_line or "OTH" in log_line:
            return "Analisis koneksi TCP tidak lengkap. Indikasi DoS Flood. Buat rule blokir."
    
    if log_type == "files":
        return "Analisis transfer file ini. Terdeteksi transfer file Executable/Binary berbahaya. Buat rule blokir."

    if log_type == "weird":
        return "Analisis anomali protokol (Weird/Malformed). Indikasi Evasion. Buat rule blokir."

    return "Analisis log jaringan ini dan buat rule mitigasi."

def generate_rule(log_line, log_type):
    instruction = get_heuristic_instruction(log_line, log_type)
    inputs = tokenizer([alpaca_prompt.format(instruction, log_line, "")], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
    decoded = tokenizer.batch_decode(outputs)[0]
    code = decoded.split("### Response:")[-1].replace("<|im_end|>", "").strip()
    
    code_clean = re.sub(r'#[^}]*', '', code).strip()
    if "event http_request" in code_clean and "version: string" not in code_clean:
        code_clean = re.sub(r'event http_request\((.*?)\)', 
                            r'event http_request(c: connection, method: string, original_URI: string, unescaped_URI: string, version: string)', 
                            code_clean)
    return code_clean, instruction

def run_zeek_validation(source_ip):
    cmd = ["zeek", FRAMEWORK_FILE, OUTPUT_RULE]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return (result.returncode == 0), result.stderr.strip().split('\n')[0] 
    except Exception as e:
        return False, str(e)

# --- MAIN LOOP ---
def main():
    print(f"Monitoring Logs: {', '.join(TARGET_LOGS)}")
    blocked_ips = set()
    files_map = {}

    while True:
        for log_name in TARGET_LOGS:
            if log_name not in files_map and os.path.exists(log_name):
                print(f"   [+] Active logs: {log_name}")
                f = open(log_name, "r")
                f.seek(0, 2) 
                files_map[log_name] = f

        data_found = False
        for filename, f_handle in list(files_map.items()):
            line = f_handle.readline()
            if not line: continue
            
            data_found = True
            if line.startswith("#"): continue
            line = line.strip()
            if not line: continue

            # Normalisasi
            if "::1" in line: line = line.replace("::1", "127.0.0.1")
            log_type = filename.split(".")[0]

            # Extract IP
            source_ip = None
            parts = line.split('\t')
            for p in parts:
                if "127.0.0.1" in p or "192.168." in p:
                    source_ip = p
                    break
            
            if not source_ip or source_ip in blocked_ips: continue

            is_threat = False
            
            if log_type == "conn" and ("REJ" in line or "RST" in line or "SH" in line or "S0" in line):
                is_threat = True
            elif log_type == "http" and ("union" in line.lower() or "select" in line.lower()):
                is_threat = True
            elif log_type == "files" and ("dosexec" in line or ".exe" in line):
                is_threat = True
            elif log_type == "weird":
                is_threat = True

            if is_threat:
                print(f"\n{'='*60}")
                print(f"THREAT ALERT! {filename.upper()}")
                
                t0 = time.time()
                rule, context = generate_rule(line, log_type)
                t1 = time.time()
                
                print(f"\nAI Analysis ({t1-t0:.2f}s):")
                print(f"   Context : {context}")
                print(f"   Solution  : {rule}")
                
                with open(OUTPUT_RULE, "a") as zf:
                    zf.write(f"\n# Alert from {filename}\n")
                    zf.write(rule + "\n")
                
                valid, msg = run_zeek_validation(source_ip)
                if valid:
                    print(f"🟢 VERIFIED: BLOCKED {source_ip}.")
                    blocked_ips.add(source_ip)
                else:
                    print(f"⛔ ERROR: {msg}")
                print(f"{'='*60}\n")

        if not data_found:
            time.sleep(0.1)

if __name__ == "__main__":
    main()