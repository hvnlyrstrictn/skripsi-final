#!/usr/bin/env python3
"""
train_ablation.py - Automated Ablation Study Training
Skripsi: Implementasi Constrained Generation pada LLM Menggunakan QLoRA
         untuk Otomasi Aturan Mitigasi Zeek IDS

Menjalankan 3 skenario fine-tuning secara berurutan dengan
hyperparameter berbeda dan mencatat semua metrik ke CSV + JSON.

Skenario:
  - Lightweight : r=8,  alpha=16, epochs=1  -> Efisiensi & Kecepatan
  - Balanced    : r=16, alpha=32, epochs=2  -> Keseimbangan Optimal
  - Deep        : r=32, alpha=64, epochs=3  -> Kapabilitas Maksimal
"""

import torch
import json
import csv
import time
import os
import re
import gc
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================================
# KONFIGURASI UTAMA
# ============================================================

DATASET_FILE    = "dataset_finetune_v3.jsonl"   # Hasil build_dataset_final.py
BASE_MODEL      = "Qwen/Qwen2.5-3B-Instruct"    # Model dasar
MAX_SEQ_LENGTH  = 2048
OUTPUT_DIR      = "ablation_results"             # Folder hasil training

METRICS_CSV     = os.path.join(OUTPUT_DIR, "ablation_metrics.csv")
METRICS_JSON    = os.path.join(OUTPUT_DIR, "ablation_metrics.json")

EVAL_SAMPLES    = 50

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

def clean_zeek_code(code: str) -> str:
    """Post-processing output model: bersihkan komentar, patch HTTP."""
    code_clean = re.sub(r'#[^}]*', '', code).strip()
    if "event http_request" in code_clean and "version: string" not in code_clean:
        code_clean = re.sub(
            r'event http_request\((.*?)\)',
            'event http_request(c: connection, method: string, '
            'original_URI: string, unescaped_URI: string, version: string)',
            code_clean
        )
    return code_clean

# ============================================================
# DEFINISI 3 SKENARIO ABLATION STUDY
# ============================================================

SCENARIOS = [
    {
        "id":          "A_lightweight",
        "label":       "Lightweight (r=8)",
        "description": "Efisiensi tinggi & training cepat. Cocok untuk resource terbatas.",
        "lora_r":       8,
        "lora_alpha":   16,
        "num_epochs":   1,
        "batch_size":   2,
        "grad_accum":   4,
        "learning_rate": 2e-4,
        "output_dir":   os.path.join(OUTPUT_DIR, "model_A_lightweight"),
    },
    {
        "id":          "B_balanced",
        "label":       "Balanced (r=16)",
        "description": "Keseimbangan antara performa dan resource. Rekomendasi umum.",
        "lora_r":       16,
        "lora_alpha":   32,
        "num_epochs":   2,
        "batch_size":   2,
        "grad_accum":   4,
        "learning_rate": 2e-4,
        "output_dir":   os.path.join(OUTPUT_DIR, "model_B_balanced"),
    },
    {
        "id":          "C_deep",
        "label":       "Deep (r=32)",
        "description": "Kapabilitas maksimal. Menangkap nuansa Zeek Script lebih dalam.",
        "lora_r":       32,
        "lora_alpha":   64,
        "num_epochs":   3,
        "batch_size":   2,
        "grad_accum":   4,
        "learning_rate": 2e-4,
        "output_dir":   os.path.join(OUTPUT_DIR, "model_C_deep"),
    },
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_prompt(instruction: str, input_text: str, output: str = "") -> str:
    return ALPACA_PROMPT.format(instruction, input_text, output)


def formatting_prompts_func(examples):
    """Formatter untuk SFTTrainer."""
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for inst, inp, out in zip(instructions, inputs, outputs):
        text = format_prompt(inst, inp, out) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}





def validate_zeek_syntax(script: str, framework_path: str = "framework.zeek") -> bool:
    """
    Validasi sintaks script Zeek menggunakan Zeek engine.
    Mengembalikan True jika valid, False jika error.
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".zeek",
                                     delete=False, dir="/tmp") as tf:
        tf.write(script + "\n")
        tmp_path = tf.name

    try:
        cmd = ["zeek", tmp_path]
        if os.path.exists(framework_path):
            cmd = ["zeek", framework_path, tmp_path]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_vram_usage_mb() -> float:
    """Dapatkan VRAM yang sedang digunakan dalam MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def get_vram_peak_mb() -> float:
    """Dapatkan peak VRAM yang pernah digunakan dalam MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def evaluate_syntax_validity(model_eval, tokenizer_eval, test_samples, scenario_id: str) -> dict:
    """
    Evaluasi post-training:
    - Jalankan inference pada test_samples
    - Hitung berapa persen yang menghasilkan valid Zeek syntax
    - Hitung average inference time
    """
    print(f"\n  [EVAL] Mengevaluasi sintaks untuk {len(test_samples)} sampel...")

    FastLanguageModel.for_inference(model_eval)

    valid_count     = 0
    has_drop_conn   = 0
    inference_times = []

    for i, sample in enumerate(test_samples):
        prompt = format_prompt(sample["instruction"], sample["input"], "")

        inputs = tokenizer_eval(
            [prompt], return_tensors="pt"
        ).to("cuda")

        t0 = time.time()
        with torch.no_grad():
            output_ids = model_eval.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                do_sample=False,    # Greedy decoding untuk konsistensi evaluasi
            )
        t1 = time.time()
        inference_times.append(t1 - t0)

        generated = tokenizer_eval.batch_decode(output_ids)[0]
        code = generated.split("### Response:")[-1].replace("<|im_end|>", "").strip()
        code_clean = clean_zeek_code(code)

        is_valid = validate_zeek_syntax(code_clean)
        if is_valid:
            valid_count += 1
        if "drop_connection" in code_clean:
            has_drop_conn += 1

        if (i + 1) % 10 == 0:
            print(f"     -> {i+1}/{len(test_samples)} selesai "
                  f"(valid sejauh ini: {valid_count})")

    avg_inference_ms = (sum(inference_times) / len(inference_times)) * 1000

    result = {
        "eval_samples":         len(test_samples),
        "syntax_valid_count":   valid_count,
        "syntax_validity_rate": round(valid_count / len(test_samples) * 100, 2),
        "has_drop_conn_count":  has_drop_conn,
        "has_drop_conn_rate":   round(has_drop_conn / len(test_samples) * 100, 2),
        "avg_inference_ms":     round(avg_inference_ms, 2),
    }

    print(f"  [EVAL] Hasil:")
    print(f"     - Syntax Validity Rate : {result['syntax_validity_rate']}%")
    print(f"     - Drop Connection Rate : {result['has_drop_conn_rate']}%")
    print(f"     - Avg Inference Time   : {result['avg_inference_ms']} ms")

    return result


# ============================================================
# LOAD DATASET (SEKALI, DIPAKAI SEMUA SKENARIO)
# ============================================================

print("=" * 60)
print(" ABLATION STUDY - QLoRA Fine-Tuning")
print(f" Waktu Mulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Buat direktori output
os.makedirs(OUTPUT_DIR, exist_ok=True)
for s in SCENARIOS:
    os.makedirs(s["output_dir"], exist_ok=True)

# Load dataset
print(f"\n[DATASET] Memuat: {DATASET_FILE}")
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {DATASET_FILE}\n"
                            f"Jalankan build_dataset_final.py terlebih dahulu!")

full_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
full_dataset = full_dataset.shuffle(seed=42)

total_samples = len(full_dataset)
print(f"  Total sampel   : {total_samples:,}")

# Pisahkan eval set (gunakan 50 sampel acak, SAMA untuk semua skenario)
eval_indices    = list(range(min(EVAL_SAMPLES, total_samples)))
test_samples    = [full_dataset[i] for i in eval_indices]
train_dataset   = full_dataset.select(range(EVAL_SAMPLES, total_samples))
print(f"  Training set   : {len(train_dataset):,}")
print(f"  Evaluation set : {len(test_samples)}")

# ============================================================
# INISIALISASI CSV UNTUK PENCATATAN METRIK
# ============================================================

CSV_HEADERS = [
    "scenario_id", "label",
    "lora_r", "lora_alpha", "num_epochs",
    "total_steps", "trainable_params", "total_params", "trainable_percent",
    "training_time_sec", "training_time_min",
    "initial_loss", "final_loss", "best_loss",
    "vram_peak_mb", "vram_peak_gb",
    "syntax_validity_rate", "has_drop_conn_rate",
    "avg_inference_ms",
    "eval_samples",
    "timestamp_start", "timestamp_end",
]

# Tulis header CSV
with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
    writer.writeheader()

# Inisialisasi JSON output
all_metrics = []

# ============================================================
# LOOP UTAMA: 3 SKENARIO
# ============================================================

tokenizer = None  # Akan diinisialisasi di dalam loop pertama

for scenario_idx, scenario in enumerate(SCENARIOS):
    print(f"\n{'=' * 60}")
    print(f" SKENARIO {scenario_idx + 1}/3: {scenario['label']}")
    print(f" Deskripsi: {scenario['description']}")
    print(f" Config   : r={scenario['lora_r']}, alpha={scenario['lora_alpha']}, "
          f"epochs={scenario['num_epochs']}")
    print(f"{'=' * 60}")

    ts_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    torch.cuda.reset_peak_memory_stats()

    # --- LOAD MODEL BARU (setiap skenario dari base model yang sama) ---
    print(f"\n[MODEL] Memuat model dasar: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name    = BASE_MODEL,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,       # Auto-detect (bfloat16 untuk RTX 4060)
        load_in_4bit   = True,       # 4-bit NF4 quantization (QLoRA)
    )

    # --- PASANG LORA ADAPTER ---
    print(f"[LORA] Konfigurasi adapter: r={scenario['lora_r']}, alpha={scenario['lora_alpha']}")
    model = FastLanguageModel.get_peft_model(
        model,
        r               = scenario["lora_r"],
        target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        lora_alpha      = scenario["lora_alpha"],
        lora_dropout    = 0,          # 0 lebih cepat (rekomendasi Unsloth)
        bias            = "none",
        use_gradient_checkpointing = "unsloth",  # Hemat VRAM ekstrem
        random_state    = 3407,
        use_rslora      = False,
        loftq_config    = None,
    )

    # Hitung trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_pct    = trainable_params / total_params * 100

    print(f"  Trainable params: {trainable_params:,} / {total_params:,} "
          f"({trainable_pct:.2f}%)")

    # --- FORMAT DATASET ---
    print("[DATA] Memformat dataset...")
    formatted_dataset = train_dataset.map(formatting_prompts_func, batched=True)

    # --- KONFIGURASI TRAINING ---
    steps_per_epoch = len(train_dataset) // (scenario["batch_size"] * scenario["grad_accum"])
    total_steps     = steps_per_epoch * scenario["num_epochs"]
    warmup_steps    = max(10, total_steps // 20)   # 5% dari total steps

    print(f"[TRAIN] Steps per epoch: {steps_per_epoch}, Total: {total_steps}, "
          f"Warmup: {warmup_steps}")

    training_args = TrainingArguments(
        per_device_train_batch_size  = scenario["batch_size"],
        gradient_accumulation_steps  = scenario["grad_accum"],
        num_train_epochs             = scenario["num_epochs"],
        learning_rate                = scenario["learning_rate"],
        fp16                         = not torch.cuda.is_bf16_supported(),
        bf16                         = torch.cuda.is_bf16_supported(),
        logging_steps                = max(1, total_steps // 20),   # Log 20x
        optim                        = "adamw_8bit",
        weight_decay                 = 0.01,
        lr_scheduler_type            = "cosine",   # Cosine decay lebih smooth
        warmup_steps                 = warmup_steps,
        seed                         = 3407,
        output_dir                   = scenario["output_dir"],
        save_strategy                = "no",       # Hemat disk
        report_to                    = "none",     # Disable wandb/tensorboard
    )

    trainer = SFTTrainer(
        model              = model,
        tokenizer          = tokenizer,
        train_dataset      = formatted_dataset,
        dataset_text_field = "text",
        max_seq_length     = MAX_SEQ_LENGTH,
        dataset_num_proc   = 2,
        packing            = False,
        args               = training_args,
    )

    # --- EKSEKUSI TRAINING ---
    print(f"\n[TRAIN] Mulai training... ({datetime.now().strftime('%H:%M:%S')})")
    t_train_start = time.time()

    trainer_stats  = trainer.train()

    t_train_end    = time.time()
    training_time  = t_train_end - t_train_start
    vram_peak_mb   = get_vram_peak_mb()

    print(f"[TRAIN] Selesai dalam {training_time:.1f} detik "
          f"({training_time/60:.1f} menit)")

    # Ambil loss dari log history
    log_history    = trainer.state.log_history
    loss_logs      = [e for e in log_history if "loss" in e]
    initial_loss   = loss_logs[0]["loss"]   if loss_logs else None
    final_loss     = loss_logs[-1]["loss"]  if loss_logs else None
    best_loss      = min(e["loss"] for e in loss_logs) if loss_logs else None

    print(f"  Loss: initial={initial_loss:.4f}, final={final_loss:.4f}, "
          f"best={best_loss:.4f}")
    print(f"  VRAM Peak: {vram_peak_mb:.1f} MB ({vram_peak_mb/1024:.2f} GB)")

    # Simpan loss history per skenario
    loss_history_path = os.path.join(scenario["output_dir"], "loss_history.json")
    with open(loss_history_path, "w") as f:
        json.dump(loss_logs, f, indent=2)
    print(f"  Loss history disimpan: {loss_history_path}")

    # --- SIMPAN MODEL ---
    print(f"[SAVE] Menyimpan adapter ke: {scenario['output_dir']}")
    model.save_pretrained(scenario["output_dir"])
    tokenizer.save_pretrained(scenario["output_dir"])

    # --- EVALUASI SINTAKS ---
    eval_results = evaluate_syntax_validity(model, tokenizer, test_samples, scenario["id"])
    ts_end       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- CATAT METRIK ---
    metrics_row = {
        "scenario_id":          scenario["id"],
        "label":                scenario["label"],
        "lora_r":               scenario["lora_r"],
        "lora_alpha":           scenario["lora_alpha"],
        "num_epochs":           scenario["num_epochs"],
        "total_steps":          trainer_stats.global_step,
        "trainable_params":     trainable_params,
        "total_params":         total_params,
        "trainable_percent":    round(trainable_pct, 4),
        "training_time_sec":    round(training_time, 2),
        "training_time_min":    round(training_time / 60, 2),
        "initial_loss":         round(initial_loss, 4) if initial_loss else None,
        "final_loss":           round(final_loss,   4) if final_loss   else None,
        "best_loss":            round(best_loss,    4) if best_loss    else None,
        "vram_peak_mb":         round(vram_peak_mb, 2),
        "vram_peak_gb":         round(vram_peak_mb / 1024, 3),
        "syntax_validity_rate": eval_results["syntax_validity_rate"],
        "has_drop_conn_rate":   eval_results["has_drop_conn_rate"],
        "avg_inference_ms":     eval_results["avg_inference_ms"],
        "eval_samples":         eval_results["eval_samples"],
        "timestamp_start":      ts_start,
        "timestamp_end":        ts_end,
    }
    metrics_row["loss_history"] = loss_logs  # Hanya untuk JSON

    # Tulis ke CSV (tanpa loss_history)
    csv_row = {k: v for k, v in metrics_row.items() if k != "loss_history"}
    with open(METRICS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow(csv_row)

    # Tambah ke list JSON
    all_metrics.append(metrics_row)

    # Tulis JSON (overwrite setiap skenario selesai, bukan di akhir)
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  [OK] Metrik skenario {scenario['id']} disimpan.")
    print(f"  CSV  : {METRICS_CSV}")
    print(f"  JSON : {METRICS_JSON}")

    # --- BERSIHKAN MEMORI sebelum skenario berikutnya ---
    print("\n[CLEANUP] Membersihkan GPU memory untuk skenario berikutnya...")
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  VRAM setelah cleanup: {get_vram_usage_mb():.1f} MB")

# ============================================================
# RINGKASAN AKHIR
# ============================================================

print(f"\n{'=' * 60}")
print(" RINGKASAN ABLATION STUDY")
print(f"{'=' * 60}")
print(f"{'Skenario':<20} {'Epochs':>6} {'r':>4} {'α':>4} "
      f"{'Loss Akhir':>10} {'Validity%':>10} {'Inf(ms)':>8} {'VRAM(GB)':>9}")
print("-" * 75)

for m in all_metrics:
    print(f"  {m['label']:<18} {m['num_epochs']:>6} {m['lora_r']:>4} {m['lora_alpha']:>4} "
          f"  {m['final_loss']:>8.4f}   {m['syntax_validity_rate']:>8.1f}%"
          f"  {m['avg_inference_ms']:>6.0f}  {m['vram_peak_gb']:>8.3f}")

print(f"\n  File metrik:")
print(f"  - CSV  : {METRICS_CSV}")
print(f"  - JSON : {METRICS_JSON}")
print(f"\n  Model tersimpan di:")
for s in SCENARIOS:
    print(f"  - {s['output_dir']}/")

print(f"\n  Selesai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)