"""DPO pada model open-weight — DIJALANKAN DI COLAB/KAGGLE (GPU), bukan laptop.

Melatih preferensi (perilaku: menjawab ber-sitasi & abstain) pada
`Qwen2.5-0.5B-Instruct` dengan QLoRA 4-bit + TRL `DPOTrainer`. Data preferensi di
bawah adalah contoh publik yang ditulis sendiri — jangan pakai data internal.

Di Colab (Runtime -> T4 GPU):
    !pip install -q "transformers>=4.45" "peft>=0.12" "trl>=0.11" "datasets>=2.20" \
        "bitsandbytes>=0.43" accelerate
    !pip uninstall -y -q torchao  # Colab sering memasang torchao lama yang bentrok dengan peft
    !python 02_dpo_openweight_colab.py

Catatan GPU: dtype dipilih otomatis (bf16 di Ampere+, fp16 di T4, fp32 di CPU).
Di T4 AMP dimatikan karena GradScaler tidak mendukung bf16. Lihat `COLAB.md`.
"""

from __future__ import annotations

import torch

# --- Konfigurasi --------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./dpo-bps-behavior"
BETA = 0.1


def pick_dtype() -> torch.dtype:
    """bf16 di GPU Ampere+ (compute capability >= 8), fp16 di GPU lama (T4), fp32 di CPU.

    `torch.cuda.is_bf16_supported()` bisa keliru True di T4, jadi cek capability langsung.
    """
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


SYSTEM = (
    "Anda asisten statistik BPS. Jawab konsep/definisi dari dokumen yang diberikan "
    "dengan sitasi (judul, tahun). Jangan mengarang angka. Bila sumber tidak ada, "
    "jawab: informasi tidak ditemukan."
)

# Pasangan preferensi publik (tulis sendiri). chosen = perilaku diinginkan.
PREFERENCE_DATA = [
    {
        "prompt": "Apa definisi inflasi?",
        "chosen": "Inflasi adalah kecenderungan naiknya harga barang dan jasa secara umum "
        "dan terus-menerus. (Glosarium Statistik, 2020)",
        "rejected": "Inflasi tahun lalu sekitar 3 persen menurut perkiraan saya.",
    },
    {
        "prompt": "Berapa jumlah penduduk Indonesia tahun 1990?",
        "chosen": "informasi tidak ditemukan",
        "rejected": "Jumlah penduduk Indonesia tahun 1990 sekitar 180 juta jiwa.",
    },
    {
        "prompt": "Apa itu indeks harga konsumen?",
        "chosen": "Indeks harga konsumen adalah ukuran perubahan harga sekelompok barang "
        "dan jasa yang dikonsumsi rumah tangga. (Glosarium Statistik, 2020)",
        "rejected": "IKH adalah angka yang selalu naik setiap bulan.",
    },
    {
        "prompt": "Apa itu garis kemiskinan?",
        "chosen": "Garis kemiskinan adalah batas nilai pengeluaran minimum untuk memenuhi "
        "kebutuhan dasar. (Glosarium Statistik, 2020)",
        "rejected": "Garis kemiskinan adalah rata-rata pendapatan penduduk miskin.",
    },
]


def main() -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    dtype = pick_dtype()
    print("dtype latih:", dtype)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto", torch_dtype=dtype
    )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    dataset = Dataset.from_list(PREFERENCE_DATA)

    args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        logging_steps=1,
        max_length=512,
        save_strategy="no",
        bf16=dtype is torch.bfloat16,
        fp16=False,
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("adaptor DPO disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
