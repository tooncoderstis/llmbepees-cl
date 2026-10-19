"""Fine-tuning LoRA model open-weight — DIJALANKAN DI GOOGLE COLAB (GPU), bukan laptop.

Contoh: Qwen2.5-0.5B-Instruct dilatih untuk satu tugas sempit — menjawab dengan
**format ber-sitasi** dan perilaku "tidak ditemukan" sesuai prinsip asisten BPS
(lihat `AGENTS.md`). Data latih sepenuhnya contoh publik yang ditulis sendiri;
jangan pernah mengirim data internal ke API/layanan mana pun.

Di Colab:
    !pip install -q "transformers>=4.45" "peft>=0.12" "trl>=0.11" "datasets>=2.20" accelerate
    !python finetune_openweight_colab.py

Detail langkah ada di `COLAB.md` di folder yang sama.
"""

from __future__ import annotations

# --- Konfigurasi --------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "./lora-bps-format"
MAX_SEQ_LEN = 512

SYSTEM = (
    "Anda asisten statistik BPS. Jawab konsep dan definisi hanya berdasarkan dokumen "
    "yang diberikan. Selalu sertakan sitasi dalam bentuk (judul, tahun). Jangan "
    "mengarang angka; angka resmi hanya dari tool. Bila sumber tidak ada, jawab: "
    "informasi tidak ditemukan."
)

# Contoh publik (tulis sendiri). Ganti dengan contoh Anda; jangan pakai data internal.
EXAMPLES = [
    {
        "user": "Apa definisi inflasi?",
        "assistant": "Inflasi adalah kecenderungan naiknya harga barang dan jasa secara "
        "umum dan terus-menerus. (Glosarium Statistik, 2020)",
    },
    {
        "user": "Apa itu indeks harga konsumen?",
        "assistant": "Indeks harga konsumen adalah ukuran perubahan harga sekelompok "
        "barang dan jasa yang dikonsumsi rumah tangga. (Glosarium Statistik, 2020)",
    },
    {
        "user": "Berapa jumlah penduduk Indonesia tahun 1990?",
        "assistant": "informasi tidak ditemukan",
    },
    {
        "user": "Jelaskan konsep garis kemiskinan.",
        "assistant": "Garis kemiskinan adalah batas nilai pengeluaran minimum yang "
        "diperlukan untuk memenuhi kebutuhan dasar. (Glosarium Statistik, 2020)",
    },
]


def build_texts(tokenizer) -> list[dict[str, str]]:
    """Ubah contoh menjadi teks sesuai chat template model (prompt + jawaban)."""
    texts = []
    for ex in EXAMPLES:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ]
        texts.append({"text": tokenizer.apply_chat_template(messages, tokenize=False)})
    return texts


def main() -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = Dataset.from_list(build_texts(tokenizer))

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        max_seq_length=MAX_SEQ_LEN,
        save_strategy="no",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("adaptor LoRA disimpan di:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
