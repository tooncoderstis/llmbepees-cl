"""Skrip 02 — Fine-tuning LoRA pada MiniGPT (studi kasus satu tugas sempit).

Alur: (1) latih model dasar pada korpus statistik, (2) ukur loss pada contoh format
"T: <konsep>\\nJ: <definisi>" yang **ditahan** (belum pernah dilihat), (3) fine-tune
hanya bobot LoRA pada contoh latih, (4) ukur ulang loss contoh tahanan + lihat
contoh hasil. Hanya sebagian kecil parameter yang dilatih.

Catatan: ini model karakter 0,1 juta parameter pada belasan contoh — tujuannya
memahami mekanika LoRA, bukan menghasilkan bahasa yang mulus.

Jalankan:
    .venv\\Scripts\\python 03-finetune-lora\\02_finetune_lora_minigpt.py
"""

from __future__ import annotations

import torch

from llmbepees_cl.lora import apply_lora, total_parameter_count, trainable_parameter_count
from llmbepees_cl.minigpt import (
    TINY_CORPUS,
    CharTokenizer,
    MiniGPT,
    encode_corpus,
    generate,
    train_language_model,
)

PAIRS = [
    ("rata-rata", "jumlah nilai dibagi banyaknya nilai"),
    ("modus", "nilai yang paling sering muncul dalam data"),
    ("inflasi", "kenaikan harga barang dan jasa secara umum"),
    ("populasi", "keseluruhan objek yang menjadi perhatian penelitian"),
    ("sampel", "bagian dari populasi yang dipilih mewakili keseluruhan"),
    ("rasio", "perbandingan antara dua nilai yang berbeda satuan"),
    ("median", "nilai tengah setelah data diurutkan"),  # tahanan
    ("proporsi", "perbandingan yang menyatakan bagian dari keseluruhan"),  # tahanan
]
train_pairs, holdout_pairs = PAIRS[:-2], PAIRS[-2:]


def as_text(pairs: list[tuple[str, str]]) -> str:
    return "".join(f"T: {k}\nJ: {v}\n\n" for k, v in pairs)


qa_train_text = as_text(train_pairs) * 8
holdout_text = as_text(holdout_pairs)
tok = CharTokenizer.from_text(TINY_CORPUS + as_text(PAIRS))

base_data = encode_corpus(TINY_CORPUS, tok)
qa_data = encode_corpus(qa_train_text, tok)
holdout = encode_corpus(holdout_text, tok)


def seq_loss(model: MiniGPT, ids: torch.Tensor, block_size: int = 64) -> float:
    """Rata-rata loss prediksi token berikutnya atas potongan sepanjang `block_size`."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, ids.shape[0] - 1, block_size):
            chunk = ids[start : start + block_size + 1]
            if chunk.shape[0] < 2:
                break
            _, loss = model(chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0))
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def jawab(model: MiniGPT) -> str:
    idx = torch.tensor([tok.encode("T: rata-rata\nJ:")], dtype=torch.long)
    out = generate(
        model,
        idx,
        max_new_tokens=45,
        temperature=0.5,
        top_k=3,
        generator=torch.Generator().manual_seed(0),
    )
    return tok.decode(out[0].tolist())


# --- (1) Model dasar ----------------------------------------------------------
model = MiniGPT(vocab_size=tok.vocab_size, n_embd=64, n_head=4, n_layer=2, block_size=64)
train_language_model(model, base_data, steps=250, batch_size=16, block_size=64, lr=3e-3)
print("model dasar selesai dilatih")

# --- (2) Sebelum fine-tuning --------------------------------------------------
print(f"loss contoh tahanan (sebelum LoRA): {seq_loss(model, holdout):.4f}")
print(f"jawaban (sebelum): {jawab(model)!r}")

# --- (3) Terapkan LoRA & fine-tune hanya adaptor ------------------------------
replaced = apply_lora(model, target_modules=("query", "key", "value", "proj"), r=4, alpha=8)
total = total_parameter_count(model)
trainable = trainable_parameter_count(model)
print(
    f"\nlapisan LoRA: {replaced} | dilatih {trainable:,} / {total:,} parameter "
    f"({100 * trainable / total:.1f}%)"
)

history = train_language_model(model, qa_data, steps=400, batch_size=8, block_size=64, lr=1.5e-2)
print(f"loss format (latih): {history[0]:.3f} -> {history[-1]:.3f}")

# --- (4) Sesudah fine-tuning --------------------------------------------------
print(f"loss contoh tahanan (sesudah LoRA): {seq_loss(model, holdout):.4f}")
print(f"jawaban (sesudah): {jawab(model)!r}")
print("\nPelajaran: LoRA menurunkan loss pada contoh yang tak dilatih dan mengubah")
print("perilaku (format) model tanpa menyentuh bobot dasar; pada model bahasa nyata")
print("ini menghemat memori training secara drastis.")
