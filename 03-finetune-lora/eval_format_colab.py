"""Evaluasi sederhana sebelum vs sesudah LoRA — DIJALANKAN DI COLAB (GPU).

Membandingkan jawaban **model dasar** dan **model + adaptor LoRA** pada beberapa
prompt uji, lalu memberi skor kasar: apakah jawaban menyertakan sitasi, apakah
menolak saat sumber tidak ada, dan apakah tidak mengarang angka. Ini baseline
kecil, bukan penilaian menyeluruh (test set BPS sesungguhnya menyusul di Fase 6).

Di Colab:
    !python eval_format_colab.py            # butuh ./lora-bps-format hasil training
"""

from __future__ import annotations

import re

from finetune_openweight_colab import MODEL_ID, SYSTEM, pick_dtype

ADAPTER_DIR = "./lora-bps-format"

TESTS = [
    {"q": "Apa definisi inflasi?", "perlu_sitasi": True, "harus_tolak": False},
    {"q": "Apa itu indeks harga konsumen?", "perlu_sitasi": True, "harus_tolak": False},
    {
        "q": "Berapa jumlah penduduk Indonesia tahun 1990?",
        "perlu_sitasi": False,
        "harus_tolak": True,
    },
]

SITASI = re.compile(r"\([^)]*\d{4}\)")


def tanya(model, tokenizer, prompt: str) -> str:
    import torch

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def skor(jawaban: str, tes: dict) -> bool:
    if tes["harus_tolak"]:
        return "tidak ditemukan" in jawaban.lower()
    return bool(SITASI.search(jawaban))


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    kwargs: dict = {"torch_dtype": pick_dtype()}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)

    benar_base = benar_lora = 0
    for tes in TESTS:
        a_base = tanya(base, tokenizer, tes["q"])
        a_lora = tanya(tuned, tokenizer, tes["q"])
        benar_base += skor(a_base, tes)
        benar_lora += skor(a_lora, tes)
        print(f"\nQ: {tes['q']}\n  dasar: {a_base!r}\n  LoRA : {a_lora!r}")

    n = len(TESTS)
    print(f"\nSkor format — dasar: {benar_base}/{n} | LoRA: {benar_lora}/{n}")


if __name__ == "__main__":
    main()
