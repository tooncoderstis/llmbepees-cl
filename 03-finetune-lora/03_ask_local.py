"""Tanya model + adaptor LoRA di laptop (CPU) — versi lokal dari `eval_format_colab.py`.

Memuat `Qwen/Qwen2.5-0.5B-Instruct` (CPU, fp32) lalu adaptor `./lora-bps-format`,
dan membandingkan jawaban model dasar vs model + LoRA. Berguna untuk memastikan
pipeline LoRA (latih di GPU → jalan di lokal) bekerja tanpa GPU.

Contoh (dari folder `03-finetune-lora`):
    $env:HF_HOME='E:\\hf'
    ..\\..\\.venv\\Scripts\\python 03_ask_local.py "Apa definisi inflasi?"
    ..\\..\\.venv\\Scripts\\python 03_ask_local.py          # memakai contoh bawaan
"""

from __future__ import annotations

import sys

from finetune_openweight_colab import MODEL_ID, SYSTEM, pick_dtype

ADAPTER_DIR = "./lora-bps-format"

PROMPTS = [
    "Apa definisi inflasi?",
    "Berapa jumlah penduduk Indonesia tahun 1990?",
]


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


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = sys.argv[1:] or PROMPTS
    dtype = pick_dtype()
    print(f"model: {MODEL_ID} | dtype: {dtype} | adaptor: {ADAPTER_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
    tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)

    params = dict(tuned.named_parameters())
    b_norms = [p.detach().abs().mean().item() for n, p in params.items() if "lora_B" in n]
    print(f"adaptor aktif: {bool(b_norms)} | lora_B absmean(max): {max(b_norms, default=0.0):.3e}")

    for p in prompts:
        a_base = tanya(base, tokenizer, p)
        a_lora = tanya(tuned, tokenizer, p)
        print(f"\nQ: {p}\n  dasar: {a_base!r}\n  LoRA : {a_lora!r}\n  sama : {a_base == a_lora}")


if __name__ == "__main__":
    main()
