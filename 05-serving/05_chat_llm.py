"""Skrip 05 — Chat lokal dengan LLM (FastAPI + adaptor LoRA) di CPU/GPU.

Jalankan (dari root repo):
    $env:HF_HOME='E:\\hf'
    .venv\\Scripts\\python 05-serving\\05_chat_llm.py
Lalu buka http://127.0.0.1:8000 di browser.

Konfigurasi lewat env: `CHAT_MODEL_ID`, `CHAT_ADAPTER_DIR` (isi `none` untuk base saja),
`CHAT_MAX_NEW_TOKENS`, `HF_HOME`.
"""

from __future__ import annotations

import uvicorn

from llmbepees_cl.serving.chat import create_chat_app

app = create_chat_app()

if __name__ == "__main__":
    print("Chat lokal di http://127.0.0.1:8000 (Ctrl+C untuk berhenti)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
