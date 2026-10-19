"""Skrip 03 — Menjalankan API FastAPI asisten BPS.

Membungkus `BPSAssistant` menjadi layanan HTTP. Endpoint:
- `GET /health`   — status + jumlah dokumen/passage
- `POST /ask`     — {question} -> {answer, citations, tool_calls, abstained}
- `GET /indicators` — daftar indikator pada tool

Jalankan (server lokal):
    .venv\\Scripts\\python 05-serving\\03_serve.py
Lalu uji dari PowerShell (terminal lain):
    Invoke-RestMethod http://127.0.0.1:8000/health
    Invoke-RestMethod http://127.0.0.1:8000/ask -Method Post `
      -ContentType "application/json" -Body '{"question":"Apa itu inflasi?"}'

Dokumentasi interaktif otomatis: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import uvicorn

from llmbepees_cl.serving.app import create_app

app = create_app()

if __name__ == "__main__":
    print("Menjalankan API di http://127.0.0.1:8000 (Ctrl+C untuk berhenti)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
