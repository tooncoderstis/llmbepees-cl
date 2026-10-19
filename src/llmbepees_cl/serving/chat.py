"""Chat lokal dengan LLM (Fase 5): FastAPI + model Hugging Face + adaptor LoRA.

Endpoint:
- `GET /`      — halaman chat (HTML inline, tanpa framework/template).
- `POST /chat` — `{messages:[{role, content}]}` -> `{reply}` (server stateless; riwayat
  dijaga di browser).

`transformers`/`peft` diimpor **di dalam** `LocalLLMEngine` agar modul ini tetap ringan
(CI menguji endpoint dengan engine palsu, tanpa paket berat). Untuk asisten BPS ber-sitasi
+ tool (mesin ekstraktif), lihat `llmbepees_cl.serving.app`.
"""

from __future__ import annotations

import os
from typing import Protocol

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

DEFAULT_MODEL_ID = os.environ.get("CHAT_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_ADAPTER_DIR = os.environ.get("CHAT_ADAPTER_DIR", "03-finetune-lora/lora-bps-format")
DEFAULT_SYSTEM = (
    "Anda asisten statistik BPS. Jawab konsep/definisi dengan sitasi (judul, tahun) bila "
    "tersedia. Jangan mengarang angka. Bila sumber tidak ada, jawab: informasi tidak ditemukan."
)
MAX_NEW_TOKENS = int(os.environ.get("CHAT_MAX_NEW_TOKENS", "256"))


def pick_dtype():
    """bf16 di GPU Ampere+ (capability >= 8), fp16 di GPU lama (T4), fp32 di CPU."""
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


class Engine(Protocol):
    """Kontrak mesin jawab: cukup punya `reply(messages) -> str`."""

    def reply(self, messages: list[dict[str, str]]) -> str: ...


class LocalLLMEngine:
    """Memuat model (+ adaptor LoRA opsional) sekali, lalu menjawab via chat template."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_dir: str | None = DEFAULT_ADAPTER_DIR,
        system: str = DEFAULT_SYSTEM,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        self.model_id = model_id
        self.adapter_dir = None if (adapter_dir or "").lower() in {"", "none"} else adapter_dir
        self.system = system
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=pick_dtype())
        if self.adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter_dir)
        self.model = model

    def reply(self, messages: list[dict[str, str]]) -> str:
        import torch

        self._load()
        convo = [{"role": "system", "content": self.system}, *messages]
        text = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str


def create_chat_app(engine: Engine | None = None) -> FastAPI:
    """Bangun aplikasi chat; `engine` bisa diganti (untuk tes tanpa model)."""
    engine = engine or LocalLLMEngine()
    app = FastAPI(title="llmbepees-cl — chat lokal", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(content=CHAT_HTML)

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        messages = [m.model_dump() for m in request.messages]
        return ChatResponse(reply=engine.reply(messages))

    return app


CHAT_HTML = """<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chat lokal — llmbepees-cl</title>
<style>
  :root { color-scheme: light dark; }
  body { margin: 0; font-family: system-ui, sans-serif; background: #0f172a; color: #e2e8f0; }
  header { padding: 12px 16px; background: #1e293b; font-weight: 600; }
  #log { padding: 16px; max-width: 820px; margin: 0 auto;
         display: flex; flex-direction: column; gap: 10px; }
  .msg { padding: 10px 12px; border-radius: 10px; white-space: pre-wrap; max-width: 85%; }
  .user { align-self: flex-end; background: #2563eb; color: #fff; }
  .bot { align-self: flex-start; background: #334155; }
  .typing { align-self: flex-start; color: #94a3b8; font-style: italic; }
  form { position: sticky; bottom: 0; display: flex; gap: 8px;
         padding: 12px 16px; background: #1e293b; }
  input { flex: 1; padding: 10px 12px; border-radius: 8px;
          border: 1px solid #475569; background: #0f172a; color: inherit; }
  button { padding: 10px 18px; border: 0; border-radius: 8px;
           background: #2563eb; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; cursor: progress; }
  small { color: #94a3b8; }
</style>
</head>
<body>
<header>Chat lokal · Qwen2.5-0.5B-Instruct + LoRA <small id="hint">(CPU, sabar ya)</small></header>
<div id="log"></div>
<form id="f">
  <input id="q" autocomplete="off" placeholder="Tulis pertanyaan…">
  <button id="s" type="submit">Kirim</button>
</form>
<script>
const msgs = [];
const log = document.getElementById('log');
function bubble(cls, text) {
  const el = document.createElement('div');
  el.className = 'msg ' + cls;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}
document.getElementById('f').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const input = document.getElementById('q');
  const btn = document.getElementById('s');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  bubble('user', text);
  msgs.push({ role: 'user', content: text });
  const typing = bubble('typing', 'mengetik…');
  btn.disabled = true;
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: msgs }),
    });
    const data = await res.json();
    typing.remove();
    if (!res.ok) { bubble('bot', 'error: ' + JSON.stringify(data.detail ?? res.status)); return; }
    bubble('bot', data.reply);
    msgs.push({ role: 'assistant', content: data.reply });
  } catch (err) {
    typing.remove();
    bubble('bot', 'error: ' + err);
  } finally {
    btn.disabled = false;
    input.focus();
  }
});
document.getElementById('q').focus();
</script>
</body>
</html>
"""
