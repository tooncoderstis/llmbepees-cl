# ROADMAP — 12 Minggu Belajar Pengembangan Model AI

Asumsi: 8–10 jam per minggu. Centang tiap item saat selesai. Setiap minggu tutup dengan ringkasan di `notes.md`.

## Setup awal (hari 1)
- [ ] Pasang PyTorch **versi CPU** (jangan CUDA di laptop ini)
- [ ] Jalankan `opencode` di folder proyek, pastikan `AGENTS.md` terbaca

## Minggu 1–2: Dasar PyTorch
- [ ] Tensor, autograd, `nn.Module`, training loop
- [ ] Latih MLP klasifikasi sederhana (misal MNIST kecil atau data tabular)
- [ ] Segarkan: aljabar linear (perkalian matriks), probabilitas, gradient descent
- **Hasil:** folder `01-pytorch-basics/` dengan notebook/skrip dan catatan

## Minggu 3–4: Transformer mini dari nol
- [ ] Tokenisasi karakter lalu BPE sederhana
- [ ] Implementasi self-attention, blok transformer, GPT mini
- [ ] Latih pada teks kecil (CPU cukup; ukuran model beberapa juta parameter)
- [ ] Ikuti materi nanoGPT/makeMore (Karpathy) sebagai acuan
- **Hasil:** `02-mini-gpt/` yang bisa menghasilkan teks

## Minggu 5–6: Fine-tuning LoRA
- [ ] Pelajari konsep LoRA/QLoRA dan kenapa hemat memori
- [ ] Jalankan di **Google Colab/Kaggle**, bukan di laptop
- [ ] Fine-tune model open-weight kecil untuk satu tugas sempit (misal format jawaban dengan sitasi)
- [ ] Bandingkan sebelum/sesudah dengan evaluasi sederhana
- **Hasil:** `03-finetune-lora/` + catatan hasil

## Minggu 7–8: RAG dan tool calling (prototipe asisten BPS)
- [ ] Kumpulkan dokumen publik (glosarium, metadata, publikasi) ke `04-rag-bps/data/`
- [ ] Chunking per bagian logis, embedding multibahasa, pencarian hybrid (vektor + BM25), reranker
- [ ] Prompt dengan sitasi wajib dan perilaku "tidak ditemukan"
- [ ] Tool `get_indicator(...)` untuk data resmi (mulai dengan data dummy/publik)
- [ ] Susun test set awal (minimal 50 pertanyaan) dan skrip evaluasi
- **Hasil:** prototipe berjalan + angka baseline akurasi/halusinasi

## Minggu 9–10: Mekanika training, scaling, dan serving dasar
- [ ] Mekanika training yang dipakai di lab: mixed precision (bf16), gradient accumulation, LR schedule, gradient clipping
- [ ] Konsep training terdistribusi: data parallel (DDP), FSDP, tensor/expert parallel (praktik kecil di Colab/GPU sewaan)
- [ ] Scaling laws: baca paper Kaplan dan Chinchilla, ulangi eksperimen kecil (variasi ukuran model vs data)
- [ ] Serving dasar: model kecil lokal (llama.cpp/Ollama), pahami kuantisasi dan KV cache, coba vLLM di GPU sewaan
- [ ] Bungkus asisten BPS menjadi API sederhana (FastAPI) dan ukur latensi
- **Hasil:** `05-serving/` (API + benchmark) dan catatan eksperimen scaling di `notes.md`

## Minggu 11–12: Jalur lab model (post-training dan evaluasi)
- [ ] Preference tuning: pahami RLHF vs DPO, jalankan DPO kecil di Colab/GPU sewaan
- [ ] Dasar RL untuk LLM (reward, policy gradient, ide di balik RL dengan reward yang bisa diverifikasi)
- [ ] Evaluasi: bangun satu benchmark kecil sendiri (bisa dari test set BPS) dan analisis kegagalan model
- [ ] Baca berurutan, satu per minggu: paper Transformer, Chinchilla, InstructGPT, DPO, satu paper teknis model open-weight terbaru
- [ ] Satu proyek terdokumentasi di GitHub dengan README yang jelas
- **Hasil:** `06-specialization/` (DPO kecil atau benchmark evaluasi) + ringkasan paper

## Setelah 12 minggu (menuju lab)
- [ ] Reproduksi hasil dari satu paper kecil dan tulis temuannya
- [ ] Kontribusi ke proyek open-source (library training/serving, evals)
- [ ] Tulisan teknis publik tentang eksperimen Anda
- [ ] Pilih satu bidang untuk didalami: infrastruktur training, evals/data, atau alignment
- [ ] Pantau halaman karier lab target untuk melihat syarat peran yang dituju

Catatan: jalur Nvidia/chip (CUDA, kernel Triton) sengaja tidak masuk roadmap ini. Bisa ditambahkan lagi kapan saja kalau arahnya berubah.

## Kebiasaan mingguan
- Baca satu paper, ringkas dengan kata sendiri
- Tulis hasil eksperimen di `notes.md` (apa dicoba, hasil, dipelajari)
- Commit rutin ke GitHub

---

## Prompt awal untuk opencode (tempel satu per satu)

**Prompt 1 — setup**
```
Baca AGENTS.md dan ROADMAP.md. Bantu aku mengerjakan bagian "Setup awal (hari 1)" langkah demi langkah di Windows PowerShell. Jalankan satu langkah, jelaskan singkat, tunggu konfirmasiku sebelum lanjut. Minta izin sebelum menginstal paket.
```

**Prompt 2 — minggu 1**
```
Mulai Minggu 1 dari ROADMAP.md. Buat folder 01-pytorch-basics dan ajari aku tensor, autograd, dan training loop lewat skrip kecil yang kujalankan sendiri. Setelah tiap skrip, jelaskan apa yang terjadi dan beri satu latihan kecil.
```

**Prompt 3 — saat siap ke RAG**
```
Mulai Minggu 7 dari ROADMAP.md. Rancang struktur 04-rag-bps sesuai aturan data di AGENTS.md (hanya dokumen publik, tidak ada kredensial di repo). Mulai dari skrip indexing paling sederhana, lalu tulis skrip evaluasi sebelum menambah fitur.
```
