# Konsep RL & post-training untuk LLM (catatan)

Ringkasan konsep — bukan kode. Tujuannya memahami **kenapa** dan **kapan** memakai
masing-masing pendekatan di jalur lab model.

## Dari SFT ke preferensi
- **SFT (supervised fine-tuning)**: meniru jawaban contoh. Cepat, tapi tidak mengajarkan mana yang *lebih* disukai saat ada beberapa jawaban masuk akal.
- **Preference tuning**: belajar dari perbandingan (A lebih baik dari B), bukan dari satu jawaban benar.

## RLHF (klasik) — tiga tahap
1. **SFT** pada demonstrasi.
2. Latih **reward model (RM)** dari data preferensi manusia.
3. **RL dengan PPO**: policy menghasilkan jawaban, RM memberi skor, policy diperbarui untuk menaikkan skor — dengan **penalti KL** ke policy SFT agar tidak "kabur" jauh.

Intuisi **policy gradient**: naikkan probabilitas aksi (token) yang menghasilkan reward tinggi.
`∇θ J = E[ ∇θ log πθ(a|s) · A ]`, dengan `A` (advantage) mengukur seberapa lebih baik aksi itu
dari rata-rata. PPO menstabilkan langkahnya (clipping ratio) dan menambah penalti KL.

Masalah RLHF: rumit (RM + RL), tidak stabil, mahal.

## DPO (Direct Preference Optimization)
Amati bahwa RM optimal punya bentuk tertutup terkait policy. DPO **melewati** RM dan RL,
langsung mengoptimalkan policy pada data preferensi:

```
loss = −log σ( β·[ (logp_θ(y_w) − logp_θ(y_l)) − (logp_ref(y_w) − logp_ref(y_l)) ] )
```

- `y_w`/`y_l` = jawaban dipilih/ditolak; `ref` = policy SFT yang **dibekukan**; `β` mengatur
  seberapa jauh boleh menyimpang dari reference.
- Sederhana seperti training supervised (tidak ada sampling/RL loop) → lebih stabil & murah.
- Kecenderungan: DPO mudah dijalankan, tapi bisa "over-optimize" pada data preferensi sempit.

## RL dengan reward yang bisa diverifikasi (verifiable reward)
Untuk tugas dengan jawaban **objektif** (matematika, kode, format, sitasi benar), reward tak
perlu dari manusia: cukup **cek programatik** (jawaban benar? sitasi ada? angka dari tool?).
Algoritma seperti **GRPO/REINFORCE** memakai reward ini. Sangat relevan untuk asisten BPS:
"angka harus dari tool" dan "sitasi wajib" adalah sinyal yang bisa diverifikasi otomatis.

## Kapan pakai apa (untuk proyek ini)
| Kebutuhan | Pendekatan masuk akal |
|---|---|
| Gaya bahasa & format sitasi | SFT kecil atau **LoRA** (Fase 3) |
| Pilih jawaban mana yang lebih disukai (mis. mengalahkan abstain keliru) | **DPO** |
| Tugas dengan cek objektif (format/sitasi/angka-via-tool) | **Verifiable reward** + GRPO |

## Bacaan wajib (ringkasan ada di `papers/`)
Transformer (2017) · Kaplan (2020) · Chinchilla (2022) · InstructGPT (2022) · DPO (2023).
