"""Skrip 01 — Tokenisasi: dari karakter ke BPE.

Tokenizer mengubah teks menjadi deret id integer, karena model bahasa bekerja
pada angka, bukan huruf. Di sini kita bandingkan dua tingkat: level karakter
(paling sederhana) dan BPE sederhana (menggabungkan pasangan yang sering muncul).

Jalankan:
    .venv\\Scripts\\python 02-mini-gpt\\01_tokenizer.py
"""

from __future__ import annotations

from llmbepees_cl.minigpt import BPETokenizer, CharTokenizer

text = "statistik menyajikan data statistik resmi"
print("teks:", text)
print("panjang teks (karakter):", len(text))

char = CharTokenizer.from_text(text)
char_ids = char.encode(text)
print("\n[char] kosakata:", "".join(char.itos), f"(ukuran {char.vocab_size})")
print("[char] id      :", char_ids)
print("[char] jumlah token:", len(char_ids))
assert char.decode(char_ids) == text, "roundtrip karakter harus utuh"
print("[char] decode  :", char.decode(char_ids))

bpe = BPETokenizer().train(text, n_merges=10)
bpe_ids = bpe.encode(text)
print("\n[bpe] ukuran kosakata:", bpe.vocab_size)
print("[bpe] beberapa token baru:", bpe.itos[char.vocab_size :][:8])
print("[bpe] id       :", bpe_ids)
print("[bpe] jumlah token:", len(bpe_ids))
assert bpe.decode(bpe_ids) == text, "roundtrip BPE harus utuh"
print("[bpe] decode   :", bpe.decode(bpe_ids))

print("\nKesimpulan: BPE mewakili teks yang sama dengan token lebih sedikit,")
print("karena potongan seperti 'ta' atau 'kan' menjadi satu token.")
