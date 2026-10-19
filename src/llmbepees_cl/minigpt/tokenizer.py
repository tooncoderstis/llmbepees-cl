"""Tokenisasi: mengubah teks menjadi deret id integer dan sebaliknya.

Model bahasa tidak melihat teks, melainkan deret id. Langkah ini jadi gerbang
pertama sebelum masuk transformer. Dua tingkat kerumitan:
- `CharTokenizer`: satu karakter = satu token (paling mudah dipahami).
- `BPETokenizer`: menggabungkan pasangan karakter yang sering muncul (BPE).
"""

from __future__ import annotations

from collections import Counter


class CharTokenizer:
    """Tokenizer paling sederhana: satu karakter = satu token.

    `vocab` adalah daftar karakter terurut; id token = indeks di daftar itu.
    Gunakan `CharTokenizer.from_text(...)` agar kosakata menyesuaikan korpus.
    """

    def __init__(self, vocab: list[str] | None = None) -> None:
        self.itos: list[str] = list(vocab) if vocab else []
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Bangun kosakata dari semua karakter unik di `text` (terurut)."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str) -> list[int]:
        """Teks -> deret id. Karakter di luar kosakata memicu `KeyError`."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        """Deret id -> teks (kebalikan `encode`)."""
        return "".join(self.itos[int(i)] for i in ids)


def _pair_counts(ids: list[int]) -> Counter[tuple[int, int]]:
    """Hitung frekuensi setiap pasangan token yang bersebelahan."""
    return Counter(zip(ids, ids[1:], strict=False))


def _merge_ids(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Ganti semua kemunculan `pair` di `ids` dengan satu token `new_id`."""
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-Pair Encoding sederhana: gabungkan pasangan yang paling sering.

    Latih dengan `train(text, n_merges)`: mulai dari kosakata karakter, lalu
    ulangi `n_merges` kali — cari pasangan token paling sering, beri id baru,
    dan gabungkan. Token jadi lebih panjang dari satu karakter, sehingga teks
    diwakili lebih sedikit token (mirip cara tokenizer LLM asli bekerja).
    """

    def __init__(self) -> None:
        self.itos: list[str] = []
        self.stoi: dict[str, int] = {}
        self.merges: dict[tuple[int, int], int] = {}
        self.merge_rank: dict[tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        return self.vocab_size

    def train(self, text: str, n_merges: int) -> BPETokenizer:
        """Bangun kosakata karakter lalu jalankan `n_merges` penggabungan BPE."""
        chars = sorted(set(text))
        self.itos = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.merges = {}
        self.merge_rank = {}

        ids = [self.stoi[ch] for ch in text]
        for _ in range(n_merges):
            counts = _pair_counts(ids)
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = len(self.itos)
            self.merges[pair] = new_id
            self.merge_rank[pair] = len(self.merges) - 1
            self.itos.append(self.itos[pair[0]] + self.itos[pair[1]])
            ids = _merge_ids(ids, pair, new_id)
        return self

    def encode(self, text: str) -> list[int]:
        """Terapkan penggabungan sesuai urutan belajar sampai tidak ada lagi yang cocok."""
        ids = [self.stoi[ch] for ch in text]
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:], strict=False))
            known = [(self.merge_rank[p], p) for p in pairs if p in self.merge_rank]
            if not known:
                break
            _, pair = min(known)
            ids = _merge_ids(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids) -> str:
        """Deret id -> teks (ketika digabung, tiap token sudah berupa potongan teks)."""
        return "".join(self.itos[int(i)] for i in ids)
