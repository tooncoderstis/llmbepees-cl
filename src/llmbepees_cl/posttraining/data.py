"""Data preferensi untuk DPO — pasangan (prompt, dipilih, ditolak).

Semua teks ditulis sendiri (publik). Contoh tugas sempit: **lebih memilih memberi
definisi daripada menolak** ketika konsepnya ada di korpus. Ganti dengan pasangan
preferensi Anda (mis. hasil anotasi) untuk tugas sesungguhnya.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreferencePair:
    """Satu perbandingan: untuk `prompt`, `chosen` lebih disukai daripada `rejected`."""

    prompt: str
    chosen: str
    rejected: str


PREFERENCE_PAIRS: list[PreferencePair] = [
    PreferencePair(
        "T: inflasi\nJ:", " naiknya harga barang dan jasa.\n", " informasi tidak ditemukan.\n"
    ),
    PreferencePair(
        "T: deflasi\nJ:", " turunnya harga barang dan jasa.\n", " informasi tidak ditemukan.\n"
    ),
    PreferencePair(
        "T: populasi\nJ:",
        " keseluruhan objek penelitian.\n",
        " informasi tidak ditemukan.\n",
    ),
    PreferencePair(
        "T: sampel\nJ:",
        " bagian dari populasi.\n",
        " informasi tidak ditemukan.\n",
    ),
    PreferencePair(
        "T: sensus\nJ:",
        " pencacahan lengkap.\n",
        " informasi tidak ditemukan.\n",
    ),
    PreferencePair(
        "T: median\nJ:",
        " nilai tengah data terurut.\n",
        " informasi tidak ditemukan.\n",
    ),
    PreferencePair(
        "T: kemiskinan\nJ:",
        " tidak mampu memenuhi kebutuhan dasar.\n",
        " informasi tidak ditemukan.\n",
    ),
    PreferencePair(
        "T: metadata\nJ:",
        " informasi yang menjelaskan data.\n",
        " informasi tidak ditemukan.\n",
    ),
]
