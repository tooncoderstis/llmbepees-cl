"""Korpus kecil bawaan + util batch untuk latihan model bahasa.

Tidak ada unduhan: korpus ditulis sendiri di file ini (teks publik, aman untuk
instruksi data). Ukurannya kecil supaya training CPU di laptop tetap cepat dan
ramah RAM, namun cukup berulang agar model mini bisa belajar pola.
"""

from __future__ import annotations

import torch

TINY_CORPUS = """Statistik adalah ilmu tentang mengumpulkan, mengolah, dan menyajikan data.
Data statistik adalah keterangan atau angka yang menggambarkan suatu keadaan.
Populasi adalah keseluruhan objek yang menjadi perhatian dalam penelitian.
Sampel adalah bagian dari populasi yang dipilih untuk mewakili keseluruhan.
Sensus adalah pencacahan lengkap terhadap seluruh populasi.
Survei adalah pencacahan sampel untuk memperkirakan keadaan populasi.
Rata-rata adalah jumlah nilai dibagi banyaknya nilai.
Median adalah nilai tengah setelah data diurutkan dari kecil ke besar.
Modus adalah nilai yang paling sering muncul dalam data.
Rasio adalah perbandingan antara dua nilai yang berbeda satuan.
Proporsi adalah perbandingan yang menyatakan bagian dari keseluruhan.
Persentase adalah proporsi yang dikalikan seratus.
Indeks adalah ukuran perubahan suatu nilai dibandingkan nilai acuan.
Inflasi adalah kenaikan harga barang dan jasa secara umum dan terus menerus.
Pertumbuhan ekonomi adalah kenaikan produk domestik bruto suatu periode.
Produk domestik bruto adalah nilai barang dan jasa akhir yang diproduksi suatu wilayah.
Nilai tukar adalah harga mata uang suatu negara terhadap mata uang lain.
Kemiskinan adalah keadaan tidak mampu memenuhi kebutuhan dasar.
Pengangguran adalah keadaan seseorang yang mencari pekerjaan namun belum memperolehnya.
Angka partisipasi sekolah adalah proporsi penduduk yang bersekolah pada umur tertentu.
Data resmi adalah data yang diterbitkan oleh lembaga statistik negara.
Penyajian data dapat berupa tabel, grafik, atau diagram.
Tabel menyajikan data dalam baris dan kolom agar mudah dibaca.
Grafik membantu melihat pola dan kecenderungan data dari waktu ke waktu.
Tingkat akurasi data bergantung pada mutu pengumpulan dan pengolahan.
Data berkualitas berarti data lengkap, konsisten, tepat waktu, dan dapat dipercaya.
Metadata adalah informasi yang menjelaskan arti dan asal suatu data.
Definisi operasional menjelaskan cara sebuah konsep diukur menjadi data.
Satuan menyatakan ukuran yang dipakai pada setiap angka data.
Periode menyatakan rentang waktu pengamatan suatu data.
Wilayah menyatakan cakupan tempat suatu data dikumpulkan.
Seorang analis selalu menyebut periode, wilayah, dan satuan saat menyajikan angka.
Angka tanpa satuan dan periode mudah disalahartikan.
Karena itu penyajian data resmi selalu menyertakan catatan kaki dan sumber.
Sumber data yang baik dapat diverifikasi dan diulang pengumpulannya.
Konsep statistik dijelaskan dari dokumen resmi, bukan dari dugaan.
Angka disajikan sesuai publikasi resmi tanpa ditambah atau dikurangi.
Jika data tidak ditemukan, jawablah bahwa data tidak tersedia.
Kejujuran data lebih penting daripada jawaban yang terdengar meyakinkan.
Belajar statistik berarti belajar membaca angka dengan hati-hati dan teliti.
"""


def encode_corpus(text: str, tokenizer) -> torch.Tensor:
    """Encode seluruh `text` memakai `tokenizer` menjadi tensor id 1D (`int64`)."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def split_data(data: torch.Tensor, val_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Potong `data` menjadi (latih, validasi) secara berurutan (bukan diacak)."""
    n_train = int(data.shape[0] * (1 - val_frac))
    return data[:n_train], data[n_train:]


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ambil `batch_size` potongan acak berukuran `block_size` dari `data`.

    Mengembalikan `(x, y)`: `x` adalah konteks dan `y` adalah token berikutnya
    yang digeser satu posisi — persis tugas "prediksi token selanjutnya".
    """
    if data.shape[0] <= block_size:
        raise ValueError("panjang data harus lebih besar dari block_size")
    max_start = data.shape[0] - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts])
    y = torch.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return x, y
