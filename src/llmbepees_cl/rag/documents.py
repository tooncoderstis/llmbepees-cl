"""Dokumen sumber untuk RAG — tipe data dan korpus contoh yang bisa dikomit.

Aturan data (lihat `AGENTS.md`): hanya **dokumen publik**. Korpus contoh di sini
adalah teks yang ditulis sendiri (mengikuti gaya glosarium statistik), aman untuk
repo. Dokumen publik nyata (PDF/teks) diletakkan pengguna di `04-rag-bps/data/`
(folder itu di-`.gitignore` kecuali README); muat dengan `load_text_files`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """Satu bagian logis dari sebuah dokumen publik.

    Definisi tiap istilah dipisah baris kosong di dalam `text`, sehingga bisa
    dipecah menjadi bagian-bagian kecil (lihat `chunking.py`).
    """

    doc_id: str
    title: str
    year: int
    section: str
    text: str

    @property
    def citation(self) -> str:
        """Rujukan ringkas: `Judul (Tahun)`."""
        return f"{self.title} ({self.year})"


# Korpus contoh (publik, ditulis sendiri). Ganti/tambah dengan dokumen publik nyata.
SAMPLE_DOCS: list[Document] = [
    Document(
        doc_id="glosarium-2020-penduduk",
        title="Glosarium Statistik",
        year=2020,
        section="Penduduk dan Ketenagakerjaan",
        text=(
            "Populasi adalah keseluruhan objek atau penduduk yang menjadi perhatian "
            "dalam suatu penelitian.\n\n"
            "Sampel adalah bagian dari populasi yang dipilih untuk mewakili keseluruhan.\n\n"
            "Sensus adalah pencacahan lengkap terhadap seluruh populasi.\n\n"
            "Survei adalah pencacahan pada sebagian populasi (sampel) untuk memperkirakan "
            "keadaan populasi.\n\n"
            "Pengangguran adalah keadaan seseorang yang termasuk angkatan kerja namun "
            "sedang mencari pekerjaan dan belum memperolehnya.\n\n"
            "Tingkat pengangguran terbuka adalah persentase jumlah pengangguran terhadap "
            "jumlah angkatan kerja."
        ),
    ),
    Document(
        doc_id="glosarium-2020-harga",
        title="Glosarium Statistik",
        year=2020,
        section="Harga dan Inflasi",
        text=(
            "Inflasi adalah kecenderungan naiknya harga barang dan jasa secara umum dan "
            "terus-menerus dalam suatu periode.\n\n"
            "Deflasi adalah kecenderungan turunnya harga barang dan jasa secara umum dan "
            "terus-menerus dalam suatu periode.\n\n"
            "Indeks harga konsumen adalah ukuran perubahan harga sekelompok barang dan jasa "
            "yang dikonsumsi rumah tangga dari waktu ke waktu.\n\n"
            "Nilai tukar adalah harga mata uang suatu negara terhadap mata uang negara lain."
        ),
    ),
    Document(
        doc_id="glosarium-2020-ekonomi",
        title="Glosarium Statistik",
        year=2020,
        section="Ekonomi",
        text=(
            "Produk domestik bruto adalah nilai seluruh barang dan jasa akhir yang "
            "diproduksi suatu wilayah dalam periode tertentu.\n\n"
            "Pertumbuhan ekonomi adalah kenaikan produk domestik bruto suatu periode "
            "dibandingkan periode sebelumnya.\n\n"
            "Kemiskinan adalah keadaan tidak mampu memenuhi kebutuhan dasar.\n\n"
            "Garis kemiskinan adalah batas nilai pengeluaran minimum yang diperlukan "
            "untuk memenuhi kebutuhan dasar."
        ),
    ),
    Document(
        doc_id="glosarium-2020-metodologi",
        title="Glosarium Statistik",
        year=2020,
        section="Metodologi",
        text=(
            "Data resmi adalah data statistik yang diterbitkan oleh lembaga statistik "
            "negara melalui publikasi resmi.\n\n"
            "Metadata adalah informasi yang menjelaskan arti, asal, dan cara pengukuran "
            "suatu data.\n\n"
            "Definisi operasional adalah penjelasan cara sebuah konsep diukur menjadi data.\n\n"
            "Satuan adalah ukuran yang dipakai pada setiap angka data, misalnya persen, "
            "ribu jiwa, atau rupiah.\n\n"
            "Periode adalah rentang waktu pengamatan suatu data.\n\n"
            "Wilayah adalah cakupan tempat suatu data dikumpulkan."
        ),
    ),
    Document(
        doc_id="panduan-penyajian-2021-prinsip",
        title="Panduan Penyajian Data Statistik",
        year=2021,
        section="Prinsip Penyajian",
        text=(
            "Penyajian data adalah cara menyampaikan data kepada pengguna, misalnya "
            "dalam bentuk tabel atau grafik.\n\n"
            "Tabel adalah penyajian data dalam baris dan kolom agar mudah dibaca.\n\n"
            "Grafik adalah penyajian data secara visual untuk melihat pola dan "
            "kecenderungan.\n\n"
            "Rasio adalah perbandingan antara dua nilai yang berbeda satuan."
        ),
    ),
    Document(
        doc_id="panduan-penyajian-2021-sumber",
        title="Panduan Penyajian Data Statistik",
        year=2021,
        section="Sumber dan Sitasi",
        text=(
            "Sumber data adalah asal data yang dapat diverifikasi dan diulang "
            "pengumpulannya.\n\n"
            "Sitasi adalah pencantuman rujukan sumber, biasanya memuat judul dan tahun "
            "penerbitan.\n\n"
            "Dokumen publik adalah dokumen yang boleh diakses dan dikutip siapa saja.\n\n"
            "Proporsi adalah perbandingan yang menyatakan bagian dari keseluruhan."
        ),
    ),
]


def load_text_files(directory: str | Path) -> list[Document]:
    """Muat dokumen publik dari file `.txt`/`.md` di `directory`.

    Setiap file dianggap satu dokumen; judul = nama file, tahun = 0 kecuali
    tertulis "Tahun: YYYY" pada baris awal. Berguna untuk dokumen yang Anda
    letakkan sendiri di `04-rag-bps/data/`.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    docs: list[Document] = []
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() not in {".txt", ".md"} or path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8")
        year = 0
        for line in text.splitlines():
            if line.lower().startswith("tahun:"):
                year = int(line.split(":", 1)[1].strip())
                break
        docs.append(
            Document(
                doc_id=path.stem,
                title=path.stem.replace("-", " ").replace("_", " ").title(),
                year=year,
                section="Isi",
                text=text,
            )
        )
    return docs
