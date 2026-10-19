"""Tool `get_indicator(...)` — satu-satunya sumber angka dalam sistem ini.

Prinsip (lihat `AGENTS.md`): **angka resmi hanya dari tool**, tidak pernah dari
ingatan model. Data di sini adalah **contoh (dummy)** untuk belajar — bukan angka
resmi dan tidak boleh dipakai sebagai rujukan. Ganti dengan sumber/API publik
resmi saat tersedia; selalu sertakan periode, wilayah, dan satuan.
"""

from __future__ import annotations

SOURCES = "Contoh data (dummy) — bukan angka resmi"

# Kunci: (indikator, wilayah, periode) — nilai semua huruf kecil.
_RAW: dict[tuple[str, str, str], dict[str, object]] = {
    ("inflasi", "indonesia", "2023"): {"value": 3.67, "unit": "persen"},
    ("inflasi", "indonesia", "2022"): {"value": 5.51, "unit": "persen"},
    ("inflasi", "indonesia", "2024"): {"value": 2.57, "unit": "persen"},
    ("tingkat pengangguran terbuka", "indonesia", "2023"): {"value": 5.32, "unit": "persen"},
    ("tingkat pengangguran terbuka", "indonesia", "2024"): {"value": 4.91, "unit": "persen"},
    ("pertumbuhan ekonomi", "indonesia", "2023"): {"value": 5.05, "unit": "persen"},
    ("pertumbuhan ekonomi", "indonesia", "2024"): {"value": 5.03, "unit": "persen"},
    ("jumlah penduduk", "indonesia", "2023"): {"value": 278.70, "unit": "juta jiwa"},
    ("jumlah penduduk", "indonesia", "2024"): {"value": 281.60, "unit": "juta jiwa"},
    ("garis kemiskinan", "indonesia", "2023"): {
        "value": 515530,
        "unit": "rupiah per kapita per bulan",
    },
}

# Sinonim/alias -> nama indikator kanonik.
ALIASES: dict[str, str] = {
    "inflasi": "inflasi",
    "tingkat pengangguran terbuka": "tingkat pengangguran terbuka",
    "pengangguran terbuka": "tingkat pengangguran terbuka",
    "pengangguran": "tingkat pengangguran terbuka",
    "tpt": "tingkat pengangguran terbuka",
    "pertumbuhan ekonomi": "pertumbuhan ekonomi",
    "jumlah penduduk": "jumlah penduduk",
    "penduduk": "jumlah penduduk",
    "garis kemiskinan": "garis kemiskinan",
}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def available_indicators() -> list[str]:
    """Daftar nama indikator kanonik yang tersedia di tool."""
    return sorted({key[0] for key in _RAW})


def detect_indicator(question: str) -> str | None:
    """Tebak indikator yang dimaksud pertanyaan (alias terpanjang menang)."""
    text = _norm(question)
    matches = [alias for alias in ALIASES if alias in text]
    if not matches:
        return None
    return ALIASES[max(matches, key=len)]


def get_indicator(
    indicator: str, region: str = "Indonesia", period: str | None = None
) -> dict[str, object] | None:
    """Ambil satu angka indikator; kembalikan `None` bila tidak ada (jangan menebak).

    Hasil selalu memuat `indicator`, `region`, `period`, `value`, `unit`, `source`.
    """
    if period is None:
        return None
    record = _RAW.get((_norm(indicator), _norm(region), _norm(period)))
    if record is None:
        return None
    return {
        "indicator": _norm(indicator),
        "region": region,
        "period": period,
        "value": record["value"],
        "unit": record["unit"],
        "source": SOURCES,
    }
