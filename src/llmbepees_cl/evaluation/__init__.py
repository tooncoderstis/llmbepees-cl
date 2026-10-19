"""Evaluasi asisten: harness + metrik dapat dipakai ulang (Fase 6).

Metrik inti (sesuai PRD):
- **akurasi**: jawaban memuat semua frasa acuan (`must_contain`);
- **ketepatan sitasi**: jawaban konsep menyitasi sumber yang benar;
- **abstain benar**: pertanyaan di luar korpus dijawab "tidak ditemukan";
- **angka via tool**: pertanyaan angka dijawab lewat tool, bukan tebakan.

Reusable dan diuji di `tests/test_evaluation.py`. Pemakaian ada di `04-rag-bps/eval/`
dan `06-specialization/eval/`.
"""

from llmbepees_cl.evaluation.failure_analysis import summarize_failures
from llmbepees_cl.evaluation.harness import Report, evaluate, load_testset

__all__ = ["Report", "evaluate", "load_testset", "summarize_failures"]
