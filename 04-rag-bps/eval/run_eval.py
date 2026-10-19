"""Skrip evaluasi asisten (Fase 4) — ukur akurasi, sitasi, abstain, dan disiplin angka.

Memakai paket bersama `llmbepees_cl.evaluation`. Test set kecil (52 pertanyaan) ada di
`testset.jsonl`; test set besar (~200) ada di `06-specialization/eval/`.

Jalankan:
    .venv\\Scripts\\python 04-rag-bps\\eval\\run_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llmbepees_cl.evaluation import evaluate, load_testset, summarize_failures
from llmbepees_cl.evaluation.failure_analysis import format_summary
from llmbepees_cl.rag import build_default_assistant

TESTSET = Path(__file__).parent / "testset.jsonl"


def main() -> None:
    items = load_testset(TESTSET)
    report = evaluate(build_default_assistant(), items)

    def line(label: str, correct: int, total: int) -> str:
        return f"{label:<23}: {report.rate(correct, total):.1%} ({correct}/{total})"

    print(f"Test set: {report.total} pertanyaan\n")
    print(line("Akurasi (konsep+angka)", report.accuracy_correct, report.accuracy_total))
    print(line("Ketepatan sitasi", report.citation_correct, report.citation_total))
    print(line("Abstain benar", report.abstain_correct, report.abstain_total))
    print(line("Angka via tool", report.tool_ok, report.tool_total))

    if report.failures:
        print("\nAnalisis kegagalan:")
        print(format_summary(summarize_failures(report.failures)))

    out = TESTSET.parent / "report.json"
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nLaporan disimpan: {out}")


if __name__ == "__main__":
    main()
