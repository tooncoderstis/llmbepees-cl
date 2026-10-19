"""Skrip evaluasi asisten (Fase 4) — ukur akurasi, sitasi, abstain, dan disiplin angka.

Menjalankan `BPSAssistant` atas `testset.jsonl` dan menghitung metrik:
- **akurasi**: jawaban konsep/angka memuat semua frasa acuan (`must_contain`);
- **ketepatan sitasi**: jawaban konsep menyitasi sumber yang benar;
- **abstain benar**: pertanyaan di luar korpus dijawab "tidak ditemukan";
- **angka via tool**: pertanyaan angka dijawab dengan memanggil tool (bukan menebak).

Jalankan:
    .venv\\Scripts\\python 04-rag-bps\\eval\\run_eval.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from llmbepees_cl.rag import build_default_assistant

TESTSET = Path(__file__).parent / "testset.jsonl"


@dataclass
class Metrics:
    total: int = 0
    accuracy_correct: int = 0
    accuracy_total: int = 0
    citation_correct: int = 0
    citation_total: int = 0
    abstain_correct: int = 0
    abstain_total: int = 0
    number_tool_ok: int = 0
    number_total: int = 0

    def rate(self, num: int, den: int) -> float:
        return num / den if den else 0.0


def load_testset(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def evaluate(items: list[dict]) -> tuple[Metrics, list[dict]]:
    assistant = build_default_assistant()
    metrics = Metrics(total=len(items))
    failures: list[dict] = []

    for item in items:
        answer = assistant.ask(item["question"])
        text = answer.text.lower()
        ok = True

        if item["type"] == "abstain":
            metrics.abstain_total += 1
            correct = answer.abstained
            metrics.abstain_correct += int(correct)
            ok = correct
        else:
            metrics.accuracy_total += 1
            contains = all(needle.lower() in text for needle in item["must_contain"])
            correct = contains and not answer.abstained
            metrics.accuracy_correct += int(correct)
            ok = correct

            if item["type"] == "number":
                metrics.number_total += 1
                tool_ok = bool(answer.tool_calls) and not answer.abstained
                metrics.number_tool_ok += int(tool_ok)
                ok = ok and tool_ok
            elif item["type"] == "concept":
                metrics.citation_total += 1
                cite_ok = item["expect_citation"] in answer.citations
                metrics.citation_correct += int(cite_ok)
                ok = ok and cite_ok

        if not ok:
            failures.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "answer": answer.text,
                    "abstained": answer.abstained,
                }
            )

    return metrics, failures


def metric_line(label: str, correct: int, total: int) -> str:
    rate = correct / total if total else 0.0
    return f"{label:<23}: {rate:.1%} ({correct}/{total})"


def main() -> None:
    items = load_testset(TESTSET)
    metrics, failures = evaluate(items)

    print(f"Test set: {metrics.total} pertanyaan\n")
    print(metric_line("Akurasi (konsep+angka)", metrics.accuracy_correct, metrics.accuracy_total))
    print(metric_line("Ketepatan sitasi", metrics.citation_correct, metrics.citation_total))
    print(metric_line("Abstain benar", metrics.abstain_correct, metrics.abstain_total))
    print(metric_line("Angka via tool", metrics.number_tool_ok, metrics.number_total))

    if failures:
        print(f"\nGagal ({len(failures)}):")
        for failure in failures:
            print(f"  [{failure['id']}] {failure['question']} -> {failure['answer'][:90]}")

    report = {
        "total": metrics.total,
        "accuracy": metrics.rate(metrics.accuracy_correct, metrics.accuracy_total),
        "citation_accuracy": metrics.rate(metrics.citation_correct, metrics.citation_total),
        "abstain_accuracy": metrics.rate(metrics.abstain_correct, metrics.abstain_total),
        "number_tool_compliance": metrics.rate(metrics.number_tool_ok, metrics.number_total),
        "failures": failures,
    }
    out = TESTSET.parent / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nLaporan disimpan: {out}")


if __name__ == "__main__":
    main()
