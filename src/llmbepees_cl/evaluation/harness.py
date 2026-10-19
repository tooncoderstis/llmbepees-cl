"""Harness evaluasi: jalankan test set pada asisten dan hitung metrik.

Test set = daftar dict dengan field `type`:
- `"concept"`: `must_contain` (frasa wajib ada) + `expect_citation`;
- `"number"`: `must_contain` + `expect_tool`;
- `"abstain"`: `should_abstain: true`.

Tidak bergantung pada implementasi asisten tertentu — cukup objek dengan `.ask(question)`
yang mengembalikan `.text`, `.citations`, `.tool_calls`, `.abstained`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Report:
    """Hasil evaluasi: hitungan per metrik + daftar kegagalan."""

    total: int = 0
    accuracy_correct: int = 0
    accuracy_total: int = 0
    citation_correct: int = 0
    citation_total: int = 0
    abstain_correct: int = 0
    abstain_total: int = 0
    tool_ok: int = 0
    tool_total: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)

    @staticmethod
    def rate(num: int, den: int) -> float:
        return num / den if den else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "accuracy": self.rate(self.accuracy_correct, self.accuracy_total),
            "citation_accuracy": self.rate(self.citation_correct, self.citation_total),
            "abstain_accuracy": self.rate(self.abstain_correct, self.abstain_total),
            "number_tool_compliance": self.rate(self.tool_ok, self.tool_total),
            "by_type": self.by_type,
            "failures": self.failures,
        }


def load_testset(path: str | Path) -> list[dict]:
    """Baca test set dari file JSONL."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _contains_all(text: str, needles: list[str]) -> bool:
    return all(needle.lower() in text for needle in needles)


def evaluate(assistant, items: list[dict]) -> Report:
    """Jalankan `assistant` atas `items` dan kembalikan `Report`."""
    report = Report(total=len(items))
    for item in items:
        kind = item["type"]
        report.by_type[kind] = report.by_type.get(kind, 0) + 1
        answer = assistant.ask(item["question"])
        text = answer.text.lower()
        ok = True

        if kind == "abstain":
            report.abstain_total += 1
            ok = bool(answer.abstained)
            report.abstain_correct += int(ok)
        else:
            report.accuracy_total += 1
            ok = _contains_all(text, item["must_contain"]) and not answer.abstained
            report.accuracy_correct += int(ok)

            if kind == "number":
                report.tool_total += 1
                tool_ok = bool(answer.tool_calls) and not answer.abstained
                report.tool_ok += int(tool_ok)
                ok = ok and tool_ok
            elif kind == "concept":
                report.citation_total += 1
                cite_ok = item["expect_citation"] in answer.citations
                report.citation_correct += int(cite_ok)
                ok = ok and cite_ok

        if not ok:
            report.failures.append(
                {
                    "id": item["id"],
                    "type": kind,
                    "question": item["question"],
                    "answer": answer.text,
                }
            )
    return report
