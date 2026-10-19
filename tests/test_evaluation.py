from llmbepees_cl.evaluation import evaluate, summarize_failures
from llmbepees_cl.rag import build_default_assistant

ITEMS = [
    {
        "id": "c1",
        "type": "concept",
        "question": "Apa itu inflasi?",
        "expect_citation": "Glosarium Statistik (2020)",
        "must_contain": ["kecenderungan naiknya harga barang"],
    },
    {
        "id": "n1",
        "type": "number",
        "question": "Berapa inflasi Indonesia 2023?",
        "must_contain": ["3.67", "persen"],
    },
    {"id": "a1", "type": "abstain", "question": "Apa warna bendera BPS?", "should_abstain": True},
    {"id": "a2", "type": "abstain", "question": "Apa itu kucing persia?", "should_abstain": True},
]


def test_evaluate_perfect_testset():
    report = evaluate(build_default_assistant(), ITEMS)
    assert report.total == 4
    assert report.rate(report.accuracy_correct, report.accuracy_total) == 1.0
    assert report.rate(report.citation_correct, report.citation_total) == 1.0
    assert report.rate(report.abstain_correct, report.abstain_total) == 1.0
    assert report.rate(report.tool_ok, report.tool_total) == 1.0
    assert report.failures == []
    assert report.by_type == {"concept": 1, "number": 1, "abstain": 2}


def test_evaluate_records_failures():
    bad = dict(ITEMS[0], id="c-bad", must_contain=["frasa yang tidak ada di jawaban"])
    report = evaluate(build_default_assistant(), [bad])
    assert len(report.failures) == 1
    assert report.failures[0]["id"] == "c-bad"
    assert report.rate(report.accuracy_correct, report.accuracy_total) == 0.0


def test_summarize_failures_groups_by_type():
    report = evaluate(build_default_assistant(), [dict(ITEMS[0], id="x", must_contain=["nope"])])
    grouped = summarize_failures(report.failures)
    assert "concept" in grouped
    assert grouped["concept"][0]["id"] == "x"


def test_report_to_dict_shape():
    data = evaluate(build_default_assistant(), ITEMS).to_dict()
    assert set(data) >= {"total", "accuracy", "citation_accuracy", "abstain_accuracy", "by_type"}
