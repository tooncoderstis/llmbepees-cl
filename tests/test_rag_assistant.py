from llmbepees_cl.rag import SYSTEM_PROMPT, BPSAssistant, build_default_assistant
from llmbepees_cl.rag.prompt import build_context, build_prompt


def test_concept_answer_cites_source_without_tool():
    answer = build_default_assistant().ask("Apa itu inflasi?")
    assert not answer.abstained
    assert "inflasi" in answer.text.lower()
    assert answer.citations  # ada sitasi
    assert answer.tool_calls == []  # konsep: tanpa tool


def test_number_question_uses_tool_with_period_region_unit():
    answer = build_default_assistant().ask("Berapa inflasi Indonesia 2023?")
    assert not answer.abstained
    assert len(answer.tool_calls) == 1
    record = answer.tool_calls[0]
    assert record["period"] == "2023"
    assert record["region"] == "Indonesia"
    assert record["unit"] == "persen"
    assert "3.67" in answer.text


def test_out_of_corpus_abstains():
    answer = build_default_assistant().ask("Apa warna bendera BPS?")
    assert answer.abstained
    assert "tidak ditemukan" in answer.text.lower()
    assert answer.tool_calls == []


def test_unknown_indicator_abstains():
    answer = build_default_assistant().ask("Berapa curah hujan Indonesia 2023?")
    assert answer.abstained


def test_number_question_without_period_abstains():
    answer = build_default_assistant().ask("Berapa inflasi Indonesia?")
    assert answer.abstained


def test_prompt_includes_rules_and_context():
    assistant = BPSAssistant()
    context = build_context(assistant.passages[:2])
    prompt = build_prompt("Apa itu inflasi?", assistant.passages[:2])
    assert SYSTEM_PROMPT in prompt
    assert "KONTEKS" in prompt
    assert context.splitlines()[0].startswith("[1]")
