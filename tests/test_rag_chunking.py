from llmbepees_cl.rag import SAMPLE_DOCS, chunk_documents
from llmbepees_cl.rag.documents import load_text_files


def test_chunk_documents_produces_passages_with_metadata():
    passages = chunk_documents(SAMPLE_DOCS)
    assert len(passages) >= len(SAMPLE_DOCS)
    for passage in passages:
        assert passage.passage_id
        assert passage.title
        assert passage.year > 0
        assert passage.section
        assert passage.text.strip()


def test_chunk_splits_paragraphs():
    passages = chunk_documents(SAMPLE_DOCS)
    texts = [p.text for p in passages]
    assert any(t.startswith("Inflasi adalah") for t in texts)
    assert any(t.startswith("Populasi adalah") for t in texts)


def test_citation_format():
    passage = chunk_documents(SAMPLE_DOCS)[0]
    assert passage.citation == f"{passage.title} ({passage.year})"


def test_load_text_files_missing_dir_is_empty(tmp_path):
    assert load_text_files(tmp_path / "tidak-ada") == []


def test_load_text_files_reads_year(tmp_path):
    (tmp_path / "publikasi-a.txt").write_text(
        "Tahun: 2019\n\nIsi dokumen publik.", encoding="utf-8"
    )
    docs = load_text_files(tmp_path)
    assert len(docs) == 1
    assert docs[0].year == 2019
    assert "Isi dokumen publik" in docs[0].text
