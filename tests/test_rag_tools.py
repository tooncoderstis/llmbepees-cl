from llmbepees_cl.rag import available_indicators, get_indicator
from llmbepees_cl.rag.tools import detect_indicator


def test_get_indicator_returns_full_metadata():
    record = get_indicator("inflasi", "Indonesia", "2023")
    assert record is not None
    assert record["value"] == 3.67
    assert record["unit"] == "persen"
    assert record["region"] == "Indonesia"
    assert record["period"] == "2023"
    assert record["source"]


def test_get_indicator_is_case_insensitive():
    assert get_indicator("INFLASI", "indonesia", "2023") is not None


def test_get_indicator_missing_period_or_unknown_returns_none():
    assert get_indicator("inflasi", "Indonesia", None) is None
    assert get_indicator("inflasi", "Indonesia", "1900") is None
    assert get_indicator("indikator-ngawur", "Indonesia", "2023") is None


def test_detect_indicator_with_alias():
    assert detect_indicator("Berapa TPT Indonesia 2024?") == "tingkat pengangguran terbuka"
    assert detect_indicator("Berapa jumlah penduduk 2023?") == "jumlah penduduk"
    assert detect_indicator("Apa itu cuaca?") is None


def test_available_indicators_non_empty():
    names = available_indicators()
    assert "inflasi" in names
    assert names == sorted(names)
