"""Tests for text normalization profiles, including:
punctuation removal, Unicode normalization, diacritic preservation, whitespace collapsing, profile selection.
"""

import pytest
from fleurs_asr.text_normalization import normalize_text, get_profile

def test_get_profile_returns_profile():
    profile = get_profile("wer_standard")
    assert profile.lowercase is True
    
def test_get_profile_unknown_raises():
    with pytest.raises(ValueError):
        get_profile("nope")

def test_wer_standard_basic():
    text =  normalize_text("Héllo, WORLD!", get_profile("wer_standard"))
    assert text == "héllo world"

def test_diacritices_preserved_in_wer_standard():
    text = normalize_text("café", get_profile("wer_standard"))
    assert text == "café"
    
def test_diacritics_stripped_in_aggressive():
    text = normalize_text("café", get_profile("aggressive"))
    assert text == "cafe"
    
def test_unicode_nfc_nfd_equal():
    text_composed = "caf\u00e9"
    text_decomposed = "cafe\u0301"
    text_composed = normalize_text(text_composed, get_profile("conservative"))
    text_decomposed = normalize_text(text_decomposed, get_profile("conservative"))
    assert text_composed == "café"
    assert text_decomposed == "café"
    
def test_whitespace_collapased():
    text = normalize_text("a b\t\nc", get_profile("conservative"))
    assert text == "a b c"