"""Named, explicit text normalization profiles for fair WER/CER scoring.

Diacritics are preserved by default; only the ``aggressive`` profile strips them.
Profiles: ``conservative``, ``wer_standard``, ``aggressive``.
"""

from dataclasses import dataclass 
import re
import unicodedata

@dataclass
class NormalizationProfile:
    lowercase: bool
    remove_punctuation: bool
    strip_diacritics: bool
    unicode_normalization: str
   
PROFILES = {
    "conservative": NormalizationProfile(
        lowercase=False,
        remove_punctuation=False,
        strip_diacritics=False,
        unicode_normalization="NFC",
    ),
    "wer_standard": NormalizationProfile(
        lowercase=True,
        remove_punctuation=True,
        strip_diacritics=False,
        unicode_normalization="NFC"
    ),
    "aggressive": NormalizationProfile(
        lowercase=True,
        remove_punctuation=True,
        strip_diacritics=True,
        unicode_normalization="NFKD"
    )
}
 
def get_profile(name):
    if name not in PROFILES:
        raise ValueError(f"{name} not in PROFILE")
    return PROFILES[name]
        
        
def normalize_text(text: str, profile: NormalizationProfile) -> str: 
    text = unicodedata.normalize(profile.unicode_normalization, text)
    if profile.lowercase:
        text = text.lower()
    if profile.remove_punctuation:
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
        
    if profile.strip_diacritics:
        text = unicodedata.normalize("NFD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text