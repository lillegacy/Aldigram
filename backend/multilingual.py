"""Text normalization and Dhivehi (Thaana) transliteration helpers.

The IACT dashboard accepts commands in English, Dhivehi written in Thaana
script, and Dhivehi written in Latin transliteration.  Multilingual
sentence-transformer models are trained mostly on Latin-script data, so Thaana
text is transliterated into a consistent Latin representation before it is
embedded.  This keeps the three script variants close together in the shared
vector space.

The transliteration is deliberately simple: it maps every Thaana codepoint to
its usual Latin equivalent and leaves the result unsegmented.  That is exactly
what a multilingual encoder needs - a stable, language-preserving spelling
rather than a scholarly romanization.
"""

from __future__ import annotations

import unicodedata

# Thaana block (U+0780 - U+07BF).  The comments show the character each entry
# maps; they are written as escapes so this file stays readable in any editor.
THAANA_TO_LATIN: dict[str, str] = {
    "\u0780": "h",    # HAA
    "\u0781": "sh",   # SHAVIYANI
    "\u0782": "n",    # NOONU
    "\u0783": "r",    # RAA
    "\u0784": "b",    # BAA
    "\u0785": "lh",   # LHAVIYANI
    "\u0786": "k",    # KAAFU
    "\u0787": "",     # ALIFU (vowel carrier)
    "\u0788": "v",    # VAAVU
    "\u0789": "m",    # MEEMU
    "\u078A": "f",    # FAAFU
    "\u078B": "dh",   # DHAALU
    "\u078C": "th",   # THAA
    "\u078D": "l",    # LAAMU
    "\u078E": "g",    # GAAFU
    "\u078F": "gn",   # GNAVIYANI
    "\u0790": "s",    # SEENU
    "\u0791": "d",    # DAVIYANI
    "\u0792": "z",    # ZAVIYANI
    "\u0793": "t",    # TAVIYANI
    "\u0794": "y",    # YAA
    "\u0795": "p",    # PAVIYANI
    "\u0796": "j",    # JECHA
    "\u0797": "ch",   # CHAVIYANI
    "\u0798": "th",   # rare voiced THAA
    "\u0799": "h",    # rare HAA
    "\u079A": "kh",   # rare KHAA
    "\u079B": "dh",   # rare DHAALU
    "\u079C": "z",    # rare ZAA
    "\u079D": "sh",   # rare SHEENU
    "\u079E": "s",    # rare SAADU
    "\u079F": "d",    # rare DAADU
    "\u07A0": "t",    # rare TAA
    "\u07A1": "z",    # rare ZAA
    "\u07A2": "'",    # AINU
    "\u07A3": "gh",   # GHAINU
    "\u07A4": "q",    # QAAFU
    "\u07A5": "w",    # WAAVU
    # Vowel diacritics (fili).
    "\u07A6": "a",    # ABAFILI
    "\u07A7": "aa",   # AABAAFILI
    "\u07A8": "i",    # IBIFILI
    "\u07A9": "ee",   # EEBEEFILI
    "\u07AA": "u",    # UBUFILI
    "\u07AB": "oo",   # OOBOOFILI
    "\u07AC": "e",    # EBEFILI
    "\u07AD": "ey",   # EYBEYFILI
    "\u07AE": "o",    # OBOFILI
    "\u07AF": "oa",   # OABOAFILI
    "\u07B0": "",     # SUKUN (no vowel)
    "\u07B1": "n",    # NAA
}

THAANA_RANGE = ("\u0780", "\u07BF")


def has_thaana(text: str) -> bool:
    """Return True when *text* contains at least one Thaana character."""
    return any(THAANA_RANGE[0] <= char <= THAANA_RANGE[1] for char in text)


def transliterate_thaana(text: str) -> str:
    """Convert Thaana text to a Latin transliteration.

    Characters outside the Thaana block are passed through unchanged, so mixed
    Dhivehi/English input (very common in real commands) works as-is.
    """
    return "".join(THAANA_TO_LATIN.get(char, char) for char in text)


def normalize_text(text: str) -> str:
    """Normalize a command for embedding.

    Steps: Unicode NFKC, lowercase, Thaana transliteration, punctuation
    flattening and whitespace collapsing.  The result is script-agnostic Latin
    text that the multilingual model can place in one shared vector space.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = transliterate_thaana(text)

    cleaned: list[str] = []
    for char in text:
        if char.isalnum() or char == "'":
            cleaned.append(char)
        elif char.isspace() or unicodedata.category(char).startswith("P"):
            cleaned.append(" ")
        elif unicodedata.category(char).startswith("Z"):
            cleaned.append(" ")

    return " ".join("".join(cleaned).split())
