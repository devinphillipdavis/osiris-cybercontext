"""Text analyzers for the BM25 backend: the original simple split, and a Lucene-style English analyzer.

``simple`` is the lowercase ``[a-z0-9]+`` split the backend has always used, and
stays the default so existing capsules and digests keep their values.

``lucene-english`` reproduces Lucene's EnglishAnalyzer, the analysis Anserini (and
so Pyserini's BEIR indexes) uses for English, in pure Python:

- StandardTokenizer, approximated from the UAX #29 word-break rules: characters that
  cannot be part of a token become separators, and an apostrophe, period or colon
  joins two letters while a period, comma or semicolon joins two digits.
- EnglishPossessiveFilter (John's -> John), lowercasing, and Lucene's 33-word English
  stop list.
- The Porter stemmer, with the 1980 rules and the two step-2 revisions Lucene uses
  (bli -> ble and logi -> log). Written out here, no dependencies.
- Symbols with the emoji property in the Latin-1 and arrow blocks (for example the
  copyright and registered signs) are tokens of their own.

Lucene's BM25 also scores with a one-byte encoding of each document's length
(lucene_length_norm), and it scores every occurrence of a repeated query term; the
BM25 backend applies both under this analyzer.

Validation: with this analyzer, every SciFact document's term vector matches the
term vector Lucene stores in Pyserini's prebuilt beir-v1.0.0-scifact.flat index, and
the top-10 rankings for all 300 judged test queries match it too (README,
"Real SciFact retrieval"). Characters outside the Latin, Greek, Cyrillic and common
symbol blocks, such as CJK ideographs (which Lucene emits one per token), are not
handled, and the emoji list covers the symbols that occur in SciFact.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

SIMPLE = "simple"
LUCENE_ENGLISH = "lucene-english"
ANALYZERS = (SIMPLE, LUCENE_ENGLISH)

LUCENE_ENGLISH_STOPWORDS = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
))

# StandardTokenizer's joiners: an apostrophe (U+0027, U+2018, U+2019), period, colon or middle dot
# between two letters; an apostrophe, period, comma or semicolon between two digits.
_LETTER_JOINERS = ".:'\u2018\u2019\u00b7"
_DIGIT_JOINERS = ".,;'\u2018\u2019"
# Symbols with the emoji property, which Lucene emits as tokens of their own (the Latin-1 and arrow
# blocks that scientific text uses).
_EMOJI_TOKENS = "\u00a9\u00ae\u2122\u203c\u2049\u2139\u2194\u2195\u2196\u2197\u2198\u2199\u21a9\u21aa"
# Format characters (soft hyphen, zero-width marks). They attach to the letter or digit before them, so they
# stay inside the token; with nothing before them in the token they are dropped.
_FORMAT_CHARS = "\u00ad\u200b\u200c\u200d\u200e\u200f\u2060\ufeff"

_SIMPLE_TOKEN = re.compile(r"[a-z0-9]+")
_LETTER = r"[^\W\d_]"
_DIGIT = r"\d"
_CHUNK = rf"(?:(?:{_LETTER}|{_DIGIT}|_)[{re.escape(_FORMAT_CHARS)}]*)"
_WORD = (
    rf"(?:{_CHUNK})+"
    rf"(?:(?:(?<={_LETTER})[{re.escape(_LETTER_JOINERS)}](?={_LETTER})"
    rf"|(?<={_DIGIT})[{re.escape(_DIGIT_JOINERS)}](?={_DIGIT}))"
    rf"(?:{_CHUNK})+)*"
)
_STANDARD_TOKEN = re.compile(rf"{_WORD}|[{re.escape(_EMOJI_TOKENS)}]")
_KEPT_CHARS = frozenset(_LETTER_JOINERS + _DIGIT_JOINERS + _FORMAT_CHARS)

_VOWELS = "aeiou"


def simple_tokens(text: str) -> List[str]:
    """Lowercase ``[a-z0-9]+`` tokens: no stemming, stopping or field weighting."""
    return _SIMPLE_TOKEN.findall(text.lower())


def _standard_text(text: str) -> str:
    """``text`` with every character that cannot be part of a token replaced by a space."""
    out = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch in _EMOJI_TOKENS:
            out.append(f" {ch} ")
        elif ch in _KEPT_CHARS or category[0] in "LM" or category in ("Nd", "Pc"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def lucene_english_tokens(text: str) -> List[str]:
    """Tokens under a Lucene-style English analyzer: possessives, stop list, Porter stemming."""
    tokens: List[str] = []
    for token in _STANDARD_TOKEN.findall(_standard_text(text)):
        if not token.strip("_"):
            continue  # a run of underscores alone is not a token
        if len(token) > 2 and token[-2] in "'\u2019" and token[-1] in "sS":
            token = token[:-2]  # EnglishPossessiveFilter: John's -> John
        token = token.lower()
        if token and token not in LUCENE_ENGLISH_STOPWORDS:
            tokens.append(porter_stem(token))
    return tokens


def _int4_encode(i: int) -> int:
    """Lucene's 4-significant-bit code for i (3 mantissa bits after the leading one, exact below 8)."""
    bits = i.bit_length()
    if bits < 4:
        return i
    shift = bits - 4
    return ((shift + 1) << 3) | ((i >> shift) & 0x07)


def _int4_decode(code: int) -> int:
    bits = code & 0x07
    shift = (code >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


def lucene_length_norm(length: int) -> int:
    """The document length Lucene's BM25 scores with: the length after its one-byte norm encoding.

    Lucene stores a field length as one byte (SmallFloat.intToByte4). Lengths below 24 are
    exact; above that only four significant bits survive, so BM25 sees a rounded length. This
    returns the value Lucene decodes, which is what the BM25 length normalization must use.
    """
    if length < 24:
        return length
    return 24 + _int4_decode(_int4_encode(length - 24))


def analyze(text: str, analyzer: str = SIMPLE) -> List[str]:
    """Tokens of ``text`` under the named analyzer (one of ``ANALYZERS``)."""
    if analyzer == SIMPLE:
        return simple_tokens(text)
    if analyzer == LUCENE_ENGLISH:
        return lucene_english_tokens(text)
    raise ValueError(f"unknown analyzer {analyzer!r}; expected one of {ANALYZERS}")


# Porter stemmer: the 1980 algorithm with the two step-2 revisions Porter made later and Lucene uses
# (bli -> ble in place of abli -> able, and logi -> log). A word is a sequence of consonant (C) and
# vowel (V) runs; its measure m is the number of VC pairs in C?(VC)^m V?.

def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    count = 0
    previous_vowel = False
    for i in range(len(stem)):
        vowel = not _is_consonant(stem, i)
        if previous_vowel and not vowel:
            count += 1
        previous_vowel = vowel
    return count


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """*o: ends consonant-vowel-consonant, where the last consonant is not w, x or y."""
    n = len(word)
    return n >= 3 and _is_consonant(word, n - 3) and not _is_consonant(word, n - 2) \
        and _is_consonant(word, n - 1) and word[-1] not in "wxy"


_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
    ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
    ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ("logi", "log"),
)
_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""),
    ("ness", ""),
)
_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou",
    "ism", "ate", "iti", "ous", "ive", "ize",
)


def _replace_longest(word: str, rules, min_measure: int) -> str:
    """Apply the longest matching suffix rule if the stem left behind has measure > min_measure."""
    matches = [(suffix, repl) for suffix, repl in rules if word.endswith(suffix)]
    if not matches:
        return word
    suffix, repl = max(matches, key=lambda rule: len(rule[0]))
    stem = word[: len(word) - len(suffix)]
    return stem + repl if _measure(stem) > min_measure else word


def porter_stem(word: str) -> str:
    """The Porter stem of a lowercase word."""
    if len(word) <= 2:
        return word

    # Step 1a: plurals
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b: -eed, -ed, -ing
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        stripped = None
        if word.endswith("ed") and _has_vowel(word[:-2]):
            stripped = word[:-2]
        elif word.endswith("ing") and _has_vowel(word[:-3]):
            stripped = word[:-3]
        if stripped is not None:
            word = stripped
            if word.endswith(("at", "bl", "iz")):
                word += "e"
            elif _ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif _measure(word) == 1 and _ends_cvc(word):
                word += "e"

    # Step 1c: y -> i after a vowel in the stem
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    word = _replace_longest(word, _STEP2, 0)
    word = _replace_longest(word, _STEP3, 0)

    # Step 4: remove the suffix when the measure exceeds 1 (-ion only after s or t)
    matches = [suffix for suffix in _STEP4 if word.endswith(suffix)]
    if matches:
        suffix = max(matches, key=len)
        stem = word[: len(word) - len(suffix)]
        if _measure(stem) > 1 and (suffix != "ion" or stem.endswith(("s", "t"))):
            word = stem

    # Step 5a: drop a final e
    if word.endswith("e"):
        stem = word[:-1]
        if _measure(stem) > 1 or (_measure(stem) == 1 and not _ends_cvc(stem)):
            word = stem

    # Step 5b: -ll -> -l when the measure exceeds 1
    if _measure(word) > 1 and _ends_double_consonant(word) and word.endswith("l"):
        word = word[:-1]
    return word
