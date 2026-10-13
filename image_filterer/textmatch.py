"""Match a typed query against text read off the photo (OCR).

OCR of stage screens is good but not clean: words run together ("DSX Flex" →
"dsxflex", "Groq 3 LPX" → "groq3lpx"), letters swap ("Blackwell" →
"blackweli"), and lines break mid-phrase. So matching is done against the
frame's text with spaces removed, and longer words tolerate a typo:

- words of 6+ characters allow one wrong letter, 10+ allow two — but such a
  near-match must start where a word starts;
- shorter words must appear exactly — "dell" must not match "bell";
- words of 3 characters or fewer must stand on their own (letters may touch
  digits: "groq3" still holds "groq", "nvl72" holds "nvl"), or "ai" would hit
  every "said" and "main".

Every word of the query has to be found (AND). Filler words ("a", "the", "on",
…) are ignored, so a query typed like a sentence still works.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

import numpy as np

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
except ImportError:  # pragma: no cover — exact matching still works without it
    fuzz = Levenshtein = None

STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "at", "to", "for", "and", "or", "with",
    "is", "it", "by", "from", "that", "this", "which", "where", "are", "be",
    "photo", "photos", "picture", "pictures", "image", "images", "shot", "shots",
    "text", "says", "saying", "reads", "written", "word", "words",
    "showing", "shows", "show", "screen", "screens", "slide", "slides",
}


def normalize(text: str) -> str:
    """Lowercase, letters and digits only, single spaces."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(text).lower()).split())


def query_terms(query: str) -> List[str]:
    """The words of ``query`` that have to be found. A quoted phrase is one
    term (matched with its spaces removed, like the photo text)."""
    terms: List[str] = []
    for phrase in re.findall(r'"([^"]+)"', query):
        t = normalize(phrase).replace(" ", "")
        if t:
            terms.append(t)
    rest = re.sub(r'"[^"]*"', " ", query)
    for w in normalize(rest).split():
        if w not in STOPWORDS and (len(w) >= 2 or w.isdigit()):
            terms.append(w)
    return terms


def _allowed_edits(term: str) -> int:
    n = len(term)
    return 2 if n >= 10 else 1 if n >= 6 else 0


def _short_pattern(term: str) -> "re.Pattern[str]":
    # Alphabetic: no letters either side. Numeric: no digits either side.
    if term.isdigit():
        return re.compile(rf"(?<!\d){re.escape(term)}(?!\d)")
    return re.compile(rf"(?<![a-z]){re.escape(term)}(?![a-z])")


class _Text:
    """One photo's text, prepared once per query."""

    def __init__(self, text: str) -> None:
        self.raw = " ".join(str(text).split())
        self.spaced = normalize(self.raw)
        self.packed = self.spaced.replace(" ", "")
        # Where each word starts in ``packed`` — a typo'd match must start on
        # one, or "SIEMENS Energy" (→ "siemensenergy") would contain "jensen".
        self.starts, pos = [], 0
        for w in self.spaced.split():
            self.starts.append(pos)
            pos += len(w)

    def find(self, term: str) -> "tuple[float, int]":
        """(score, position in ``packed``): 1.0 for an exact hit, a bit less
        for a typo'd one, 0 for none (position -1)."""
        if len(term) <= 3:
            m = _short_pattern(term).search(self.spaced)
            return (1.0, len(self.spaced[:m.start()].replace(" ", ""))) if m else (0.0, -1)
        k = self.packed.find(term)
        if k >= 0:
            return 1.0, k
        edits = _allowed_edits(term)
        if not edits or fuzz is None or len(self.packed) < len(term) - edits:
            return 0.0, -1
        # Cheap screen first (one C call over the whole text); partial_ratio
        # counts a substitution as two indel edits, hence 1 - k/len.
        if not fuzz.partial_ratio(term, self.packed, score_cutoff=100.0 * (1.0 - edits / len(term)) - 1e-6):
            return 0.0, -1
        best, at = edits + 1, -1
        for pos in self.starts:
            for n in range(len(term) - edits, len(term) + edits + 1):
                d = Levenshtein.distance(term, self.packed[pos:pos + n], score_cutoff=edits)
                if d < best:
                    best, at = d, pos
        return (1.0 - best / len(term), at) if at >= 0 else (0.0, -1)

    def snippet(self, packed_pos: int, width: int) -> str:
        """~``width`` characters of the original text around ``packed_pos``."""
        words, pos, k = self.raw.split(" "), 0, 0
        for k, w in enumerate(words):
            pos += len(normalize(w).replace(" ", ""))
            if pos > packed_pos:
                break
        a = max(0, len(" ".join(words[:k])) - width // 3)
        return ("…" if a else "") + self.raw[a:a + width] + ("…" if a + width < len(self.raw) else "")


def term_score(term: str, text: str) -> float:
    return _Text(text).find(term)[0]


def match_scores(texts: Sequence[Optional[str]], query: str) -> np.ndarray:
    """Per text: mean term score if every term is found, else 0. ``None``
    (not read yet) and "" score 0."""
    terms = query_terms(query)
    out = np.zeros(len(texts), dtype=np.float32)
    if not terms:
        return out
    for i, text in enumerate(texts):
        if not text:
            continue
        t = _Text(text)
        total = 0.0
        for term in terms:
            sc, _ = t.find(term)
            if sc <= 0:
                total = 0.0
                break
            total += sc
        out[i] = total / len(terms)
    return out


def snippet(text: str, query: str, width: int = 60) -> str:
    """A short piece of ``text`` around the first query word found, for display."""
    t = _Text(text)
    terms = query_terms(query)
    # The whole phrase if it's there ("Vera Rubin", not the "Vera BlueField"
    # earlier on the slide), else the most specific (longest) word.
    for term in ["".join(terms)] * (len(terms) > 1) + sorted(terms, key=len, reverse=True):
        sc, pos = t.find(term)
        if sc > 0:
            return t.snippet(pos, width)
    return t.raw[:width] + ("…" if len(t.raw) > width else "")


def line_matches(lines: Sequence[str], query: str) -> List[bool]:
    """Which lines of a photo to highlight for ``query``: the lines holding
    every word if any do ("Vera Rubin", not the "Vera BlueField" beside it),
    otherwise each line holding any of them (a phrase split over two lines)."""
    terms = query_terms(query)
    if not terms:
        return [False] * len(lines)
    found = [[_Text(line).find(t)[0] > 0 for t in terms] for line in lines]
    if any(all(f) for f in found):
        return [all(f) for f in found]
    return [any(f) for f in found]
