"""
Match an OCR'd word against the RxNorm drug vocabulary.

We hit the public RxNav approximateTerm endpoint -- it's free, no API key,
and surprisingly forgiving of misspellings. The response gives us a ranked
list of candidates, each with a score we can use to decide how confident
we are.

Network errors are swallowed on purpose: the OCR result should still render
even if RxNav is having a bad day. We just return [] and let the UI show
"no match".

The Levenshtein helpers live here too, used by the /evaluate route to
compute character error rate against the held-out test split.
"""

from __future__ import annotations

from typing import List, Dict

import requests


RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"


def rxnorm_approximate(name: str, max_entries: int = 3, timeout: int = 10) -> List[Dict]:
    """Look up `name` in RxNorm and return up to `max_entries` candidates.

    Each candidate is a dict from the RxNav API. The keys we care about
    downstream are 'rxcui', 'name', and 'score'.
    """
    if not name:
        return []
    try:
        r = requests.get(
            f"{RXNORM_BASE}/approximateTerm.json",
            params={"term": name, "maxEntries": max_entries},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("approximateGroup", {}).get("candidate", []) or []
    except requests.RequestException:
        # Network down, RxNav 5xx, timeout -- all the same to us. Empty list.
        return []


def candidate_score(candidate: Dict | None) -> float:
    """Pull the numeric score out of a candidate dict, defensively.

    The API gives us a string; sometimes it's missing entirely. We treat
    "missing" or "unparseable" as zero, which makes the side-by-side
    comparison code in app.py simpler.
    """
    if not candidate:
        return 0.0
    raw = candidate.get("score")
    if not raw:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


# --- Levenshtein / CER for the eval route ---------------------------------

def levenshtein(a: str, b: str) -> int:
    """Classic edit distance. Iterative, two-row table -- O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + (0 if ca == cb else 1),   # match / substitution
            )
        prev = curr
    return prev[-1]


def char_error_rate(pred: str, truth: str) -> float:
    """Edit distance normalised by truth length.

    Edge case: if the truth string is empty we can't divide by zero, so we
    return 1.0 if the prediction is also non-empty (clearly wrong) or 0.0
    if both are empty (trivially right).
    """
    if not truth:
        return 1.0 if pred else 0.0
    return levenshtein(pred, truth) / len(truth)
