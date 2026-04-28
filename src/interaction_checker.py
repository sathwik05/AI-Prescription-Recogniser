"""
Drug-drug interaction lookup via RxNav.

Once we've matched a few words to RxNorm concept IDs (rxcuis), we ask the
interaction/list endpoint for any known interactions between them. The API
takes a '+'-separated list of rxcuis and returns a fairly nested JSON; we
flatten it into a simple list of {source, severity, description} dicts so
the template can iterate over it without any logic.

A request is only worth making if we have at least two distinct rxcuis --
otherwise there's nothing to interact with. We also dedupe before sending,
because if the OCR found "metformin" twice we don't want to ask whether
metformin interacts with itself.
"""

from __future__ import annotations

from typing import List, Dict

import requests

from .drug_matching import RXNORM_BASE


def rxnorm_interactions(rxcuis: List[str], timeout: int = 20) -> List[Dict]:
    """Return a flat list of interaction pairs for the given rxcuis.

    Each item: {"source": ..., "severity": ..., "description": ...}.
    Returns [] if there are <2 unique rxcuis or if the call fails.
    """
    # dict.fromkeys preserves order while deduping -- good enough here.
    unique = [c for c in dict.fromkeys(rxcuis) if c]
    if len(unique) < 2:
        return []

    try:
        r = requests.get(
            f"{RXNORM_BASE}/interaction/list.json",
            params={"rxcuis": "+".join(unique)},
            timeout=timeout,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        # Same policy as drug_matching -- swallow and return empty.
        return []

    flat: List[Dict] = []

    # The response shape:
    #   fullInteractionTypeGroup -> [ { sourceName, fullInteractionType -> [
    #       { interactionPair -> [ { severity, description, ... } ] }
    #   ] } ]
    # We flatten the whole tree into one list because the UI doesn't care
    # about the grouping.
    for group in data.get("fullInteractionTypeGroup", []) or []:
        source = group.get("sourceName", "")
        for ftype in group.get("fullInteractionType", []) or []:
            for pair in ftype.get("interactionPair", []) or []:
                flat.append({
                    "source": source,
                    "severity": pair.get("severity", ""),
                    "description": pair.get("description", ""),
                })

    return flat
