"""Lightweight claim extraction helpers."""

from __future__ import annotations

import re

from eigentruth.verify.protocols import Claim

_SENTENCE_RE = re.compile(r"[^.!?。！？]+[.!?。！？]?")


def extract_claims(text: str, *, min_chars: int = 3) -> tuple[Claim, ...]:
    """Split text into simple sentence-level atomic claim candidates.

    This is intentionally conservative and dependency-free. Production systems
    can replace it with a stronger extractor behind the same `Claim` contract.
    """
    claims = []
    for idx, match in enumerate(_SENTENCE_RE.finditer(text), start=1):
        claim_text = match.group(0).strip()
        if len(claim_text) < min_chars:
            continue
        claims.append(
            Claim(
                text=claim_text,
                claim_id=f"c{len(claims) + 1}",
                span=(match.start(), match.end()),
                metadata={"extractor": "sentence_split", "source_index": idx},
            )
        )
    return tuple(claims)
