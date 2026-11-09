"""Slug generation per docs/methodology.md

Algorithm:
  1. ASCII-transliterate (``unidecode``)
  2. lowercase
  3. non-alphanumeric → ``-``
  4. collapse runs of ``-``
  5. trim leading/trailing ``-``
  6. truncate to ``max_length``; if cut, append ``-<hash6>`` of original
  7. deduplicate against ``existing_slugs`` by appending ``-1``, ``-2``, …
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import Any, Final, cast

from unidecode import unidecode as _unidecode_raw  # type: ignore[import-untyped]


def _unidecode(text: str) -> str:
    return cast(str, cast(Any, _unidecode_raw)(text))


_NON_ALNUM: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")


def _base_slug(title: str) -> str:
    transliterated = _unidecode(title)
    lowered = transliterated.lower()
    return _NON_ALNUM.sub("-", lowered).strip("-")


def _hash_suffix(title: str) -> str:
    return hashlib.sha1(title.encode("utf-8"), usedforsecurity=False).hexdigest()[:6]


def make_slug(
    title: str,
    existing_slugs: Iterable[str] = (),
    max_length: int = 60,
) -> str:
    """Generate a canonical kebab-case slug from ``title``."""
    base = _base_slug(title)
    if len(base) > max_length:
        suffix = _hash_suffix(title)
        # keep room for "-<6hex>"
        base = f"{base[: max_length - 7].rstrip('-')}-{suffix}"

    existing_set = set(existing_slugs)
    if base not in existing_set:
        return base

    n = 1
    while True:
        candidate = f"{base}-{n}"
        if candidate not in existing_set:
            return candidate
        n += 1


__all__ = ["make_slug"]
