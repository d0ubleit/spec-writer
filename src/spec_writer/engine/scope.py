"""Scope filter — narrow the corpus the engine considers for a run.

The :class:`Scope` model (:mod:`spec_writer.types`) is declarative: it
captures which specs the user wants the agent to look at. This module
turns it into a predicate over loaded :class:`Spec` objects.

Filters combine with AND semantics; ``None`` on a dimension means "no
constraint". ``folder_pattern`` is matched against
``layer.name/<spec-id>``-style relative paths so methodology mixins
can carve out e.g. ``api/`` anchors.
"""

from __future__ import annotations

import fnmatch

from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Scope, Spec


def _layer_dir_for(methodology: ResolvedMethodology, layer_id: str) -> str:
    for layer in methodology.layers:
        if layer.id == layer_id:
            return layer.name
    return layer_id


def spec_matches_scope(spec: Spec, scope: Scope, methodology: ResolvedMethodology) -> bool:
    """True iff ``spec`` satisfies every dimension of ``scope``."""
    if scope.layers and spec.layer not in scope.layers:
        return False
    if scope.types and spec.type not in scope.types:
        return False
    if scope.tags and not any(t in spec.tags for t in scope.tags):
        return False
    if scope.spec_ids and spec.id not in scope.spec_ids:
        return False
    if scope.folder_pattern:
        layer_name = _layer_dir_for(methodology, spec.layer)
        # Synthetic path used purely for fnmatch — avoids requiring on-disk lookup.
        synthetic = f"{layer_name}/{spec.id}"
        if not fnmatch.fnmatchcase(synthetic, scope.folder_pattern):
            return False
    # from_finding is a Phase-4-REV / sync concept; ignored here.
    return True


def filter_corpus(
    corpus: dict[str, Spec], scope: Scope | None, methodology: ResolvedMethodology
) -> dict[str, Spec]:
    """Return the subset of ``corpus`` that matches ``scope`` (None = passthrough)."""
    if scope is None:
        return dict(corpus)
    return {
        sid: spec for sid, spec in corpus.items() if spec_matches_scope(spec, scope, methodology)
    }


__all__ = ["filter_corpus", "spec_matches_scope"]
