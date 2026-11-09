"""S2.2: TokenLedger — per-call recording + cap detection + usage log."""

from __future__ import annotations

import json
from pathlib import Path

from spec_writer.config import TokenLimits
from spec_writer.engine.token_ledger import TokenLedger
from spec_writer.types import TokenUsage


def _limits(**kwargs) -> TokenLimits:
    return TokenLimits(**kwargs)


def test_record_updates_totals() -> None:
    ledger = TokenLedger(limits=_limits())
    ledger.record("L1", TokenUsage(input=10, output=20, total=30))
    assert ledger.per_session_total == 30
    assert ledger.per_project_total == 30
    assert ledger.per_layer_totals == {"L1": 30}


def test_record_aggregates_per_layer() -> None:
    ledger = TokenLedger(limits=_limits())
    ledger.record("L1", TokenUsage(input=5, output=5, total=10))
    ledger.record("L2", TokenUsage(input=15, output=15, total=30))
    ledger.record("L1", TokenUsage(input=5, output=5, total=10))
    assert ledger.per_layer_totals == {"L1": 20, "L2": 30}
    assert ledger.per_session_total == 50


def test_check_cap_returns_none_when_safe() -> None:
    ledger = TokenLedger(limits=_limits(per_session=100))
    ledger.record("L1", TokenUsage(input=10, output=10, total=20))
    assert ledger.check_cap("L1", prospective_tokens=50) is None


def test_check_cap_session() -> None:
    ledger = TokenLedger(limits=_limits(per_session=50))
    ledger.record("L1", TokenUsage(input=10, output=10, total=20))
    assert ledger.check_cap("L1", prospective_tokens=40) == "per_session"


def test_check_cap_layer() -> None:
    ledger = TokenLedger(limits=_limits(per_layer=30))
    ledger.record("L1", TokenUsage(input=10, output=10, total=20))
    assert ledger.check_cap("L1", prospective_tokens=20) == "per_layer"


def test_check_cap_project() -> None:
    ledger = TokenLedger(limits=_limits(per_project=40))
    ledger.record("L1", TokenUsage(input=10, output=10, total=20))
    assert ledger.check_cap("L1", prospective_tokens=30) == "per_project"


def test_check_cap_session_takes_precedence() -> None:
    ledger = TokenLedger(
        limits=_limits(per_session=10, per_layer=10, per_project=10),
    )
    assert ledger.check_cap("L1", prospective_tokens=20) == "per_session"


def test_no_cap_when_limits_none() -> None:
    ledger = TokenLedger(limits=_limits(per_session=None, per_layer=None, per_project=None))
    ledger.record("L1", TokenUsage(input=100_000, output=100_000, total=200_000))
    assert ledger.check_cap("L1", prospective_tokens=10_000_000) is None


def test_jsonl_log_written(tmp_path: Path) -> None:
    ledger = TokenLedger(limits=_limits(), usage_dir=tmp_path)
    ledger.record("L1", TokenUsage(input=5, output=10, total=15), call_label="seed")
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["layer"] == "L1"
    assert record["total"] == 15
    assert record["label"] == "seed"


def test_jsonl_log_appends(tmp_path: Path) -> None:
    ledger = TokenLedger(limits=_limits(), usage_dir=tmp_path)
    ledger.record("L1", TokenUsage(input=1, output=1, total=2))
    ledger.record("L2", TokenUsage(input=3, output=3, total=6))
    files = list(tmp_path.iterdir())
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_no_log_when_usage_dir_none() -> None:
    ledger = TokenLedger(limits=_limits())
    ledger.record("L1", TokenUsage(input=1, output=1, total=2))
    # No exception, no file — that's the contract.


def test_total_fallback_when_zero() -> None:
    ledger = TokenLedger(limits=_limits())
    ledger.record("L1", TokenUsage(input=7, output=3, total=0))
    assert ledger.per_session_total == 10


def test_jsonl_log_writes_cache_fields(tmp_path: Path) -> None:
    """: cache fields land in the JSONL record."""
    ledger = TokenLedger(limits=_limits(), usage_dir=tmp_path)
    ledger.record(
        "chat",
        TokenUsage(
            input=100,
            output=50,
            total=150,
            cache_creation_input=2000,
            cache_read_input=8000,
        ),
        call_label="sess-1:turn-3",
    )
    files = list(tmp_path.iterdir())
    record = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert record["cache_creation_input"] == 2000
    assert record["cache_read_input"] == 8000
    assert record["input"] == 100
    assert record["output"] == 50
    assert record["label"] == "sess-1:turn-3"
