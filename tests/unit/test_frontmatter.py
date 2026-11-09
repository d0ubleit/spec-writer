"""S1.4: canonical frontmatter formatter + round-trip parser."""

from __future__ import annotations

from typing import Any

from spec_writer.io.frontmatter import parse_frontmatter, write_canonical

# ── field order ──────────────────────────────────────────────────────────


def test_field_order_engine_then_vertical_then_conventional() -> None:
    fm: dict[str, Any] = {
        "tags": ["a"],
        "title": "X",
        "up_to": [],
        "type": "vision",
        "id": "L1-VIS-001",
        "layer": "L1",
    }
    out = write_canonical(fm, body="")
    lines = [line for line in out.splitlines() if ":" in line]
    keys = [line.split(":", 1)[0] for line in lines]
    assert keys[:5] == ["id", "type", "layer", "title", "up_to"]
    assert keys[-1] == "tags"


def test_field_order_methodology_mid() -> None:
    fm: dict[str, Any] = {
        "id": "L4-API-001",
        "type": "api-contract",
        "layer": "L4",
        "title": "X",
        "up_to": ["L3-BHV-001"],
        "implemented_by": [],
        "method": "POST",
        "path": "/u",
        "auth": "authenticated",
        "tags": ["api"],
        "status": "draft",
    }
    out = write_canonical(fm, body="", methodology_required_order=["method", "path", "auth"])
    keys = [
        line.split(":", 1)[0]
        for line in out.splitlines()
        if ":" in line and not line.startswith(" ")
    ]
    assert keys.index("implemented_by") < keys.index("method")
    assert keys.index("method") < keys.index("path") < keys.index("auth")
    assert keys.index("auth") < keys.index("tags")


def test_field_order_user_extra_alphabetical() -> None:
    fm: dict[str, Any] = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "zebra": "z",
        "alpha": "a",
        "mango": "m",
    }
    out = write_canonical(fm, body="")
    lines = [line for line in out.splitlines() if ":" in line]
    keys = [line.split(":", 1)[0] for line in lines]
    extras = [k for k in keys if k in ("alpha", "mango", "zebra")]
    assert extras == ["alpha", "mango", "zebra"]


# ── value formatting ─────────────────────────────────────────────────────


def test_title_always_double_quoted() -> None:
    fm = {"id": "L1-VIS-001", "type": "vision", "layer": "L1", "title": "Simple", "up_to": []}
    out = write_canonical(fm, body="")
    assert 'title: "Simple"\n' in out


def test_list_flow_style() -> None:
    fm = {
        "id": "L2-CAP-001",
        "type": "capability",
        "layer": "L2",
        "title": "X",
        "up_to": ["L1-VIS-001"],
        "tags": ["auth", "user-management"],
    }
    out = write_canonical(fm, body="")
    assert "up_to: [L1-VIS-001]" in out
    assert "tags: [auth, user-management]" in out
    assert "- L1-VIS-001" not in out  # no block style


def test_empty_list_renders_brackets() -> None:
    fm = {"id": "L1-VIS-001", "type": "vision", "layer": "L1", "title": "X", "up_to": []}
    out = write_canonical(fm, body="")
    assert "up_to: []\n" in out


def test_implemented_by_empty_renders() -> None:
    fm = {
        "id": "L4-API-001",
        "type": "api-contract",
        "layer": "L4",
        "title": "X",
        "up_to": ["L3-BHV-001"],
        "implemented_by": [],
    }
    out = write_canonical(fm, body="")
    assert "implemented_by: []\n" in out


def test_implemented_by_with_paths() -> None:
    fm = {
        "id": "L4-API-001",
        "type": "api-contract",
        "layer": "L4",
        "title": "X",
        "up_to": ["L3-BHV-001"],
        "implemented_by": ["src/api/profile.py:42-87", "src/api/service.py:15-60"],
    }
    out = write_canonical(fm, body="")
    assert 'implemented_by: ["src/api/profile.py:42-87", "src/api/service.py:15-60"]' in out


def test_string_bare_when_safe() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "status": "draft",
    }
    out = write_canonical(fm, body="")
    assert "status: draft\n" in out


def test_string_quoted_when_reserved_char() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "note": "has: colon",
    }
    out = write_canonical(fm, body="")
    assert 'note: "has: colon"\n' in out


def test_integer_bare() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "iteration": 3,
    }
    out = write_canonical(fm, body="")
    assert "iteration: 3\n" in out


def test_relates_to_inline_objects() -> None:
    fm = {
        "id": "L2-CAP-001",
        "type": "capability",
        "layer": "L2",
        "title": "X",
        "up_to": ["L1-VIS-001"],
        "relates_to": [{"id": "L2-CAP-002", "kind": "depends_on", "reason": "needs session"}],
    }
    out = write_canonical(fm, body="")
    assert 'relates_to: [{id: L2-CAP-002, kind: depends_on, reason: "needs session"}]' in out


def test_provenance_inline_object() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "provenance": {"kind": "doc", "ref": "biz-req.pdf.md", "confidence": 0.9},
    }
    out = write_canonical(fm, body="")
    assert 'provenance: {kind: doc, ref: "biz-req.pdf.md", confidence: 0.9}' in out


# ── parse + round-trip ───────────────────────────────────────────────────


def test_parse_basic() -> None:
    src = (
        "---\n"
        "id: L1-VIS-001\n"
        "type: vision\n"
        "layer: L1\n"
        'title: "Platform Vision"\n'
        "up_to: []\n"
        "---\n"
        "\n"
        "## Purpose\n"
        "Body content.\n"
    )
    fm, body = parse_frontmatter(src)
    assert fm["id"] == "L1-VIS-001"
    assert fm["title"] == "Platform Vision"
    assert fm["up_to"] == []
    assert "## Purpose" in body


def test_parse_no_frontmatter() -> None:
    fm, body = parse_frontmatter("# just markdown\n")
    assert fm == {}
    assert body == "# just markdown\n"


def test_parse_block_style_lists_supported() -> None:
    src = (
        "---\n"
        "id: L2-CAP-001\n"
        "type: capability\n"
        "layer: L2\n"
        'title: "X"\n'
        "up_to:\n"
        "  - L1-VIS-001\n"
        "tags:\n"
        "  - auth\n"
        "  - identity\n"
        "---\n"
        "body\n"
    )
    fm, body = parse_frontmatter(src)
    assert fm["up_to"] == ["L1-VIS-001"]
    assert fm["tags"] == ["auth", "identity"]
    assert body == "body\n"


def test_normalize_at_write_round_trip() -> None:
    """Read block-style → write canonical → re-parse — semantic equivalence."""
    src = (
        "---\n"
        "id: L2-CAP-001\n"
        "type: capability\n"
        "layer: L2\n"
        "title: User Profile\n"
        "up_to:\n"
        "  - L1-VIS-001\n"
        "tags: [identity]\n"
        "---\n"
        "## Body\n"
    )
    fm, body = parse_frontmatter(src)
    canonical = write_canonical(fm, body)
    assert "up_to: [L1-VIS-001]" in canonical
    assert 'title: "User Profile"' in canonical
    fm2, body2 = parse_frontmatter(canonical)
    assert fm2 == fm
    assert body2.strip() == body.strip()


# ── body handling ────────────────────────────────────────────────────────


def test_body_preserved() -> None:
    fm = {"id": "L1-VIS-001", "type": "vision", "layer": "L1", "title": "X", "up_to": []}
    out = write_canonical(fm, body="## Body\n\nText.\n")
    assert out.endswith("## Body\n\nText.\n")


def test_body_empty() -> None:
    fm = {"id": "L1-VIS-001", "type": "vision", "layer": "L1", "title": "X", "up_to": []}
    out = write_canonical(fm, body="")
    assert out.endswith("---\n\n")


# ── golden L1-L4 fixtures (from docs/methodology.md) ───────────────────


def test_golden_l1_vision() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "SaaS Content Platform — Vision",
        "up_to": [],
        "tags": ["platform-meta"],
        "status": "approved",
        "provenance": {
            "kind": "doc",
            "ref": "Site_Matrix_Business_Req.pdf.md",
            "confidence": 0.95,
        },
    }
    out = write_canonical(fm, body="")
    expected = (
        "---\n"
        "id: L1-VIS-001\n"
        "type: vision\n"
        "layer: L1\n"
        'title: "SaaS Content Platform — Vision"\n'
        "up_to: []\n"
        "tags: [platform-meta]\n"
        "status: approved\n"
        'provenance: {kind: doc, ref: "Site_Matrix_Business_Req.pdf.md", confidence: 1.0}\n'  # 0.95 stays
        "---\n\n"
    )
    # confidence 0.95 — but _format_number for non-int floats keeps repr
    expected = expected.replace("confidence: 1.0", "confidence: 0.95")
    assert out == expected


def test_golden_l2_capability() -> None:
    fm = {
        "id": "L2-CAP-007",
        "type": "capability",
        "layer": "L2",
        "title": "User Profile Management",
        "up_to": ["L1-VIS-001"],
        "tags": ["user-management", "identity"],
        "status": "draft",
        "relates_to": [{"id": "L2-CAP-002", "kind": "depends_on", "reason": "needs session"}],
    }
    out = write_canonical(fm, body="")
    assert "up_to: [L1-VIS-001]\n" in out
    assert "tags: [user-management, identity]\n" in out
    assert 'relates_to: [{id: L2-CAP-002, kind: depends_on, reason: "needs session"}]\n' in out


def test_golden_l3_behavior() -> None:
    fm = {
        "id": "L3-BHV-008",
        "type": "behavior",
        "layer": "L3",
        "title": "Edit User Profile",
        "up_to": ["L2-CAP-007"],
        "down_to": ["L4-API-012", "L4-EVT-005"],
        "tags": ["user-management"],
        "status": "draft",
    }
    out = write_canonical(fm, body="## Acceptance Criteria\n")
    assert "up_to: [L2-CAP-007]\n" in out
    assert "down_to: [L4-API-012, L4-EVT-005]\n" in out
    assert out.endswith("## Acceptance Criteria\n")


def test_parse_non_dict_yaml_returns_empty() -> None:
    src = "---\n- just_a_list\n- of_items\n---\nbody\n"
    fm, body = parse_frontmatter(src)
    assert fm == {}
    assert body == "body\n"


def test_write_with_no_fields() -> None:
    out = write_canonical({}, body="just body\n")
    assert out == "---\n---\n\njust body\n"


def test_value_none_renders_null() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "note": None,
    }
    assert "note: null\n" in write_canonical(fm, body="")


def test_value_bool_renders_lowercase() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "approved": True,
        "deprecated": False,
    }
    out = write_canonical(fm, body="")
    assert "approved: true\n" in out
    assert "deprecated: false\n" in out


def test_iso_datetime_bare() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "modified_at": "2026-05-14T12:30:00Z",
    }
    out = write_canonical(fm, body="")
    assert "modified_at: 2026-05-14T12:30:00Z\n" in out


def test_whitespace_padded_string_quoted() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "note": "  trailing  ",
    }
    out = write_canonical(fm, body="")
    assert 'note: "  trailing  "\n' in out


def test_empty_mapping_renders_braces() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "metadata": {},
    }
    out = write_canonical(fm, body="")
    assert "metadata: {}\n" in out


def test_int_value_bare() -> None:
    fm = {
        "id": "L1-VIS-001",
        "type": "vision",
        "layer": "L1",
        "title": "X",
        "up_to": [],
        "iteration": 7,
    }
    out = write_canonical(fm, body="")
    assert "iteration: 7\n" in out


def test_golden_l4_api_contract() -> None:
    fm = {
        "id": "L4-API-012",
        "type": "api-contract",
        "layer": "L4",
        "title": "PATCH /api/v1/profiles/{id}",
        "up_to": ["L3-BHV-008"],
        "down_to": [],
        "implemented_by": [
            "src/Http/Controllers/Api/V1/ProfileController.php:42-87",
            "src/Services/ProfileUpdateService.php:15-60",
        ],
        "method": "PATCH",
        "path": "/api/v1/profiles/{id}",
        "auth": "authenticated",
        "tags": ["user-management", "http-api"],
        "status": "draft",
    }
    out = write_canonical(
        fm,
        body="## Purpose\n",
        methodology_required_order=["method", "path", "auth"],
    )
    # field order: id, type, layer, title, up_to, down_to, implemented_by,
    # method, path, auth, tags, status
    expected_order_substring = (
        'title: "PATCH /api/v1/profiles/{id}"\n'
        "up_to: [L3-BHV-008]\n"
        "down_to: []\n"
        'implemented_by: ["src/Http/Controllers/Api/V1/ProfileController.php:42-87", '
        '"src/Services/ProfileUpdateService.php:15-60"]\n'
        "method: PATCH\n"
        'path: "/api/v1/profiles/{id}"\n'
        "auth: authenticated\n"
        "tags: [user-management, http-api]\n"
        "status: draft\n"
    )
    assert expected_order_substring in out
