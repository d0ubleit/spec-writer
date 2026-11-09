"""S1.4: slug generation per docs/methodology.md"""

from __future__ import annotations

from spec_writer.io.slug import make_slug


def test_ascii_kebab() -> None:
    assert make_slug("User Profile Management") == "user-profile-management"


def test_lowercase() -> None:
    assert make_slug("API CONTRACT") == "api-contract"


def test_unicode_transliterated() -> None:
    # unidecode chooses two-letter romanization for ё.
    assert make_slug("ёлка") == "iolka"
    assert make_slug("Café Münster") == "cafe-munster"
    assert make_slug("Привет мир") == "privet-mir"


def test_punctuation_normalised() -> None:
    assert make_slug("hello, world!") == "hello-world"


def test_multiple_dashes_collapsed() -> None:
    assert make_slug("a---b__c") == "a-b-c"


def test_trim_leading_trailing_dashes() -> None:
    assert make_slug("  -- hello -- ") == "hello"


def test_truncation_appends_hash_suffix() -> None:
    title = "x" * 200
    slug = make_slug(title, max_length=60)
    assert len(slug) == 60
    assert slug[-7] == "-"  # `-<hash6>`
    # deterministic — same input yields same suffix
    assert make_slug(title, max_length=60) == slug


def test_truncation_only_when_needed() -> None:
    title = "short title"
    slug = make_slug(title, max_length=60)
    assert slug == "short-title"


def test_deduplication_appends_number() -> None:
    existing = {"user-profile", "user-profile-1"}
    assert make_slug("User Profile", existing_slugs=existing) == "user-profile-2"


def test_deduplication_first_collision() -> None:
    assert make_slug("Foo", existing_slugs={"foo"}) == "foo-1"


def test_no_collision_returns_base() -> None:
    assert make_slug("New Idea", existing_slugs={"unrelated"}) == "new-idea"


def test_empty_title_yields_empty_or_safe() -> None:
    # Don't crash on degenerate input — return something stable.
    assert make_slug("") == ""
    assert make_slug("---") == ""
