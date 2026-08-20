"""Tests for the free-music search, download guards, and attribution.

The licence tests are the important ones: mixing music under narration creates
a commercial derivative work, so a NonCommercial or NoDerivatives track is
legally unusable no matter how well it fits. Everything else here guards the
handling of untrusted data from a third-party API.
"""

from __future__ import annotations

import pytest

from core.errors import StudioError
from core.music_library import (
    ALLOWED_LICENCES,
    MAX_PAGE_SIZE,
    Track,
    _broadening_variants,
    _extension_for,
    _parse_track,
    _safe_stem,
    _sanitise,
    build_credits,
    search_music,
    write_credits,
)


def api_result(**overrides):
    """A realistic Openverse result payload, overridable per test."""
    payload = {
        "id": "abc-123",
        "title": "Ambient Flight",
        "creator": "Zeropage",
        "url": "https://prod-1.storage.jamendo.com/?trackid=20237&format=mp32",
        "foreign_landing_url": "https://www.jamendo.com/track/20237",
        "license": "by",
        "license_version": "3.0",
        "license_url": "https://creativecommons.org/licenses/by/3.0/",
        "duration": 264000,
        "source": "jamendo",
        "attribution": '"Ambient Flight" by Zeropage is licensed under CC BY 3.0.',
    }
    payload.update(overrides)
    return payload


class TestLicenceFiltering:
    """No track that forbids remixing or commercial use may ever surface."""

    @pytest.mark.parametrize("licence", ["cc0", "by"])
    def test_permitted_licences_are_accepted(self, licence):
        assert _parse_track(api_result(license=licence)) is not None

    @pytest.mark.parametrize(
        "licence",
        ["by-nc", "by-nd", "by-nc-nd", "by-nc-sa", "by-sa", "sampling+", "", "unknown"],
    )
    def test_forbidden_licences_are_rejected(self, licence):
        # by-nc / by-nd forbid this use outright; by-sa would virally impose
        # its terms on the user's finished video.
        assert _parse_track(api_result(license=licence)) is None

    def test_licence_matching_is_case_insensitive(self):
        assert _parse_track(api_result(license="CC0")) is not None

    def test_allowed_set_is_exactly_cc0_and_by(self):
        assert ALLOWED_LICENCES == {"cc0", "by"}


class TestUrlSafety:
    @pytest.mark.parametrize(
        "url",
        [
            "http://insecure.example/track.mp3",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "ftp://example.com/a.mp3",
            "",
        ],
    )
    def test_non_https_urls_are_rejected(self, url):
        assert _parse_track(api_result(url=url)) is None

    def test_https_url_is_accepted(self):
        assert _parse_track(api_result(url="https://cdn.example/a.mp3")) is not None


class TestSanitisation:
    """Titles and creators are attacker-controlled text from a public API."""

    @pytest.mark.parametrize(
        "raw",
        [
            "Title | with | pipes",
            "Title\nwith\nnewlines",
            "<script>alert(1)</script>",
            "**bold** and `code`",
            "[link](http://evil.example)",
        ],
    )
    def test_markup_and_table_breakers_are_stripped(self, raw):
        cleaned = _sanitise(raw)
        assert not any(ch in cleaned for ch in "|<>[]`*_\n\r")

    def test_empty_input_becomes_a_placeholder(self):
        assert _sanitise("") == "Untitled"
        assert _sanitise(None) == "Untitled"

    def test_long_titles_are_truncated(self):
        assert len(_sanitise("x" * 500, limit=40)) <= 40

    def test_ordinary_unicode_survives(self):
        assert _sanitise("Antigone et Créon") == "Antigone et Créon"


class TestFilenameSafety:
    @pytest.mark.parametrize(
        "identifier,expected",
        [
            ("../../../etc/passwd", "etcpasswd"),
            ("a/b/c", "abc"),
            ("", "track"),
            ("....", "track"),
            ("good-id_123", "good-id_123"),
        ],
    )
    def test_identifiers_cannot_escape_the_directory(self, identifier, expected):
        assert _safe_stem(Track(identifier, "T", "C", "https://x", "", "by", "4.0", "", 0, "s", "")) == expected

    def test_extension_follows_content_type(self):
        track = Track("i", "T", "C", "https://x/y", "", "cc0", "1.0", "", 0, "s", "")
        assert _extension_for("audio/mpeg; charset=utf-8", track) == ".mp3"
        assert _extension_for("audio/x-wav", track) == ".wav"

    def test_extension_falls_back_to_the_url(self):
        track = Track("i", "T", "C", "https://x/y.flac?token=1", "", "cc0", "1.0", "", 0, "s", "")
        assert _extension_for("application/octet-stream", track) == ".flac"


class TestQueryBroadening:
    """Openverse ANDs every term, so precise phrases match nothing."""

    def test_three_words_fall_back_twice(self):
        assert _broadening_variants("calm cinematic piano") == [
            "calm cinematic piano", "cinematic piano", "cinematic",
        ]

    def test_two_words_fall_back_once(self):
        assert _broadening_variants("cinematic piano") == ["cinematic piano", "cinematic"]

    def test_single_word_has_no_fallback(self):
        assert _broadening_variants("piano") == ["piano"]

    def test_variants_are_deduplicated(self):
        assert len(_broadening_variants("piano piano")) == len(
            set(v.casefold() for v in _broadening_variants("piano piano"))
        )


class TestTrackPresentation:
    def test_cc0_needs_no_attribution(self):
        track = _parse_track(api_result(license="cc0", license_version="1.0"))
        assert not track.requires_attribution

    def test_by_requires_attribution(self):
        assert _parse_track(api_result()).requires_attribution

    def test_duration_is_shown_as_minutes_and_seconds(self):
        assert _parse_track(api_result(duration=264000)).duration_label == "4:24"
        assert _parse_track(api_result(duration=59000)).duration_label == "0:59"

    def test_missing_duration_degrades_gracefully(self):
        assert _parse_track(api_result(duration=None)).duration_label == "?"

    def test_choice_label_states_the_credit_obligation(self):
        assert "credit required" in _parse_track(api_result()).choice_label
        assert "no credit needed" in _parse_track(api_result(license="cc0")).choice_label

    def test_credit_line_falls_back_when_api_omits_attribution(self):
        line = _parse_track(api_result(attribution="")).credit_line()
        assert "Ambient Flight" in line and "Zeropage" in line


class TestCredits:
    def test_by_credits_state_the_obligation(self):
        text = build_credits(_parse_track(api_result()), "narration.mp3")
        assert "MUST reproduce this credit" in text
        assert "Zeropage" in text
        assert "creativecommons.org/licenses/by/3.0/" in text
        assert "narration.mp3" in text

    def test_cc0_credits_say_none_is_required(self):
        text = build_credits(_parse_track(api_result(license="cc0")), "narration.mp3")
        assert "no credit is legally required" in text

    def test_write_credits_lands_beside_the_audio(self, tmp_path):
        narration = tmp_path / "narration_123.mp3"
        narration.write_bytes(b"not really audio")
        written = write_credits(_parse_track(api_result()), narration)

        assert written.name == "narration_123.credits.txt"
        assert written.parent == narration.parent
        assert "Zeropage" in written.read_text(encoding="utf-8")


class TestSearchGuards:
    @pytest.mark.parametrize("query", ["", "   ", None])
    def test_blank_query_is_rejected_before_any_request(self, query):
        with pytest.raises(StudioError, match="Type something to search"):
            search_music(query)

    def test_page_size_cap_matches_the_anonymous_limit(self):
        assert MAX_PAGE_SIZE == 20


@pytest.mark.network
class TestAgainstLiveApi:
    def test_search_returns_only_permitted_licences(self):
        result = search_music("piano")
        assert result.tracks, "expected results for a broad query"
        assert {t.licence for t in result.tracks} <= ALLOWED_LICENCES

    def test_broadening_rescues_an_over_specific_query(self):
        result = search_music("calm cinematic piano")
        assert result.tracks
        assert result.was_broadened


class TestOptionalFieldPlaceholders:
    """An absent optional field must stay falsy, not become a visible label."""

    def test_missing_attribution_stays_empty(self):
        assert _parse_track(api_result(attribution="")).attribution == ""
        assert _parse_track(api_result(attribution=None)).attribution == ""

    def test_missing_title_still_gets_a_placeholder(self):
        assert _parse_track(api_result(title="")).title == "Untitled"


class TestPreviewConfiguration:
    """An audition clip must stay small enough to fetch in a couple of seconds."""

    def test_preview_is_a_small_fraction_of_the_download_ceiling(self):
        from core.music_library import MAX_DOWNLOAD_BYTES, PREVIEW_BYTES
        assert PREVIEW_BYTES < MAX_DOWNLOAD_BYTES / 10

    def test_preview_length_is_bounded(self):
        from core.music_library import PREVIEW_FADE_MS, PREVIEW_SECONDS
        assert 0 < PREVIEW_SECONDS <= 60
        assert 0 < PREVIEW_FADE_MS < PREVIEW_SECONDS * 1000


class TestCachePruning:
    def test_keeps_the_newest_and_evicts_the_rest(self, tmp_path):
        import os
        from core.music_library import prune_cache

        for i in range(10):
            f = tmp_path / f"f{i}.mp3"
            f.write_bytes(b"x")
            os.utime(f, (1_000 + i, 1_000 + i))  # deterministic ages

        assert prune_cache(tmp_path, max_files=4) == 6
        survivors = sorted(f.name for f in tmp_path.iterdir())
        assert survivors == ["f6.mp3", "f7.mp3", "f8.mp3", "f9.mp3"]

    def test_under_the_limit_nothing_is_removed(self, tmp_path):
        from core.music_library import prune_cache
        (tmp_path / "a.mp3").write_bytes(b"x")
        assert prune_cache(tmp_path, max_files=10) == 0
        assert (tmp_path / "a.mp3").exists()

    def test_missing_directory_is_harmless(self, tmp_path):
        from core.music_library import prune_cache
        assert prune_cache(tmp_path / "nope") == 0


class TestDownloadCaching:
    """Auditioning then using a track must not download it twice."""

    def test_a_cached_file_is_returned_without_a_request(self, tmp_path, monkeypatch):
        import httpx as _httpx
        from core.music_library import download_track

        track = _parse_track(api_result(id="cafe-1234"))
        cached = tmp_path / "bgm_cafe-1234.mp3"
        cached.write_bytes(b"already here")

        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("network was used despite a cache hit")

        monkeypatch.setattr(_httpx, "stream", explode)
        assert download_track(track, tmp_path) == cached

    def test_preview_and_full_downloads_do_not_collide(self, tmp_path, monkeypatch):
        import httpx as _httpx
        from core.music_library import download_track

        track = _parse_track(api_result(id="cafe-1234"))
        (tmp_path / "preview_cafe-1234.mp3").write_bytes(b"partial clip")

        def explode(*args, **kwargs):
            raise AssertionError("full download must not reuse the preview file")

        monkeypatch.setattr(_httpx, "stream", explode)
        # The preview is cached under a different prefix, so asking for the full
        # track must still attempt a real download rather than serving the clip.
        with pytest.raises(AssertionError, match="must not reuse"):
            download_track(track, tmp_path)

    def test_an_empty_cached_file_is_ignored(self, tmp_path, monkeypatch):
        import httpx as _httpx
        from core.music_library import download_track

        track = _parse_track(api_result(id="cafe-1234"))
        (tmp_path / "bgm_cafe-1234.mp3").write_bytes(b"")

        def explode(*args, **kwargs):
            raise AssertionError("re-download attempted")

        monkeypatch.setattr(_httpx, "stream", explode)
        with pytest.raises(AssertionError, match="re-download attempted"):
            download_track(track, tmp_path)


class TestPreviewCaching:
    def test_re_auditioning_never_touches_the_network(self, tmp_path, monkeypatch):
        """A track already heard must replay instantly, not re-download."""
        import httpx as _httpx
        from core.music_library import fetch_preview

        track = _parse_track(api_result(id="cafe-1234"))
        (tmp_path / "clip_cafe-1234.mp3").write_bytes(b"cached clip")

        def explode(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("network used despite a cached preview clip")

        monkeypatch.setattr(_httpx, "stream", explode)
        assert fetch_preview(track, tmp_path) == tmp_path / "clip_cafe-1234.mp3"


class TestPreviewDeadline:
    """Auditioning must never hang the UI on a slow host.

    httpx timeouts are per-operation, so a host that trickles bytes keeps
    resetting the read timer. Only a wall-clock budget bounds the total.
    """

    def test_deadline_is_far_shorter_than_the_full_download_timeout(self):
        from core.music_library import DOWNLOAD_TIMEOUT_S, PREVIEW_DEADLINE_S
        assert PREVIEW_DEADLINE_S < DOWNLOAD_TIMEOUT_S / 4

    def test_minimum_playable_size_is_below_the_preview_target(self):
        from core.music_library import MIN_PLAYABLE_BYTES, PREVIEW_BYTES
        assert 0 < MIN_PLAYABLE_BYTES < PREVIEW_BYTES

    def test_slow_trickle_yields_a_short_clip_instead_of_hanging(self, tmp_path, monkeypatch):
        import httpx as _httpx
        from core import music_library as ml

        class SlowResponse:
            status_code = 200
            headers = {"Content-Type": "audio/mpeg"}

            def iter_bytes(self, size):
                # Emits 64 KB per "second" of simulated clock, forever.
                while True:
                    yield b"\0" * (64 * 1024)

            def __enter__(self): return self
            def __exit__(self, *a): return False

        clock = {"t": 0.0}
        monkeypatch.setattr(ml.time, "monotonic", lambda: clock["t"])

        original = SlowResponse.iter_bytes
        def ticking(self, size):
            for chunk in original(self, size):
                clock["t"] += 1.0     # each chunk costs a second
                yield chunk
        monkeypatch.setattr(SlowResponse, "iter_bytes", ticking)
        monkeypatch.setattr(_httpx, "stream", lambda *a, **k: SlowResponse())

        track = _parse_track(api_result(id="slow-1"))
        out = ml.download_track(
            track, tmp_path, limit_bytes=10 * 1024 * 1024,
            prefix="preview", deadline_s=5.0,
        )
        # Stopped on the deadline with a usable amount, not on the byte limit.
        assert 0 < out.stat().st_size < 10 * 1024 * 1024
        assert out.stat().st_size >= ml.MIN_PLAYABLE_BYTES

    def test_deadline_without_partial_mode_is_an_error(self, tmp_path, monkeypatch):
        import httpx as _httpx
        from core import music_library as ml
        from core.errors import StudioError as SE

        class Trickle:
            status_code = 200
            headers = {"Content-Type": "audio/mpeg"}
            def iter_bytes(self, size):
                while True:
                    clock["t"] += 10.0
                    yield b"\0" * 1024
            def __enter__(self): return self
            def __exit__(self, *a): return False

        clock = {"t": 0.0}
        monkeypatch.setattr(ml.time, "monotonic", lambda: clock["t"])
        monkeypatch.setattr(_httpx, "stream", lambda *a, **k: Trickle())

        track = _parse_track(api_result(id="slow-2"))
        with pytest.raises(SE, match="too slowly"):
            ml.download_track(track, tmp_path, deadline_s=5.0)


class TestPreviewTimeoutPhases:
    def test_each_phase_is_bounded_well_inside_the_deadline(self):
        """Connect + read must not be able to stack past the wall-clock budget."""
        from core.music_library import PREVIEW_DEADLINE_S, PREVIEW_TIMEOUT
        assert PREVIEW_TIMEOUT.connect + PREVIEW_TIMEOUT.read <= PREVIEW_DEADLINE_S
