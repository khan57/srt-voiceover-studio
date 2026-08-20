"""Tests for SRT parsing, encoding detection, and text sanitation."""

from __future__ import annotations

import pytest

from core.errors import SubtitleError
from core.parser import clean_text, devanagari_ratio, parse_srt

HINDI_SRT = """1
00:00:00,500 --> 00:00:04,000
नमस्ते दोस्तों, आज हम बात करेंगे।

2
00:00:05,000 --> 00:00:09,500
यह टूल सबटाइटल को आवाज़ में बदलता है।
"""


def write(tmp_path, name, text, encoding="utf-8"):
    """Write a subtitle fixture and return its path."""
    path = tmp_path / name
    path.write_text(text, encoding=encoding)
    return path


class TestCleanText:
    def test_strips_html_tags(self):
        assert clean_text("<i>Hello</i> <b>world</b>") == "Hello world"

    def test_strips_ass_overrides(self):
        assert clean_text(r"{\an8}Positioned line") == "Positioned line"

    def test_joins_wrapped_lines_with_a_space(self):
        assert clean_text("first half\nsecond half") == "first half second half"

    def test_collapses_whitespace_runs(self):
        assert clean_text("too    many\t\tspaces") == "too many spaces"

    def test_preserves_devanagari(self):
        text = "नमस्ते दुनिया"
        assert clean_text(f"<i>{text}</i>") == text

    def test_markup_only_cue_becomes_empty(self):
        assert clean_text(r"{\an8}<i></i>   ") == ""


class TestParseSrt:
    def test_parses_hindi_utf8(self, tmp_path):
        report = parse_srt(write(tmp_path, "hi.srt", HINDI_SRT))
        assert len(report.cues) == 2
        assert report.cues[0].text.startswith("नमस्ते")
        assert report.cues[0].start_ms == 500
        assert report.cues[0].end_ms == 4000
        assert report.cues[0].duration_ms == 3500

    def test_handles_utf8_bom(self, tmp_path):
        report = parse_srt(write(tmp_path, "bom.srt", HINDI_SRT, encoding="utf-8-sig"))
        assert len(report.cues) == 2
        # A mishandled BOM would leave a stray ﻿ glued to the first word.
        assert "﻿" not in report.cues[0].text

    def test_total_duration_is_last_end(self, tmp_path):
        report = parse_srt(write(tmp_path, "hi.srt", HINDI_SRT))
        assert report.total_duration_ms == 9500

    def test_drops_empty_cues_and_counts_them(self, tmp_path):
        srt = (
            "1\n00:00:00,000 --> 00:00:02,000\n♪\n\n"
            "2\n00:00:03,000 --> 00:00:05,000\nReal line.\n"
        ).replace("♪", r"{\an8}")
        report = parse_srt(write(tmp_path, "sparse.srt", srt))
        assert len(report.cues) == 1
        assert report.skipped_empty == 1
        assert any("Skipped 1" in w for w in report.warnings)

    def test_sorts_out_of_order_cues(self, tmp_path):
        srt = (
            "1\n00:00:10,000 --> 00:00:12,000\nSecond.\n\n"
            "2\n00:00:01,000 --> 00:00:03,000\nFirst.\n"
        )
        report = parse_srt(write(tmp_path, "unordered.srt", srt))
        assert [c.text for c in report.cues] == ["First.", "Second."]

    def test_warns_about_overlapping_slots(self, tmp_path):
        srt = (
            "1\n00:00:00,000 --> 00:00:05,000\nOne.\n\n"
            "2\n00:00:03,000 --> 00:00:07,000\nTwo.\n"
        )
        report = parse_srt(write(tmp_path, "overlap.srt", srt))
        assert any("overlap" in w for w in report.warnings)

    def test_rejects_zero_length_cue(self, tmp_path):
        srt = "1\n00:00:02,000 --> 00:00:02,000\nInstant.\n"
        with pytest.raises(SubtitleError, match="ends at or before it starts"):
            parse_srt(write(tmp_path, "zero.srt", srt))

    def test_rejects_reversed_timecodes(self, tmp_path):
        srt = "1\n00:00:05,000 --> 00:00:02,000\nBackwards.\n"
        with pytest.raises(SubtitleError, match="ends at or before it starts"):
            parse_srt(write(tmp_path, "reversed.srt", srt))

    def test_rejects_empty_file(self, tmp_path):
        with pytest.raises(SubtitleError, match="is empty"):
            parse_srt(write(tmp_path, "empty.srt", ""))

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(SubtitleError, match="not found"):
            parse_srt(tmp_path / "nope.srt")

    def test_rejects_file_with_only_empty_cues(self, tmp_path):
        srt = "1\n00:00:00,000 --> 00:00:02,000\n<i></i>\n"
        with pytest.raises(SubtitleError, match="no speakable text"):
            parse_srt(write(tmp_path, "blank.srt", srt))


class TestScriptDetection:
    def test_pure_devanagari(self):
        assert devanagari_ratio("नमस्ते दुनिया") == 1.0

    def test_pure_latin(self):
        assert devanagari_ratio("Hello world") == 0.0

    def test_ignores_digits_and_punctuation(self):
        # "2024!" contributes no letters, so it must not dilute the ratio.
        assert devanagari_ratio("नमस्ते 2024!") == 1.0

    def test_empty_and_symbol_only_text_is_not_devanagari(self):
        assert devanagari_ratio("") == 0.0
        assert devanagari_ratio("123 -- !!") == 0.0

    def test_mixed_script_reports_a_fraction(self):
        ratio = devanagari_ratio("नमस्ते hello")
        assert 0.0 < ratio < 1.0

    def test_report_flags_a_hindi_file(self, tmp_path):
        report = parse_srt(write(tmp_path, "hi.srt", HINDI_SRT))
        assert report.is_devanagari
        assert report.devanagari_ratio == pytest.approx(1.0)

    def test_report_does_not_flag_an_english_file(self, tmp_path):
        srt = "1\n00:00:00,000 --> 00:00:02,000\nHello there, friend.\n"
        report = parse_srt(write(tmp_path, "en.srt", srt))
        assert not report.is_devanagari

    def test_transliterated_hindi_counts_as_latin(self, tmp_path):
        # "Namaste doston" is Hindi, but an English voice can read it, so it
        # must not trip the Devanagari guard.
        srt = "1\n00:00:00,000 --> 00:00:02,000\nNamaste doston, kaise ho.\n"
        assert not parse_srt(write(tmp_path, "roman.srt", srt)).is_devanagari
