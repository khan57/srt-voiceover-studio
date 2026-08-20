"""Tests for the voice catalogue.

The offline tests guard the catalogue's internal consistency. The test marked
``network`` checks the roster against the live Edge-TTS service, which is the
only way to catch Microsoft retiring or renaming a voice; run it with
``pytest -m network``.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import (
    DEFAULT_VOICE_LABEL,
    VOICE_CATALOGUE,
    VOICES,
    VOICES_BY_LABEL,
    VoiceKind,
)
from core.tts import list_available_voices


class TestCatalogueIntegrity:
    def test_labels_are_unique(self):
        labels = [v.label for v in VOICE_CATALOGUE]
        assert len(labels) == len(set(labels))

    def test_short_names_are_unique(self):
        names = [v.short_name for v in VOICE_CATALOGUE]
        assert len(names) == len(set(names))

    def test_lookup_maps_agree_with_the_catalogue(self):
        assert len(VOICES) == len(VOICE_CATALOGUE)
        assert set(VOICES) == set(VOICES_BY_LABEL)
        for voice in VOICE_CATALOGUE:
            assert VOICES[voice.label] == voice.short_name

    def test_default_is_selectable(self):
        assert DEFAULT_VOICE_LABEL in VOICES

    def test_hindi_is_offered_first(self):
        # Hindi is the project's priority language; it must head the dropdown.
        assert VOICE_CATALOGUE[0].language == "Hindi"

    def test_both_native_hindi_voices_are_present(self):
        # The free endpoint offers exactly these two; if either disappears from
        # the catalogue, Hindi coverage has silently regressed.
        hindi = {v.short_name for v in VOICE_CATALOGUE if v.language == "Hindi" and v.is_native}
        assert hindi == {"hi-IN-MadhurNeural", "hi-IN-SwaraNeural"}

    def test_multilingual_labels_disclose_the_accent(self):
        """A non-native voice must say so in its own label.

        These voices render Devanagari with a foreign accent. Someone scanning
        the dropdown has to see that trade-off without consulting the docs.
        """
        for voice in VOICE_CATALOGUE:
            if voice.kind is VoiceKind.MULTILINGUAL:
                assert "non-native" in voice.label.lower(), (
                    f"{voice.short_name} is multilingual but its label hides that"
                )

    def test_native_flag_tracks_the_kind(self):
        for voice in VOICE_CATALOGUE:
            assert voice.is_native == (voice.kind is VoiceKind.NATIVE)


@pytest.mark.network
class TestAgainstLiveService:
    def test_every_catalogued_voice_still_exists(self):
        """Catches Microsoft retiring or renaming a voice we depend on."""
        available = set(asyncio.run(list_available_voices()))
        missing = [v.short_name for v in VOICE_CATALOGUE if v.short_name not in available]
        assert not missing, f"No longer offered by Edge-TTS: {missing}"


class TestScriptCapability:
    """Guards the Hindi-text-to-English-voice mismatch.

    An English-only voice given Devanagari returns no audio at all, which the
    pipeline would otherwise report as a network failure.
    """

    def test_every_hindi_option_can_speak_devanagari(self):
        for voice in VOICE_CATALOGUE:
            if voice.language == "Hindi":
                assert voice.speaks_devanagari, f"{voice.short_name} cannot speak Hindi"

    def test_english_only_voices_are_flagged_incapable(self):
        english_only = [
            v for v in VOICE_CATALOGUE
            if v.language.startswith("English") and v.kind is VoiceKind.NATIVE
        ]
        assert english_only, "expected some English-only voices in the catalogue"
        for voice in english_only:
            assert not voice.speaks_devanagari

    def test_there_are_more_hindi_options_than_native_ones(self):
        # The whole point of the multilingual entries: Microsoft ships two
        # native Hindi voices, and the roster should beat that.
        hindi = [v for v in VOICE_CATALOGUE if v.language == "Hindi"]
        assert len(hindi) > 2
        assert any("Male" in v.label for v in hindi if not v.is_native)
