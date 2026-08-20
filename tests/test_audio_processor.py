"""Tests for tempo fitting, timeline placement, and the ducking envelope.

These run entirely offline: narration is stood in for by locally generated
tones, so nothing here touches the Edge-TTS service.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from core import audio_processor as ap
from core.config import EXPORT_SAMPLE_RATE, VOICE_TARGET_DBFS
from core.parser import Cue
from core.tts import SynthesisResult, format_pitch, format_rate

ap.ensure_ffmpeg()


def tone(duration_ms: int, freq: int = 440, gain_db: float = -6.0) -> AudioSegment:
    """Generate a stereo test tone of a given length."""
    return (
        Sine(freq, sample_rate=EXPORT_SAMPLE_RATE)
        .to_audio_segment(duration=duration_ms)
        .apply_gain(gain_db)
        .set_channels(2)
    )


def make_results(tmp_path: Path, cues, lengths_ms):
    """Render a stand-in clip per cue and wrap them as synthesis results."""
    results = []
    for cue, length in zip(cues, lengths_ms):
        if length is None:
            results.append(SynthesisResult(cue=cue, audio_path=None, error="stubbed failure"))
            continue
        path = tmp_path / f"cue_{cue.index}.wav"
        tone(length).export(path, format="wav")
        results.append(SynthesisResult(cue=cue, audio_path=path))
    return results


# ==========================================================================
# Edge-TTS parameter formatting
# ==========================================================================

class TestParameterFormatting:
    @pytest.mark.parametrize(
        "value,expected", [(0, "+0%"), (15, "+15%"), (-30, "-30%"), (7.4, "+7%")]
    )
    def test_rate_is_always_signed(self, value, expected):
        # Edge-TTS rejects unsigned values, so "0%" would be a hard failure.
        assert format_rate(value) == expected

    @pytest.mark.parametrize(
        "value,expected", [(0, "+0Hz"), (5, "+5Hz"), (-20, "-20Hz")]
    )
    def test_pitch_is_always_signed(self, value, expected):
        assert format_pitch(value) == expected


# ==========================================================================
# atempo
# ==========================================================================

class TestAtempoChain:
    def test_single_stage_within_limits(self):
        assert ap.atempo_chain(1.5) == pytest.approx([1.5])

    def test_chains_above_two(self):
        stages = ap.atempo_chain(3.0)
        assert len(stages) == 2
        assert math.prod(stages) == pytest.approx(3.0)
        assert all(s <= 2.0 for s in stages)

    def test_chains_far_above_two(self):
        stages = ap.atempo_chain(7.5)
        assert math.prod(stages) == pytest.approx(7.5)
        assert all(0.5 <= s <= 2.0 for s in stages)

    def test_chains_below_half(self):
        stages = ap.atempo_chain(0.25)
        assert math.prod(stages) == pytest.approx(0.25)
        assert all(s >= 0.5 for s in stages)


class TestFitToSlot:
    def test_leaves_short_clip_alone(self, tmp_path):
        clip = tone(1_000)
        fitted = ap.fit_to_slot(clip, budget_ms=3_000, max_tempo=1.5, workdir=tmp_path)
        assert fitted.tempo == 1.0
        assert not fitted.was_compressed
        assert len(fitted.audio) == len(clip)

    def test_ignores_overrun_within_tolerance(self, tmp_path):
        # 30 ms over a 2 s slot is inaudible; compressing it is not worth it.
        fitted = ap.fit_to_slot(tone(2_030), budget_ms=2_000, max_tempo=1.5, workdir=tmp_path)
        assert fitted.tempo == 1.0

    def test_compresses_to_fit(self, tmp_path):
        fitted = ap.fit_to_slot(tone(3_000), budget_ms=2_000, max_tempo=2.0, workdir=tmp_path)
        assert fitted.was_compressed
        assert fitted.tempo == pytest.approx(1.5, abs=0.01)
        assert len(fitted.audio) <= 2_000 + 60
        assert fitted.overflow_ms == 0

    def test_respects_the_tempo_cap(self, tmp_path):
        # Needs 3x but is only allowed 1.5x, so it must still overrun.
        fitted = ap.fit_to_slot(tone(3_000), budget_ms=1_000, max_tempo=1.5, workdir=tmp_path)
        assert fitted.tempo == pytest.approx(1.5, abs=0.01)
        assert len(fitted.audio) == pytest.approx(2_000, abs=80)
        assert fitted.overflow_ms > 500

    def test_preserves_pitch(self, tmp_path):
        """atempo must time-stretch, not resample -- the tone stays at 440 Hz."""
        fitted = ap.fit_to_slot(tone(3_000, freq=440), 2_000, 2.0, tmp_path)
        frames, _ = ap._to_float_frames(fitted.audio)
        mono = frames[:, 0]
        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        peak_hz = np.fft.rfftfreq(len(mono), 1 / fitted.audio.frame_rate)[np.argmax(spectrum)]
        # Naive frame-rate resampling would have shifted this to ~660 Hz.
        assert peak_hz == pytest.approx(440, abs=5)


# ==========================================================================
# Timeline assembly
# ==========================================================================

class TestBuildVoiceTrack:
    def test_places_clips_at_absolute_timestamps(self, tmp_path):
        cues = [
            Cue(index=1, start_ms=1_000, end_ms=3_000, text="one"),
            Cue(index=2, start_ms=5_000, end_ms=7_000, text="two"),
        ]
        results = make_results(tmp_path, cues, [1_500, 1_500])
        track = ap.build_voice_track(cues, results, 1.5, tmp_path, normalise=False)

        assert track.intervals == [(1_000, 2_500), (5_000, 6_500)]
        # Gaps must be silent; the slots must not be.
        assert track.audio[0:900].dBFS == float("-inf")
        assert track.audio[3_100:4_900].dBFS == float("-inf")
        assert track.audio[1_100:2_400].dBFS > -40

    def test_total_length_matches_the_subtitle_timeline(self, tmp_path):
        cues = [
            Cue(index=1, start_ms=0, end_ms=2_000, text="a"),
            Cue(index=2, start_ms=4_000, end_ms=8_000, text="b"),
        ]
        results = make_results(tmp_path, cues, [1_000, 1_000])
        track = ap.build_voice_track(cues, results, 1.5, tmp_path, normalise=False)
        assert len(track.audio) == 8_000

    def test_no_cumulative_drift_when_lines_overrun(self, tmp_path):
        """Ten lines that each blow past their slot must not shift line ten."""
        cues = [
            Cue(index=i, start_ms=i * 2_000, end_ms=i * 2_000 + 1_000, text=f"line {i}")
            for i in range(10)
        ]
        # Every clip is 1.8 s in a 1.0 s slot, and the 1.2x cap cannot fix that.
        results = make_results(tmp_path, cues, [1_800] * 10)
        track = ap.build_voice_track(cues, results, 1.2, tmp_path, normalise=False)

        assert [start for start, _ in track.intervals] == [i * 2_000 for i in range(10)]
        assert track.compressed_lines == 10
        assert track.overflowing_lines == 10
        assert any("compression limit" in w for w in track.warnings)

    def test_failed_line_becomes_silence_without_aborting(self, tmp_path):
        cues = [
            Cue(index=1, start_ms=0, end_ms=2_000, text="ok"),
            Cue(index=2, start_ms=3_000, end_ms=5_000, text="broken"),
        ]
        results = make_results(tmp_path, cues, [1_500, None])
        track = ap.build_voice_track(cues, results, 1.5, tmp_path, normalise=False)

        assert track.failed_lines == 1
        assert len(track.intervals) == 1
        assert track.audio[3_000:5_000].dBFS == float("-inf")

    def test_normalisation_targets_speech_not_silence(self, tmp_path):
        """Loudness is measured over the speech only, ignoring the long gaps.

        The speech here occupies one second of a ten-second timeline. Measuring
        the whole canvas would report a near-silent level and provoke a massive
        boost; measuring the speech alone lands it on the target instead.
        """
        cues = [Cue(index=1, start_ms=0, end_ms=1_000, text="x")]
        results = make_results(tmp_path, cues, [1_000])
        cues.append(Cue(index=2, start_ms=9_000, end_ms=10_000, text="y"))
        results.extend(make_results(tmp_path, cues[1:], [1_000]))

        track = ap.build_voice_track(cues, results, 1.5, tmp_path, normalise=True)
        assert len(track.audio) == 10_000
        assert track.audio[0:1_000].dBFS == pytest.approx(VOICE_TARGET_DBFS, abs=1.0)

    def test_normalisation_gain_is_clamped(self, tmp_path):
        """A pathologically quiet render is lifted, but never without limit."""
        cues = [Cue(index=1, start_ms=0, end_ms=2_000, text="x")]
        path = tmp_path / "faint.wav"
        tone(2_000, gain_db=-60.0).export(path, format="wav")
        results = [SynthesisResult(cue=cues[0], audio_path=path)]

        track = ap.build_voice_track(cues, results, 1.5, tmp_path, normalise=True)
        # Source RMS is about -63 dBFS; reaching -16 would need +47 dB, but the
        # clamp allows only +12, so it must land near -51 dBFS.
        assert track.audio[200:1_800].dBFS == pytest.approx(-51.0, abs=1.5)


# ==========================================================================
# Ducking
# ==========================================================================

class TestMergeIntervals:
    def test_merges_close_neighbours(self):
        merged = ap.merge_intervals([(0, 1_000), (1_100, 2_000)], merge_gap_ms=500)
        assert merged == [(0, 2_000)]

    def test_keeps_distant_regions_separate(self):
        merged = ap.merge_intervals([(0, 1_000), (5_000, 6_000)], merge_gap_ms=500)
        assert merged == [(0, 1_000), (5_000, 6_000)]

    def test_sorts_and_absorbs_nested_regions(self):
        merged = ap.merge_intervals([(2_000, 3_000), (0, 5_000)], merge_gap_ms=100)
        assert merged == [(0, 5_000)]

    def test_handles_no_intervals(self):
        assert ap.merge_intervals([], merge_gap_ms=500) == []


class TestDuckEnvelope:
    def _envelope(self, intervals, total_ms=10_000, duck_db=-12.0, fade_ms=300, hold_ms=200):
        rate = 1_000  # 1 frame per ms keeps index maths readable in assertions
        return ap.build_duck_envelope(
            intervals, total_ms, duck_db, fade_ms, hold_ms, rate, total_ms
        )

    def test_full_level_when_nothing_is_speaking(self):
        assert np.allclose(self._envelope([]), 1.0)

    def test_dips_to_the_ducked_level_during_speech(self):
        env = self._envelope([(3_000, 6_000)])
        assert env[4_500] == pytest.approx(10 ** (-12.0 / 20.0), abs=1e-6)

    def test_restores_full_level_between_regions(self):
        env = self._envelope([(1_000, 2_000), (7_000, 8_000)])
        assert env[4_500] == pytest.approx(1.0, abs=1e-6)

    def test_ramp_is_complete_before_the_first_syllable(self):
        env = self._envelope([(3_000, 6_000)], fade_ms=300)
        assert env[2_699] == pytest.approx(1.0, abs=1e-6)   # before the ramp
        assert env[3_000] == pytest.approx(10 ** (-12 / 20), abs=1e-6)  # fully ducked
        assert 0.25 < env[2_850] < 1.0                       # mid-ramp

    def test_holds_before_releasing(self):
        env = self._envelope([(3_000, 6_000)], fade_ms=300, hold_ms=200)
        assert env[6_100] == pytest.approx(10 ** (-12 / 20), abs=1e-6)  # still held
        assert env[6_500] == pytest.approx(1.0, abs=1e-6)               # released

    def test_envelope_is_monotonic_safe_with_adjacent_lines(self):
        # Back-to-back lines would otherwise produce overlapping ramps.
        intervals = [(i * 1_000, i * 1_000 + 900) for i in range(9)]
        env = self._envelope(intervals)
        assert np.all(env > 0) and np.all(env <= 1.0 + 1e-9)

    def test_zero_depth_is_a_no_op(self):
        assert np.allclose(self._envelope([(1_000, 2_000)], duck_db=0.0), 1.0)


class TestApplyDucking:
    def test_music_is_quieter_under_speech_than_in_the_gaps(self, tmp_path):
        bgm = tone(10_000, freq=220, gain_db=-3.0)
        ducked = ap.apply_ducking(
            bgm, [(3_000, 6_000)], duck_db=-12.0, fade_ms=300, hold_ms=200
        )
        under_speech = ducked[4_000:5_000].dBFS
        in_the_gap = ducked[8_000:9_000].dBFS

        assert in_the_gap - under_speech == pytest.approx(12.0, abs=0.5)
        assert len(ducked) == len(bgm)

    def test_no_speech_leaves_music_untouched(self, tmp_path):
        bgm = tone(4_000, freq=220)
        ducked = ap.apply_ducking(bgm, [], duck_db=-12.0, fade_ms=300, hold_ms=200)
        assert ducked.dBFS == pytest.approx(bgm.dBFS, abs=0.1)


class TestPrepareBgm:
    def test_loops_short_music_to_fill_the_timeline(self, tmp_path):
        path = tmp_path / "short.wav"
        tone(2_000, freq=330).export(path, format="wav")
        result = ap.prepare_bgm(path, target_ms=9_000, base_db=-18.0)

        assert len(result.audio) == 9_000
        assert result.loops > 1
        assert result.source_ms == 2_000

    def test_trims_long_music_to_the_timeline(self, tmp_path):
        path = tmp_path / "long.wav"
        tone(20_000, freq=330).export(path, format="wav")
        result = ap.prepare_bgm(path, target_ms=5_000, base_db=-18.0)

        assert len(result.audio) == 5_000
        assert result.loops == 1

    def test_levels_music_to_the_requested_base(self, tmp_path):
        path = tmp_path / "hot.wav"
        tone(6_000, freq=330, gain_db=-0.5).export(path, format="wav")
        result = ap.prepare_bgm(path, target_ms=6_000, base_db=-18.0)
        # Measured away from the 1 s edge fades.
        assert result.audio[2_000:4_000].max_dBFS == pytest.approx(-18.0, abs=1.0)

    def test_rejects_unusably_short_music(self, tmp_path):
        path = tmp_path / "blip.wav"
        tone(120).export(path, format="wav")
        with pytest.raises(ap.AudioProcessingError, match="only"):
            ap.prepare_bgm(path, target_ms=5_000, base_db=-18.0)

    def test_rejects_a_file_that_is_not_audio(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("this is not audio")
        with pytest.raises(ap.AudioProcessingError, match="Could not read"):
            ap.prepare_bgm(path, target_ms=5_000, base_db=-18.0)


# ==========================================================================
# Mastering
# ==========================================================================

class TestMixMaster:
    def test_bypasses_mixing_when_there_is_no_music(self):
        voice = tone(3_000)
        assert ap.mix_master(voice, None).dBFS == pytest.approx(voice.dBFS, abs=0.1)

    def test_pulls_the_peak_back_to_the_ceiling(self):
        hot = tone(2_000, gain_db=0.0)
        master = ap.mix_master(hot, tone(2_000, freq=220, gain_db=0.0))
        assert master.max_dBFS == pytest.approx(-1.0, abs=0.2)

    def test_exports_a_playable_mp3(self, tmp_path):
        destination = ap.export_master(tone(2_000), tmp_path / "out.mp3")
        assert destination.exists() and destination.stat().st_size > 0
        # Round-trip it to prove the encode is valid, not just non-empty.
        assert len(AudioSegment.from_file(destination)) == pytest.approx(2_000, abs=120)
