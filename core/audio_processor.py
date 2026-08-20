"""Timeline assembly, tempo fitting, BGM ducking, and mastering.

This module owns every decision about *when* a sound happens and *how loud* it
is.  Two design choices drive the implementation:

**Absolute placement, not concatenation.**  Rather than appending clips and
silences end-to-end, we allocate one silent canvas the length of the whole
timeline and overlay each rendered line at its exact ``start_ms``.  Gaps are
therefore silent by construction, and a line that runs long cannot push every
subsequent line later -- cumulative drift is structurally impossible instead of
merely corrected for.

**Ducking driven by rendered speech, not subtitle blocks.**  A two-second line
sitting in a five-second slot gets its music back for the three-second tail,
which is what a human mix engineer would do.  The gain envelope is built once
as a NumPy array and applied in a single vectorised multiply, rather than by
chaining thousands of whole-segment ``pydub`` fades.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from pydub import AudioSegment

from .config import (
    ATEMPO_MAX_PER_STAGE,
    BGM_EDGE_FADE_MS,
    BGM_LOOP_CROSSFADE_MS,
    EXPORT_BITRATE,
    EXPORT_CHANNELS,
    EXPORT_SAMPLE_RATE,
    FIT_TOLERANCE_MS,
    MASTER_CEILING_DBFS,
    VOICE_TARGET_DBFS,
)
from .errors import AudioProcessingError, FFmpegMissingError
from .parser import Cue
from .tts import SynthesisResult

logger = logging.getLogger(__name__)

#: A half-open ``[start_ms, end_ms)`` region during which narration is audible.
SpeechInterval = tuple[int, int]

#: Largest gain change we will apply when normalising the narration bed. Keeps
#: a pathologically quiet or loud render from being amplified into distortion.
_MAX_NORMALISATION_GAIN_DB = 12.0


# ==========================================================================
# ffmpeg availability
# ==========================================================================

def ensure_ffmpeg() -> tuple[str, str]:
    """Locate ffmpeg/ffprobe and point pydub at them.

    pydub shells out to these binaries for every non-WAV decode and for the
    final mp3 encode, but fails with an opaque ``CouldntDecodeError`` when they
    are absent.  Checking up front lets us give an actionable message instead.

    Returns:
        The resolved ``(ffmpeg, ffprobe)`` paths.

    Raises:
        FFmpegMissingError: If either binary is not on ``PATH``.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")

    if not ffmpeg or not ffprobe:
        missing = ", ".join(n for n, p in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if not p)
        raise FFmpegMissingError(
            f"Required audio tool(s) not found on PATH: {missing}.",
            hint=(
                "Install ffmpeg and restart the app:\n"
                "  macOS:  brew install ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg"
            ),
        )

    AudioSegment.converter = ffmpeg
    AudioSegment.ffmpeg = ffmpeg
    AudioSegment.ffprobe = ffprobe
    return ffmpeg, ffprobe


# ==========================================================================
# NumPy <-> AudioSegment bridge
# ==========================================================================

_DTYPE_FOR_WIDTH: dict[int, type[np.signedinteger]] = {
    1: np.int8,
    2: np.int16,
    4: np.int32,
}


def _to_float_frames(segment: AudioSegment) -> tuple[np.ndarray, type[np.signedinteger]]:
    """Expose a segment's samples as a float array shaped ``(frames, channels)``.

    Args:
        segment: The audio to read.

    Returns:
        A ``(frames, dtype)`` pair, where ``dtype`` is the integer type the
        samples must be cast back to.

    Raises:
        AudioProcessingError: If the sample width is not one pydub supports.
    """
    dtype = _DTYPE_FOR_WIDTH.get(segment.sample_width)
    if dtype is None:
        raise AudioProcessingError(
            f"Unsupported sample width: {segment.sample_width} bytes."
        )
    samples = np.array(segment.get_array_of_samples(), dtype=np.float64)
    return samples.reshape((-1, segment.channels)), dtype


def _from_float_frames(
    frames: np.ndarray,
    dtype: type[np.signedinteger],
    template: AudioSegment,
) -> AudioSegment:
    """Rebuild an :class:`AudioSegment` from a float frame array.

    Args:
        frames: Array shaped ``(frames, channels)``.
        dtype: Integer type to quantise back to.
        template: Segment whose sample rate/width/channel count to reuse.

    Returns:
        A new segment carrying the modified samples.
    """
    info = np.iinfo(dtype)
    clipped = np.clip(np.rint(frames), info.min, info.max).astype(dtype)
    return template._spawn(clipped.reshape(-1).tobytes())


# ==========================================================================
# Tempo fitting
# ==========================================================================

def atempo_chain(factor: float) -> list[float]:
    """Split a tempo factor into stages ffmpeg's ``atempo`` filter accepts.

    ``atempo`` is limited to 0.5-2.0 per instance, so larger changes must be
    produced by chaining instances whose factors multiply to the target.

    Args:
        factor: Desired overall speed multiplier (>1 speeds up).

    Returns:
        Factors to apply in series; their product equals ``factor``.
    """
    if factor <= 0:
        raise AudioProcessingError(f"Invalid tempo factor: {factor}")

    stages: list[float] = []
    remaining = float(factor)
    while remaining > ATEMPO_MAX_PER_STAGE:
        stages.append(ATEMPO_MAX_PER_STAGE)
        remaining /= ATEMPO_MAX_PER_STAGE
    while remaining < 0.5:
        stages.append(0.5)
        remaining /= 0.5
    stages.append(round(remaining, 6))
    return stages


def _apply_atempo(segment: AudioSegment, factor: float, workdir: Path) -> AudioSegment:
    """Change playback speed without changing pitch, via ffmpeg.

    ``pydub``'s ``speedup``/frame-rate tricks resample the waveform, which
    shifts pitch and makes narration sound like a chipmunk.  ``atempo`` is a
    proper time-stretch and keeps the voice's timbre intact.

    Args:
        segment: Audio to stretch.
        factor: Speed multiplier (>1 shortens).
        workdir: Scratch directory for the intermediate WAV files.

    Returns:
        The time-stretched audio.

    Raises:
        AudioProcessingError: If ffmpeg fails.
    """
    filters = ",".join(f"atempo={stage:.6f}" for stage in atempo_chain(factor))

    with tempfile.TemporaryDirectory(dir=workdir) as scratch:
        source = Path(scratch) / "in.wav"
        target = Path(scratch) / "out.wav"
        segment.export(source, format="wav")

        command = [
            AudioSegment.converter, "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(source), "-filter:a", filters, str(target),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not target.exists():
            raise AudioProcessingError(
                "ffmpeg could not adjust the speed of a narration line.",
                hint=completed.stderr.strip()[:500] or "No stderr output.",
            )
        return AudioSegment.from_file(target, format="wav")


@dataclass
class FitResult:
    """Outcome of fitting one clip into its subtitle slot.

    Attributes:
        audio: The (possibly time-compressed) clip.
        tempo: Speed multiplier that was applied; 1.0 means untouched.
        overflow_ms: How far the clip *still* exceeds its slot after fitting.
    """

    audio: AudioSegment
    tempo: float
    overflow_ms: int

    @property
    def was_compressed(self) -> bool:
        """True when the clip had to be sped up to fit."""
        return self.tempo > 1.0


def fit_to_slot(
    segment: AudioSegment,
    budget_ms: int,
    max_tempo: float,
    workdir: Path,
) -> FitResult:
    """Compress a clip to fit its subtitle slot, never exceeding ``max_tempo``.

    Clips within :data:`~core.config.FIT_TOLERANCE_MS` of the budget are left
    alone: the correction would be inaudible and not worth the artefacts.

    Args:
        segment: The synthesised narration line.
        budget_ms: Length of the subtitle slot, in milliseconds.
        max_tempo: Hard ceiling on the speed-up, so speech stays intelligible.
        workdir: Scratch directory for ffmpeg intermediates.

    Returns:
        A :class:`FitResult` describing what was done and what still overruns.
    """
    length = len(segment)
    if budget_ms <= 0 or length <= budget_ms + FIT_TOLERANCE_MS:
        return FitResult(audio=segment, tempo=1.0, overflow_ms=max(0, length - budget_ms))

    required = length / budget_ms
    tempo = min(required, max(1.0, max_tempo))

    if tempo <= 1.0:
        return FitResult(audio=segment, tempo=1.0, overflow_ms=length - budget_ms)

    stretched = _apply_atempo(segment, tempo, workdir)
    overflow = max(0, len(stretched) - budget_ms)

    if required > tempo:
        logger.info(
            "Line capped at %.2fx (needed %.2fx); still %d ms over slot.",
            tempo, required, overflow,
        )
    return FitResult(audio=stretched, tempo=tempo, overflow_ms=overflow)


# ==========================================================================
# Timeline assembly
# ==========================================================================

@dataclass
class VoiceTrack:
    """The assembled narration bed plus the statistics the UI reports.

    Attributes:
        audio: Full-length narration with silence in every gap.
        intervals: Where narration is actually audible, used for ducking.
        compressed_lines: How many lines were sped up to fit their slot.
        overflowing_lines: How many still exceed their slot after the cap.
        failed_lines: How many lines could not be synthesised at all.
        max_tempo_used: Largest speed-up actually applied.
        warnings: Notes worth showing the user.
    """

    audio: AudioSegment
    intervals: list[SpeechInterval]
    compressed_lines: int = 0
    overflowing_lines: int = 0
    failed_lines: int = 0
    max_tempo_used: float = 1.0
    warnings: list[str] = field(default_factory=list)


def _speech_rms_dbfs(track: AudioSegment, intervals: Sequence[SpeechInterval]) -> float:
    """Measure loudness across the speaking parts only.

    Measuring the whole canvas would average in every silent gap and report an
    absurdly low level, causing normalisation to over-boost the voice.

    Args:
        track: The assembled narration bed.
        intervals: Regions where narration is audible.

    Returns:
        RMS level in dBFS, or ``-inf`` if there is no measurable signal.
    """
    if not intervals:
        return float("-inf")

    frames, dtype = _to_float_frames(track)
    rate = track.frame_rate
    pieces = [
        frames[int(start * rate / 1000): int(end * rate / 1000)]
        for start, end in intervals
    ]
    speech = np.concatenate([p for p in pieces if p.size]) if any(p.size for p in pieces) else None
    if speech is None or speech.size == 0:
        return float("-inf")

    rms = float(np.sqrt(np.mean(np.square(speech))))
    if rms <= 0:
        return float("-inf")
    return 20.0 * float(np.log10(rms / np.iinfo(dtype).max))


def build_voice_track(
    cues: Sequence[Cue],
    results: Sequence[SynthesisResult],
    max_tempo: float,
    workdir: Path,
    *,
    normalise: bool = True,
) -> VoiceTrack:
    """Lay every synthesised line onto a silent canvas at its exact timestamp.

    Args:
        cues: The validated cues, in timeline order.
        results: Synthesis results aligned 1:1 with ``cues``.
        max_tempo: Ceiling on the speed-up used to fit over-long lines.
        workdir: Scratch directory for ffmpeg intermediates.
        normalise: Whether to bring the narration to the target loudness.

    Returns:
        A :class:`VoiceTrack` with the audio and per-run statistics.

    Raises:
        AudioProcessingError: If a rendered clip cannot be decoded.
    """
    if len(cues) != len(results):
        raise AudioProcessingError(
            f"Internal mismatch: {len(cues)} cues but {len(results)} synthesis results."
        )

    track = VoiceTrack(audio=AudioSegment.empty(), intervals=[])
    clips: list[tuple[int, AudioSegment]] = []
    timeline_end_ms = max((cue.end_ms for cue in cues), default=0)

    for cue, result in zip(cues, results):
        if not result.ok:
            track.failed_lines += 1
            track.warnings.append(f"Line #{cue.index} could not be voiced; left silent.")
            continue

        try:
            clip = AudioSegment.from_file(result.audio_path)
        except Exception as exc:
            raise AudioProcessingError(
                f"Could not decode the audio generated for line #{cue.index}.",
                hint=str(exc),
            ) from exc

        # The slot is the subtitle's own window. Where the next cue starts
        # later than this one ends, the intervening silence is spare room the
        # line may bleed into if it still overruns after capped compression.
        fitted = fit_to_slot(clip, cue.duration_ms, max_tempo, workdir)

        if fitted.was_compressed:
            track.compressed_lines += 1
            track.max_tempo_used = max(track.max_tempo_used, fitted.tempo)
        if fitted.overflow_ms > FIT_TOLERANCE_MS:
            track.overflowing_lines += 1

        start = cue.start_ms
        end = start + len(fitted.audio)
        clips.append((start, fitted.audio))
        track.intervals.append((start, end))
        timeline_end_ms = max(timeline_end_ms, end)

    if not clips:
        raise AudioProcessingError(
            "No narration could be generated for this subtitle file.",
            hint="Every line failed to synthesise or decode.",
        )

    # Build the canvas once at its final length, then stamp each clip in place.
    canvas = AudioSegment.silent(duration=timeline_end_ms, frame_rate=EXPORT_SAMPLE_RATE)
    canvas = canvas.set_channels(EXPORT_CHANNELS)
    for start_ms, clip in clips:
        canvas = canvas.overlay(
            clip.set_frame_rate(EXPORT_SAMPLE_RATE).set_channels(EXPORT_CHANNELS),
            position=start_ms,
        )

    if normalise:
        measured = _speech_rms_dbfs(canvas, track.intervals)
        if measured != float("-inf"):
            gain = float(np.clip(
                VOICE_TARGET_DBFS - measured,
                -_MAX_NORMALISATION_GAIN_DB,
                _MAX_NORMALISATION_GAIN_DB,
            ))
            logger.info("Narration measured %.1f dBFS; applying %+.1f dB.", measured, gain)
            canvas = canvas.apply_gain(gain)

    if track.overflowing_lines:
        track.warnings.append(
            f"{track.overflowing_lines} line(s) are still longer than their subtitle "
            f"slot at the {max_tempo:.2f}x compression limit. Raise the limit, or "
            "shorten those subtitles, for a tighter fit."
        )

    track.audio = canvas
    return track


# ==========================================================================
# Background music
# ==========================================================================

@dataclass
class BgmResult:
    """A prepared music bed and how it was fitted to the timeline.

    Attributes:
        audio: Music, at base level, exactly ``target_ms`` long.
        source_ms: Original length of the uploaded file.
        loops: How many times the source was repeated (1 = trimmed, no loop).
    """

    audio: AudioSegment
    source_ms: int
    loops: int


def prepare_bgm(path: str | Path, target_ms: int, base_db: float) -> BgmResult:
    """Decode, loop or trim, and level the music so it spans the timeline.

    Looping uses a short crossfade at each seam; a hard splice would produce an
    audible click wherever the waveform does not happen to cross zero.

    Args:
        path: Uploaded music file (any format ffmpeg can decode).
        target_ms: Exact length the bed must end up.
        base_db: Resting level of the music, relative to its own peak.

    Returns:
        A :class:`BgmResult`.

    Raises:
        AudioProcessingError: If the file cannot be decoded or is unusably short.
    """
    path = Path(path)
    try:
        source = AudioSegment.from_file(path)
    except Exception as exc:
        raise AudioProcessingError(
            f"Could not read the background music file '{path.name}'.",
            hint=f"{exc}\n\nTry a standard MP3 or WAV file.",
        ) from exc

    source_ms = len(source)
    if source_ms < 500:
        raise AudioProcessingError(
            f"The background music is only {source_ms} ms long.",
            hint="Please use a music file of at least half a second.",
        )

    source = source.set_frame_rate(EXPORT_SAMPLE_RATE).set_channels(EXPORT_CHANNELS)

    # Level the music relative to its own peak first, so the base_db setting
    # means the same thing regardless of how hot the uploaded file was mastered.
    if source.max_dBFS != float("-inf"):
        source = source.apply_gain(-source.max_dBFS)

    loops = 1
    bed = source
    if len(bed) < target_ms:
        crossfade = min(BGM_LOOP_CROSSFADE_MS, len(source) // 4)
        while len(bed) < target_ms:
            bed = bed.append(source, crossfade=crossfade)
            loops += 1

    bed = bed[:target_ms]
    bed = bed.apply_gain(base_db)

    edge_fade = min(BGM_EDGE_FADE_MS, max(0, len(bed) // 4))
    if edge_fade:
        bed = bed.fade_in(edge_fade).fade_out(edge_fade)

    logger.info(
        "Prepared BGM: source %d ms -> %d ms (%d loop(s)) at %.1f dB.",
        source_ms, len(bed), loops, base_db,
    )
    return BgmResult(audio=bed, source_ms=source_ms, loops=loops)


def merge_intervals(
    intervals: Sequence[SpeechInterval],
    merge_gap_ms: int,
) -> list[SpeechInterval]:
    """Coalesce speech regions that are too close together to duck separately.

    Without this, back-to-back subtitle lines make the music surge up and drop
    again in the fraction of a second between them -- the classic "pumping"
    artefact.

    Args:
        intervals: Speech regions, in any order.
        merge_gap_ms: Regions separated by less than this are joined.

    Returns:
        Sorted, non-overlapping regions.
    """
    if not intervals:
        return []

    ordered = sorted(intervals)
    merged: list[SpeechInterval] = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start - last_end < merge_gap_ms:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def build_duck_envelope(
    intervals: Sequence[SpeechInterval],
    total_ms: int,
    duck_db: float,
    fade_ms: int,
    hold_ms: int,
    frame_rate: int,
    frame_count: int,
) -> np.ndarray:
    """Compute a per-frame linear gain multiplier for the music bed.

    The envelope sits at 1.0 (music at its base level) and dips to the ducked
    level around each speech region.  The ramp *starts* ``fade_ms`` before
    speech so the music is already out of the way by the first syllable, and
    releases ``hold_ms`` after it ends so a short breath between sentences does
    not let the music jump back up.

    Args:
        intervals: Regions where narration is audible.
        total_ms: Length of the music bed.
        duck_db: How far to drop the music, in dB (negative attenuates).
        fade_ms: Duration of each ramp.
        hold_ms: How long ducking persists after speech stops.
        frame_rate: Sample rate of the bed.
        frame_count: Number of frames in the bed.

    Returns:
        A float array of length ``frame_count`` in the range ``(0, 1]``.
    """
    if frame_count <= 0:
        return np.ones(0, dtype=np.float64)

    ducked_gain = float(10.0 ** (duck_db / 20.0))
    if not intervals or duck_db >= 0:
        return np.ones(frame_count, dtype=np.float64)

    # Merging must cover both ramps and the hold, or adjacent envelopes would
    # interleave and break the monotonic x-axis np.interp requires.
    merged = merge_intervals(intervals, merge_gap_ms=2 * fade_ms + hold_ms)

    times_ms: list[float] = [0.0]
    gains: list[float] = [1.0]
    for start, end in merged:
        ramp_down_start = max(0.0, start - fade_ms)
        release_start = min(float(total_ms), end + hold_ms)
        ramp_up_end = min(float(total_ms), release_start + fade_ms)

        times_ms.extend([ramp_down_start, float(start), release_start, ramp_up_end])
        gains.extend([1.0, ducked_gain, ducked_gain, 1.0])

    times_ms.append(float(total_ms))
    gains.append(1.0)

    # Guard against any residual non-monotonicity (e.g. a region clipped by the
    # end of the track) so np.interp cannot silently misbehave.
    x = np.maximum.accumulate(np.asarray(times_ms, dtype=np.float64))
    y = np.asarray(gains, dtype=np.float64)

    frame_times_ms = np.arange(frame_count, dtype=np.float64) * (1000.0 / frame_rate)
    return np.interp(frame_times_ms, x, y)


def apply_ducking(
    bgm: AudioSegment,
    intervals: Sequence[SpeechInterval],
    duck_db: float,
    fade_ms: int,
    hold_ms: int,
) -> AudioSegment:
    """Attenuate the music wherever narration is playing.

    Args:
        bgm: Prepared music bed, already at its base level.
        intervals: Regions where narration is audible.
        duck_db: Additional attenuation during speech (negative).
        fade_ms: Ramp duration on either side of a speech region.
        hold_ms: How long ducking persists after speech stops.

    Returns:
        The music bed with its ducking envelope applied.
    """
    frames, dtype = _to_float_frames(bgm)
    envelope = build_duck_envelope(
        intervals=intervals,
        total_ms=len(bgm),
        duck_db=duck_db,
        fade_ms=fade_ms,
        hold_ms=hold_ms,
        frame_rate=bgm.frame_rate,
        frame_count=frames.shape[0],
    )
    # `envelope[:, None]` broadcasts the per-frame gain across every channel.
    return _from_float_frames(frames * envelope[:, None], dtype, bgm)


# ==========================================================================
# Mastering
# ==========================================================================

def mix_master(voice: AudioSegment, bgm: AudioSegment | None) -> AudioSegment:
    """Combine narration and music and trim the peak back to a safe ceiling.

    Args:
        voice: The narration bed.
        bgm: The ducked music bed, or ``None`` to bypass music entirely.

    Returns:
        The finished master.
    """
    master = voice if bgm is None else voice.overlay(bgm)

    peak = master.max_dBFS
    if peak != float("-inf") and peak > MASTER_CEILING_DBFS:
        master = master.apply_gain(MASTER_CEILING_DBFS - peak)
    return master


def export_master(segment: AudioSegment, destination: str | Path) -> Path:
    """Encode the master to mp3 at the configured quality.

    Args:
        segment: The finished mix.
        destination: Where to write the mp3.

    Returns:
        The path written.

    Raises:
        AudioProcessingError: If encoding fails.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        segment.set_frame_rate(EXPORT_SAMPLE_RATE).set_channels(EXPORT_CHANNELS).export(
            destination, format="mp3", bitrate=EXPORT_BITRATE,
        )
    except Exception as exc:
        raise AudioProcessingError(
            "Failed to encode the final MP3.",
            hint=f"{exc}\n\nCheck that ffmpeg has MP3 (libmp3lame) support.",
        ) from exc
    return destination
