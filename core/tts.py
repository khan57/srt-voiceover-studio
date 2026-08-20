"""Asynchronous Edge-TTS synthesis worker.

Edge-TTS is a remote websocket service, so the dominant cost of rendering a
subtitle file is round-trip latency, not CPU.  This module therefore
synthesises several lines concurrently behind a semaphore, retries transient
failures, and -- crucially -- refuses to let a single bad line abort a long
render.  A cue that cannot be synthesised after every retry is reported as a
failure and the pipeline substitutes silence for it.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import edge_tts
from edge_tts.exceptions import EdgeTTSException, NoAudioReceived

from .config import TTS_CONCURRENCY, TTS_MAX_ATTEMPTS
from .errors import SynthesisError
from .parser import Cue

logger = logging.getLogger(__name__)

#: Called as ``on_progress(completed, total)`` after each line finishes.
ProgressCallback = Callable[[int, int], None]


@dataclass
class SynthesisResult:
    """The outcome of synthesising a single cue.

    Attributes:
        cue: The cue this result belongs to.
        audio_path: Location of the rendered mp3, or ``None`` if synthesis
            failed and the pipeline should substitute silence.
        error: Human-readable failure reason when ``audio_path`` is ``None``.
    """

    cue: Cue
    audio_path: Path | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when this cue produced usable audio."""
        return self.audio_path is not None


def format_rate(percent: int | float) -> str:
    """Render a speech-rate percentage in the form Edge-TTS requires.

    The service rejects unsigned values, so ``0`` must be sent as ``"+0%"``
    rather than ``"0%"``.

    Args:
        percent: Rate adjustment, where 0 means the voice's natural speed.

    Returns:
        A signed percentage string such as ``"+15%"`` or ``"-30%"``.
    """
    return f"{int(round(percent)):+d}%"


def format_pitch(hertz: int | float) -> str:
    """Render a pitch offset in the form Edge-TTS requires.

    Args:
        hertz: Pitch adjustment in Hz, where 0 means the voice's natural pitch.

    Returns:
        A signed frequency string such as ``"+5Hz"`` or ``"-20Hz"``.
    """
    return f"{int(round(hertz)):+d}Hz"


async def _synthesise_one(
    cue: Cue,
    voice: str,
    rate: str,
    pitch: str,
    destination: Path,
) -> SynthesisResult:
    """Synthesise a single cue, retrying transient service failures.

    Args:
        cue: The subtitle line to speak.
        voice: Edge-TTS voice short-name, e.g. ``"hi-IN-MadhurNeural"``.
        rate: Pre-formatted signed rate string.
        pitch: Pre-formatted signed pitch string.
        destination: File path the mp3 should be written to.

    Returns:
        A :class:`SynthesisResult`; on total failure its ``audio_path`` is
        ``None`` and ``error`` explains why.
    """
    last_error: str = "unknown error"

    for attempt in range(1, TTS_MAX_ATTEMPTS + 1):
        try:
            communicate = edge_tts.Communicate(cue.text, voice, rate=rate, pitch=pitch)
            await communicate.save(str(destination))

            # `save` can complete having written nothing at all if the service
            # returned an empty stream; treat that as a retryable failure.
            if not destination.exists() or destination.stat().st_size == 0:
                raise NoAudioReceived("Edge-TTS returned an empty audio stream")

            return SynthesisResult(cue=cue, audio_path=destination)

        except (EdgeTTSException, OSError) as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
        except Exception as exc:  # network stack raises a wide variety of types
            last_error = f"{exc.__class__.__name__}: {exc}"

        if attempt < TTS_MAX_ATTEMPTS:
            backoff = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s, ...
            logger.warning(
                "Cue #%d failed (attempt %d/%d): %s -- retrying in %.1fs",
                cue.index, attempt, TTS_MAX_ATTEMPTS, last_error, backoff,
            )
            await asyncio.sleep(backoff)

    logger.error("Cue #%d permanently failed: %s", cue.index, last_error)
    destination.unlink(missing_ok=True)
    return SynthesisResult(cue=cue, audio_path=None, error=last_error)


async def synthesize_all(
    cues: Sequence[Cue],
    voice: str,
    rate: int | float,
    pitch: int | float,
    workdir: Path,
    *,
    concurrency: int = TTS_CONCURRENCY,
    on_progress: ProgressCallback | None = None,
) -> list[SynthesisResult]:
    """Synthesise every cue concurrently and report progress as lines land.

    Args:
        cues: Validated cues from :func:`core.parser.parse_srt`.
        voice: Edge-TTS voice short-name.
        rate: Speech rate adjustment in percent (-30..+30).
        pitch: Pitch adjustment in Hz (-20..+20).
        workdir: Directory to write the per-line mp3 files into. Must exist.
        concurrency: Maximum simultaneous connections to the voice service.
        on_progress: Optional callback invoked as each line completes.

    Returns:
        Results in the same order as ``cues``.

    Raises:
        SynthesisError: If *every* line failed, which indicates a systemic
            problem (no network, invalid voice) rather than a flaky line.
    """
    if not cues:
        raise SynthesisError("There is nothing to synthesise.")

    rate_str = format_rate(rate)
    pitch_str = format_pitch(pitch)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    total = len(cues)
    completed = 0
    lock = asyncio.Lock()

    async def worker(position: int, cue: Cue) -> tuple[int, SynthesisResult]:
        nonlocal completed
        async with semaphore:
            destination = workdir / f"line_{position:05d}.mp3"
            result = await _synthesise_one(cue, voice, rate_str, pitch_str, destination)

        # Progress is reported from inside the task rather than by awaiting in
        # completion order, so the counter stays accurate without forcing the
        # caller to consume an iterator.
        async with lock:
            completed += 1
            if on_progress is not None:
                on_progress(completed, total)

        return position, result

    logger.info(
        "Synthesising %d line(s) with %s (rate=%s, pitch=%s, concurrency=%d)",
        total, voice, rate_str, pitch_str, concurrency,
    )

    gathered = await asyncio.gather(
        *(worker(position, cue) for position, cue in enumerate(cues))
    )

    ordered: list[SynthesisResult] = [result for _, result in sorted(gathered, key=lambda p: p[0])]

    if not any(result.ok for result in ordered):
        first_error = next((r.error for r in ordered if r.error), "unknown error")
        raise SynthesisError(
            "Every line failed to synthesise.",
            hint=(
                "This usually means there is no internet connection, or the "
                f"selected voice is unavailable.\n\nFirst error was: {first_error}"
            ),
        )

    return ordered


def synthesize_all_sync(
    cues: Sequence[Cue],
    voice: str,
    rate: int | float,
    pitch: int | float,
    workdir: Path,
    *,
    concurrency: int = TTS_CONCURRENCY,
    on_progress: ProgressCallback | None = None,
) -> list[SynthesisResult]:
    """Blocking wrapper around :func:`synthesize_all`.

    Gradio event handlers run synchronously in a worker thread, so the async
    worker pool is driven by its own event loop for the duration of the call.

    Args: See :func:`synthesize_all`.

    Returns:
        Results in the same order as ``cues``.
    """
    return asyncio.run(
        synthesize_all(
            cues, voice, rate, pitch, workdir,
            concurrency=concurrency, on_progress=on_progress,
        )
    )


async def list_available_voices(prefix: Iterable[str] = ()) -> list[str]:
    """Query the service for installed voices, optionally filtered by locale.

    Useful for diagnostics when a curated voice stops being offered.

    Args:
        prefix: Locale prefixes to keep, e.g. ``("hi-IN", "en-IN")``. Empty
            means return everything.

    Returns:
        Sorted voice short-names.
    """
    voices = await edge_tts.VoicesManager.create()
    names = [v["ShortName"] for v in voices.voices]
    prefixes = tuple(prefix)
    if prefixes:
        names = [n for n in names if n.startswith(prefixes)]
    return sorted(names)
