"""Exception hierarchy for the SRT Voiceover Studio.

Every error that is *expected* -- a bad subtitle file, a missing binary, an
unreachable voice service -- derives from :class:`StudioError` and carries a
message written for the end user rather than for a stack trace.  ``app.py``
catches this base class and surfaces ``str(exc)`` directly in the UI; anything
that is *not* a ``StudioError`` is treated as a bug, logged with a full
traceback, and reported generically.
"""

from __future__ import annotations


class StudioError(Exception):
    """Base class for all user-facing, recoverable errors."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.message}\n\n{self.hint}" if self.hint else self.message


class SubtitleError(StudioError):
    """The `.srt` file is missing, unreadable, empty, or malformed."""


class FFmpegMissingError(StudioError):
    """The `ffmpeg`/`ffprobe` binaries could not be located on PATH."""


class SynthesisError(StudioError):
    """Speech synthesis failed in a way that could not be recovered from."""


class AudioProcessingError(StudioError):
    """Decoding, mixing, or exporting audio failed."""
