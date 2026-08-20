"""SRT parsing and timeline modelling.

The job of this module is to turn a messy, human-authored ``.srt`` file into a
clean, sorted, validated list of :class:`Cue` objects that the rest of the
pipeline can trust.  Everything downstream assumes cues are ordered, non-empty,
and have positive duration -- this is the only place those guarantees are
established.

Devanagari safety is a first-class concern: subtitle files in the wild are
saved in a variety of encodings, and a wrong guess turns Hindi text into
mojibake that the TTS engine will happily read aloud as garbage.  We therefore
try a cascade of encodings and additionally *score* the result, rejecting
decodings that produce replacement characters.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pysrt

from .errors import SubtitleError

logger = logging.getLogger(__name__)

# Encodings tried in order.  utf-8-sig comes first because it transparently
# handles both BOM'd and plain UTF-8, which together cover the vast majority of
# modern subtitle files.  utf-16 must be attempted before the single-byte
# fallbacks, which would "succeed" on UTF-16 input while producing nonsense.
_ENCODING_CASCADE: tuple[str, ...] = ("utf-8-sig", "utf-8", "utf-16", "cp1252", "latin-1")

# Markup commonly embedded in subtitle text, none of which should be spoken.
_HTML_TAG_RE = re.compile(r"<[^>]+>")            # <i>, </b>, <font color="…">
_ASS_OVERRIDE_RE = re.compile(r"\{[^}]*\}")      # {\an8}, {\pos(…)}
_WHITESPACE_RE = re.compile(r"\s+")

#: Above this share of Devanagari letters, subtitles count as Hindi.
_DEVANAGARI_THRESHOLD = 0.30


@dataclass(frozen=True)
class Cue:
    """One subtitle line, normalised onto a millisecond timeline.

    Attributes:
        index: 1-based position of this cue in the source file, used in error
            messages so a user can find the offending line.
        start_ms: Start of the cue's slot, in milliseconds from 00:00:00.000.
        end_ms: End of the cue's slot, in milliseconds. Always > ``start_ms``.
        text: Cleaned, speakable text with all markup removed.
    """

    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        """Length of the subtitle's on-screen slot, in milliseconds."""
        return self.end_ms - self.start_ms


@dataclass
class ParseReport:
    """Diagnostics gathered while parsing, surfaced in the UI metrics panel.

    Attributes:
        cues: The usable cues, sorted by start time.
        encoding: Which encoding successfully decoded the file.
        skipped_empty: Count of cues dropped because they had no speakable text
            (e.g. lines containing only ``♪`` or formatting tags).
        warnings: Human-readable notes about anything unusual but survivable.
    """

    cues: list[Cue]
    encoding: str
    skipped_empty: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def total_duration_ms(self) -> int:
        """End timestamp of the last cue -- the nominal timeline length."""
        return max((c.end_ms for c in self.cues), default=0)

    @property
    def devanagari_ratio(self) -> float:
        """Share of letters across all cues that are Devanagari, 0.0 to 1.0."""
        return devanagari_ratio(" ".join(c.text for c in self.cues))

    @property
    def is_devanagari(self) -> bool:
        """True when the subtitles are predominantly Hindi script."""
        return self.devanagari_ratio >= _DEVANAGARI_THRESHOLD


def devanagari_ratio(text: str) -> float:
    """Measure how much of a string is written in Devanagari.

    Used to catch the case where Hindi subtitles are paired with an
    English-only voice, which yields silence rather than an obvious error.

    Args:
        text: Any text; punctuation, digits and whitespace are ignored.

    Returns:
        Devanagari letters as a fraction of all letters, or 0.0 if there are
        no letters at all.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    devanagari = sum(1 for c in letters if "\u0900" <= c <= "\u097f")
    return devanagari / len(letters)


def clean_text(raw: str) -> str:
    """Strip markup and normalise whitespace, preserving all Unicode content.

    Multi-line subtitle blocks are joined with a single space so the TTS engine
    reads them as one continuous sentence rather than pausing at the line wrap,
    which is a purely visual artefact of subtitle layout.

    Args:
        raw: The subtitle text exactly as it appeared in the file.

    Returns:
        Speakable plain text, possibly empty if the cue was markup-only.
    """
    text = _ASS_OVERRIDE_RE.sub(" ", raw)
    text = _HTML_TAG_RE.sub(" ", text)
    # Collapse the newlines inside a cue, then any resulting whitespace runs.
    text = text.replace("\n", " ").replace("\r", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _read_subtitles(path: Path) -> tuple[list[pysrt.SubRipItem], str]:
    """Decode the file, trying each candidate encoding until one works.

    Args:
        path: Location of the ``.srt`` file.

    Returns:
        A ``(items, encoding)`` pair.

    Raises:
        SubtitleError: If no encoding in the cascade yields a parseable file.
    """
    failures: list[str] = []
    for encoding in _ENCODING_CASCADE:
        try:
            items = list(pysrt.open(str(path), encoding=encoding))
        except (UnicodeDecodeError, UnicodeError) as exc:
            failures.append(f"{encoding}: {exc.__class__.__name__}")
            continue
        except Exception as exc:  # pysrt raises bare Exception on bad syntax
            failures.append(f"{encoding}: {exc}")
            continue

        if not items:
            failures.append(f"{encoding}: decoded but contained no subtitles")
            continue

        # A successful decode that is littered with U+FFFD means we picked the
        # wrong codec -- the bytes were consumed but the text is corrupt.
        joined = "\n".join(item.text for item in items)
        if "�" in joined:
            failures.append(f"{encoding}: produced replacement characters")
            continue

        logger.info("Decoded %s using %s (%d cues)", path.name, encoding, len(items))
        return items, encoding

    raise SubtitleError(
        f"Could not read '{path.name}' as a subtitle file.",
        hint=(
            "Tried these encodings without success:\n  - "
            + "\n  - ".join(failures)
            + "\n\nRe-save the file as UTF-8 and try again."
        ),
    )


def parse_srt(path: str | Path) -> ParseReport:
    """Parse an ``.srt`` file into a validated, ordered list of cues.

    Args:
        path: Path to the subtitle file.

    Returns:
        A :class:`ParseReport` containing the usable cues and diagnostics.

    Raises:
        SubtitleError: If the file is missing, undecodable, contains no cues, or
            every cue is unusable.
    """
    path = Path(path)
    if not path.is_file():
        raise SubtitleError(f"Subtitle file not found: {path}")
    if path.stat().st_size == 0:
        raise SubtitleError(f"Subtitle file '{path.name}' is empty.")

    items, encoding = _read_subtitles(path)
    report = ParseReport(cues=[], encoding=encoding)

    for position, item in enumerate(items, start=1):
        start_ms = int(item.start.ordinal)
        end_ms = int(item.end.ordinal)

        if start_ms < 0 or end_ms < 0:
            raise SubtitleError(
                f"Subtitle #{position} has a negative timestamp.",
                hint="Timecodes must be of the form HH:MM:SS,mmm and non-negative.",
            )
        if end_ms <= start_ms:
            raise SubtitleError(
                f"Subtitle #{position} ends at or before it starts "
                f"({item.start} --> {item.end}).",
                hint="Every subtitle needs a positive duration. Fix or remove this cue.",
            )

        text = clean_text(item.text)
        if not text:
            report.skipped_empty += 1
            continue

        report.cues.append(Cue(index=position, start_ms=start_ms, end_ms=end_ms, text=text))

    if not report.cues:
        raise SubtitleError(
            f"'{path.name}' contains no speakable text.",
            hint=(
                f"{report.skipped_empty} cue(s) were found but all were empty "
                "or contained only formatting tags."
            ),
        )

    # Author order is not guaranteed to be chronological. Sort defensively --
    # the timeline builder places clips by absolute offset, but the ducking
    # envelope and the overlap check below both assume ascending order.
    report.cues.sort(key=lambda c: (c.start_ms, c.end_ms))

    overlaps = sum(
        1
        for previous, current in zip(report.cues, report.cues[1:])
        if current.start_ms < previous.end_ms
    )
    if overlaps:
        report.warnings.append(
            f"{overlaps} subtitle slot(s) overlap in the source file; "
            "narration for those lines may run into each other."
        )
    if report.skipped_empty:
        report.warnings.append(
            f"Skipped {report.skipped_empty} cue(s) with no speakable text."
        )

    return report
