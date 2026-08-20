"""Search and fetch freely-licensed background music from Openverse.

Openverse aggregates Creative Commons audio from Jamendo, Freesound and others,
and serves it over an API that needs **no key and no account** -- which keeps
this app's "nothing to sign up for" promise intact.

Licensing is the whole difficulty here, not the HTTP.  Mixing a music bed under
narration produces a *derivative work* and is usually *commercial*, so the two
CC clauses that forbid exactly that -- ``ND`` (NoDerivatives) and ``NC``
(NonCommercial) -- make a track unusable no matter how good it sounds.  An
unfiltered search for "cinematic background" returns 16 such tracks out of 20.
This module therefore refuses to surface anything except ``CC0`` and ``CC BY``,
and it enforces that twice: once in the query and again on every result that
comes back, so a server-side change cannot quietly widen what the UI offers.

Everything the API returns is untrusted input.  Titles and creator names are
escaped before they reach the Markdown UI, and downloads are constrained by
scheme, size, content type, and a decode check before being handed to the mixer.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import httpx
import time

from .errors import StudioError

logger = logging.getLogger(__name__)

API_URL: Final[str] = "https://api.openverse.org/v1/audio/"
USER_AGENT: Final[str] = "srt-voiceover-studio/1.0 (+local desktop app)"

#: The only licences under which a track may be mixed beneath narration.
#: ``cc0`` is public domain; ``by`` requires credit but permits both commercial
#: use and derivative works. Everything else -- ``nc``, ``nd``, ``sa`` -- is
#: excluded: the first two forbid this use outright, and ShareAlike would
#: virally impose its terms on the user's finished video.
ALLOWED_LICENCES: Final[frozenset[str]] = frozenset({"cc0", "by"})

#: Openverse caps anonymous requests at 20 results per page.
MAX_PAGE_SIZE: Final[int] = 20

#: Refuse to pull down anything larger than this. Music beds are minutes long,
#: not hours, and this bounds the damage from a mislabelled URL.
MAX_DOWNLOAD_BYTES: Final[int] = 40 * 1024 * 1024

#: Openverse can take a while on filtered queries, so this is deliberately
#: generous. A slow search is worth waiting for; a spurious failure is not.
#: Bytes fetched for an audition clip. MP3 is frame-based, so a truncated
#: stream still decodes: ~384 KB yields roughly 15-20 seconds of audio in about
#: two seconds, whatever the length of the full track.
PREVIEW_BYTES: Final[int] = 384 * 1024

#: Audition clips are trimmed to this, with a fade so they do not cut abruptly.
PREVIEW_SECONDS: Final[int] = 30
PREVIEW_FADE_MS: Final[int] = 1_500

#: Cap on files kept in the download cache before the oldest are evicted.
CACHE_MAX_FILES: Final[int] = 40

SEARCH_TIMEOUT_S: Final[float] = 45.0

#: One extra attempt when a search times out, since slowness is intermittent.
SEARCH_ATTEMPTS: Final[int] = 2
DOWNLOAD_TIMEOUT_S: Final[float] = 90.0

#: Total wall-clock budget for an audition clip. httpx timeouts are
#: per-operation, so a slow trickle can keep resetting the read timer and drag
#: a "quick" preview out for minutes. Auditioning has to feel instant, so the
#: streaming loop enforces this deadline itself.
PREVIEW_DEADLINE_S: Final[float] = 15.0

#: Per-phase timeouts for auditioning. Bounded tightly because httpx applies
#: each phase separately: a single generous number would let a stalled connect
#: and a stalled read stack up to twice the budget.
PREVIEW_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=5.0, read=10.0, write=10.0, pool=5.0
)

#: Enough bytes to decode into a few seconds of audio. Below this a truncated
#: MP3 is not worth offering as a preview.
MIN_PLAYABLE_BYTES: Final[int] = 32 * 1024

_LICENCE_NAMES: Final[dict[str, str]] = {
    "cc0": "CC0 (public domain)",
    "by": "CC BY (credit required)",
}

# Collapses anything that could break out of a Markdown table cell or inject
# markup into the results list.
_UNSAFE_DISPLAY = re.compile(r"[|\r\n<>\[\]`*_]")


def _sanitise(text: str, limit: int = 90, placeholder: str = "Untitled") -> str:
    """Make an API-supplied string safe to render in the Markdown UI.

    Track titles and creator names are arbitrary user-submitted text from a
    third-party service. They are displayed, never executed, but they still
    must not be able to inject markup or break the results table.

    Args:
        text: Untrusted text from the API.
        limit: Maximum length before truncation.
        placeholder: Returned when the text is empty after cleaning. Pass an
            empty string for optional fields, where a visible placeholder
            would be worse than nothing -- an absent attribution must stay
            falsy so the caller can compose its own credit line.

    Returns:
        A single-line, markup-free string.
    """
    cleaned = _UNSAFE_DISPLAY.sub(" ", html.unescape(str(text or ""))).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return placeholder
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


@dataclass(frozen=True)
class Track:
    """One freely-licensed music track offered by the search.

    Attributes:
        identifier: Openverse UUID for the track.
        title: Display title, already sanitised.
        creator: Artist name, already sanitised.
        audio_url: Direct URL to the audio file.
        landing_url: Human-facing page for the track on its source site.
        licence: Short licence code, guaranteed to be in
            :data:`ALLOWED_LICENCES`.
        licence_version: Version string, e.g. ``"4.0"``.
        licence_url: Canonical URL of the licence deed.
        duration_ms: Track length in milliseconds, 0 when unreported.
        source: Which provider the track came from, e.g. ``"jamendo"``.
        attribution: The provider's ready-made credit line.
    """

    identifier: str
    title: str
    creator: str
    audio_url: str
    landing_url: str
    licence: str
    licence_version: str
    licence_url: str
    duration_ms: int
    source: str
    attribution: str

    @property
    def requires_attribution(self) -> bool:
        """True when this licence obliges the user to credit the artist."""
        return self.licence != "cc0"

    @property
    def licence_label(self) -> str:
        """Human-readable licence name for display."""
        base = _LICENCE_NAMES.get(self.licence, self.licence.upper())
        return f"{base} {self.licence_version}".strip()

    @property
    def duration_label(self) -> str:
        """Track length as ``M:SS``, or ``"?"`` when the API omitted it."""
        if self.duration_ms <= 0:
            return "?"
        seconds = round(self.duration_ms / 1000)
        return f"{seconds // 60}:{seconds % 60:02d}"

    @property
    def choice_label(self) -> str:
        """Single-line summary used as the radio-button label in the UI."""
        credit = "credit required" if self.requires_attribution else "no credit needed"
        return f"{self.title} — {self.creator}  [{self.duration_label} · {credit}]"

    def credit_line(self) -> str:
        """The attribution text to reproduce alongside the finished video."""
        if self.attribution:
            return self.attribution
        return (
            f'"{self.title}" by {self.creator} is licensed under '
            f"{self.licence_label}. {self.licence_url}".strip()
        )


def _parse_track(payload: dict) -> Track | None:
    """Convert one API result into a :class:`Track`, or reject it.

    Args:
        payload: A single object from the API's ``results`` array.

    Returns:
        The parsed track, or ``None`` if it is unusable or not on an
        allowed licence.
    """
    licence = str(payload.get("license") or "").lower().strip()
    audio_url = str(payload.get("url") or "")

    # Belt-and-braces: the query already filters by licence, but a result that
    # slipped through must never reach the user.
    if licence not in ALLOWED_LICENCES:
        logger.debug("Discarding %r on disallowed licence %r", payload.get("title"), licence)
        return None

    # Only ever fetch over TLS, and never follow a non-HTTP scheme.
    if not audio_url.lower().startswith("https://"):
        logger.debug("Discarding %r on non-HTTPS url", payload.get("title"))
        return None

    try:
        duration = int(payload.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0

    return Track(
        identifier=str(payload.get("id") or ""),
        title=_sanitise(payload.get("title")),
        creator=_sanitise(payload.get("creator") or "Unknown artist", limit=50),
        audio_url=audio_url,
        landing_url=str(payload.get("foreign_landing_url") or ""),
        licence=licence,
        licence_version=str(payload.get("license_version") or ""),
        licence_url=str(payload.get("license_url") or ""),
        duration_ms=duration,
        source=_sanitise(payload.get("source") or "openverse", limit=24),
        attribution=_sanitise(payload.get("attribution") or "", limit=300, placeholder=""),
    )


def _search_once(
    query: str,
    *,
    limit: int = MAX_PAGE_SIZE,
    music_only: bool = True,
) -> list[Track]:
    """Run a single Openverse query and return its permitted-licence results.

    Args:
        query: Free-text search, e.g. ``"calm cinematic piano"``.
        limit: Maximum results, capped at :data:`MAX_PAGE_SIZE`.
        music_only: Restrict to the ``music`` category, excluding the sound
            effects and field recordings that otherwise dominate results.

    Returns:
        Tracks on a permitted licence, best match first. May be empty.

    Raises:
        StudioError: If the query is blank, the service is unreachable, or the
            anonymous rate limit has been exhausted.
    """
    query = (query or "").strip()
    if not query:
        raise StudioError("Type something to search for, such as “calm piano”.")

    params = {
        "q": query,
        "page_size": str(max(1, min(limit, MAX_PAGE_SIZE))),
        # Request only the licences we are prepared to accept.
        "license": ",".join(sorted(ALLOWED_LICENCES)),
    }
    if music_only:
        params["category"] = "music"

    response = None
    for attempt in range(1, SEARCH_ATTEMPTS + 1):
        try:
            response = httpx.get(
                API_URL,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=SEARCH_TIMEOUT_S,
                follow_redirects=True,
            )
            break
        except httpx.TimeoutException as exc:
            # Distinct from an unreachable host: the service answered slowly or
            # not at all, which telling the user to check their wifi would only
            # send them chasing the wrong problem.
            logger.warning("Music search timed out (attempt %d): %s", attempt, exc)
            if attempt == SEARCH_ATTEMPTS:
                raise StudioError(
                    "The music library took too long to respond.",
                    hint=(
                        "Openverse is occasionally slow on filtered searches. "
                        "Try again in a moment, or use a broader single word."
                    ),
                ) from exc
        except httpx.RequestError as exc:
            raise StudioError(
                "Could not reach the music library.",
                hint=f"Check your internet connection.\n\n{exc.__class__.__name__}: {exc}",
            ) from exc

    if response is None:  # pragma: no cover - defensive; loop always sets or raises
        raise StudioError("The music library could not be reached.")

    if response.status_code == 429:
        raise StudioError(
            "The music library's rate limit has been reached.",
            hint=(
                "Openverse allows 20 searches per minute and 200 per day without "
                "an account. Wait a moment and try again."
            ),
        )
    if response.status_code >= 400:
        raise StudioError(
            f"The music library returned an error (HTTP {response.status_code}).",
            hint="This is usually temporary. Try again shortly.",
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise StudioError("The music library sent a malformed response.") from exc

    results = payload.get("results") or []
    tracks = [t for t in (_parse_track(r) for r in results) if t is not None]

    logger.info(
        "Music search %r: %d result(s), %d usable after licence filtering.",
        query, len(results), len(tracks),
    )
    return tracks


def _broadening_variants(query: str) -> list[str]:
    """Progressively looser versions of a query, most specific first.

    Openverse ANDs every search term, so precision collapses fast: "calm
    cinematic piano" matches nothing at all, while "cinematic piano" matches
    32 tracks. Rather than dead-ending on a reasonable-sounding phrase, fall
    back to the last two words, then to the single longest (most distinctive)
    word.

    Args:
        query: The user's original search text.

    Returns:
        Queries to try in order, deduplicated.
    """
    words = query.split()
    variants = [query]
    if len(words) > 2:
        variants.append(" ".join(words[-2:]))
    if len(words) > 1:
        variants.append(max(words, key=len))

    seen: set[str] = set()
    ordered: list[str] = []
    for variant in variants:
        key = variant.casefold()
        if key not in seen:
            seen.add(key)
            ordered.append(variant)
    return ordered


@dataclass(frozen=True)
class SearchResult:
    """What a music search produced, and which query actually produced it.

    Attributes:
        tracks: Matching tracks, all on a permitted licence.
        query: The user's original search text.
        effective_query: The query that actually returned ``tracks``. Differs
            from ``query`` when the search had to be broadened.
    """

    tracks: list[Track]
    query: str
    effective_query: str

    @property
    def was_broadened(self) -> bool:
        """True when the original query matched nothing and had to be loosened."""
        return self.effective_query.casefold() != self.query.casefold()


def search_music(
    query: str,
    *,
    limit: int = MAX_PAGE_SIZE,
    music_only: bool = True,
) -> SearchResult:
    """Search Openverse for background music that may legally be remixed.

    Broadens the query automatically if the exact phrase matches nothing.

    Args:
        query: Free-text search, e.g. ``"calm cinematic piano"``.
        limit: Maximum results, capped at :data:`MAX_PAGE_SIZE`.
        music_only: Restrict to the ``music`` category, excluding the sound
            effects and field recordings that otherwise dominate results.

    Returns:
        A :class:`SearchResult`; its ``tracks`` may be empty.

    Raises:
        StudioError: If the query is blank, the service is unreachable, or the
            anonymous rate limit has been exhausted.
    """
    query = (query or "").strip()
    if not query:
        raise StudioError("Type something to search for, such as \u201ccalm piano\u201d.")

    variants = _broadening_variants(query)
    for attempt, variant in enumerate(variants):
        tracks = _search_once(variant, limit=limit, music_only=music_only)
        if tracks:
            if attempt:
                logger.info("Broadened %r to %r to find results.", query, variant)
            return SearchResult(tracks=tracks, query=query, effective_query=variant)

    return SearchResult(tracks=[], query=query, effective_query=variants[-1])


#: Content types we know how to name on disk. Anything else falls back to a
#: sensible default and is left to ffmpeg to sniff.
_EXTENSION_BY_TYPE: Final[dict[str, str]] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/ogg": ".ogg",
    "audio/vorbis": ".ogg",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
}


def _extension_for(content_type: str, track: Track) -> str:
    """Choose a file extension for a downloaded track.

    Args:
        content_type: The response's ``Content-Type`` header, lowercased.
        track: The track being downloaded, used as a secondary hint.

    Returns:
        A dotted extension such as ``".mp3"``.
    """
    for prefix, extension in _EXTENSION_BY_TYPE.items():
        if content_type.startswith(prefix):
            return extension
    suffix = Path(track.audio_url.split("?")[0]).suffix.lower()
    if suffix in {".wav", ".ogg", ".flac", ".m4a", ".aac", ".mp3"}:
        return suffix
    return ".mp3"


def _safe_stem(track: Track) -> str:
    """Build a filesystem-safe stem identifying a track.

    The identifier is supplied by an external service, so it is reduced to
    characters that cannot traverse out of the destination directory.

    Args:
        track: The track being downloaded.

    Returns:
        A short alphanumeric stem, never empty.
    """
    stem = re.sub(r"[^A-Za-z0-9_-]", "", track.identifier or "")[:40]
    return stem or "track"


def download_track(
    track: Track,
    destination_dir: Path,
    *,
    limit_bytes: int | None = None,
    prefix: str = "bgm",
    deadline_s: float | None = None,
    timeout: float | httpx.Timeout = DOWNLOAD_TIMEOUT_S,
) -> Path:
    """Fetch a track's audio to disk, enforcing safety limits as it streams.

    The URL comes from a third-party API, so the download is constrained on
    every axis that matters: HTTPS only, a hard byte ceiling enforced *during*
    streaming rather than by trusting ``Content-Length``, a declared audio
    content type, and a filename derived from a sanitised identifier.

    Args:
        track: The track to fetch.
        destination_dir: Directory to write into; created if absent.
        limit_bytes: Stop after this many bytes, yielding a partial but still
            playable clip. ``None`` fetches the whole track.
        prefix: Filename prefix, keeping partial audition clips distinct from
            complete downloads in the cache.
        deadline_s: Total wall-clock budget. On expiry the download stops and
            keeps whatever arrived, provided that is enough to decode; without
            a ``limit_bytes`` partial fetch, expiry is an error instead.
        timeout: httpx timeout. Note these are *per-operation*, so connect and
            read each get the full budget -- use an :class:`httpx.Timeout` with
            tight phases when the total matters, as previews do.

    Returns:
        Path to the downloaded file. An already-cached file is returned
        without re-fetching it.

    Raises:
        StudioError: If the download fails, is too large, or is not audio.
    """
    if not track.audio_url.lower().startswith("https://"):
        raise StudioError("That track's download link is not secure; skipping it.")

    destination_dir.mkdir(parents=True, exist_ok=True)

    # Auditioning a track then using it must not download it twice, so any
    # non-empty cached file for this prefix and id is reused as-is.
    cached = next(
        (
            candidate
            for candidate in destination_dir.glob(f"{prefix}_{_safe_stem(track)}.*")
            if candidate.is_file() and candidate.stat().st_size > 0
        ),
        None,
    )
    if cached is not None:
        logger.info("Reusing cached %s for %r", cached.name, track.title)
        return cached

    destination: Path | None = None
    written = 0
    started = time.monotonic()

    def out_of_time() -> bool:
        return deadline_s is not None and (time.monotonic() - started) > deadline_s

    try:
        with httpx.stream(
            "GET",
            track.audio_url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        ) as response:
            if response.status_code >= 400:
                raise StudioError(
                    f"Could not download “{track.title}” "
                    f"(HTTP {response.status_code}).",
                    hint="The source site may have removed it. Try another track.",
                )

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not (
                content_type.startswith("audio/")
                or content_type.startswith("application/octet-stream")
            ):
                raise StudioError(
                    f"“{track.title}” did not download as an audio file.",
                    hint=f"The server described it as “{content_type}”.",
                )

            # Named after the track so the UI can recognise the file later, with
            # a real extension so Gradio and ffmpeg both handle it happily.
            destination = destination_dir / (
                f"{prefix}_{_safe_stem(track)}{_extension_for(content_type, track)}"
            )

            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(64 * 1024):
                    # Checked before writing, and against bytes actually
                    # received, so neither an absent nor a dishonest
                    # Content-Length can get past the ceiling.
                    if written + len(chunk) > MAX_DOWNLOAD_BYTES:
                        raise StudioError(
                            f"“{track.title}” exceeds the "
                            f"{MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB limit.",
                            hint="Pick a shorter track.",
                        )
                    handle.write(chunk)
                    written += len(chunk)
                    # A partial MP3 still decodes, so an audition clip simply
                    # stops reading once it has enough audio to be useful.
                    if limit_bytes is not None and written >= limit_bytes:
                        break
                    if out_of_time():
                        # Settle for a shorter clip rather than keep a user
                        # waiting on a slow host.
                        if limit_bytes is not None and written >= MIN_PLAYABLE_BYTES:
                            logger.info(
                                "Preview of %r cut short at %d KB after %.0fs.",
                                track.title, written // 1024, deadline_s,
                            )
                            break
                        raise StudioError(
                            f"“{track.title}” is downloading too slowly.",
                            hint="The source host is not responding quickly. "
                                 "Try a different track.",
                        )

    except StudioError:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise
    except httpx.TimeoutException as exc:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise StudioError(
            f"“{track.title}” is downloading too slowly.",
            hint="The source host is not responding quickly. Try another track.",
        ) from exc
    except httpx.RequestError as exc:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise StudioError(
            f"Downloading “{track.title}” failed.",
            hint=f"{exc.__class__.__name__}: {exc}",
        ) from exc

    if destination is None or written == 0:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise StudioError(f"“{track.title}” downloaded as an empty file.")

    logger.info("Downloaded %r (%d bytes) from %s", track.title, written, track.source)
    return destination


def prune_cache(cache_dir: Path, max_files: int = CACHE_MAX_FILES) -> int:
    """Evict the oldest cached downloads once the cache grows too large.

    Args:
        cache_dir: Directory holding cached tracks and audition clips.
        max_files: How many files to keep.

    Returns:
        The number of files removed.
    """
    if not cache_dir.is_dir():
        return 0

    files = sorted(
        (f for f in cache_dir.iterdir() if f.is_file()),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for stale in files[max_files:]:
        try:
            stale.unlink()
            removed += 1
        except OSError:  # pragma: no cover - best-effort housekeeping
            logger.debug("Could not evict %s", stale.name)
    if removed:
        logger.info("Evicted %d old file(s) from the music cache.", removed)
    return removed


def fetch_preview(track: Track, cache_dir: Path) -> Path:
    """Produce a short audition clip so a track can be heard before committing.

    Only the opening of the file is fetched -- enough to judge whether a bed
    suits the narration, in roughly two seconds rather than the ten to twenty a
    full track takes. The clip is then trimmed and faded so it ends cleanly
    instead of stopping mid-note on a truncated frame.

    Args:
        track: The track to audition.
        cache_dir: Directory for cached downloads.

    Returns:
        Path to a playable preview clip.

    Raises:
        StudioError: If the download fails or the clip cannot be decoded.
    """
    # Imported here so the module stays importable (and unit-testable) without
    # a working ffmpeg, which only the trimming step below actually needs.
    from pydub import AudioSegment

    # Checked before any network call: re-auditioning a track the user has
    # already heard must be instant, not a silent re-download.
    clip_path = cache_dir / f"clip_{_safe_stem(track)}.mp3"
    if clip_path.is_file() and clip_path.stat().st_size > 0:
        logger.debug("Reusing cached preview clip for %r", track.title)
        return clip_path

    raw = download_track(
        track,
        cache_dir,
        limit_bytes=PREVIEW_BYTES,
        prefix="preview",
        deadline_s=PREVIEW_DEADLINE_S,
        timeout=PREVIEW_TIMEOUT,
    )

    try:
        clip = AudioSegment.from_file(raw)[: PREVIEW_SECONDS * 1000]
        if len(clip) > PREVIEW_FADE_MS:
            clip = clip.fade_out(PREVIEW_FADE_MS)
        clip.export(clip_path, format="mp3", bitrate="128k")
        # The raw partial download has served its purpose; the trimmed clip is
        # what gets replayed, so keeping both would double the cache for nothing.
        raw.unlink(missing_ok=True)
    except Exception as exc:
        raise StudioError(
            f"Could not play a preview of “{track.title}”.",
            hint=f"The file may be in an unsupported format.\n\n{exc}",
        ) from exc

    prune_cache(cache_dir)
    logger.info("Prepared %.1fs preview of %r", len(clip) / 1000, track.title)
    return clip_path


def build_credits(track: Track, narration_filename: str) -> str:
    """Compose the credits text that accompanies a rendered narration.

    Args:
        track: The music track that was mixed in.
        narration_filename: Name of the exported audio file.

    Returns:
        The full text of the credits file.
    """
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    obligation = (
        "You MUST reproduce this credit wherever the audio is published."
        if track.requires_attribution
        else "This track is public domain: no credit is legally required, "
             "but crediting the artist is still good practice."
    )

    return "\n".join(
        [
            "BACKGROUND MUSIC CREDITS",
            "=" * 24,
            f"For: {narration_filename}",
            f"Generated: {stamp}",
            "",
            obligation,
            "",
            "-" * 24,
            track.credit_line(),
            "-" * 24,
            "",
            f"Title:    {track.title}",
            f"Artist:   {track.creator}",
            f"Licence:  {track.licence_label}",
            f"Terms:    {track.licence_url}",
            f"Source:   {track.landing_url or track.source}",
            "",
            "Found via Openverse (https://openverse.org).",
            "",
        ]
    )


def write_credits(track: Track, narration_path: Path) -> Path:
    """Write the credits file next to a rendered narration track.

    Args:
        track: The music track that was mixed in.
        narration_path: Path of the exported narration audio.

    Returns:
        Path of the credits file that was written.
    """
    destination = narration_path.with_suffix(".credits.txt")
    destination.write_text(build_credits(track, narration_path.name), encoding="utf-8")
    logger.info("Wrote credits to %s", destination.name)
    return destination
