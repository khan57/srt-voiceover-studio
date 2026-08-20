"""SRT Voiceover Studio -- Gradio front end.

This module is deliberately thin: it describes the layout, binds events, and
acts as the error boundary between the processing pipeline and the browser.
All timing, synthesis, and mixing logic lives in :mod:`core`.

Run with::

    python app.py
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from core import audio_processor as ap
from core.config import (
    BGM_BASE_DB_DEFAULT,
    BGM_BASE_DB_MAX,
    BGM_BASE_DB_MIN,
    DEFAULT_VOICE_LABEL,
    DUCK_DEPTH_DB_DEFAULT,
    DUCK_DEPTH_DB_MAX,
    DUCK_DEPTH_DB_MIN,
    DUCK_FADE_MS_DEFAULT,
    DUCK_FADE_MS_MAX,
    DUCK_FADE_MS_MIN,
    DUCK_HOLD_MS_DEFAULT,
    MAX_TEMPO_DEFAULT,
    MAX_TEMPO_MAX,
    MAX_TEMPO_MIN,
    OUTPUT_DIR,
    PITCH_DEFAULT,
    PITCH_MAX,
    PITCH_MIN,
    RATE_DEFAULT,
    RATE_MAX,
    RATE_MIN,
    VOICE_CATALOGUE,
    VOICES,
    VOICES_BY_LABEL,
)
from core.errors import StudioError
from core.music_library import (
    PREVIEW_SECONDS,
    SearchResult,
    Track,
    download_track,
    fetch_preview,
    search_music,
    write_credits,
)
from core.parser import parse_srt
from core.tts import synthesize_all_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("srt_studio")

#: Where tracks fetched from the music library are kept. Deliberately outside
#: the per-render temp directory so a downloaded bed survives repeated renders
#: instead of being re-fetched every time.
MUSIC_CACHE_DIR = OUTPUT_DIR.parent / ".music_cache"


# ==========================================================================
# Presentation helpers
# ==========================================================================

def format_timecode(milliseconds: int) -> str:
    """Render a millisecond count as ``HH:MM:SS.mmm``.

    Args:
        milliseconds: Duration or offset to format.

    Returns:
        A zero-padded timecode string.
    """
    ms = max(0, int(milliseconds))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def _idle_metrics() -> str:
    """Placeholder shown in the metrics panel before the first render."""
    return (
        "### Ready\n"
        "Upload an `.srt` file and press **Generate Narration**.\n\n"
        "| Metric | Value |\n| --- | --- |\n"
        "| Subtitles | — |\n| Total duration | — |\n| Ducking | — |"
    )


def _build_metrics(
    *,
    report,
    voice_track,
    voice_label: str,
    rate: int,
    pitch: int,
    duration_ms: int,
    bgm: ap.BgmResult | None,
    duck_db: float,
    base_db: float,
    elapsed_s: float,
    output_path: Path,
    track: Track | None = None,
    credits_path: Path | None = None,
) -> str:
    """Assemble the Markdown summary shown beside the audio player.

    Args:
        report: The :class:`~core.parser.ParseReport` for this run.
        voice_track: The :class:`~core.audio_processor.VoiceTrack` built.
        voice_label: Display name of the chosen voice.
        rate: Applied speech-rate percentage.
        pitch: Applied pitch offset in Hz.
        duration_ms: Length of the finished master.
        bgm: Prepared music bed, or ``None`` when ducking was bypassed.
        duck_db: Ducking depth that was applied.
        base_db: Resting music level.
        elapsed_s: Wall-clock time the render took.
        output_path: Where the mp3 was written.
        track: The library track used as music, if any.
        credits_path: Attribution file written beside the audio, if any.

    Returns:
        A Markdown string.
    """
    speech_ms = sum(end - start for start, end in voice_track.intervals)
    coverage = (speech_ms / duration_ms * 100) if duration_ms else 0.0

    if bgm is None:
        ducking = "Bypassed — pure voice track"
        music_row = "| Background music | None |\n"
    else:
        ducking = f"Active — {base_db:+.0f} dB base, {duck_db:+.0f} dB under speech"
        music_row = (
            f"| Background music | {format_timecode(bgm.source_ms)} source, "
            f"{bgm.loops} loop(s) |\n"
        )

    tempo_note = (
        f"{voice_track.compressed_lines} line(s), up to {voice_track.max_tempo_used:.2f}×"
        if voice_track.compressed_lines
        else "None needed"
    )

    rows = (
        "| Metric | Value |\n| --- | --- |\n"
        f"| Subtitles voiced | {len(report.cues) - voice_track.failed_lines} of {len(report.cues)} |\n"
        f"| Total duration | {format_timecode(duration_ms)} |\n"
        f"| Speech coverage | {coverage:.0f}% |\n"
        f"| Voice | {voice_label} |\n"
        f"| Rate / Pitch | {rate:+d}% / {pitch:+d} Hz |\n"
        f"| Time-compressed | {tempo_note} |\n"
        f"| Ducking | {ducking} |\n"
        + music_row
        + f"| Source encoding | {report.encoding} |\n"
        f"| Render time | {elapsed_s:.1f}s |\n"
    )

    licence_note = ""
    if track is not None:
        licence_note = (
            f"\n\n**Music licence**\n- {track.credit_line()}\n"
            f"- Licence: {track.licence_label}"
        )
        if credits_path is not None:
            licence_note += f"\n- Credit saved to `{credits_path.name}`"

    warnings = [*report.warnings, *voice_track.warnings]
    notes = ""
    if warnings:
        notes = "\n\n**Notes**\n" + "\n".join(f"- {w}" for w in warnings)

    return f"### ✅ Render complete\n`{output_path.name}`\n\n{rows}{licence_note}{notes}"


# ==========================================================================
# Music library
# ==========================================================================

def _search_music(query: str) -> tuple[SearchResult | None, dict, str]:
    """Search the free music library and populate the results picker.

    Args:
        query: Free-text search entered by the user.

    Returns:
        A triple of ``(search_result, radio_update, status_markdown)``.

    Raises:
        gr.Error: If the library is unreachable or rate-limited.
    """
    try:
        result = search_music(query)
    except StudioError as exc:
        logger.warning("Music search failed: %s", exc.message)
        raise gr.Error(str(exc)) from exc

    if not result.tracks:
        return (
            None,
            gr.update(choices=[], value=None),
            f"No freely-reusable music found for **{query}**. "
            "Try a single broad word such as *piano*, *ambient* or *cinematic*.",
        )

    # Values are indices, so two tracks with identical labels stay distinct.
    choices = [(t.choice_label, i) for i, t in enumerate(result.tracks)]
    note = (
        f"Nothing matched **{result.query}**, so this shows results for "
        f"**{result.effective_query}** instead.\n\n"
        if result.was_broadened
        else ""
    )
    status = (
        f"{note}**{len(result.tracks)} track(s)** you may legally mix under "
        "narration — CC0 and CC BY only. Select one and press *▶ Preview*."
    )
    return result, gr.update(choices=choices, value=0), status


def _clear_preview(
    result: SearchResult | None,
    selection: int | None,
) -> tuple[None, str]:
    """Drop the previous audition clip when a different result is selected.

    Deliberately does no network work: browsing twenty results should not mean
    twenty downloads. The clip only arrives when the user presses Preview.

    Args:
        result: The active search result set.
        selection: Index of the newly highlighted track.

    Returns:
        ``(None, status_markdown)`` -- clearing the player and describing the
        highlighted track.
    """
    if result is None or selection is None:
        return None, "Search for some music, then press **▶ Preview** to hear it."

    try:
        track = result.tracks[int(selection)]
    except (IndexError, ValueError):
        return None, "That track is no longer in the results; search again."

    credit = "credit required" if track.requires_attribution else "public domain"
    return None, (
        f"**{track.title}** — {track.creator}\n\n"
        f"{track.duration_label} · {track.licence_label} · {credit}.\n\n"
        "Press **▶ Preview** to hear it, or **Use this track** to mix it in."
    )


def _preview_track(
    result: SearchResult | None,
    selection: int | None,
) -> tuple[str | None, str]:
    """Fetch a short audition clip for the highlighted search result.

    Only the opening of the track is downloaded, so browsing the results stays
    responsive instead of pulling several megabytes per click.

    Args:
        result: The active search result set.
        selection: Index of the highlighted track.

    Returns:
        ``(clip_path, status_markdown)``; the path is ``None`` if nothing is
        selected.
    """
    if result is None or selection is None:
        return None, "Select a track from the results first."

    try:
        track = result.tracks[int(selection)]
    except (IndexError, ValueError):
        return None, "That track is no longer in the results; search again."

    try:
        clip = fetch_preview(track, MUSIC_CACHE_DIR)
    except StudioError as exc:
        # A single unplayable track should not derail browsing, so this is
        # reported inline rather than raised as a blocking error.
        logger.warning("Preview failed for %r: %s", track.title, exc.message)
        return None, f"⚠️ {exc.message}"

    credit = (
        "Credit required" if track.requires_attribution else "Public domain"
    )
    status = (
        f"**Previewing:** {track.title} — {track.creator}\n\n"
        f"First {PREVIEW_SECONDS}s · {track.duration_label} full length · "
        f"{track.licence_label} · {credit}.\n\n"
        "Press **Use this track** to mix it under your narration."
    )
    return str(clip), status


def _use_track(
    result: SearchResult | None,
    selection: int | None,
    known: dict[str, Track],
) -> tuple[str | None, dict[str, Track], str]:
    """Download the chosen track and load it as the background music.

    Args:
        result: The active search result set.
        selection: Index of the chosen track within ``result.tracks``.
        known: Basename -> track map of everything downloaded this session.

    Returns:
        ``(audio_path, updated_known_map, status_markdown)``.

    Raises:
        gr.Error: If nothing is selected or the download fails.
    """
    if result is None or selection is None:
        raise gr.Error("Search for some music and select a track first.")
    try:
        track = result.tracks[int(selection)]
    except (IndexError, ValueError) as exc:
        raise gr.Error("That track is no longer in the results; search again.") from exc

    try:
        path = download_track(track, MUSIC_CACHE_DIR)
    except StudioError as exc:
        logger.warning("Track download failed: %s", exc.message)
        raise gr.Error(str(exc)) from exc

    # Keyed by basename because Gradio copies uploads into a hashed directory
    # but preserves the filename, which is how the credits are matched back up
    # at render time.
    known = {**known, Path(path).name: track}

    credit = (
        f"Credit required — a `.credits.txt` file will be saved next to your "
        f"narration.\n\n> {track.credit_line()}"
        if track.requires_attribution
        else "Public domain — no credit required."
    )
    status = (
        f"**Loaded:** {track.title} — {track.creator}\n\n"
        f"Licence: {track.licence_label}. {credit}"
    )
    gr.Info(f"Loaded “{track.title}” as background music.")
    return str(path), known, status


# ==========================================================================
# Main handler
# ==========================================================================

def generate_narration(
    srt_file: str | None,
    voice_label: str,
    rate: float,
    pitch: float,
    max_tempo: float,
    bgm_file: str | None,
    enable_ducking: bool,
    base_db: float,
    duck_db: float,
    fade_ms: float,
    known_tracks: dict[str, Track] | None = None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str]:
    """Render a subtitle file into a synchronised master narration track.

    Args:
        srt_file: Path to the uploaded ``.srt``.
        voice_label: Key into :data:`~core.config.VOICES`.
        rate: Speech-rate adjustment in percent.
        pitch: Pitch adjustment in Hz.
        max_tempo: Ceiling on time-compression for over-long lines.
        bgm_file: Optional path to a background music file.
        enable_ducking: Whether to mix and duck the music at all.
        base_db: Resting music level in dB.
        duck_db: Extra attenuation applied to music during speech.
        fade_ms: Ducking ramp duration in milliseconds.
        known_tracks: Basename -> track map for music fetched from the library,
            used to emit the correct attribution file.
        progress: Gradio progress reporter, injected by the framework.

    Returns:
        A ``(audio_path, metrics_markdown)`` pair for the output components.

    Raises:
        gr.Error: For every failure mode, carrying a message written for the
            person using the app rather than a traceback.
    """
    started = time.monotonic()
    workdir: Path | None = None

    try:
        # --- Preconditions -------------------------------------------------
        progress(0.0, desc="Checking audio tools…")
        ap.ensure_ffmpeg()

        if not srt_file:
            raise StudioError("Please upload a subtitle (.srt) file first.")
        if voice_label not in VOICES:
            raise StudioError(f"Unknown voice: {voice_label!r}.")

        voice = VOICES[voice_label]
        rate_i, pitch_i = int(round(rate)), int(round(pitch))

        # --- Parse ---------------------------------------------------------
        progress(0.02, desc="Reading subtitles…")
        report = parse_srt(srt_file)
        total = len(report.cues)
        logger.info(
            "Parsed %d cue(s), timeline %s.", total,
            format_timecode(report.total_duration_ms),
        )

        # Pre-flight: an English-only voice handed Devanagari returns no audio
        # at all, so catch the mismatch here rather than after a full round of
        # retries against every line.
        selected = VOICES_BY_LABEL[voice_label]
        if report.is_devanagari and not selected.speaks_devanagari:
            capable = [v.label for v in VOICE_CATALOGUE if v.speaks_devanagari]
            raise StudioError(
                f"These subtitles are in Hindi, but “{voice_label}” cannot "
                "pronounce Devanagari.",
                hint=(
                    "Pick a Hindi-capable voice instead:\n  - "
                    + "\n  - ".join(capable)
                ),
            )

        workdir = Path(tempfile.mkdtemp(prefix="srtvo_"))

        # --- Synthesise ----------------------------------------------------
        # Synthesis dominates wall-clock time, so it owns the bulk of the bar.
        def report_progress(done: int, count: int) -> None:
            progress(0.05 + 0.70 * (done / count), desc=f"Synthesising line {done}/{count}…")

        results = synthesize_all_sync(
            report.cues, voice, rate_i, pitch_i, workdir, on_progress=report_progress,
        )

        # --- Assemble the timeline ------------------------------------------
        progress(0.78, desc="Aligning to subtitle timestamps…")
        track = ap.build_voice_track(report.cues, results, max_tempo, workdir)

        # --- Music & ducking -------------------------------------------------
        bgm_result: ap.BgmResult | None = None
        ducked = None
        if enable_ducking and bgm_file:
            progress(0.86, desc="Preparing background music…")
            bgm_result = ap.prepare_bgm(bgm_file, len(track.audio), base_db)

            progress(0.90, desc="Ducking music under narration…")
            ducked = ap.apply_ducking(
                bgm_result.audio,
                track.intervals,
                duck_db=duck_db,
                fade_ms=int(round(fade_ms)),
                hold_ms=DUCK_HOLD_MS_DEFAULT,
            )

        # --- Master & export --------------------------------------------------
        progress(0.94, desc="Mixing master…")
        master = ap.mix_master(track.audio, ducked)

        progress(0.97, desc="Encoding MP3…")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = ap.export_master(master, OUTPUT_DIR / f"narration_{stamp}.mp3")

        # A track from the library may carry an attribution obligation, so the
        # credit is written beside the audio rather than left to the user.
        credits_path: Path | None = None
        library_track = (known_tracks or {}).get(Path(bgm_file).name) if bgm_file else None
        if bgm_result is not None and library_track is not None:
            credits_path = write_credits(library_track, output_path)

        elapsed = time.monotonic() - started
        logger.info("Rendered %s in %.1fs.", output_path.name, elapsed)
        progress(1.0, desc="Done")

        for warning in (*report.warnings, *track.warnings):
            gr.Warning(warning)

        metrics = _build_metrics(
            report=report, voice_track=track, voice_label=voice_label,
            rate=rate_i, pitch=pitch_i, duration_ms=len(master),
            bgm=bgm_result, duck_db=duck_db, base_db=base_db,
            elapsed_s=elapsed, output_path=output_path,
            track=library_track, credits_path=credits_path,
        )
        return str(output_path), metrics

    except StudioError as exc:
        # Expected, explainable failures: show the message as written.
        logger.warning("Render aborted: %s", exc.message)
        raise gr.Error(str(exc)) from exc

    except Exception as exc:
        # Anything else is a bug. Log it in full, report it briefly.
        logger.exception("Unexpected failure during render")
        raise gr.Error(
            f"Something went wrong while generating the narration.\n\n"
            f"{exc.__class__.__name__}: {exc}\n\n"
            "The full traceback has been written to the terminal."
        ) from exc

    finally:
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)


# ==========================================================================
# Interface
# ==========================================================================

CUSTOM_CSS = """
#studio-header h1 { margin-bottom: 0.15rem; font-weight: 700; letter-spacing: -0.02em; }
#studio-header p  { margin-top: 0; opacity: 0.72; }
#generate-btn     { font-size: 1.05rem; font-weight: 600; padding: 0.85rem 1rem; }
#metrics-panel    { min-height: 320px; }
.gradio-container { max-width: 1240px !important; }
"""


def build_interface() -> gr.Blocks:
    """Construct the Gradio Blocks app.

    Returns:
        The assembled, un-launched interface.
    """
    theme = gr.themes.Ocean(font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

    with gr.Blocks(
        theme=theme,
        css=CUSTOM_CSS,
        title="SRT Voiceover Studio",
        # This is a local, offline-first tool; do not phone home.
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            "# 🎙️ SRT Voiceover Studio\n"
            "Turn subtitles into a perfectly timed narration track — with "
            "background music that ducks itself out of the way.",
            elem_id="studio-header",
        )

        with gr.Row(equal_height=False):
            # ---------------- Left: inputs ----------------
            with gr.Column(scale=5):
                srt_input = gr.File(
                    label="Subtitle file",
                    file_types=[".srt"],
                    file_count="single",
                    type="filepath",
                    height=140,
                )

                voice_input = gr.Dropdown(
                    label="Narration voice",
                    choices=list(VOICES.keys()),
                    value=DEFAULT_VOICE_LABEL,
                    filterable=True,
                    info=(
                        "Hindi and Indian English are listed first. Voices marked "
                        "non-native render Devanagari with a foreign accent."
                    ),
                )

                with gr.Accordion("🎛️  Fine-Tune Voice", open=False):
                    rate_input = gr.Slider(
                        RATE_MIN, RATE_MAX, value=RATE_DEFAULT, step=1,
                        label="Speech rate (%)",
                        info="Negative is slower, positive is faster.",
                    )
                    pitch_input = gr.Slider(
                        PITCH_MIN, PITCH_MAX, value=PITCH_DEFAULT, step=1,
                        label="Voice pitch (Hz)",
                        info="Shifts the voice lower or higher.",
                    )
                    tempo_input = gr.Slider(
                        MAX_TEMPO_MIN, MAX_TEMPO_MAX, value=MAX_TEMPO_DEFAULT, step=0.05,
                        label="Max time-compression",
                        info=(
                            "Lines longer than their subtitle slot are sped up "
                            "(without changing pitch) by at most this much."
                        ),
                    )

                with gr.Accordion("🎵  Background Music & Ducking (optional)", open=False):
                    ducking_input = gr.Checkbox(
                        label="Mix in background music",
                        value=True,
                        info="Uncheck to export a pure voice track.",
                    )
                    with gr.Tab("Find free music"):
                        gr.Markdown(
                            "Search Creative Commons music that may legally be "
                            "mixed under narration. Only **CC0** and **CC BY** "
                            "are offered — NonCommercial and NoDerivatives "
                            "licences forbid this use."
                        )
                        with gr.Row():
                            music_query = gr.Textbox(
                                label="Search",
                                placeholder="piano, ambient, cinematic…",
                                scale=3,
                                container=True,
                            )
                            music_search_btn = gr.Button(
                                "Search", variant="secondary", scale=1,
                            )
                        music_results = gr.Radio(
                            label="Results", choices=[], value=None,
                        )
                        with gr.Row():
                            music_preview_btn = gr.Button("▶  Preview", scale=1)
                            music_use_btn = gr.Button(
                                "Use this track", variant="secondary", scale=1,
                            )
                        music_preview = gr.Audio(
                            label="Preview",
                            type="filepath",
                            interactive=False,
                            autoplay=True,
                            show_download_button=False,
                        )
                        music_status = gr.Markdown(
                            "Single broad words work best — the library matches "
                            "*every* word you type."
                        )

                    with gr.Tab("Upload your own"):
                        bgm_input = gr.Audio(
                            label="Music file (MP3 / WAV)",
                            type="filepath",
                            sources=["upload"],
                        )
                    base_db_input = gr.Slider(
                        BGM_BASE_DB_MIN, BGM_BASE_DB_MAX, value=BGM_BASE_DB_DEFAULT, step=1,
                        label="Music level (dB)",
                        info="Resting volume of the music between lines.",
                    )
                    duck_db_input = gr.Slider(
                        DUCK_DEPTH_DB_MIN, DUCK_DEPTH_DB_MAX, value=DUCK_DEPTH_DB_DEFAULT, step=1,
                        label="Ducking depth (dB)",
                        info="How much further the music drops while the voice speaks.",
                    )
                    fade_input = gr.Slider(
                        DUCK_FADE_MS_MIN, DUCK_FADE_MS_MAX, value=DUCK_FADE_MS_DEFAULT, step=25,
                        label="Duck fade (ms)",
                        info="Length of the volume ramp on either side of a line.",
                    )

                generate_btn = gr.Button(
                    "Generate Narration", variant="primary", size="lg", elem_id="generate-btn",
                )

            # ---------------- Right: output ----------------
            with gr.Column(scale=4):
                audio_output = gr.Audio(
                    label="Master narration track",
                    type="filepath",
                    interactive=False,
                    show_download_button=True,
                    waveform_options=gr.WaveformOptions(show_recording_waveform=True),
                )
                metrics_output = gr.Markdown(_idle_metrics(), elem_id="metrics-panel")

        gr.Markdown(
            "<sub>Voices are synthesised locally through Microsoft Edge's free "
            "neural TTS service — no API key needed. Rendered files are saved to "
            "`outputs/`.</sub>"
        )

        # Holds the active search results and every track fetched this session.
        search_state = gr.State(None)
        known_tracks_state = gr.State({})

        music_search_btn.click(
            fn=_search_music,
            inputs=[music_query],
            outputs=[search_state, music_results, music_status],
        )
        music_query.submit(
            fn=_search_music,
            inputs=[music_query],
            outputs=[search_state, music_results, music_status],
        )
        # Selecting a result only clears the previous clip -- nothing is
        # fetched until the user actually asks to hear it, so browsing the list
        # never triggers a download.
        music_results.change(
            fn=_clear_preview,
            inputs=[search_state, music_results],
            outputs=[music_preview, music_status],
        )
        music_preview_btn.click(
            fn=_preview_track,
            inputs=[search_state, music_results],
            outputs=[music_preview, music_status],
        )
        music_use_btn.click(
            fn=_use_track,
            inputs=[search_state, music_results, known_tracks_state],
            outputs=[bgm_input, known_tracks_state, music_status],
        )

        generate_btn.click(
            fn=generate_narration,
            inputs=[
                srt_input, voice_input, rate_input, pitch_input, tempo_input,
                bgm_input, ducking_input, base_db_input, duck_db_input, fade_input,
                known_tracks_state,
            ],
            outputs=[audio_output, metrics_output],
            show_progress="full",
        )

    return demo


def main() -> None:
    """Parse CLI arguments, run start-up checks, and launch the server."""
    parser = argparse.ArgumentParser(description="SRT Voiceover Studio")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to bind.")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = parser.parse_args()

    try:
        ffmpeg, _ = ap.ensure_ffmpeg()
        logger.info("Using ffmpeg at %s", ffmpeg)
    except StudioError as exc:
        # Surfaced now as a clear terminal message; the handler repeats it in
        # the UI, so the app still starts and stays diagnosable.
        logger.error("%s", exc)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    build_interface().queue().launch(
        server_name=args.host, server_port=args.port, share=args.share, show_error=True,
    )


if __name__ == "__main__":
    main()
