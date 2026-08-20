"""Static configuration: the curated voice catalogue and tunable defaults.

Keeping these out of ``app.py`` means the UI file only describes *layout and
wiring*, and the processing modules only describe *logic*.  Anything a user
might reasonably want to change without reading code lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
SAMPLES_DIR: Final[Path] = PROJECT_ROOT / "samples"

# --------------------------------------------------------------------------
# Voice catalogue
# --------------------------------------------------------------------------
# The free Edge endpoint exposes exactly two native Hindi voices, and no voice
# on it advertises a style/emotion list -- SSML `mstts:express-as` is escaped
# and read aloud rather than interpreted. The roster below therefore leans on
# the two levers that do work: native voices where they exist, and the
# multilingual voices, which render Devanagari with a non-native accent.
#
# That accent trade-off is encoded in the *label* so it is visible at the
# moment of selection, not buried in documentation.


class VoiceKind(str, Enum):
    """Whether a voice is a native speaker of the text it will be given."""

    NATIVE = "native"
    MULTILINGUAL = "multilingual"


@dataclass(frozen=True)
class Voice:
    """One selectable narration voice.

    Attributes:
        label: What the dropdown shows. Carries the accent caveat for
            multilingual voices so the trade-off is visible at selection time.
        short_name: Edge-TTS voice identifier.
        language: Human-readable language this entry is intended for.
        kind: Native speaker of ``language``, or a multilingual stand-in.
    """

    label: str
    short_name: str
    language: str
    kind: VoiceKind = VoiceKind.NATIVE

    @property
    def is_native(self) -> bool:
        """True when this voice natively speaks its intended language."""
        return self.kind is VoiceKind.NATIVE

    @property
    def speaks_devanagari(self) -> bool:
        """True when this voice can pronounce Devanagari text at all.

        An English-only voice handed Hindi returns *no audio whatsoever* rather
        than mispronouncing it, so this distinction is the difference between a
        usable render and a total failure.
        """
        return self.language == "Hindi" or self.kind is VoiceKind.MULTILINGUAL


VOICE_CATALOGUE: Final[tuple[Voice, ...]] = (
    # -- Native Hindi. The complete hi-IN roster; Microsoft offers no others.
    Voice("🇮🇳 Hindi — Madhur (Male)", "hi-IN-MadhurNeural", "Hindi"),
    Voice("🇮🇳 Hindi — Swara (Female)", "hi-IN-SwaraNeural", "Hindi"),

    # -- Native Indian English.
    Voice("🇮🇳 English (India) — Prabhat (Male)", "en-IN-PrabhatNeural", "English (India)"),
    Voice("🇮🇳 English (India) — Neerja (Female)", "en-IN-NeerjaNeural", "English (India)"),
    Voice(
        "🇮🇳 English (India) — Neerja Expressive (Female)",
        "en-IN-NeerjaExpressiveNeural",
        "English (India)",
    ),

    # -- Native US English.
    Voice("🇺🇸 English (US) — Christopher (Male)", "en-US-ChristopherNeural", "English (US)"),
    Voice("🇺🇸 English (US) — Jenny (Female)", "en-US-JennyNeural", "English (US)"),

    # -- Multilingual voices rendering Hindi.
    # These are not hi-IN voices; they auto-detect Devanagari and speak it with
    # the accent of their own locale. They exist here purely because Microsoft
    # offers only two native Hindi voices, and eight imperfect extra options
    # beat none. Every label says "non-native" so nobody picks one by accident
    # for a native-audience project.
    Voice("🌐 Hindi — Ava (Female, non-native)",
          "en-US-AvaMultilingualNeural", "Hindi", VoiceKind.MULTILINGUAL),
    Voice("🌐 Hindi — Emma (Female, non-native)",
          "en-US-EmmaMultilingualNeural", "Hindi", VoiceKind.MULTILINGUAL),
    Voice("🌐 Hindi — Seraphina (Female, non-native)",
          "de-DE-SeraphinaMultilingualNeural", "Hindi", VoiceKind.MULTILINGUAL),
    Voice("🌐 Hindi — Vivienne (Female, non-native)",
          "fr-FR-VivienneMultilingualNeural", "Hindi", VoiceKind.MULTILINGUAL),
    Voice("🌐 Hindi — Andrew (Male, non-native)",
          "en-US-AndrewMultilingualNeural", "Hindi", VoiceKind.MULTILINGUAL),
    Voice("🌐 Hindi — Brian (Male, non-native)",
          "en-US-BrianMultilingualNeural", "Hindi", VoiceKind.MULTILINGUAL),
    Voice("🌐 Hindi — William (Male, non-native)",
          "en-AU-WilliamMultilingualNeural", "Hindi", VoiceKind.MULTILINGUAL),
    Voice("🌐 Hindi — Hyunsu (Male, non-native)",
          "ko-KR-HyunsuMultilingualNeural", "Hindi", VoiceKind.MULTILINGUAL),
)

#: Display label -> Edge-TTS short-name, in dropdown order.
VOICES: Final[dict[str, str]] = {v.label: v.short_name for v in VOICE_CATALOGUE}

#: Display label -> full :class:`Voice` record, for callers that need the
#: accent metadata rather than just the identifier.
VOICES_BY_LABEL: Final[dict[str, Voice]] = {v.label: v for v in VOICE_CATALOGUE}

DEFAULT_VOICE_LABEL: Final[str] = VOICE_CATALOGUE[0].label

# --------------------------------------------------------------------------
# Voice synthesis defaults
# --------------------------------------------------------------------------

RATE_MIN: Final[int] = -30
RATE_MAX: Final[int] = 30
RATE_DEFAULT: Final[int] = 0

PITCH_MIN: Final[int] = -20
PITCH_MAX: Final[int] = 20
PITCH_DEFAULT: Final[int] = 0

#: Number of subtitle lines synthesised concurrently.  Edge-TTS is a remote
#: websocket service; a handful of parallel connections is a large speedup,
#: but pushing it high invites throttling and dropped audio.
TTS_CONCURRENCY: Final[int] = 5

#: Attempts per line before it degrades to silence.
TTS_MAX_ATTEMPTS: Final[int] = 3

# --------------------------------------------------------------------------
# Timeline fitting
# --------------------------------------------------------------------------

#: A line may run this far past its slot before we bother compressing it.
#: Below this threshold the correction is inaudible and not worth the
#: resampling artefacts.
FIT_TOLERANCE_MS: Final[int] = 50

#: Upper bound on the pitch-preserving speed-up applied to over-long lines.
#: Past roughly 1.6x, Hindi and English narration both start to sound rushed.
MAX_TEMPO_MIN: Final[float] = 1.0
MAX_TEMPO_MAX: Final[float] = 2.5
MAX_TEMPO_DEFAULT: Final[float] = 1.5

#: ffmpeg's `atempo` filter accepts 0.5-2.0 per instance; larger factors must
#: be produced by chaining several instances together.
ATEMPO_MAX_PER_STAGE: Final[float] = 2.0

# --------------------------------------------------------------------------
# Mixing / mastering
# --------------------------------------------------------------------------

#: Target loudness for the narration bed before the music is mixed under it.
VOICE_TARGET_DBFS: Final[float] = -16.0

#: Peak ceiling for the finished master, leaving headroom for mp3 encoding.
MASTER_CEILING_DBFS: Final[float] = -1.0

EXPORT_BITRATE: Final[str] = "192k"
EXPORT_SAMPLE_RATE: Final[int] = 44_100
EXPORT_CHANNELS: Final[int] = 2

# --------------------------------------------------------------------------
# Background music & ducking
# --------------------------------------------------------------------------

BGM_BASE_DB_DEFAULT: Final[float] = -18.0
BGM_BASE_DB_MIN: Final[float] = -40.0
BGM_BASE_DB_MAX: Final[float] = 0.0

#: How much *further* the music drops while narration is playing.
DUCK_DEPTH_DB_DEFAULT: Final[float] = -12.0
DUCK_DEPTH_DB_MIN: Final[float] = -30.0
DUCK_DEPTH_DB_MAX: Final[float] = 0.0

#: Duration of the volume ramp on either side of a speech region.
DUCK_FADE_MS_DEFAULT: Final[int] = 300
DUCK_FADE_MS_MIN: Final[int] = 100
DUCK_FADE_MS_MAX: Final[int] = 1_000

#: How long the music stays ducked after speech stops, before it climbs back.
#: Prevents the music surging up inside a short breath between sentences.
DUCK_HOLD_MS_DEFAULT: Final[int] = 200

#: Fade applied at the very start and end of the music bed.
BGM_EDGE_FADE_MS: Final[int] = 1_000

#: Crossfade applied at each loop seam so a repeating track does not click.
BGM_LOOP_CROSSFADE_MS: Final[int] = 500
