"""Core processing pipeline for the SRT Voiceover Studio.

The package is deliberately split by responsibility so that each stage can be
tested in isolation:

* :mod:`core.parser`           -- ``.srt`` -> validated, ordered cues
* :mod:`core.tts`              -- cues -> one synthesised audio file per cue
* :mod:`core.audio_processor`  -- clips -> a single timed, mixed master track

:mod:`core.config` holds tunable constants and :mod:`core.errors` the
user-facing exception hierarchy.
"""

__all__ = ["audio_processor", "config", "errors", "parser", "tts"]
