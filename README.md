# 🎙️ SRT Voiceover Studio

Turn an `.srt` subtitle file into a single, perfectly timed master `.mp3`
narration track — with optional background music that automatically ducks out
of the way whenever the narrator speaks.

Built for **Hindi (hi-IN)** and **English (en-IN, en-US)** first. Runs entirely
on your machine, uses Microsoft Edge's free neural voices, and needs **no API
key and no account**.

15 voices, **10 of them usable for Hindi**.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue.svg)](https://www.python.org/downloads/)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-145%20passing-brightgreen.svg)](tests/)

```bash
git clone git@github.com:khan57/srt-voiceover-studio.git && cd srt-voiceover-studio
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
./.venv/bin/python app.py
```

Then open <http://127.0.0.1:7860>. Full setup, including the Python 3.13
`audioop` caveat, is in [Setup](#setup).

---

## Why it exists

Text-to-speech gives you audio. It does not give you audio that *lines up with
your video*. This tool solves the two problems that stand between the two:

**Timing.** Every line is stamped onto the timeline at its exact subtitle
timestamp, on a pre-allocated silent canvas — not concatenated end to end. A
line that runs long therefore cannot push every later line out of place, so
cumulative drift is structurally impossible rather than merely corrected for.
Lines that overshoot their slot are time-compressed with ffmpeg's `atempo`,
which preserves pitch (naive resampling turns narration into a chipmunk).

**Ducking.** Music is attenuated around the regions where narration is
*actually audible*, not for the whole subtitle block. A two-second line sitting
in a five-second slot gets its music back for the three-second tail. The
envelope ramps down *before* the first syllable and holds briefly after the
last, so the music never surges up inside a breath between sentences.

---

## Requirements

* Python **3.10 – 3.12** recommended (3.13+ works — see the note below)
* **ffmpeg** on your `PATH`

---

## Setup

### macOS (Apple Silicon M1/M2/M3 and Intel)

```bash
brew install ffmpeg python@3.11
```

```bash
git clone <your-repo-url> tts && cd tts
```

Using [uv](https://github.com/astral-sh/uv) (fast, recommended):

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
```

Or with plain `venv`:

```bash
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

### Linux (Debian / Ubuntu)

```bash
sudo apt update && sudo apt install -y ffmpeg python3.11 python3.11-venv
```

```bash
python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

### Run it

```bash
./.venv/bin/python app.py
```

Then open <http://127.0.0.1:7860>. Useful flags: `--port 7861`, `--host 0.0.0.0`,
`--share`.

---

## Using it

1. **Upload an `.srt`.** UTF-8, UTF-16, and BOM'd files all work; Devanagari is
   preserved end to end.
2. **Pick a voice.** 15 curated neural voices, Hindi and Indian English first.
   See [Voices](#voices) for the native-vs-multilingual trade-off.
3. **Fine-Tune Voice** *(optional)* — speech rate (−30%…+30%), pitch
   (−20…+20 Hz), and the max time-compression cap.
4. **Background Music & Ducking** *(optional)* — either **search the built-in
   free music library** (with **▶ Preview** to audition before committing) or
   upload your own MP3/WAV. Either way the track is
   looped or trimmed to the exact timeline length automatically. Leave the box
   unchecked to export a pure voice track. See [Free music](#free-music).
5. **Generate Narration.** Progress runs line by line. The finished file lands
   in `outputs/` and is playable and downloadable in the browser.

### The controls that matter

| Control | What it does | Good default |
| --- | --- | --- |
| **Max time-compression** | Ceiling on how much an over-long line may be sped up to fit its slot. Higher = tighter sync, lower = more natural delivery. | `1.5×` |
| **Music level** | Resting volume of the music between lines. | `−18 dB` |
| **Ducking depth** | How much *further* the music drops while the voice speaks. | `−12 dB` |
| **Duck fade** | Length of the volume ramp on either side of a line. Longer is smoother but needs wider gaps. | `300 ms` |

> **A note on tight subtitles.** The music needs roughly
> `fade + hold + fade` (≈ 800 ms at the defaults) of clear gap to climb all the
> way back to its resting level. With gaps shorter than that it stays partly
> ducked between lines — which is what you want, and what a human mix engineer
> would do. Shorten the fade if you would rather hear the music breathe
> between closely spaced lines.

### Reading the metrics panel

* **Subtitles voiced** — lines that produced audio. A line that fails every
  retry is left silent rather than aborting the whole render.
* **Time-compressed** — how many lines had to be sped up, and by how much. If
  this is high, your subtitles are dense for the slots they occupy.
* **Notes** — flags overlapping slots, skipped empty cues, and lines that still
  overrun at the compression cap.

---

## Free music

The **Find free music** tab searches [Openverse](https://openverse.org) for
Creative Commons audio. No API key and no account — the same standard the rest
of the app holds to.

### Only two licences are offered, and that is deliberate

Mixing a bed under narration creates a **derivative work**, and publishing the
result is usually **commercial**. The two Creative Commons clauses that forbid
exactly those things are the two most common ones in the wild:

| Clause | Meaning | Usable here |
| --- | --- | --- |
| `CC0` | Public domain | ✅ no credit needed |
| `BY` | Credit the artist | ✅ credit required |
| `BY-SA` | ShareAlike | ❌ would force your video under the same licence |
| `BY-NC` | NonCommercial | ❌ forbids commercial use |
| `BY-ND` | NoDerivatives | ❌ forbids mixing at all |

An unfiltered search for *cinematic background* returns **16 unusable tracks
out of 20** — mostly `BY-NC-ND`. The app therefore filters to `CC0` and `CC BY`
in the query *and* re-checks every result before displaying it, so a change at
the far end cannot quietly widen what you are offered.

### Credits are written for you

Pick a `CC BY` track and the app saves
`narration_<stamp>.credits.txt` beside the MP3, containing the ready-made
attribution line, the artist, and the licence URL. Reproduce it wherever you
publish the audio — that is the whole obligation the licence imposes.

`CC0` tracks get a credits file too, noting that no attribution is required.

**Uploading your own music writes no credits file.** The app only vouches for
what it found itself; rights for anything you supply are yours to clear.

### Auditioning before you commit

Select a result and press **▶ Preview** to hear it. Only the opening of the
track is fetched — MP3 is frame-based, so a truncated download still decodes —
which makes an audition about **2 seconds** instead of the ten to twenty a full
track takes. The clip is trimmed to 30 seconds and faded out so it ends cleanly.

Previewing is **opt-in on purpose**. Selecting a result downloads nothing, so
clicking down a list of twenty costs nothing and does not hammer the source
CDN. Nothing is fetched until you ask to hear it.

Auditioning a track and then using it does **not** download it twice, and
re-previewing something you have already heard is instant — both are served
from `.music_cache/`, which keeps the 40 most recent files and evicts the rest.

Source CDNs vary in speed. A preview is bounded by a 15-second wall-clock
budget; if a host is too slow, the app says so inline and you can pick another
track rather than sitting through it.

### Searching effectively

Openverse matches **every** word you type, so precise phrases match nothing —
*calm cinematic piano* returns zero results. The app handles this by retrying
automatically with progressively broader queries (*cinematic piano*, then
*cinematic*) and telling you when it did. Single broad words still work best.

Anonymous use is limited to **20 searches per minute and 200 per day**, which
is ample for normal use. Downloads come from the original host and can take
10–20 seconds.

---

## Voices

### The Hindi ceiling, and how this roster gets around it

Microsoft's free endpoint carries 322 voices but exactly **two native Hindi
voices** — `Madhur` and `Swara`. There is no third, and no amount of
configuration produces one.

The way past that is the **multilingual** voices, which auto-detect Devanagari
and speak it in the accent of their own locale. They are not Hindi voices and
this app never pretends otherwise: every one is labelled *non-native* in the
dropdown itself, so the trade-off is visible when you choose rather than buried
in this file. Eight are offered, which takes Hindi from 2 options to **10** —
and male Hindi from 1 to 5, which was the sharpest gap.

| | Native Hindi | Multilingual (non-native) |
| --- | --- | --- |
| Female | Swara | Ava, Emma, Seraphina, Vivienne |
| Male | Madhur | Andrew, Brian, William, Hyunsu |

Judge them by ear before shipping one to a native-speaking audience.

### Mixing up script and voice

An English-only voice handed Devanagari returns **no audio at all** — it does
not mispronounce it, it produces silence. The app checks the script of your
subtitles against the chosen voice up front and refuses with a list of
Hindi-capable voices, rather than failing line by line and blaming the network.

Romanised Hindi ("Namaste doston") is Latin script, so it passes the check and
any English voice will read it.

### Why there is no emotion control

The free Edge endpoint reports **zero voices with a style list** — none of the
`cheerful` / `sad` / `angry` styles that Azure's paid Speech service exposes
through `mstts:express-as`.

Worse, sending SSML does not fail cleanly: the markup is escaped and **read
aloud**. In testing, one word wrapped in a `<prosody>` tag went from 1776 ms to
5184 ms because the voice narrated the tag. The parser strips `<…>` and `{…}`
from subtitle text, so this cannot leak into your audio — but it does mean
emotional delivery is out of reach here.

What *does* work is the **rate**, **pitch** and **volume** parameters, which are
applied as real synthesis settings rather than markup. For genuine emotional
control you would need Azure Speech (paid) or a local model such as AI4Bharat's
Indic Parler-TTS.

---

## Project layout

```
app.py                     Gradio UI, event binding, error boundary
core/
  config.py                Voice catalogue and tunable defaults
  errors.py                User-facing exception hierarchy
  music_library.py         Openverse search, safe download, attribution
  parser.py                .srt -> validated, ordered cues
  tts.py                   Async Edge-TTS worker pool with retry
  audio_processor.py       Timeline assembly, atempo fit, ducking, mastering
tests/                     Offline test suite (no network required)
requirements-dev.txt       Test-only dependencies (pytest)
samples/                   Demo Hindi and English subtitle files
outputs/                   Generated .mp3 files and .credits.txt
.music_cache/              Tracks downloaded from the music library
```

Run the tests with:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt && ./.venv/bin/python -m pytest
```

Tests that hit the live Edge-TTS and Openverse services are deselected by
default. Run them — they verify every catalogued voice still exists and that
the music search still returns only permitted licences — with:

```bash
./.venv/bin/python -m pytest -m network
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'audioop'`**
`pydub` depends on the stdlib `audioop` module, which **Python 3.13 removed**
(PEP 594). `requirements.txt` already pulls in the `audioop-lts` backport on
3.13+, so a clean install handles this. If you hit it anyway, either install
the shim directly or drop to Python 3.11:

```bash
pip install audioop-lts
```

**`Required audio tool(s) not found on PATH: ffmpeg`**
Install ffmpeg (`brew install ffmpeg` / `sudo apt install ffmpeg`) and restart
the app. The check runs at startup and again on every render.

**`Every line failed to synthesise`**
Edge-TTS is an online service. Check your internet connection, and any proxy or
firewall that might block outbound WebSocket connections.

**`These subtitles are in Hindi, but ... cannot pronounce Devanagari`**
You picked an English-only voice for a Hindi file. Choose one of the Hindi
voices the message lists — see [Voices](#voices).

**`The music library's rate limit has been reached`**
Openverse allows 200 anonymous searches per day. Wait, or upload your own music
on the *Upload your own* tab.

**`The music library took too long to respond`**
Openverse is intermittently slow on filtered searches. The app already retries
once; try again, or use a shorter query.

**`<track> is downloading too slowly`**
That track's source host is not responding quickly enough for the 15-second
preview budget. Pick another track — availability varies by host, not by
anything on your machine.

**Narration sounds rushed**
Too many lines are hitting the compression cap. Lower **Max time-compression**
for more natural delivery, use a negative **Speech rate**, or lengthen the
subtitle slots in your `.srt`.

**Could not read the subtitle file**
Re-save it as UTF-8. The app tries UTF-8, UTF-16, and two single-byte fallbacks,
and rejects any decode that produces replacement characters rather than
silently narrating mojibake.

---

## How it works

```
.srt ──▶ parser.py ──▶ validated cues (sorted, non-empty, positive duration)
                            │
                            ▼
                        tts.py ──▶ one mp3 per line
                     (5 concurrent, 3 retries each,
                      a failed line degrades to silence)
                            │
                            ▼
                   audio_processor.py
                     ├─ fit each clip to its slot (atempo, capped)
                     ├─ overlay onto a silent canvas at absolute timestamps
                     ├─ normalise loudness over the speech only
                     ├─ loop/trim music, build a NumPy gain envelope, duck
                     └─ mix, limit to −1 dBFS, encode 192 kbps MP3
                            │
                            ▼
                     outputs/narration_<timestamp>.mp3
```

---

## Contributing

Issues and pull requests are welcome. Install the dev dependencies first —
`pytest` is not a runtime dependency, so it lives in its own file:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

Then run the suite:

```bash
./.venv/bin/python -m pytest
```

That is 145 offline tests and needs no internet connection. Three further tests
check that the live voice and music APIs still behave as expected:

```bash
./.venv/bin/python -m pytest -m network
```

Please keep new behaviour covered by an offline test where possible — the suite
is deliberately runnable without an internet connection.

---

## Licence

This project is licensed under the [MIT Licence](LICENSE).

### Third-party terms you are responsible for

**Speech** is generated through Microsoft Edge's public neural TTS endpoint via
[`edge-tts`](https://github.com/rany2/edge-tts). Review Microsoft's terms before
using the output commercially. This project is not affiliated with or endorsed
by Microsoft.

**Background music** found through the in-app library comes from
[Openverse](https://openverse.org) and is limited to `CC0` and `CC BY`, both of
which permit commercial use and remixing. `CC BY` obliges you to credit the
artist — the app writes that credit to a `.credits.txt` file for you, but
reproducing it wherever you publish is your responsibility. Rights for any
music you upload yourself are entirely yours to clear.
