# F5-TTS Multi-Speaker Audio Synthesis

A Kaggle notebook pipeline that converts multi-character scripts into natural-sounding audio using [F5-TTS](https://github.com/SWivid/F5-TTS), with per-speaker voice assignment, turn-taking control, and pause management.

## Demo

| Sample1 (2 men followed by narrator) | [▶ Play](https://yunakam.github.io/audio-book/player.html?src=../samples/sample1.mp3&label=Sample+1) |
| Sample2 (man & woman, girl & father) | [▶ Play](https://yunakam.github.io/audio-book/player.html?src=../samples/sample2.mp3&label=Sample+2) |

## Features

- **Multi-speaker synthesis** — assign distinct reference voices to 15+ named characters (e.g. `david`, `elspeth`, `epictetus`) via a TOML config file
- **Turn-taking control** — automatic silence insertion between speaker turns (`TURN_PAUSE`)
- **Fine-grained pause handling** — `[PAUSE]` tags map to configurable silence durations within a single speaker's segment
- **Smart sentence chunking** — overrides F5-TTS's default random chunking with sentence-boundary-aware splitting (≤35 words/chunk), resolving a `ThreadPoolExecutor` bug that caused audio artifacts
- **Per-speaker speed control** — each voice has an independent `speed` parameter; section-level overrides are also supported (e.g., slow down section 3 by 0.1×)
- **Batch inference loop** — processes all `config_sec*.toml` files in sequence, exporting `.wav` and converting to `.mp3` via ffmpeg
- **Script preprocessing pipeline** — splits a single multi-section script into per-section files, strips production notes / sound cues, and extracts speaker tags automatically

## Architecture

```
script.txt
    │
    ▼
load_and_separate_script()   # Split by "# Section N" headers → sec1.txt, sec2.txt, …
    │
    ▼
clean_script()               # Remove [PRODUCTION:…], [SOUND …], * markers → *_cleaned.txt
    │
    ▼
generate_toml_configs()      # Create config_sec1.toml … with voice assignments & paths
    │
    ▼
process_script(cfg_path)     # Main orchestrator per section
    ├── load_config_and_model()   # Load F5TTS_v1_Base checkpoint + Vocos vocoder
    ├── parse_multispeaker_text() # Split text into (speaker, content) chunks by [speaker_tag]
    └── generate_chunk()          # Infer audio per chunk → concatenate with pauses
    │
    ▼
output/sec1.wav … → ffmpeg → sec1.mp3 → output.zip
```

## Script Format

Scripts use bracket tags to indicate speakers and pauses:

```
[epictetus] You decide what you will do next, and you do it. The letter is not in your hands.
[callias] And if nothing changes.
[epictetus] Then you work with what you have. That has always been in your hands.
[main] Arrian writes.
Callias sets the scroll on the bench beside him.
He says nothing more, and the lamp hisses on its peg. [PAUSE] [PAUSE]
[SOUND] Oil lamp hiss and soft crackle — close and steady.
```

| Tag | Behavior |
|-----|----------|
| `[speaker_name]` | Switch active speaker |
| `[PAUSE]` | Standard pause (configurable ms) |
| `[PRODUCTION: …]` | Stripped during preprocessing (production note) |
| `[SOUND …]` | Stripped during preprocessing (sound cue) |

## Tech Stack

| Component | Detail |
|-----------|--------|
| **Model** | F5-TTS `F5TTS_v1_Base` (1,250,000-step checkpoint) |
| **Vocoder** | Vocos (mel-24kHz) |
| **Platform** | Kaggle Notebooks (Python 3.12, CUDA) |
| **Key libs** | `f5_tts`, `pydub`, `torchaudio`, `toml`, `soundfile` |
| **Audio export** | WAV → MP3 via ffmpeg (192kbps) |

## Setup

### 1. Kaggle datasets required

Upload the following as Kaggle private datasets before running:

| Dataset slug | Contents |
|---|---|
| `f5-tts-ckpt` | F5TTS_v1_Base checkpoint + vocab |
| `voice-sample` | Reference `.wav` files for generated voices |
| `script` | `.txt`/`.md` files containing script, separatable with `Section` |

### 2. Hugging Face token

Add `HF_TOKEN` to Kaggle Secrets (required for model download via `huggingface_hub`).

### 3. Run order

```
1. Environment setup   # Install F5-TTS, ffmpeg, re-pin torchaudio → restart kernel
2. Select model        # Load checkpoint paths
3. Utilities           # Define preprocessing + inference functions
4. Setting voices      # Assign ref_audio and speed per character
5. Preprocessing       # Run load_and_separate_script() + clean_script()
6. Generate configs    # Run generate_toml_configs()
7. RUN LOOP INFERENCE  # process_script() for each config_sec*.toml
8. Play / Download     # Listen in-notebook or download output.zip
```

## Known Issues & Workarounds

**F5-TTS `ThreadPoolExecutor` bug**
F5-TTS's internal `split_text_into_chunks()` uses random character-count splits, which can cut mid-word and cause audio glitches. This notebook monkey-patches `utils_infer.split_text_into_chunks` with `smart_sentence_chunk()`, which splits on sentence boundaries (`.!?`) and further on commas/semicolons for long sentences, capping at 35 words per chunk.

```python
utils_infer.split_text_into_chunks = smart_sentence_chunk
```

## Repository
- Notebook: https://www.kaggle.com/code/yunakam907/public-f5-tts-multi-speaker-audio-synthesis

## License

Code: MIT  
F5-TTS model weights: [CC-BY-NC-4.0](https://github.com/SWivid/F5-TTS/blob/main/LICENSE)
