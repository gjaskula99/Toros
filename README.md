# TOROS

TOROS is a Spanish-to-Polish TV voiceover prototype for a Vu+/OpenATV receiver. It reads a direct OpenWebif audio stream, detects Spanish speech, transcribes it, translates it, and plays the Polish result through the computer's audio output. The receiver continues playing the original broadcast; TOROS does not mix or lower that audio.

**Initial design was made for Spanish to Polish voiceover for bullfights, however it can be adapted for other variants.** Testing was perfomed on macOS 26.

Speech recognition (ASR), translation and text to speech (TTS) are selected independently. **The defaults are local ASR, local Ollama translation and local macOS `say` speech.** Ollama receives text only through a literal loopback address; local faster-whisper and Piper load files from disk. Model downloads described below happen during setup, not during live operation. The TV stream is read from the receiver, normally on the local network.

When no OpenAI backend is selected, TOROS does not create an OpenAI client or send captured audio, transcripts or translations to OpenAI. Selecting an OpenAI backend sends only that stage's input to OpenAI:

| Selected stage | Data sent to OpenAI |
| --- | --- |
| `--asr openai` | Captured Spanish audio segments |
| `--translation openai` | Spanish transcripts |
| `--tts openai` | Polish translations to speak |

An OpenAI client is initialized only if at least one of those options is selected. `OPENAI_API_KEY` is needed only in that case. A local backend never falls back to OpenAI after an error.

## Install and prepare models

Use Python 3.10 or newer and FFmpeg. On macOS, install the common dependencies and the local backends with the same Python interpreter that will run TOROS:

```sh
brew install ffmpeg
python3 -m pip install webrtcvad-wheels faster-whisper piper-tts
```

For any OpenAI stage, also install the SDK and set a key in your shell:

```sh
python3 -m pip install openai
export OPENAI_API_KEY='YOUR_KEY'
```

For local ASR, download a *multilingual* [faster-whisper model](https://github.com/SYSTRAN/faster-whisper#usage) in advance. The default path is `models/faster-whisper-small` beside the script. The small model is about 486 MB; [tiny](https://huggingface.co/Systran/faster-whisper-tiny/tree/main) is smaller but may transcribe less accurately. The `hf` command is installed with `faster-whisper` through its Hugging Face dependency.

```sh
hf download Systran/faster-whisper-small \
  model.bin config.json tokenizer.json vocabulary.txt \
  --local-dir models/faster-whisper-small
```

TOROS requires those four files before starting local ASR and passes `local_files_only=True` when loading the model. To try the tiny model instead, download the same four filenames from `Systran/faster-whisper-tiny` into `models/faster-whisper-tiny` and use `--local-asr-model models/faster-whisper-tiny`.

For local translation, install [Ollama](https://ollama.com/download), start its server, and download the model before running TOROS:

```sh
ollama pull translategemma:4b
```

For Piper speech, download the three [Polish voices](https://huggingface.co/rhasspy/piper-voices/blob/main/voices.json) in advance:

```sh
python3 -m piper.download_voices --download-dir voices \
  pl_PL-bass-high pl_PL-darkman-medium pl_PL-gosia-medium
```

Each Piper voice needs both `.onnx` and `.onnx.json` files. `--piper-voice bass`, `dark` and `gosia` select them; `dark` maps to Piper's `darkman` model. A model directory can be changed with `--piper-model-dir`.

## Run

Replace the stream URL with the direct MPEG-TS URL for your channel. The built-in URL points to the original receiver and will likely need changing. You can check a URL with `ffplay -nodisp 'http://RECEIVER:8001/SERVICE_REFERENCE'`. Stop TOROS with Ctrl-C.

### Full OpenAI

Spanish audio goes to OpenAI ASR, the Spanish transcript goes to OpenAI translation, and Polish text goes to OpenAI TTS. This requires an API key, SDK and `ffplay`.

```sh
python3 toros_live_cli.py \
  --asr openai --translation openai --tts openai \
  --voice alloy \
  --stream-url 'http://RECEIVER:8001/SERVICE_REFERENCE'
```

The default `gpt-transcribe` and `gpt-6-luna` model IDs are listed in the [official OpenAI model catalog](https://developers.openai.com/api/docs/models). The script's current speech endpoint uses `gpt-4o-mini-tts`, which [OpenAI lists for shutdown on January 6, 2027](https://developers.openai.com/api/docs/deprecations); it will need an API migration after that date.

### Full local

Faster-whisper transcribes on this computer, Ollama translates over loopback, and Piper speaks on this computer. **No OpenAI key or SDK is required and no audio or text is sent to OpenAI.** Prepare all three local models first.

```sh
python3 toros_live_cli.py \
  --asr local --translation ollama --tts piper \
  --local-asr-model models/faster-whisper-small \
  --piper-voice gosia \
  --stream-url 'http://RECEIVER:8001/SERVICE_REFERENCE'
```

The default command also stays local, using macOS `say` instead of Piper for speech, provided the local ASR and Ollama models are ready:

```sh
python3 toros_live_cli.py --stream-url 'http://RECEIVER:8001/SERVICE_REFERENCE'
```

### Hybrid examples

Cloud ASR with local translation and local speech sends **Spanish audio** to OpenAI:

```sh
python3 toroslive_cli.py \
  --asr openai --translation ollama --tts piper --piper-voice dark \
  --stream-url 'http://RECEIVER:8001/SERVICE_REFERENCE'
```

Local ASR with OpenAI translation and local speech sends **Spanish text** to OpenAI:

```sh
python3 toros_live_cli.py \
  --asr local --translation openai --tts piper --piper-voice bass \
  --stream-url 'http://RECEIVER:8001/SERVICE_REFERENCE'
```

Local ASR and translation with OpenAI speech sends **Polish text** to OpenAI:

```sh
python3 toros_live_cli.py \
  --asr local --translation ollama --tts openai \
  --stream-url 'http://RECEIVER:8001/SERVICE_REFERENCE'
```

## Parameters

The options below can be combined. Run `python3 toros_live_cli.py --help` for the CLI's short reference.

### Backends, models and connection

| Parameter | Description and default |
| --- | --- |
| `mode` (`local` or `oai`) | Optional positional translation shortcut. Defaults to `local`; it does not select ASR or TTS. |
| `--engine local\|oai` | Older translation selector. Overrides positional `mode`; `--translation` overrides both. |
| `--translation ollama\|openai` | Text translation backend. Defaults to `ollama` through the default mode. `ollama` sends Spanish text only to the loopback service. |
| `--model NAME` | Translation model. Defaults to `translategemma:4b` with Ollama or `gpt-6-luna` with OpenAI. The default follows the final translation backend. |
| `--asr local\|openai` | Spanish speech recognition backend. Defaults to `local`. Local uses a preloaded faster-whisper model; OpenAI uploads the audio segment. |
| `--asr-model NAME` | OpenAI transcription model, default `gpt-transcribe`. Ignored by local ASR. |
| `--local-asr-model PATH` | Directory containing the pre-downloaded faster-whisper files. Default: `models/faster-whisper-small` beside the script. Ignored by OpenAI ASR. |
| `--local-asr-beam-size N` | Local ASR search width, 1–10; default `1`. Higher values may improve accuracy but cost time. Ignored by OpenAI ASR. |
| `--tts say\|piper\|openai` | Polish speech backend. Defaults to local macOS `say`. Piper uses a local model; OpenAI uploads Polish text. |
| `--voice NAME` | OpenAI speech voice, default `alloy`. Used only with `--tts openai`. |
| `--say-voice NAME` | macOS `say` voice, default `Zosia`. Used only with `--tts say`; check installed voices with `say -v '?'`. |
| `--voice-rate N` | macOS `say` speaking rate in words per minute, 80–500; default `250`. Used only with `--tts say`. |
| `--piper-voice bass\|dark\|gosia` | Piper Polish voice, default `gosia`. `dark` selects the `darkman` model. Used only with `--tts piper`. |
| `--piper-model-dir PATH` | Directory of downloaded `.onnx` and `.onnx.json` files. Default: `voices` beside the script. |
| `--piper-length-scale N` | Piper phoneme length, 0.5–2.0; default `1.0`. Lower values speak faster. |
| `--ollama-url URL` | Ollama API base URL, default `http://127.0.0.1:11434`. Only literal loopback HTTP addresses (`127.0.0.1` or `::1`) are allowed; TOROS does not use a proxy or follow redirects. |
| `--stream-url URL` | Direct audio stream URL from OpenWebif. The built-in default is tied to the original receiver; specify your own. |

### Capture, latency and diagnostics

| Parameter | Description and default |
| --- | --- |
| `--max-segment SECONDS` | Hard maximum speech segment length, 2–30 seconds; default `5`. A short value reduces wait time but may split a sentence. |
| `--end-silence-ms MS` | Non-speech duration that ends a phrase, 100–2000 ms; default `450`. Shorter values respond sooner and split more often. |
| `--min-voiced-ms MS` | Minimum detected speech in a segment; positive integer, default `160`. Filters very brief noise. |
| `--pre-roll-ms MS` | Audio kept before detected speech, 0–1000 ms; default `240`. Helps preserve the first syllable. |
| `--vad-mode 0\|1\|2\|3` | WebRTC VAD aggressiveness; default `2`. Higher values reject more non-speech and can miss quiet speech. |
| `--max-age SECONDS` | Positive maximum age of a segment, default `10`. Age starts when capture completes the segment. Older queued work is dropped. |
| `--log-silence` | Print a periodic heartbeat while capture runs without a completed speech segment. Off by default. |
| `--test-voices` | Generate Piper samples for bass, dark and gosia and exit. It bypasses the receiver, ASR, translation and API client. |
| `--test-output-dir PATH` | Directory for `--test-voices` WAVs; default `voice_samples` under the current working directory. |

For example, shorter capture and faster Piper speech can reduce delay, with a greater chance of split or missed phrases:

```sh
python3 toros_live_cli.py --asr local --translation ollama --tts piper \
  --max-segment 4 --end-silence-ms 300 --max-age 8 --piper-length-scale 0.85 \
  --stream-url 'http://RECEIVER:8001/SERVICE_REFERENCE'
```

## Test the voices and backend routing

Generate three sample WAV files without a receiver, API key or Ollama:

```sh
python3 toros_live_cli.py --test-voices
python3 -m unittest discover -s tests -v
```

Use `--piper-model-dir PATH` if the voices are elsewhere. The `voice_samples/` directory contains samples generated during development.

## Live behavior and limits

* One-item queues drop older work when processing falls behind to stay live.
* VAD can mistake noise for speech or miss quiet speech; a hard segment limit can cut a sentence.
* The age limit cannot interrupt speech already playing, and the age shown in logs excludes capture time before the segment ends and broadcast delay. 
* Local ASR speed and quality depend on the downloaded model and computer. Testing was performed on Apple M4 Pro with 24 GB unified RAM. Voiceover delay stayed around ~7 seconds for local and ~11 seconds for OpenAI.
* Initial designed was made for Spanish to Polish voiceover, however different adaption is possible.
* Translation accuracy depends on the segment length. Long segment provides better context with delay, whereas short segment is more responsive but may split words or sentences. 5-second segment is believed to be versatile.

See [CODEBASE.md](CODEBASE.md) for architecture and verification details.
