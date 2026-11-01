# Codebase and architecture

## Files

- `toros__live_cli.py` is the only runnable application.
- `README.md` describes setup, every CLI parameter, backend combinations and data flow for end users.
- `tests/test_backends.py` checks default locality, selective OpenAI initialization, local model fail-closed behavior, Ollama's loopback restriction and hybrid routing.
- `voice_samples/` contains generated Piper WAV samples. It does not contain the model weights.

There is no package structure or dependency lock file. The workspace root has no Git metadata. Local models are expected in `models/faster-whisper-small/` and `voices/` by default, but those directories are not included in the source.

## Worker pipeline

`main()` parses the CLI, checks external commands, loads only the selected backends, and starts four daemon workers. `capture()` launches FFmpeg against the selected direct stream, converts the first audio track to 16 kHz mono signed 16-bit PCM, and feeds 20 ms frames to WebRTC VAD. `SpeechSegmenter` uses onset voting, pre-roll, a silence endpoint, minimum voiced duration and a maximum segment length. A completed segment is timestamped at the end of capture.

`recognize_and_translate()` calls `transcribe_audio()` on a segment and then `translate_text()` on the Spanish result. Local ASR uses an in-memory float32 waveform with a preloaded faster-whisper model, forced Spanish language, no separate VAD and no previous-text context. The OpenAI alternative wraps PCM in WAV and calls the transcription API. Ollama translation connects directly to `/api/chat` on a literal loopback IP; the OpenAI alternative calls the Responses API. The resulting Polish text is queued for `synthesize()`.

`synthesize_text()` sends Polish text to OpenAI speech, produces an in-memory WAV with a preloaded local Piper model, or passes the text onward to `say`. `playback()` plays WAV through `ffplay` or calls macOS `say`. One-item capture, translation and playback queues replace older queued items under backlog. Every stage checks `--max-age`, which excludes capture time before the segment ends and cannot stop playback already underway. Capture retries after stream failure. Errors are logged and the pipeline continues. Logs include Spanish transcripts and Polish translations.

## Configuration and data locality

`--asr`, `--translation` and `--tts` select three independent backends. Their defaults are `local`, `ollama` and `say`. The positional `local`/`oai` mode and `--engine` remain translation shortcuts; `--translation` takes precedence over both. `--model` defaults according to the final translation backend. No OpenAI client is initialized unless OpenAI is explicitly selected for ASR, translation or TTS. Thus an all-local configuration does not require the OpenAI SDK or `OPENAI_API_KEY`. Each OpenAI stage sends only its own input: audio for ASR, Spanish text for translation, or Polish text for TTS. A local backend has no cloud fallback.

Local faster-whisper requires `model.bin`, `config.json`, `tokenizer.json` and `vocabulary.txt` in a local model directory. The program checks all four files before starting, sets `HF_HUB_OFFLINE=1`, passes the directory path to `WhisperModel` with `local_files_only=True`, and never specifies a remote model name. The model loads once at startup. Piper requires an existing `.onnx` and matching `.onnx.json` file and also loads once at startup; ONNX Runtime telemetry events are disabled before loading it. Its aliases map `bass` to `pl_PL-bass-high`, `dark` to `pl_PL-darkman-medium`, and `gosia` to `pl_PL-gosia-medium`. Both local ASR and Piper TTS run in process without a speech service or model download during live operation.

Ollama accepts only `http://127.0.0.1:PORT` or `http://[::1]:PORT`. The `http.client.HTTPConnection` call goes directly to the parsed loopback IP, without a proxy or redirect. A non-200 response or other failure drops the segment rather than retrying through OpenAI. The stream URL is user-configurable and normally points to a local receiver. The app reads audio from that URL; no captured audio is posted back to the receiver.

`--test-voices` bypasses the TV stream, all ASR and translation backends and the OpenAI client. It generates three Piper WAVs from the same Polish sentence. `--piper-length-scale` is passed to Piper's synthesis configuration.

## Limits and verification

VAD may confuse speech and noise; hard splits have no overlap, and the queue and age checks intentionally discard material. The receiver's original audio is not controlled. FFmpeg is logged as connected immediately after process launch, before successful audio is confirmed. Shutdown does not join workers or cancel in-flight requests or speech. OpenAI model names and request shapes were checked against the installed SDK signature and official documentation but not exercised with a live account. The current OpenAI speech endpoint uses `gpt-4o-mini-tts`, which [OpenAI schedules to shut down on January 6, 2027](https://developers.openai.com/api/docs/deprecations); migration to a Realtime API voice model is future work. A complete live receiver and Ollama translation run was not available.

The nine backend tests pass. In a temporary Python 3.13 environment with faster-whisper 1.2.1 and a downloaded multilingual tiny model, local ASR transcribed a macOS-generated Spanish sentence as `Buenos días, hoy hace mucho sol en Sevilla.` without OpenAI installed or network access. An all-local startup with faster-whisper, Ollama translation and Piper reached the capture loop in that same environment; the deliberately unreachable loopback stream prevented a live pipeline test. Piper 1.8.0 generated nonempty 22,050 Hz WAV samples for bass, darkman and gosia. These checks do not measure real-time performance, translation quality or subjective voice quality.
