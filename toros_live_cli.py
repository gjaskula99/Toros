#!/usr/bin/env python3
"""Speech-aware Vu+ Spanish TV to Polish voiceover.

ASR, translation and TTS are independent choices. Local ASR, Ollama translation
and Piper TTS use prepared local models without downloading at runtime.
"""

import argparse
import http.client
import io
import ipaddress
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_STREAM_URL = "http://<RECEIVER>:8001/<SERVICE_REF>"
SAMPLE_RATE = 16000
FRAME_MS = 20    # WebRTC VAD requires 10, 20 or 30 ms
FRAME_BYTES = SAMPLE_RATE * FRAME_MS // 1000 * 2
PIPER_VOICES = {
    "bass": "pl_PL-bass-high",
    "dark": "pl_PL-darkman-medium",
    "gosia": "pl_PL-gosia-medium",
}
DEFAULT_PIPER_DIR = Path(__file__).resolve().parent / "voices"
DEFAULT_LOCAL_ASR_MODEL = Path(__file__).resolve().parent / "models" / "faster-whisper-small"
VOICE_TEST_TEXT = "Dzień dobry. To jest próbka polskiego lektora programu TOROS."


def loopback_ollama_address(url):
    """Reject remote hosts, proxies and redirects for local translation."""
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        address = ipaddress.ip_address(host) if host else None
        port = parsed.port if parsed.port is not None else 11434
    except ValueError as exc:
        raise ValueError("--ollama-url must use a loopback IP and valid port") from exc
    if (parsed.scheme != "http" or not address or not address.is_loopback or
            parsed.username or parsed.password or parsed.path not in ("", "/") or
            parsed.query or parsed.fragment or not 1 <= port <= 65535):
        raise ValueError("--ollama-url must be http://127.0.0.1:PORT or http://[::1]:PORT")
    return str(address), port


def parse_cli(argv=None):
    p = argparse.ArgumentParser(
        description="Spanish TV -> Polish lektor with independent ASR, translation and TTS backends."
    )
    p.add_argument("mode", nargs="?", choices=("oai", "local"), default="local",
                   help="Translation mode: local (default) or oai (OpenAI).")
    p.add_argument("--engine", choices=("oai", "local"),
                   help="Alternative to positional mode; overrides it.")
    p.add_argument("--translation", choices=("openai", "ollama"),
                   help="Translation backend, independent of --tts; overrides mode/--engine.")
    p.add_argument("--model", help="Translation model; default: translategemma:4b "
                                  "(local) or gpt-6-luna (OpenAI).")
    p.add_argument("--stream-url", default=DEFAULT_STREAM_URL,
                   help="Direct MPEG-TS stream URL from OpenWebif.")
    p.add_argument("--asr", choices=("openai", "local"), default="local",
                   help="Speech recognition: local faster-whisper (default) or OpenAI.")
    p.add_argument("--asr-model", default="gpt-transcribe",
                   help="OpenAI speech-recognition model (default: gpt-transcribe).")
    p.add_argument("--local-asr-model", type=Path, default=DEFAULT_LOCAL_ASR_MODEL,
                   help="Directory of a pre-downloaded faster-whisper model.")
    p.add_argument("--local-asr-beam-size", type=int, default=1,
                   help="Local ASR beam size, 1..10 (default: 1; larger may be slower).")
    p.add_argument("--tts", choices=("say", "piper", "openai"), default="say",
                   help="Polish speech: local macOS say (default), local Piper, or OpenAI.")
    p.add_argument("--voice", default="alloy", help="OpenAI TTS voice.")
    p.add_argument("--say-voice", default="Zosia", help="macOS say voice (default: Zosia).")
    p.add_argument("--voice-rate", type=int, default=250,
                   help="macOS say words per minute (default: 250).")
    p.add_argument("--piper-voice", choices=tuple(PIPER_VOICES), default="gosia",
                   help="Piper Polish voice: bass, dark (darkman), or gosia (default).")
    p.add_argument("--piper-model-dir", type=Path, default=DEFAULT_PIPER_DIR,
                   help="Directory of pre-downloaded Piper .onnx and .onnx.json files.")
    p.add_argument("--piper-length-scale", type=float, default=1.0,
                   help="Piper speech length scale; lower is faster (default: 1.0).")
    p.add_argument("--test-voices", action="store_true",
                   help="Synthesize bass, dark and gosia samples to --test-output-dir; no API or stream.")
    p.add_argument("--test-output-dir", type=Path, default=Path("voice_samples"),
                   help="WAV output directory for --test-voices (default: voice_samples).")
    p.add_argument("--max-age", type=float, default=10.0,
                   help="Maximum segment age in seconds (default: 10).")
    p.add_argument("--max-segment", type=float, default=5.0,
                   help="Maximum speech segment duration in seconds (default: 5).")
    p.add_argument("--end-silence-ms", type=int, default=450,
                   help="Milliseconds of non-speech to end phrase (default: 450).")
    p.add_argument("--min-voiced-ms", type=int, default=160,
                   help="Minimum detected speech duration (default: 160).")
    p.add_argument("--pre-roll-ms", type=int, default=240,
                   help="Keep this much audio before speech onset (default: 240).")
    p.add_argument("--vad-mode", type=int, choices=range(4), default=2,
                   metavar="{0,1,2,3}", help="WebRTC VAD aggressiveness (default: 2).")
    p.add_argument("--log-silence", action="store_true",
                   help="Print periodic capture heartbeat during silence.")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434",
                   help="Loopback Ollama API URL (default: http://127.0.0.1:11434).")
    args = p.parse_args(argv)
    args.engine = args.engine or args.mode
    args.translation_engine = args.translation or (
        "openai" if args.engine == "oai" else "ollama"
    )
    args.translation_model = args.model or (
        "gpt-6-luna" if args.translation_engine == "openai" else "translategemma:4b"
    )
    if not (2 <= args.max_segment <= 30 and args.max_age > 0):
        p.error("--max-segment must be 2..30 and --max-age must be positive")
    if not (0 <= args.pre_roll_ms <= 1000 and
            100 <= args.end_silence_ms <= 2000 and
            args.min_voiced_ms > 0):
        p.error("invalid pre-roll, end-silence or minimum voiced duration")
    if not (80 <= args.voice_rate <= 500):
        p.error("--voice-rate must be between 80 and 500")
    if not (0.5 <= args.piper_length_scale <= 2.0):
        p.error("--piper-length-scale must be between 0.5 and 2.0")
    if not (1 <= args.local_asr_beam_size <= 10):
        p.error("--local-asr-beam-size must be between 1 and 10")
    if args.translation_engine == "ollama" and not args.test_voices:
        try:
            loopback_ollama_address(args.ollama_url)
        except ValueError as exc:
            p.error(str(exc))
    return args


cfg = None  # set once in main(), then read by the worker threads
client = None
webrtcvad = None
local_asr_model = None
piper_voice = None
piper_syn_config = None
stop_event = threading.Event()
print_lock = threading.Lock()
capture_queue = queue.Queue(maxsize=1)
translation_queue = queue.Queue(maxsize=1)
playback_queue = queue.Queue(maxsize=1)
_ffmpeg_process = None


@dataclass
class Segment:
    number: int
    finished_at: float
    data: bytes


@dataclass
class Translation:
    number: int
    finished_at: float
    text: str


@dataclass
class Voice:
    number: int
    finished_at: float
    polish: str
    wav: bytes | None


def log(message):
    with print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def age(item):
    return time.monotonic() - item.finished_at


def put_latest(q, item, stage):
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        old = q.get_nowait()
        log(f"{stage}: dropped #{old.number} (backlog)")
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        log(f"{stage}: dropped new #{item.number} (queue full)")


def take(q):
    try:
        return q.get(timeout=0.4)
    except queue.Empty:
        return None


def make_wav(pcm):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return buffer.getvalue()


def needs_openai():
    return (cfg.asr == "openai" or cfg.translation_engine == "openai" or
            cfg.tts == "openai")


def load_local_asr_model():
    model_dir = cfg.local_asr_model
    required = ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise SystemExit(f"Local ASR model is incomplete at {model_dir}: "
                         f"missing {', '.join(missing)}. Download it before running TOROS.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit("Missing local ASR. Install: python3 -m pip install faster-whisper") from exc
    return WhisperModel(str(model_dir), device="cpu", compute_type="int8",
                        local_files_only=True)


def transcribe_audio(pcm):
    """Return Spanish text using only the selected ASR backend."""
    if cfg.asr == "local":
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = local_asr_model.transcribe(
            audio, language="es", task="transcribe",
            beam_size=cfg.local_asr_beam_size, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
    asr_args = {
        "model": cfg.asr_model,
        "file": ("segment.wav", make_wav(pcm), "audio/wav"),
    }
    if cfg.asr_model == "gpt-transcribe":
        asr_args["languages"] = ["es"]
    transcript = client.audio.transcriptions.create(**asr_args)
    return transcript.text.strip()


class SpeechSegmenter:
    """Apply a short vote filter, pre-roll, endpointing, and a hard max length.

    ingest() returns a complete PCM segment or None. It never sends silence-only
    segments to a network API. A segment that hits max length is cut without
    overlap, so extremely long utterances can still be split mid-sentence.
    """

    def __init__(self):
        self.pre_roll = deque(maxlen=max(1, cfg.pre_roll_ms // FRAME_MS))
        self.votes = deque(maxlen=5)
        self.max_frames = max(1, int(cfg.max_segment * 1000 / FRAME_MS))
        self.end_quiet_frames = max(1, cfg.end_silence_ms // FRAME_MS)
        self.minimum_voiced_frames = max(1, cfg.min_voiced_ms // FRAME_MS)
        self.reset_active()

    def reset_active(self):
        self.active = False
        self.frames = []
        self.voiced_frames = 0
        self.quiet_frames = 0

    def ingest(self, frame, voiced):
        if not self.active:
            self.pre_roll.append(frame)
            self.votes.append(int(voiced))
            if sum(self.votes) < 2:
                return None
            self.active = True
            self.frames = list(self.pre_roll)
            self.voiced_frames = sum(self.votes)
            self.quiet_frames = 0
            self.pre_roll.clear()
            self.votes.clear()
            return None

        self.frames.append(frame)
        if voiced:
            self.voiced_frames += 1
            self.quiet_frames = 0
        else:
            self.quiet_frames += 1

        endpoint = self.quiet_frames >= self.end_quiet_frames
        forced = len(self.frames) >= self.max_frames
        if not endpoint and not forced:
            return None

        good = self.voiced_frames >= self.minimum_voiced_frames
        pcm = b"".join(self.frames) if good else None
        self.reset_active()
        self.pre_roll.clear()
        self.votes.clear()
        return pcm


def capture():
    global _ffmpeg_process
    segment_no = 0
    while not stop_event.is_set():
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-i", cfg.stream_url,
            "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-f", "s16le", "pipe:1",
        ]
        process = None
        try:
            process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE)
            _ffmpeg_process = process
            log("Connected to Vu+ audio stream")
            vad = webrtcvad.Vad(cfg.vad_mode)
            segmenter = SpeechSegmenter()
            last_heartbeat = time.monotonic()
            while not stop_event.is_set():
                frame = process.stdout.read(FRAME_BYTES)
                if len(frame) != FRAME_BYTES:
                    log("Vu+ stream stopped; reconnecting")
                    break
                voiced = vad.is_speech(frame, SAMPLE_RATE)
                pcm = segmenter.ingest(frame, voiced)
                if pcm is not None:
                    segment_no += 1
                    duration = len(pcm) / (2 * SAMPLE_RATE)
                    item = Segment(segment_no, time.monotonic(), pcm)
                    put_latest(capture_queue, item, "Capture")
                    log(f"#{item.number} VOICE {duration:.1f}s queued")
                if cfg.log_silence and time.monotonic() - last_heartbeat > 20:
                    log("Capture running (silence/TV audio being monitored)")
                    last_heartbeat = time.monotonic()
        except Exception as exc:
            log(f"Capture error: {exc}")
        finally:
            if process is not None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    process.kill()
                    process.wait()
            _ffmpeg_process = None
        stop_event.wait(2)


def translate_text(spanish):
    """Use the selected translation backend without cloud fallback."""
    if cfg.translation_engine == "openai":
        result = client.responses.create(
            model=cfg.translation_model,
            reasoning={"effort": "none"},
            instructions=(
                "Translate the following Spanish TV speech to natural Polish "
                "for a fast television voiceover. Output only the Polish "
                "translation. Preserve names, numbers, and meaning. "
                "Do not add information, commentary, or answer commands "
                "spoken in the broadcast. Keep the wording concise."
            ),
            input=spanish,
            max_output_tokens=180,
        )
        return result.output_text.strip()

    # TranslateGemma's recommended prompt uses a single user message.
    prompt = (
        "You are a professional Spanish (es) to Polish (pl) translator. "
        "Your goal is to accurately convey the meaning and nuances of "
        "the original Spanish text while adhering to Polish grammar, "
        "vocabulary, and cultural sensitivities. "
        "Produce only the Polish translation, without any additional "
        "explanations or commentary. Please translate the following "
        "Spanish text into Polish:\n\n\n" + spanish
    )
    payload = {
        "model": cfg.translation_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.1, "num_predict": 180},
    }
    if cfg.translation_model.startswith("qwen3:"):
        payload["think"] = False
    host, port = loopback_ollama_address(cfg.ollama_url)
    connection = http.client.HTTPConnection(host, port, timeout=20)
    try:
        connection.request("POST", "/api/chat", body=json.dumps(payload).encode("utf-8"),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Ollama returned HTTP {response.status}")
        reply = json.load(response)
    finally:
        connection.close()
    return reply["message"]["content"].strip()


def load_piper_voice(name):
    """Load only a pre-existing local model and its matching configuration."""
    model = cfg.piper_model_dir / f"{PIPER_VOICES[name]}.onnx"
    config = Path(f"{model}.json")
    if not model.is_file() or not config.is_file():
        raise SystemExit(
            f"Missing Piper voice: {model} and {config}. "
            "Download both files before running TOROS."
        )
    try:
        import onnxruntime
        onnxruntime.disable_telemetry_events()
        from piper import PiperVoice
    except ImportError as exc:
        raise SystemExit("Missing Piper. Install: python3 -m pip install piper-tts") from exc
    return PiperVoice.load(str(model))


def piper_wav(voice, text):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file, syn_config=piper_syn_config)
    return buffer.getvalue()


def piper_synthesis_config():
    try:
        from piper import SynthesisConfig
    except ImportError as exc:
        raise SystemExit("Missing Piper. Install: python3 -m pip install piper-tts") from exc
    return SynthesisConfig(length_scale=cfg.piper_length_scale)


def test_voices():
    global piper_syn_config
    piper_syn_config = piper_synthesis_config()
    cfg.test_output_dir.mkdir(parents=True, exist_ok=True)
    for name in PIPER_VOICES:
        voice = load_piper_voice(name)
        wav = piper_wav(voice, VOICE_TEST_TEXT)
        path = cfg.test_output_dir / f"{PIPER_VOICES[name]}.wav"
        path.write_bytes(wav)
        log(f"Piper {name}: wrote {path} ({len(wav)} bytes)")


def recognize_and_translate():
    while not stop_event.is_set():
        item = take(capture_queue)
        if item is None:
            continue
        if age(item) > cfg.max_age:
            log(f"ASR: stale #{item.number} skipped")
            continue
        try:
            started = time.monotonic()
            spanish = transcribe_audio(item.data)
            asr_sec = time.monotonic() - started
            if not spanish:
                log(f"#{item.number} empty transcript: skipped")
                continue
            if age(item) > cfg.max_age:
                log(f"#{item.number} ASR arrived too late: skipped")
                continue

            started = time.monotonic()
            polish = translate_text(spanish)
            llm_sec = time.monotonic() - started
            if not polish:
                continue
            if age(item) > cfg.max_age:
                log(f"#{item.number} translation arrived too late: skipped")
                continue
            log(f"#{item.number} ES: {spanish}")
            log(f"#{item.number} PL: {polish}")
            log(f"#{item.number} ASR {asr_sec:.2f}s | LLM {llm_sec:.2f}s "
                f"| source age {age(item):.1f}s")
            put_latest(translation_queue,
                       Translation(item.number, item.finished_at, polish), "Translation")
        except Exception as exc:
            log(f"ASR/translation error #{item.number}: {exc}")


def synthesize():
    while not stop_event.is_set():
        item = take(translation_queue)
        if item is None:
            continue
        if age(item) > cfg.max_age:
            log(f"TTS: stale #{item.number} skipped")
            continue
        try:
            wav = synthesize_text(item.text)
            if age(item) > cfg.max_age:
                log(f"TTS output stale #{item.number}: skipped")
                continue
            put_latest(playback_queue,
                       Voice(item.number, item.finished_at, item.text, wav), "TTS")
        except Exception as exc:
            log(f"TTS error #{item.number}: {exc}")


def synthesize_text(text):
    """Use the selected TTS backend without switching to another provider."""
    if cfg.tts == "openai":
        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=cfg.voice, input=text,
            instructions="Speak Polish clearly and briskly.",
            response_format="wav", speed=1.30,
        )
        return speech.read()
    if cfg.tts == "piper":
        return piper_wav(piper_voice, text)
    return None


def playback():
    while not stop_event.is_set():
        item = take(playback_queue)
        if item is None:
            continue
        if age(item) > cfg.max_age:
            log(f"PLAY: stale #{item.number} skipped")
            continue
        try:
            log(f"#{item.number} PLAY | age since source END {age(item):.1f}s")
            if item.wav is None:
                subprocess.run(
                    ["say", "-v", cfg.say_voice, "-r", str(cfg.voice_rate), item.polish],
                    check=True, timeout=30,
                )
            else:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error",
                     "-i", "pipe:0"],
                    input=item.wav, check=True, timeout=30,
                )
            if age(item) > cfg.max_age:
                log(f"#{item.number} speech finished {age(item):.1f}s after source end; "
                    "consider increasing --voice-rate or shortening translations")
        except Exception as exc:
            log(f"Playback error #{item.number}: {exc}")


def main(argv=None):
    global cfg, client, webrtcvad, local_asr_model, piper_voice, piper_syn_config
    cfg = parse_cli(argv)
    client = None
    if cfg.test_voices:
        test_voices()
        return
    commands = ("ffmpeg", "say" if cfg.tts == "say" else "ffplay")
    for command in commands:
        if not shutil.which(command):
            raise SystemExit(f"Missing command: {command}")
    # Load only the selected backends after CLI parsing.
    try:
        import webrtcvad as vad_module
    except ImportError as exc:
        raise SystemExit("Missing VAD. Install: python3 -m pip install webrtcvad-wheels") from exc
    webrtcvad = vad_module
    if cfg.asr == "local":
        local_asr_model = load_local_asr_model()
    if cfg.tts == "piper":
        piper_syn_config = piper_synthesis_config()
        piper_voice = load_piper_voice(cfg.piper_voice)
    if needs_openai():
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Missing OpenAI SDK. Install: python3 -m pip install openai") from exc
        try:
            client = OpenAI(timeout=18.0, max_retries=0)
        except Exception as exc:
            raise SystemExit(f"OpenAI initialization failed: {exc}") from exc
    model_name = cfg.translation_model
    asr_name = cfg.asr_model if cfg.asr == "openai" else cfg.local_asr_model
    log(f"TOROS Live | ASR={cfg.asr}:{asr_name} | translator={cfg.translation_engine}:{model_name} "
        f"| TTS={cfg.tts} | VAD={cfg.vad_mode} | max segment={cfg.max_segment:.1f}s "
        f"| max age={cfg.max_age:.1f}s")
    for target in (capture, recognize_and_translate, synthesize, playback):
        threading.Thread(target=target, name=target.__name__, daemon=True).start()
    try:
        while not stop_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        log("Stopping TOROS...")
    finally:
        stop_event.set()
        if _ffmpeg_process is not None:
            _ffmpeg_process.terminate()


if __name__ == "__main__":
    main()
