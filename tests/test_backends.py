"""Check backend independence and the local translation boundary."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import toros_luna_live_cli as toros


class BackendTests(unittest.TestCase):
    def setUp(self):
        toros.client = Mock()

    def fake_ollama(self):
        response = Mock(status=200)
        response.read.return_value = json.dumps(
            {"message": {"content": "Dzień dobry"}}
        ).encode("utf-8")
        connection = Mock()
        connection.getresponse.return_value = response
        return connection

    def test_defaults_use_only_local_backends(self):
        toros.cfg = toros.parse_cli([])
        self.assertEqual((toros.cfg.asr, toros.cfg.translation_engine, toros.cfg.tts),
                         ("local", "ollama", "say"))
        self.assertFalse(toros.needs_openai())

    def test_openai_client_is_needed_only_for_an_explicit_cloud_stage(self):
        for options in (["--asr", "openai"], ["--translation", "openai"],
                        ["--tts", "openai"]):
            with self.subTest(options=options):
                toros.cfg = toros.parse_cli(options)
                self.assertTrue(toros.needs_openai())

    def test_missing_local_asr_model_fails_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            toros.cfg = toros.parse_cli(["--local-asr-model", directory])
            with self.assertRaisesRegex(SystemExit, "missing model.bin"):
                toros.load_local_asr_model()

    def test_local_asr_uses_loaded_model_without_openai(self):
        try:
            import numpy  # noqa: F401 - optional local ASR dependency
        except ImportError:
            self.skipTest("numpy is installed with optional faster-whisper")
        toros.cfg = toros.parse_cli([])
        toros.local_asr_model = Mock()
        toros.local_asr_model.transcribe.return_value = (
            iter([SimpleNamespace(text=" Hola "), SimpleNamespace(text=" mundo ")]), None
        )
        self.assertEqual(toros.transcribe_audio(b"\x00\x00" * 320), "Hola mundo")
        toros.local_asr_model.transcribe.assert_called_once()
        toros.client.audio.transcriptions.create.assert_not_called()

    def test_ollama_accepts_only_literal_loopback(self):
        self.assertEqual(toros.loopback_ollama_address("http://127.0.0.1:11434"),
                         ("127.0.0.1", 11434))
        self.assertEqual(toros.loopback_ollama_address("http://[::1]:11434"),
                         ("::1", 11434))
        for url in ("http://example.com:11434", "http://localhost:11434",
                    "https://127.0.0.1:11434", "http://127.0.0.1:11434/remote",
                    "http://127.0.0.1:11434@remote.example",
                    "http://127.0.0.1:0", "http://[::1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                toros.loopback_ollama_address(url)

    def test_local_translation_and_piper_never_call_cloud_backends(self):
        toros.cfg = toros.parse_cli(["--translation", "ollama", "--tts", "piper"])
        connection = self.fake_ollama()
        with patch.object(toros.http.client, "HTTPConnection", return_value=connection), \
                patch.object(toros, "piper_wav", return_value=b"local wav") as piper:
            self.assertEqual(toros.translate_text("Buenos días"), "Dzień dobry")
            self.assertEqual(toros.synthesize_text("Dzień dobry"), b"local wav")
        connection.request.assert_called_once()
        self.assertEqual(connection.request.call_args.args[:2], ("POST", "/api/chat"))
        piper.assert_called_once()
        toros.client.responses.create.assert_not_called()
        toros.client.audio.speech.create.assert_not_called()

    def test_openai_translation_with_local_piper_speech(self):
        toros.cfg = toros.parse_cli(["--translation", "openai", "--tts", "piper"])
        toros.client.responses.create.return_value.output_text = "Dzień dobry"
        with patch.object(toros, "piper_wav", return_value=b"local wav"):
            self.assertEqual(toros.translate_text("Buenos días"), "Dzień dobry")
            self.assertEqual(toros.synthesize_text("Dzień dobry"), b"local wav")
        toros.client.responses.create.assert_called_once()
        toros.client.audio.speech.create.assert_not_called()

    def test_local_translation_with_openai_speech(self):
        toros.cfg = toros.parse_cli(["--translation", "ollama", "--tts", "openai"])
        toros.client.audio.speech.create.return_value.read.return_value = b"cloud wav"
        with patch.object(toros.http.client, "HTTPConnection",
                          return_value=self.fake_ollama()):
            self.assertEqual(toros.translate_text("Buenos días"), "Dzień dobry")
        self.assertEqual(toros.synthesize_text("Dzień dobry"), b"cloud wav")
        toros.client.responses.create.assert_not_called()
        toros.client.audio.speech.create.assert_called_once()

    def test_ollama_redirect_is_not_followed_or_retried_in_cloud(self):
        toros.cfg = toros.parse_cli(["--translation", "ollama", "--tts", "piper"])
        connection = self.fake_ollama()
        connection.getresponse.return_value.status = 302
        with patch.object(toros.http.client, "HTTPConnection", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                toros.translate_text("Buenos días")
        toros.client.responses.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
