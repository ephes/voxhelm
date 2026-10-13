import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from synthesis.service import SynthesisResult, export_audio
from transcriptions.diarization import DiarizationError, load_audio_for_pyannote
from transcriptions.known_speaker import KnownSpeakerError, decode_mono_16k
from transcriptions.service import _normalize_audio_for_whispercpp


@pytest.mark.parametrize(
    "decoder,error,message",
    [
        ("whispercpp", RuntimeError, "configured decode deadline"),
        ("diarization", DiarizationError, "configured decode deadline"),
        ("known_speaker", KnownSpeakerError, "configured decode deadline"),
    ],
)
def test_decode_deadline_kills_ffmpeg(tmp_path, settings, monkeypatch, decoder, error, message):
    argv_path = tmp_path / "argv.json"
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys, time\n"
        f"pathlib.Path({str(argv_path)!r}).write_text(json.dumps(sys.argv))\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'partial')\n"
        "time.sleep(5)\n"
        "sys.exit(1)\n"
    )
    fake_ffmpeg.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)
    settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS = 1
    # No optional model stack is needed before the failing decode.
    monkeypatch.setitem(sys.modules, "torch", Mock())
    monkeypatch.setitem(sys.modules, "numpy", Mock())
    source = tmp_path / "input.mp3"
    source.write_bytes(b"media")
    started = time.monotonic()
    with pytest.raises(error, match=message) as caught:
        if decoder == "whispercpp":
            _normalize_audio_for_whispercpp(input_path=source, output_path=tmp_path / "output.wav")
        elif decoder == "diarization":
            load_audio_for_pyannote(source)
        else:
            decode_mono_16k(source)
    assert isinstance(caught.value.__cause__, subprocess.TimeoutExpired)
    assert time.monotonic() - started < 4
    argv = json.loads(argv_path.read_text())
    assert "-nostdin" in argv
    if decoder != "whispercpp":
        assert not Path(argv[-1]).parent.exists()
    assert source.exists()


@pytest.mark.parametrize("output_format", ["mp3", "ogg"])
def test_speech_export_disables_stdin(tmp_path, settings, monkeypatch, output_format):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")

    def fake_run(argv, **kwargs):
        assert "-nostdin" in argv
        Path(argv[-1]).write_bytes(b"converted")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("synthesis.service.subprocess.run", fake_run)
    result = SynthesisResult(source, "piper", "piper", "voice", "en", 16000, 2, 1, 1.0)
    exported = export_audio(result, output_format=output_format)
    exported.path.unlink()
