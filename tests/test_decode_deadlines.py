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
from transcriptions.service import normalize_audio_to_wav_16k_mono


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
            normalize_audio_to_wav_16k_mono(input_path=source, output_path=tmp_path / "output.wav")
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
def test_speech_export_disables_stdin(tmp_path, settings, output_format):
    source = tmp_path / "speech.wav"
    source.write_bytes(b"wav")
    argv_path = tmp_path / "argv.json"
    fake_ffmpeg = tmp_path / "ffmpeg"
    fake_ffmpeg.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(argv_path)!r}).write_text(json.dumps(sys.argv))\n"
        "assert sys.stdin.read() == ''\n"
        "pathlib.Path(sys.argv[-1]).write_bytes(b'converted')\n"
    )
    fake_ffmpeg.chmod(0o755)
    settings.VOXHELM_FFMPEG_BIN = str(fake_ffmpeg)
    result = SynthesisResult(source, "piper", "piper", "voice", "en", 16000, 2, 1, 1.0)
    exported = export_audio(result, output_format=output_format)
    assert "-nostdin" in json.loads(argv_path.read_text())
    exported.path.unlink()


def test_pyannote_decoder_retires_successful_leaders_child(tmp_path):
    from transcriptions.audio_decode import run_audio_decode

    ready = tmp_path / "ready"
    late = tmp_path / "late"
    code = (
        "import os, pathlib, time\n"
        "if os.fork() == 0:\n"
        "    os.close(2)\n"
        f"    pathlib.Path({str(ready)!r}).touch()\n"
        "    time.sleep(.4)\n"
        f"    pathlib.Path({str(late)!r}).touch()\n"
        "    os._exit(0)\n"
        f"while not pathlib.Path({str(ready)!r}).exists(): time.sleep(.01)\n"
    )
    run_audio_decode([sys.executable, "-c", code], timeout=2)
    time.sleep(0.6)
    assert ready.exists()
    assert not late.exists()


@pytest.mark.parametrize("outcome", ["diagnostic", "deadline"])
def test_pyannote_decode_primary_survives_failed_retirement(monkeypatch, outcome):
    from transcriptions import audio_decode, owned_process
    from transcriptions.owned_process import process_ownership_uncertain

    # Real child exits; inject a failed kill and reap at the cleanup seam only.
    original = owned_process._cleanup_process

    def cleanup(process, *, primary):
        process.wait(timeout=1)

        def denied():
            raise PermissionError("private detail")

        monkeypatch.setattr(owned_process, "_kill_process_group", lambda process: denied())
        monkeypatch.setattr(
            process,
            "wait",
            lambda *, timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired("private", timeout)),
        )
        original(process, primary=primary)

    monkeypatch.setattr(owned_process, "_cleanup_process", cleanup)
    started = time.monotonic()
    error_type = (
        audio_decode.AudioDecodeError if outcome == "diagnostic" else subprocess.TimeoutExpired
    )
    code = (
        'import os; os.write(2,b"x"*70000)'
        if outcome == "diagnostic"
        else "import time; time.sleep(.2)"
    )
    with pytest.raises(error_type) as caught:
        audio_decode.run_audio_decode(
            [sys.executable, "-c", code], timeout=2 if outcome == "diagnostic" else 0.05
        )
    if outcome == "diagnostic":
        assert "diagnostic" in str(caught.value)
    assert time.monotonic() - started < 3
    assert process_ownership_uncertain(caught.value)
    assert "private" not in str(caught.value)
