import importlib
import subprocess
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from transcriptions import diarization as d


@pytest.mark.parametrize("failure", ["placement", "device"])
def test_failed_pipeline_initialization_is_retryable(monkeypatch, failure):
    candidates = [Mock(), Mock()]
    load = Mock(side_effect=candidates)
    if failure == "placement":
        candidates[0].to.side_effect = RuntimeError("synthetic placement failure")
    device = (
        Mock(side_effect=[RuntimeError("synthetic device failure"), "cpu"])
        if (failure == "device")
        else Mock(return_value="cpu")
    )
    monkeypatch.setattr(d, "resolve_pyannote_device", device)
    original = importlib.import_module
    monkeypatch.setattr(
        d.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=load))
            if name == "pyannote.audio"
            else original(name)
        ),
    )
    backend = d.PyannoteDiarizationBackend(model_name="synthetic", auth_token="synthetic")
    with pytest.raises(RuntimeError):
        backend._load_pipeline()
    assert backend._load_pipeline() is candidates[1]
    assert backend._load_pipeline() is candidates[1]
    assert load.call_count == 2
    candidates[1].to.assert_called_once_with("cpu")


@pytest.mark.parametrize("geometry", ["oversize", "rate", "channels", "width", "short", "normal"])
def test_decoded_geometry_checked_before_pcm_and_tensor(monkeypatch, tmp_path, geometry):
    monkeypatch.setattr(d, "PYANNOTE_MAX_FRAMES", 16, raising=False)
    wav = Mock()
    wav.getnchannels.return_value = 2 if geometry == "channels" else 1
    wav.getsampwidth.return_value = 1 if geometry == "width" else 2
    wav.getframerate.return_value = 8000 if geometry == "rate" else 16000
    wav.getnframes.return_value = 17 if geometry == "oversize" else 16
    wav.readframes.return_value = b"\0\0" * (15 if geometry == "short" else 16)
    context = Mock(__enter__=Mock(return_value=wav), __exit__=Mock(return_value=False))
    monkeypatch.setattr(d.wave, "open", lambda *args: context)
    monkeypatch.setattr(d.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(d, "run_audio_decode", lambda *args, **kwargs: None, raising=False)
    original = importlib.import_module
    tensor = Mock()
    monkeypatch.setattr(
        d.importlib,
        "import_module",
        lambda name: SimpleNamespace(from_numpy=tensor) if name == "torch" else original(name),
    )
    if geometry == "normal":
        assert d.load_audio_for_pyannote(tmp_path / "input")["sample_rate"] == 16000
        tensor.assert_called_once()
    else:
        with pytest.raises(d.DiarizationError):
            d.load_audio_for_pyannote(tmp_path / "input")
        tensor.assert_not_called()
        if geometry != "short":
            wav.readframes.assert_not_called()


@pytest.mark.parametrize("mode", ["stdout", "stderr", "failure", "timeout", "normal"])
def test_decoder_has_bounded_output_deadline_and_cleanup(tmp_path, settings, monkeypatch, mode):
    script = tmp_path / "decoder.py"
    script.write_text(
        "import os, sys, time, wave\n"
        "with wave.open(sys.argv[-1], 'wb') as out:\n"
        "    out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))\n"
        "    out.writeframes(b'\\0\\0' * 16)\n"
        + (
            f"os.write({1 if mode == 'stdout' else 2}, b'private-sentinel' * 100000)\n"
            if mode in {"stdout", "stderr"}
            else ""
        )
        + ("sys.exit(1)\n" if mode == "failure" else "")
        + ("time.sleep(10)\n" if mode == "timeout" else "")
    )
    original = subprocess.Popen
    owned = []
    commands = []

    def spawn(argv, **kwargs):
        commands.append(argv)
        process = original([sys.executable, str(script), *argv[1:]], **kwargs)
        owned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn)
    original_import = importlib.import_module
    monkeypatch.setattr(
        d.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(from_numpy=Mock()) if name == "torch" else original_import(name)
        ),
    )
    settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS = 0.5
    started = time.monotonic()
    if mode in {"failure", "timeout", "stderr"}:
        with pytest.raises(d.DiarizationError) as caught:
            d.load_audio_for_pyannote(tmp_path / "input")
        assert "private-sentinel" not in str(caught.value)
    else:
        assert d.load_audio_for_pyannote(tmp_path / "input")["sample_rate"] == 16000
    assert time.monotonic() - started < 4
    assert owned[0].poll() is not None
    assert not Path(commands[0][-1]).parent.exists()
    assert "-t" in commands[0]
    assert owned[0].stdout is None  # Unused stdout is discarded by the OS.
    assert owned[0].stderr is not None and owned[0].stderr.closed


def test_real_compressed_decode_refuses_over_budget(tmp_path, monkeypatch, settings):
    import shutil

    settings.VOXHELM_FFMPEG_BIN = shutil.which("ffmpeg")
    assert settings.VOXHELM_FFMPEG_BIN
    source = tmp_path / "synthetic.wav"
    with wave.open(str(source), "wb") as out:
        out.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        out.writeframes(b"\0\0" * 32000)
    compressed = source.with_suffix(".mp3")
    subprocess.run(
        [settings.VOXHELM_FFMPEG_BIN, "-y", "-i", str(source), str(compressed)],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(d, "PYANNOTE_MAX_FRAMES", 16000, raising=False)
    with pytest.raises(d.DiarizationError, match="frame limit"):
        d.load_audio_for_pyannote(compressed)
