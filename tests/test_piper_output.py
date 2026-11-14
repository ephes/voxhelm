"""Piper PCM limits and readback completeness; no real model is loaded."""

import struct
import sys
import tempfile
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from synthesis import service
from tests.test_synthesis import _piper_backend, _piper_params


def chunk(frames=2, *, rate=22050, width=2, channels=1, pcm=None):
    return SimpleNamespace(
        sample_rate=rate,
        sample_width=width,
        sample_channels=channels,
        audio_int16_bytes=b"\0" * (frames * width * channels) if pcm is None else pcm,
    )


class FakeVoice:
    def __init__(self, parts):
        self.parts = parts
        self.consumed = []
        self.configs = []

    def synthesize(self, text, syn_config=None):
        self.configs.append(syn_config)
        for part in self.parts:
            self.consumed.append(part)
            if isinstance(part, BaseException):
                raise part
            yield part

    def synthesize_wav(self, text, writer, config):
        # Reproduce the old backend path so these tests exercise its failure.
        writer.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        first = True
        for part in self.synthesize(text, config):
            if first:
                writer.setparams(
                    (
                        part.sample_channels,
                        part.sample_width,
                        part.sample_rate,
                        0,
                        "NONE",
                        "not compressed",
                    )
                )
                first = False
            writer.writeframes(part.audio_int16_bytes)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(output))
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(SynthesisConfig=SimpleNamespace))
    backend = _piper_backend(tmp_path)

    def run(parts, *, speed=1.0):
        voice = FakeVoice(parts)
        monkeypatch.setattr(service, "load_piper_voice", lambda _: voice)
        return voice, lambda: backend.synthesize(
            "Synthetic.", replace(_piper_params(), speed=speed)
        )

    return output, run


@pytest.mark.parametrize("limit", ["duration", "bytes"])
def test_piper_stops_before_output_cap(setup, monkeypatch, limit):
    output, run = setup
    monkeypatch.setattr(service, "PIPER_MAX_OUTPUT_SECONDS", 1, raising=False)
    monkeypatch.setattr(
        service, "PIPER_MAX_OUTPUT_BYTES", 8 if limit == "bytes" else 100, raising=False
    )
    # Two legal chunks exactly fill the cap; the third must never be written.
    voice, synthesize = run([chunk(2, rate=4 if limit == "duration" else 22050)] * 10)
    writes = []
    original_write = wave.Wave_write.writeframes

    def write(writer, data):
        writes.append(len(data))
        return original_write(writer, data)

    monkeypatch.setattr(wave.Wave_write, "writeframes", write)
    with pytest.raises(RuntimeError, match="output limit"):
        synthesize()
    assert len(voice.consumed) == 3
    assert writes == [4, 4]
    assert not list(output.iterdir())


@pytest.mark.parametrize("channels", [1, 2])
def test_piper_exact_frames_speed_and_geometry(setup, channels):
    output, run = setup
    voice, synthesize = run([chunk(2205, channels=channels)], speed=2.0)
    result = synthesize()
    try:
        assert result.duration_seconds == 0.1
        assert (result.channels, result.sample_width, result.sample_rate) == (channels, 2, 22050)
        assert result.backend_name == "piper"
        assert result.voice_name == "en_US-lessac-medium"
        assert result.language == "en"
        assert voice.configs[0].length_scale == 0.5
        with wave.open(str(result.audio_path), "rb") as reader:
            assert reader.getnframes() == 2205
            assert len(reader.readframes(2206)) == 2205 * 2 * channels
    finally:
        result.audio_path.unlink()
    assert not list(output.iterdir())


@pytest.mark.parametrize(
    "parts",
    [
        [],
        [chunk(0)],
        [chunk(pcm=b"\0")],
        [chunk(width=1)],
        [chunk(channels=3)],
        [chunk(), chunk(rate=24000)],
    ],
)
def test_piper_refuses_empty_misaligned_or_changing_pcm(setup, parts):
    output, run = setup
    _, synthesize = run(parts)
    with pytest.raises(RuntimeError, match="Piper"):
        synthesize()
    assert not list(output.iterdir())


def test_piper_late_failure_preserves_error_and_cleans_output(setup):
    output, run = setup
    failure = RuntimeError("late synthetic inference failure")
    _, synthesize = run([chunk(), failure])
    with pytest.raises(RuntimeError) as caught:
        synthesize()
    assert caught.value is failure
    assert not list(output.iterdir())


@pytest.mark.parametrize(
    "damage", ["truncated", "empty", "riff-size", "data-size", "partial-frame"]
)
def test_piper_checks_complete_wav_before_returning(setup, monkeypatch, damage):
    output, run = setup
    _, synthesize = run([chunk(10)])
    original_open = wave.open

    def open_wave(path, mode):
        if mode == "rb":
            audio = Path(path)
            data = bytearray(audio.read_bytes())
            if damage == "truncated":
                data = data[:-2]
                # Keep RIFF consistent so the payload check must detect truncation.
                struct.pack_into("<I", data, 4, len(data) - 8)
            elif damage == "empty":
                data = data[:44]
                struct.pack_into("<I", data, 4, 36)
                struct.pack_into("<I", data, 40, 0)
            elif damage == "riff-size":
                struct.pack_into("<I", data, 4, len(data) + 100)
            elif damage == "data-size":
                struct.pack_into("<I", data, 40, 2)
            else:
                data += b"\0"
                struct.pack_into("<I", data, 4, len(data) - 8)
                struct.pack_into("<I", data, 40, 21)
            audio.write_bytes(data)
        return original_open(path, mode)

    monkeypatch.setattr(wave, "open", open_wave)
    with pytest.raises(RuntimeError, match="Piper"):
        synthesize()
    assert not list(output.iterdir())


def test_piper_readback_uses_bounded_reads(setup, monkeypatch):
    _, run = setup
    _, synthesize = run([chunk(40000, channels=2)])
    original_read = wave.Wave_read.readframes
    reads = []

    def read(reader, frames):
        requested = frames * reader.getsampwidth() * reader.getnchannels()
        assert requested <= 65536
        reads.append(requested)
        return original_read(reader, frames)

    monkeypatch.setattr(wave.Wave_read, "readframes", read)
    result = synthesize()
    result.audio_path.unlink()
    assert reads[:3] == [65536, 65536, 28928]


@pytest.mark.parametrize("mode", ["sync", "batch"])
@pytest.mark.django_db
def test_invalid_piper_output_never_publishes(setup, monkeypatch, client, settings, mode):
    import json

    from jobs.models import JobArtifact
    from tests.test_jobs import build_synthesis_payload, configure_task_backend

    output, run = setup
    _, synthesize = run([chunk(0)])

    def fail(text, params):
        return synthesize()

    if mode == "sync":
        monkeypatch.setattr("synthesis.views.synthesize_text", fail)
        response = client.post(
            "/v1/audio/speech",
            data=json.dumps({"model": "tts-1", "input": "Synthetic."}),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
        assert response.status_code == 500
        assert response.json()["error"]["message"] == "Speech synthesis failed."
    else:
        configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
        monkeypatch.setattr("jobs.services.synthesize_text", fail)
        response = client.post(
            "/v1/jobs",
            data=json.dumps(build_synthesis_payload()),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
        assert response.status_code == 201
        assert response.json()["state"] == "failed"
        assert not JobArtifact.objects.exists()
    assert not list(output.iterdir())
