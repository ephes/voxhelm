"""Synthetic inference and installed-style admission contracts."""

import json
import tempfile

import numpy as np
import pytest

from jobs.models import JobArtifact
from synthesis import kokoro
from synthesis.service import BackendUnavailableError
from tests.test_jobs import build_synthesis_payload, configure_task_backend
from tests.test_kokoro import FakeSession, _params, _stub_backend


@pytest.mark.parametrize(
    "output",
    [
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([-np.inf]),
        np.zeros((2, 3)),
        np.zeros((1, 1, 3)),
        np.array(0.5),
        np.zeros(0),
    ],
)
@pytest.mark.parametrize("mode", ["direct", "sync", "batch"])
@pytest.mark.django_db
def test_invalid_inference_never_publishes(monkeypatch, tmp_path, client, settings, output, mode):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    backend, session = _stub_backend(monkeypatch)
    calls = []

    def run(*args):
        calls.append(True)
        return [output]

    monkeypatch.setattr(session, "run", run)
    if mode == "direct":
        with pytest.raises(RuntimeError, match="audio"):
            backend.synthesize("Synthetic.", _params())
    elif mode == "sync":
        monkeypatch.setattr("synthesis.views.synthesize_text", backend.synthesize)
        response = client.post(
            "/v1/audio/speech",
            data=json.dumps(
                {
                    "model": "tts-1",
                    "input": "Synthetic.",
                    "voice": "kokoro-test",
                }
            ),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
        assert response.status_code == 500
        assert "audio" in response.json()["error"]["message"]
    else:
        configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
        monkeypatch.setattr("jobs.services.synthesize_text", backend.synthesize)
        payload = build_synthesis_payload()
        payload["voice"] = "kokoro-test"
        response = client.post(
            "/v1/jobs",
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
        assert response.status_code == 201
        assert response.json()["state"] == "failed"
        assert not JobArtifact.objects.exists()
    assert calls == [True]
    assert not list(tmp_path.glob("*.wav"))


@pytest.mark.parametrize(
    "samples", [np.array([-2, -0.5, 0, 0.5, 2]), np.array([[-2, -0.5, 0, 0.5, 2]])]
)
def test_supported_audio_keeps_exact_pcm(monkeypatch, samples):
    import wave

    backend, session = _stub_backend(monkeypatch)
    monkeypatch.setattr(session, "run", lambda *a: [samples])
    result = backend.synthesize("I", _params())
    try:
        with wave.open(str(result.audio_path), "rb") as reader:
            assert reader.getnframes() == 5
            assert np.frombuffer(reader.readframes(5), dtype="<i2").tolist() == [
                -32767,
                -16383,
                0,
                16383,
                32767,
            ]
    finally:
        result.audio_path.unlink()


@pytest.mark.parametrize(
    "style",
    [
        np.zeros((0, 1, 256)),
        np.zeros((3, 256)),
        np.zeros((3, 2, 256)),
        np.zeros((3, 1, 255)),
        np.full((3, 1, 256), np.nan),
        np.full((3, 1, 256), np.inf),
        np.full((3, 1, 256), "1"),
    ],
)
def test_bad_pack_is_not_cached_and_replacement_recovers(tmp_path, monkeypatch, style):
    kokoro.reset_caches()
    pack = tmp_path / "styles.npz"
    model = tmp_path / "model.onnx"
    model.write_bytes(b"synthetic")
    np.savez(pack, synthetic=style)
    config = kokoro.KokoroModelConfig("kokoro-test", model, pack, "synthetic", "en-us")
    backend = kokoro.KokoroBackend(models={"kokoro-test": config})
    try:
        with pytest.raises(BackendUnavailableError, match="[Vv]oicepack"):
            backend._voicepack(pack, "synthetic")
        assert not kokoro._VOICEPACK_CACHE
        replacement = tmp_path / "replacement.npz"
        np.savez(replacement, synthetic=np.ones((3, 1, 256), dtype="float32"))
        replacement.replace(pack)
        assert backend._voicepack(pack, "synthetic").shape == (3, 1, 256)
        assert len(kokoro._VOICEPACK_CACHE) == 1
        monkeypatch.setattr(backend, "_session", lambda *a: FakeSession())
        monkeypatch.setattr(backend, "_phonemize_to_tokens", lambda *a: [43])
        result = backend.synthesize("I", _params())
        assert result.audio_path.stat().st_size > 44
        result.audio_path.unlink()
    finally:
        kokoro.reset_caches()
