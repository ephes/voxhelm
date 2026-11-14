"""Voice identity and failure cleanup contracts, without real inference."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from synthesis import kokoro, service
from tests.test_kokoro import _params, _stub_backend
from tests.test_synthesis import _piper_backend, _piper_params, write_voice_fixture


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("kind", ["registry", "piper", "kokoro"])
def test_ambiguous_alias_refuses_but_exact_keys_are_stable(tmp_path, reverse, kind):
    backend: Any
    resolve: Any
    keys = ["Voice", "voice"]
    if reverse:
        keys.reverse()
    if kind == "kokoro":
        configs = {
            key: kokoro.KokoroModelConfig(key, tmp_path / "model", tmp_path / "pack", key, "en")
            for key in keys
        }
        backend = kokoro.KokoroBackend(models=configs, default_voice="Voice")
        resolve = backend._resolve_model
        assert resolve(None).voice_key == "Voice"
        assert [resolve(key).voice_key for key in keys] == keys
    else:
        for key in keys:
            write_voice_fixture(tmp_path, key)
        voices = service.discover_installed_voices(voice_dir=tmp_path, configured_voices=keys)
        if kind == "registry":
            # Include cross-backend collisions in the combined resolution path.
            from dataclasses import replace

            records = [
                replace(voices[key], backend="kokoro" if key == "Voice" else "piper")
                for key in keys
            ]
            registry = service.VoiceRegistry(tuple(records), "piper")
            resolve = registry.lookup
        else:
            backend = service.PiperBackend(
                voice_dir=tmp_path,
                configured_voices=keys,
                default_voice="Voice",
                language_voices={"en": "Voice"},
            )

            def resolve(key):
                return backend.resolve_voice(voice=key, language=None)

            assert resolve(None).key == "Voice"
        assert [resolve(key).key for key in keys] == keys
    with pytest.raises(service.SynthesisInputError, match="[Aa]mbiguous"):
        resolve("vOiCe")


@pytest.mark.parametrize("kind", ["piper", "kokoro"])
@pytest.mark.parametrize(
    "failure", [RuntimeError("synthetic inference failure"), KeyboardInterrupt()]
)
def test_cleanup_refusal_preserves_primary_failure(tmp_path, monkeypatch, caplog, kind, failure):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    if kind == "kokoro":
        backend, session = _stub_backend(monkeypatch)

        def run(*args):
            raise failure

        monkeypatch.setattr(session, "run", run)
        params = _params()
    else:
        backend = _piper_backend(tmp_path)

        class Voice:
            def synthesize(self, text, syn_config=None):
                raise failure

        monkeypatch.setattr(service, "load_piper_voice", lambda _: Voice())
        monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(SynthesisConfig=SimpleNamespace))
        params = _piper_params()
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.suffix == ".wav":
            raise PermissionError("sensitive-path-and-text")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(type(failure)) as caught:
        backend.synthesize("Synthetic private text.", params)
    assert caught.value is failure
    assert len(list(tmp_path.glob("*.wav"))) == 1
    assert "cleanup failed" in caplog.text
    assert "sensitive-path-and-text" not in caplog.text
    assert "Synthetic private text" not in caplog.text
    assert str(tmp_path) not in caplog.text
    for orphan in tmp_path.glob("*.wav"):
        original_unlink(orphan)


def test_cleanup_attempts_every_owned_path(tmp_path, monkeypatch, caplog):
    paths = [tmp_path / "one.wav", tmp_path / "two.wav"]
    calls = []

    def unlink(path, **kwargs):
        calls.append(path)
        raise PermissionError("private")

    monkeypatch.setattr(Path, "unlink", unlink)
    service.cleanup_failed_synthesis(*paths)
    assert calls == paths
    assert caplog.text.count("cleanup failed") == 2


@pytest.mark.parametrize("mode", ["sync", "batch"])
@pytest.mark.django_db
def test_primary_input_error_survives_denied_cleanup_at_public_boundaries(
    tmp_path, monkeypatch, client, settings, mode
):
    import json

    from jobs.models import JobArtifact
    from tests.test_jobs import build_synthesis_payload, configure_task_backend

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    backend, session = _stub_backend(monkeypatch)

    def fail(*args):
        raise service.SynthesisInputError("Synthetic input refused.")

    monkeypatch.setattr(session, "run", fail)
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path.suffix == ".wav":
            raise PermissionError("private cleanup diagnostics")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    if mode == "sync":
        monkeypatch.setattr("synthesis.views.synthesize_text", backend.synthesize)
        response = client.post(
            "/v1/audio/speech",
            data=json.dumps({"input": "Synthetic.", "model": "tts-1"}),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
        assert response.status_code == 400
        assert response.json()["error"]["message"] == "Synthetic input refused."
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
        assert response.json()["error"]["message"] == "Synthetic input refused."
        assert not JobArtifact.objects.exists()
    assert "private cleanup diagnostics" not in response.content.decode()
    for orphan in tmp_path.glob("*.wav"):
        original_unlink(orphan)
