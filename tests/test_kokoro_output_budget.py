import asyncio
import gc
import json
import tempfile
import wave
import weakref

import numpy as np
import pytest

from jobs.models import JobArtifact
from synthesis import kokoro
from synthesis.service import SynthesisInputError
from tests.test_jobs import build_synthesis_payload, configure_task_backend
from tests.test_kokoro import _params, _stub_backend


@pytest.mark.django_db
@pytest.mark.parametrize("mode", ["direct", "sync", "batch"])
@pytest.mark.parametrize("phonemes", ["", "§☃"])
def test_nonblank_text_without_supported_tokens_fails(
    monkeypatch, tmp_path, client, settings, mode, phonemes
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    backend, session = _stub_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(backend, "_ensure_espeak", lambda: None)
    monkeypatch.setattr(
        backend, "_phonemize_to_tokens", kokoro.KokoroBackend._phonemize_to_tokens.__get__(backend)
    )
    monkeypatch.setattr(kokoro, "_phonemize", lambda *a: phonemes)
    if mode == "direct":
        with pytest.raises(RuntimeError, match="supported speech tokens"):
            backend.synthesize("Unsupported text", _params())
    elif mode == "sync":
        monkeypatch.setattr("synthesis.views.synthesize_text", backend.synthesize)
        response = client.post(
            "/v1/audio/speech",
            data=json.dumps(
                {
                    "model": "tts-1",
                    "input": "Unsupported text",
                    "voice": "kokoro-test",
                }
            ),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
        assert response.status_code == 400
        assert "supported speech tokens" in response.json()["error"]["message"]
    else:
        configure_task_backend(settings, "django_tasks.backends.immediate.ImmediateBackend")
        monkeypatch.setattr("jobs.services.synthesize_text", backend.synthesize)
        payload = build_synthesis_payload("Unsupported text")
        payload["voice"] = "kokoro-test"
        response = client.post(
            "/v1/jobs",
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_AUTHORIZATION="Bearer test-token",
        )
        assert response.status_code == 201
        assert response.json()["state"] == "failed"
        assert "supported speech tokens" in response.json()["error"]["message"]
        assert not JobArtifact.objects.exists()
    assert not session.calls
    assert not list(tmp_path.glob("*.wav"))


def _punctuation_preserving_phonemizer(text: str, _language: str) -> str:
    # Like espeak with preserve_punctuation=True: unsupported words vanish, but
    # their sentence punctuation survives as pause tokens.
    if text.startswith("Hello"):
        return "a" + "".join(char for char in text if char in ".!?")
    return "".join(char for char in text if char in ".!?")


def _real_tokenizer_backend(monkeypatch, tmp_path):
    backend, _ = _stub_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(backend, "_ensure_espeak", lambda: None)
    monkeypatch.setattr(
        backend, "_phonemize_to_tokens", kokoro.KokoroBackend._phonemize_to_tokens.__get__(backend)
    )
    monkeypatch.setattr(kokoro, "_phonemize", _punctuation_preserving_phonemizer)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Kokoro attempted to load artifacts for unsupported input")

    monkeypatch.setattr(backend, "_session", unexpected_load)
    monkeypatch.setattr(backend, "_voicepack", unexpected_load)
    return backend


@pytest.mark.parametrize("unsupported", ["雪", "٣", "雪.", "٣!", "雪 …"])
def test_mixed_text_rejects_sentence_without_supported_tokens(monkeypatch, tmp_path, unsupported):
    backend = _real_tokenizer_backend(monkeypatch, tmp_path)

    with pytest.raises(SynthesisInputError, match="supported speech tokens in sentence 2"):
        backend.synthesize(f"Hello. {unsupported}", _params())


@pytest.mark.parametrize("unsupported", ["雪.", "٣!"])
def test_single_spoken_sentence_with_only_punctuation_tokens_fails(
    monkeypatch, tmp_path, unsupported
):
    backend = _real_tokenizer_backend(monkeypatch, tmp_path)

    with pytest.raises(SynthesisInputError) as excinfo:
        backend.synthesize(unsupported, _params())
    assert str(excinfo.value) == (
        "Kokoro found no supported speech tokens. Try text in the voice's language."
    )


def test_mixed_text_keeps_skipping_tokenless_punctuation_and_whitespace(monkeypatch, tmp_path):
    backend, session = _stub_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(
        backend, "_phonemize_to_tokens", lambda text, _: [43] if text == "I." else []
    )
    result = backend.synthesize("I.   \n\n   §", _params())
    try:
        assert len(session.calls) == 1
        assert session.calls[0]["tokens"].tolist() == [[0, 43, 0]]
    finally:
        result.audio_path.unlink()


@pytest.mark.parametrize("text", ["I", ".", "I."])
def test_one_character_and_punctuation_token_policy(monkeypatch, tmp_path, text):
    backend, _ = _stub_backend(monkeypatch, tmp_path)
    result = backend.synthesize(text, _params())
    try:
        with wave.open(str(result.audio_path), "rb") as reader:
            assert reader.getnframes() > 0
    finally:
        result.audio_path.unlink()


def test_many_chunks_keep_bounded_live_audio_and_exact_frames(monkeypatch, tmp_path):
    backend, session = _stub_backend(monkeypatch, tmp_path)
    refs, live = [], []

    def run(*args):
        gc.collect()
        audio = np.full(101, 0.5, dtype="float32")
        refs.append(weakref.ref(audio))
        live.append(sum(ref() is not None for ref in refs))
        return [audio]

    monkeypatch.setattr(session, "run", run)
    result = backend.synthesize("a" * (kokoro.KOKORO_MAX_TOKENS * 12), _params())
    try:
        assert len(refs) == 12
        assert max(live) <= 2
        with wave.open(str(result.audio_path), "rb") as reader:
            assert reader.getnframes() == 12 * 101
            assert np.frombuffer(reader.readframes(12 * 101), dtype="<i2").tolist() == [16383] * (
                12 * 101
            )
        assert result.duration_seconds == round(12 * 101 / kokoro.KOKORO_SAMPLE_RATE, 3)
    finally:
        result.audio_path.unlink()


@pytest.mark.parametrize("failure", ["inference", "cancel", "cap"])
def test_late_failure_or_budget_overflow_removes_partial_wav(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(kokoro, "KOKORO_MAX_OUTPUT_FRAMES", 150, raising=False)
    backend, session = _stub_backend(monkeypatch, tmp_path)
    calls = []
    marker = asyncio.CancelledError() if failure == "cancel" else RuntimeError("late inference")

    def run(*args):
        calls.append(True)
        if len(calls) == 2 and failure != "cap":
            raise marker
        return [np.full(100, 0.5, dtype="float32")]

    monkeypatch.setattr(session, "run", run)
    expected = RuntimeError if failure == "cap" else type(marker)
    with pytest.raises(expected) as caught:
        backend.synthesize("a" * (kokoro.KOKORO_MAX_TOKENS + 1), _params())
    if failure == "cap":
        assert "output limit" in str(caught.value)
    else:
        assert caught.value is marker
    assert len(calls) == 2
    assert not list(tmp_path.glob("*.wav"))


def test_output_at_exact_frame_budget_succeeds(monkeypatch, tmp_path):
    backend, session = _stub_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(kokoro, "KOKORO_MAX_OUTPUT_FRAMES", 100, raising=False)
    monkeypatch.setattr(session, "run", lambda *a: [np.zeros(100, dtype="float32")])
    result = backend.synthesize("I", _params())
    try:
        with wave.open(str(result.audio_path), "rb") as reader:
            assert reader.getnframes() == 100
    finally:
        result.audio_path.unlink()
