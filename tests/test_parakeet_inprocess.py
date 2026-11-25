from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import transcriptions.service as service
from transcriptions.service import (
    BackendUnavailableError,
    ParakeetInProcessBackend,
    TranscribeParams,
    TranscriptionResult,
    get_backend_services_for_params,
    transcribe_audio,
)
from transcriptions.wyoming import WyomingSttConfig, explicit_backend


@dataclass
class FakeToken:
    text: str
    start: float
    end: float


@dataclass
class FakeSentence:
    text: str
    start: float
    end: float
    tokens: list[FakeToken] = field(default_factory=list)


@dataclass
class FakeResult:
    text: str
    sentences: list[FakeSentence]


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, path: str) -> FakeResult:
        self.calls.append(path)
        return FakeResult(
            text=" Mach das Licht aus.",
            sentences=[
                FakeSentence(
                    " Mach das Licht aus.",
                    0.1,
                    1.2,
                    [
                        FakeToken(" Mach", 0.1, 0.3),
                        FakeToken(" das", 0.3, 0.5),
                        FakeToken(" Licht", 0.5, 0.8),
                        FakeToken(" aus", 0.8, 1.1),
                        FakeToken(".", 1.1, 1.2),
                    ],
                )
            ],
        )


@pytest.fixture
def fake_parakeet(monkeypatch, tmp_path: Path) -> dict[str, Any]:
    loads: list[str] = []
    model = FakeModel()

    def from_pretrained(path: str) -> FakeModel:
        loads.append(path)
        return model

    module = types.ModuleType("parakeet_mlx")
    module.from_pretrained = from_pretrained  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "parakeet_mlx", module)
    monkeypatch.setattr(service, "_PARAKEET_INPROCESS_MODELS", {})
    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    return {"loads": loads, "model": model, "dir": model_dir}


def params(language: str | None = "de", backend: str | None = None) -> TranscribeParams:
    return TranscribeParams(
        request_model="parakeet-tdt-0.6b-v3", prompt=None, language=language, backend=backend
    )


def test_inprocess_backend_loads_the_model_once_and_returns_sentences(
    fake_parakeet, tmp_path
) -> None:
    backend = ParakeetInProcessBackend(
        model_dir=str(fake_parakeet["dir"]), model_name="parakeet-tdt-0.6b-v3"
    )
    audio = tmp_path / "a.wav"

    first = backend.transcribe(audio, params())
    second = backend.transcribe(audio, params())

    assert fake_parakeet["loads"] == [str(fake_parakeet["dir"].resolve())]
    assert len(fake_parakeet["model"].calls) == 2
    assert first.text == "Mach das Licht aus."
    assert first.backend_name == "parakeet-mlx"
    assert first.model_name == "parakeet-tdt-0.6b-v3"
    assert [segment.text for segment in first.segments] == ["Mach das Licht aus."]
    assert [word.text for word in first.segments[0].words] == ["Mach", "das", "Licht", "aus."]
    assert second.text == first.text


@pytest.mark.parametrize("model_dir", ["", "missing"])
def test_inprocess_backend_is_unavailable_without_a_model(
    fake_parakeet, tmp_path, model_dir
) -> None:
    directory = str(tmp_path / model_dir) if model_dir else ""
    backend = ParakeetInProcessBackend(model_dir=directory, model_name="parakeet-tdt-0.6b-v3")
    with pytest.raises(BackendUnavailableError):
        backend.transcribe(tmp_path / "a.wav", params())


def test_inprocess_backend_is_unavailable_for_unsupported_language(fake_parakeet, tmp_path) -> None:
    backend = ParakeetInProcessBackend(model_dir=str(fake_parakeet["dir"]), model_name="x")
    with pytest.raises(BackendUnavailableError):
        backend.transcribe(tmp_path / "a.wav", params(language="ja"))


def test_inprocess_backend_is_unavailable_without_the_package(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "parakeet_mlx", None)
    backend = ParakeetInProcessBackend(model_dir=str(tmp_path), model_name="x")
    with pytest.raises(BackendUnavailableError, match="parakeet"):
        backend.transcribe(tmp_path / "a.wav", params())


def test_explicit_backend_chain_falls_back_to_mlx(settings) -> None:
    settings.VOXHELM_MLX_MODEL = "mlx-community/whisper-large-v3-mlx"
    settings.VOXHELM_PARAKEET_MODEL_NAME = "parakeet-tdt-0.6b-v3"
    chain = get_backend_services_for_params(params(backend="parakeet-mlx"))
    assert [invocation.name for invocation in chain] == ["parakeet-mlx", "mlx"]
    assert isinstance(chain[0].service, ParakeetInProcessBackend)
    assert chain[1].service.model_name == "mlx-community/whisper-large-v3-mlx"  # type: ignore[attr-defined]


def test_transcribe_audio_uses_mlx_when_parakeet_is_unavailable(
    settings, monkeypatch, tmp_path
) -> None:
    settings.VOXHELM_PARAKEET_MODEL_DIR = ""
    settings.VOXHELM_LANE_SCHEDULER_ENABLED = False
    seen: list[str] = []

    def fake_mlx(self, audio_path, p) -> TranscriptionResult:
        seen.append("mlx")
        return TranscriptionResult(text="ok", language="de", segments=[])

    monkeypatch.setattr(service.MlxWhisperBackend, "transcribe", fake_mlx)
    result = transcribe_audio(tmp_path / "a.wav", params(backend="parakeet-mlx"))
    assert result.text == "ok"
    assert seen == ["mlx"]


def make_config(backend: str) -> WyomingSttConfig:
    return WyomingSttConfig(
        host="127.0.0.1",
        port=10300,
        backend=backend,
        model="m",
        language="de",
        languages=("de",),
        prompt=None,
    )


def test_wyoming_names_only_the_inprocess_parakeet_backend() -> None:
    assert explicit_backend(make_config("parakeet-mlx")) == "parakeet-mlx"
    assert explicit_backend(make_config("mlx")) is None
    assert explicit_backend(make_config("whispercpp")) is None
