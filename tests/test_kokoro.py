from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from config.settings import env_kokoro_models
from synthesis.kokoro import (
    KOKORO_MAX_TOKENS,
    KOKORO_VOCAB,
    KokoroBackend,
    KokoroModelConfig,
    build_token_chunks,
    kokoro_registry_voices,
    select_style_row,
    split_sentences,
    tokenize,
)
from synthesis.service import BackendUnavailableError, SynthesizeParams, build_voice_registry

# numpy ships only with the optional `kokoro` extra; skip the whole module (rather
# than crash collection) when it is absent, while still importing it eagerly under
# type checking so the numpy-typed annotations below resolve.
if TYPE_CHECKING:
    import numpy as np
else:
    np = pytest.importorskip("numpy")

FIXTURE_PACK = Path(__file__).parent / "fixtures" / "kokoro_synthetic_pack.npz"

# Real-artifact locations for tier-2 tests. Skipped entirely when absent.
KOKORO_DIR = Path(__file__).resolve().parent.parent / "var" / "kokoro"
AF_HEART_PACK = KOKORO_DIR / "voices-v1.0.bin"
MARTIN_PACK = KOKORO_DIR / "voices-martin.npz"
OFFICIAL_MODEL = KOKORO_DIR / "kokoro-v1.0.onnx"
MARTIN_MODEL = KOKORO_DIR / "kokoro-martin.onnx"


def espeak_available() -> bool:
    try:
        import espeakng_loader  # noqa: F401
        import phonemizer  # noqa: F401
    except ImportError:
        return False
    return True


requires_models = pytest.mark.requires_models


# --------------------------------------------------------------------------- #
# Pure helpers (always run; no espeak, no models)
# --------------------------------------------------------------------------- #


def test_split_sentences_splits_on_boundaries() -> None:
    assert split_sentences("Hallo Welt. Wie geht es? Gut!") == [
        "Hallo Welt.",
        "Wie geht es?",
        "Gut!",
    ]


def test_split_sentences_drops_only_empty_fragments() -> None:
    # Blank/whitespace-only lines are dropped, but a one-character line ("A") is
    # real content and must be kept, not silently discarded.
    assert split_sentences("Los!\n\n  \nA\nEcht lang genug.") == [
        "Los!",
        "A",
        "Echt lang genug.",
    ]


def test_split_sentences_keeps_single_char_input() -> None:
    assert split_sentences("I") == ["I"]


def test_split_sentences_keeps_one_char_sentence_in_longer_text() -> None:
    assert split_sentences("I am here. I go.") == ["I am here.", "I go."]


def test_tokenize_maps_vocab_and_drops_unknown() -> None:
    # 'ʣ'->18, ' '->16, 'a'->43; '§' and 'Z' are not in the vocab.
    assert tokenize("ʣ a§Z") == [KOKORO_VOCAB["ʣ"], KOKORO_VOCAB[" "], KOKORO_VOCAB["a"]]


def test_build_token_chunks_single_under_budget() -> None:
    chunks = build_token_chunks([[1, 2, 3], [4, 5]], budget=KOKORO_MAX_TOKENS)
    space = KOKORO_VOCAB[" "]
    assert chunks == [[1, 2, 3, space, 4, 5]]


def test_build_token_chunks_two_sentences_cross_budget() -> None:
    first = [1] * 300
    second = [2] * 300
    chunks = build_token_chunks([first, second], budget=KOKORO_MAX_TOKENS)
    assert len(chunks) == 2
    assert chunks[0] == first
    assert chunks[1] == second


def test_build_token_chunks_hard_splits_oversized_sentence() -> None:
    oversized = list(range(KOKORO_MAX_TOKENS + 40))
    chunks = build_token_chunks([oversized], budget=KOKORO_MAX_TOKENS)
    assert [len(chunk) for chunk in chunks] == [KOKORO_MAX_TOKENS, 40]
    assert chunks[0] + chunks[1] == oversized


def test_build_token_chunks_flushes_before_oversized_sentence() -> None:
    oversized = list(range(KOKORO_MAX_TOKENS + 1))
    chunks = build_token_chunks([[9, 9], oversized], budget=KOKORO_MAX_TOKENS)
    assert chunks[0] == [9, 9]
    assert [len(chunk) for chunk in chunks[1:]] == [KOKORO_MAX_TOKENS, 1]


# --------------------------------------------------------------------------- #
# Style-vector selection (tier 1: checked-in synthetic pack)
# --------------------------------------------------------------------------- #


def test_select_style_row_by_token_count() -> None:
    pack = np.load(FIXTURE_PACK)["tiny"]  # shape (8, 1, 4), row i == i
    assert select_style_row(pack, 3).ravel().tolist() == [3.0, 3.0, 3.0, 3.0]
    assert select_style_row(pack, 0).ravel().tolist() == [0.0, 0.0, 0.0, 0.0]


def test_select_style_row_clamps_at_edge() -> None:
    pack = np.load(FIXTURE_PACK)["tiny"]  # 8 rows -> max index 7
    # Token counts at/over the pack length clamp to the final row, never raise.
    assert select_style_row(pack, 7).ravel().tolist() == [7.0, 7.0, 7.0, 7.0]
    assert select_style_row(pack, 8).ravel().tolist() == [7.0, 7.0, 7.0, 7.0]
    assert select_style_row(pack, 999).ravel().tolist() == [7.0, 7.0, 7.0, 7.0]


def test_select_style_row_shape_is_single_style_vector() -> None:
    pack = np.load(FIXTURE_PACK)["tiny"]
    assert select_style_row(pack, 2).shape == (1, 4)


# --------------------------------------------------------------------------- #
# Settings parsing
# --------------------------------------------------------------------------- #


def test_env_kokoro_models_parses_entries(monkeypatch) -> None:
    monkeypatch.setenv(
        "VOXHELM_KOKORO_MODELS",
        "kokoro-af_heart=kokoro-v1.0.onnx:voices-v1.0.bin:af_heart:en-us,"
        "kokoro-martin=kokoro-martin.onnx:voices-martin.npz:martin:de",
    )
    parsed = env_kokoro_models("VOXHELM_KOKORO_MODELS")
    assert parsed["kokoro-martin"] == {
        "model": "kokoro-martin.onnx",
        "voicepack": "voices-martin.npz",
        "voicepack_key": "martin",
        "language": "de",
    }
    assert parsed["kokoro-af_heart"]["voicepack_key"] == "af_heart"


def test_env_kokoro_models_rejects_malformed_entry(monkeypatch) -> None:
    monkeypatch.setenv("VOXHELM_KOKORO_MODELS", "kokoro-x=only:three:fields")
    with pytest.raises(ValueError, match="Invalid VOXHELM_KOKORO_MODELS entry"):
        env_kokoro_models("VOXHELM_KOKORO_MODELS")


def test_env_kokoro_models_empty_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("VOXHELM_KOKORO_MODELS", raising=False)
    assert env_kokoro_models("VOXHELM_KOKORO_MODELS") == {}


# --------------------------------------------------------------------------- #
# Registry integration
# --------------------------------------------------------------------------- #


def _write_kokoro_artifacts(tmp_path: Path) -> None:
    (tmp_path / "kokoro-martin.onnx").write_bytes(b"onnx")
    (tmp_path / "voices-martin.npz").write_bytes(b"npz")


def test_kokoro_registry_voices_lists_configured_voice(tmp_path: Path) -> None:
    _write_kokoro_artifacts(tmp_path)
    voices = kokoro_registry_voices(
        models={
            "kokoro-martin": {
                "model": "kokoro-martin.onnx",
                "voicepack": "voices-martin.npz",
                "voicepack_key": "martin",
                "language": "de",
            }
        },
        model_dir=tmp_path,
    )
    assert list(voices) == ["kokoro-martin"]
    voice = voices["kokoro-martin"]
    assert voice.backend == "kokoro"
    assert voice.languages == ("de",)
    assert voice.artifacts["voicepack"] == tmp_path / "voices-martin.npz"


def test_kokoro_registry_voices_skips_missing_files(tmp_path: Path) -> None:
    voices = kokoro_registry_voices(
        models={
            "kokoro-martin": {
                "model": "absent.onnx",
                "voicepack": "absent.npz",
                "voicepack_key": "martin",
                "language": "de",
            }
        },
        model_dir=tmp_path,
    )
    assert voices == {}


def test_build_voice_registry_merges_piper_and_kokoro(tmp_path: Path, settings) -> None:
    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    (piper_dir / "en_US-lessac-medium.onnx").write_bytes(b"model")
    (piper_dir / "en_US-lessac-medium.onnx.json").write_text("{}", encoding="utf-8")
    kokoro_dir = tmp_path / "kokoro"
    kokoro_dir.mkdir()
    _write_kokoro_artifacts(kokoro_dir)

    settings.VOXHELM_TTS_BACKEND = "piper"
    settings.VOXHELM_PIPER_VOICE_DIR = piper_dir
    settings.VOXHELM_PIPER_VOICES = ["en_US-lessac-medium"]
    settings.VOXHELM_KOKORO_MODEL_DIR = kokoro_dir
    settings.VOXHELM_KOKORO_MODELS = {
        "kokoro-martin": {
            "model": "kokoro-martin.onnx",
            "voicepack": "voices-martin.npz",
            "voicepack_key": "martin",
            "language": "de",
        }
    }

    registry = build_voice_registry()
    keys = {voice.key: voice.backend for voice in registry.voices}
    assert keys == {"en_US-lessac-medium": "piper", "kokoro-martin": "kokoro"}
    assert registry.resolve_backend(voice="kokoro-martin", request_model="auto") == "kokoro"


def test_build_voice_registry_without_kokoro_models_is_piper_only(
    tmp_path: Path, settings
) -> None:
    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    (piper_dir / "de_DE-thorsten-high.onnx").write_bytes(b"model")
    (piper_dir / "de_DE-thorsten-high.onnx.json").write_text("{}", encoding="utf-8")
    settings.VOXHELM_TTS_BACKEND = "piper"
    settings.VOXHELM_PIPER_VOICE_DIR = piper_dir
    settings.VOXHELM_PIPER_VOICES = ["de_DE-thorsten-high"]
    settings.VOXHELM_KOKORO_MODELS = {}

    registry = build_voice_registry()
    assert [voice.backend for voice in registry.voices] == ["piper"]


# --------------------------------------------------------------------------- #
# Backend synthesis with a stubbed onnxruntime session
# --------------------------------------------------------------------------- #

_SAMPLES_PER_TOKEN = 100


class FakeSession:
    """Stubbed ort session: audio length scales with the token count."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, _outputs, inputs):
        self.calls.append(inputs)
        token_count = int(np.asarray(inputs["tokens"]).shape[1])
        return [np.full(token_count * _SAMPLES_PER_TOKEN, 0.5, dtype=np.float32)]


def _stub_backend(monkeypatch, language: str = "en-us", *, pack: np.ndarray | None = None):
    config = KokoroModelConfig(
        voice_key="kokoro-test",
        model_path=Path("/models/test.onnx"),
        voicepack_path=Path("/models/test.npz"),
        voicepack_key="test",
        phoneme_language=language,
    )
    backend = KokoroBackend(models={"kokoro-test": config})
    session = FakeSession()
    if pack is None:
        pack = np.zeros((KOKORO_MAX_TOKENS + 2, 1, 256), dtype=np.float32)
    voicepack = pack
    monkeypatch.setattr(backend, "_session", lambda _path: session)
    monkeypatch.setattr(backend, "_voicepack", lambda _path, _key: voicepack)
    # Token count per sentence == its stripped character count (espeak bypassed).
    monkeypatch.setattr(
        backend,
        "_phonemize_to_tokens",
        lambda text, lang: [KOKORO_VOCAB["a"]] * len(text.strip()),
    )
    return backend, session


def _params() -> SynthesizeParams:
    return SynthesizeParams(request_model="auto", voice="kokoro-test", language=None, speed=1.0)


def _read_frames(path: Path) -> int:
    import wave

    with wave.open(str(path), "rb") as reader:
        assert reader.getframerate() == 24000
        assert reader.getsampwidth() == 2
        return reader.getnframes()


def test_synthesize_single_chunk_one_call(monkeypatch) -> None:
    backend, session = _stub_backend(monkeypatch)
    text = "a" * 100 + "."  # one 101-char sentence, under budget
    result = backend.synthesize(text, _params())
    assert len(session.calls) == 1
    assert result.backend_name == "kokoro"
    # 101 phoneme tokens + 2 pad tokens fed to the (single) inference call.
    assert _read_frames(result.audio_path) == (101 + 2) * _SAMPLES_PER_TOKEN
    result.audio_path.unlink()


def test_synthesize_wav_write_failure_removes_temp_file(monkeypatch, tmp_path: Path) -> None:
    import tempfile
    import wave

    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    backend, _session = _stub_backend(monkeypatch)
    real_open = wave.open

    class _FailingWriter:
        def __init__(self, path: str, mode: str) -> None:
            self._inner = real_open(path, mode)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> None:
            self._inner.close()

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

        def writeframes(self, data: bytes) -> None:
            raise OSError("disk full")

    monkeypatch.setattr(wave, "open", _FailingWriter)

    with pytest.raises(OSError, match="disk full"):
        backend.synthesize("Hello.", _params())

    assert list(temp_dir.iterdir()) == []


def test_synthesize_success_returns_temp_file_for_caller(monkeypatch, tmp_path: Path) -> None:
    import tempfile

    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    backend, _session = _stub_backend(monkeypatch)

    result = backend.synthesize("Hello.", _params())

    assert result.audio_path.parent == temp_dir
    assert result.audio_path.exists()
    result.audio_path.unlink()


def test_synthesize_single_char_input_produces_nonempty_audio(monkeypatch) -> None:
    # "I" is a valid one-character utterance; it must synthesize real audio
    # rather than being dropped into a zero-frame WAV.
    backend, session = _stub_backend(monkeypatch)
    result = backend.synthesize("I", _params())
    assert len(session.calls) == 1
    frames = _read_frames(result.audio_path)
    assert frames > 0
    # 1 phoneme token + 2 pad tokens.
    assert frames == (1 + 2) * _SAMPLES_PER_TOKEN
    assert result.duration_seconds > 0
    result.audio_path.unlink()


def test_phonemize_runs_under_phonemizer_lock(monkeypatch) -> None:
    import synthesis.kokoro as kokoro_module

    observed: list[bool] = []

    def fake_phonemize(text: str, language: str) -> str:
        observed.append(kokoro_module._PHONEMIZER_LOCK.locked())
        return "a" * len(text.strip())

    backend = KokoroBackend(models={"kokoro-test": _config("kokoro-test")})
    # Exercise the real _phonemize_to_tokens to prove init + phonemize run
    # while _PHONEMIZER_LOCK is held.
    monkeypatch.setattr(backend, "_ensure_espeak", lambda: None)
    monkeypatch.setattr(kokoro_module, "_phonemize", fake_phonemize)
    tokens = backend._phonemize_to_tokens("hello.", "en-us")
    assert tokens == [KOKORO_VOCAB["a"]] * len("hello.")
    assert observed == [True]
    assert not kokoro_module._PHONEMIZER_LOCK.locked()


def test_synthesize_two_sentences_cross_budget_concatenates(monkeypatch) -> None:
    backend, session = _stub_backend(monkeypatch)
    text = "a" * 300 + ". " + "b" * 300 + "."  # two ~301-token sentences
    result = backend.synthesize(text, _params())
    assert len(session.calls) == 2
    # Two chunks, each 301 tokens + 2 pad tokens; durations concatenate (sum).
    total_padded = (301 + 2) + (301 + 2)
    assert _read_frames(result.audio_path) == total_padded * _SAMPLES_PER_TOKEN
    assert result.duration_seconds == round(total_padded * _SAMPLES_PER_TOKEN / 24000, 3)
    result.audio_path.unlink()


def test_synthesize_oversized_sentence_hard_splits(monkeypatch) -> None:
    backend, session = _stub_backend(monkeypatch)
    text = "a" * 700 + "."  # single 701-token sentence
    result = backend.synthesize(text, _params())
    assert len(session.calls) == 2
    token_counts = [int(np.asarray(call["tokens"]).shape[1]) for call in session.calls]
    # Each call is padded with a leading + trailing 0 token.
    assert token_counts == [KOKORO_MAX_TOKENS + 2, 191 + 2]
    result.audio_path.unlink()


def test_synthesize_selects_style_by_token_count_with_clamp(monkeypatch) -> None:
    pack = np.stack(
        [np.full((1, 256), i, dtype=np.float32) for i in range(KOKORO_MAX_TOKENS + 2)]
    )
    backend, session = _stub_backend(monkeypatch, pack=pack)
    result = backend.synthesize("a" * 40 + ".", _params())  # 41 tokens
    style = np.asarray(session.calls[0]["style"])
    assert float(style.ravel()[0]) == 41.0
    result.audio_path.unlink()


def test_synthesize_applies_german_rules_only_for_german(monkeypatch) -> None:
    seen: list[str] = []

    def capture(text: str, lang: str) -> list[int]:
        seen.append(text)
        return [KOKORO_VOCAB["a"]] * max(len(text.strip()), 1)

    backend, _ = _stub_backend(monkeypatch, language="de")
    monkeypatch.setattr(backend, "_phonemize_to_tokens", capture)
    backend.synthesize("Das kostet 5 EUR heute.", _params()).audio_path.unlink()
    assert any("fünf Euro" in text for text in seen)

    seen.clear()
    backend_en, _ = _stub_backend(monkeypatch, language="en-us")
    monkeypatch.setattr(backend_en, "_phonemize_to_tokens", capture)
    backend_en.synthesize("Das kostet 5 EUR heute.", _params()).audio_path.unlink()
    assert any("5 EUR" in text for text in seen)


def test_synthesize_unknown_voice_raises(monkeypatch) -> None:
    backend, _ = _stub_backend(monkeypatch)
    params = SynthesizeParams(
        request_model="auto", voice="kokoro-missing", language=None, speed=1.0
    )
    with pytest.raises(RuntimeError, match="is not configured"):
        backend.synthesize("Hallo.", params)


def _config(voice_key: str, language: str = "en-us") -> KokoroModelConfig:
    return KokoroModelConfig(
        voice_key=voice_key,
        model_path=Path(f"/models/{voice_key}.onnx"),
        voicepack_path=Path(f"/models/{voice_key}.npz"),
        voicepack_key=voice_key,
        phoneme_language=language,
    )


def test_resolve_model_uses_configured_default_voice() -> None:
    models = {
        "kokoro-af_heart": _config("kokoro-af_heart"),
        "kokoro-martin": _config("kokoro-martin", "de"),
    }
    backend = KokoroBackend(models=models, default_voice="kokoro-martin")
    # An omitted/blank voice resolves to the explicit default, not the
    # sorted-first entry (which would be kokoro-af_heart).
    assert backend._resolve_model(None).voice_key == "kokoro-martin"
    assert backend._resolve_model("   ").voice_key == "kokoro-martin"


def test_resolve_model_default_falls_back_to_sorted_first_when_unset() -> None:
    # Insertion order is martin-then-af_heart; sorted-key order is af_heart first.
    models = {
        "kokoro-martin": _config("kokoro-martin", "de"),
        "kokoro-af_heart": _config("kokoro-af_heart"),
    }
    assert KokoroBackend(models=models)._resolve_model(None).voice_key == "kokoro-af_heart"
    # An unknown configured default is ignored, falling back the same way.
    unknown = KokoroBackend(models=models, default_voice="kokoro-nope")
    assert unknown._resolve_model(None).voice_key == "kokoro-af_heart"


def test_resolve_model_default_no_models_raises() -> None:
    with pytest.raises(RuntimeError, match="No Kokoro voices are configured"):
        KokoroBackend(models={})._resolve_model(None)


def test_synthesize_without_voice_uses_default(monkeypatch) -> None:
    backend, session = _stub_backend(monkeypatch)
    params = SynthesizeParams(request_model="auto", voice=None, language=None, speed=1.0)
    result = backend.synthesize("a" * 20 + ".", params)
    assert result.voice_name == "kokoro-test"
    assert len(session.calls) == 1
    result.audio_path.unlink()


def test_require_numpy_missing_raises_backend_unavailable(monkeypatch) -> None:
    import synthesis.kokoro as kokoro_module

    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(BackendUnavailableError, match="kokoro' extra is not installed"):
        kokoro_module._require_numpy()


# --------------------------------------------------------------------------- #
# Tier 2: real artifacts (skipped when absent)
# --------------------------------------------------------------------------- #

# Style-row sha256 checksums at index 5 (pinned during Slice 2 against the
# checksum-verified downloads); guards against a silent wrong-voice regression.
_REAL_STYLE_SHA256 = {
    "af_heart": "379e6515ba42434246901a598d00f97b0fea48933346ae13ee2b555ae710acff",
    "martin": "238913bf8412d81ed0aea3d8b16590aa7e7cd00aa353f25f9fc29904e192d5ba",
}


@requires_models
@pytest.mark.parametrize(
    "pack_path, key",
    [(AF_HEART_PACK, "af_heart"), (MARTIN_PACK, "martin")],
)
def test_real_voicepack_embedding_checksums(pack_path: Path, key: str) -> None:
    if not pack_path.exists():
        pytest.skip(f"missing artifact: {pack_path}")
    pack = np.load(pack_path)[key]
    assert pack.shape == (510, 1, 256)
    row = np.ascontiguousarray(select_style_row(pack, 5), dtype=np.float32)
    assert hashlib.sha256(row.tobytes()).hexdigest() == _REAL_STYLE_SHA256[key]


@requires_models
@pytest.mark.parametrize(
    "model, pack, key, language, text",
    [
        (OFFICIAL_MODEL, AF_HEART_PACK, "af_heart", "en-us", "Hello there"),
        (MARTIN_MODEL, MARTIN_PACK, "martin", "de", "Guten Tag"),
    ],
)
def test_real_inference_smoke(
    model: Path, pack: Path, key: str, language: str, text: str
) -> None:
    if not (model.exists() and pack.exists() and espeak_available()):
        pytest.skip("Kokoro models or espeak-ng unavailable")
    config = KokoroModelConfig(
        voice_key=f"kokoro-{key}",
        model_path=model,
        voicepack_path=pack,
        voicepack_key=key,
        phoneme_language=language,
    )
    backend = KokoroBackend(models={f"kokoro-{key}": config})
    params = SynthesizeParams(
        request_model="auto", voice=f"kokoro-{key}", language=None, speed=1.0
    )
    result = backend.synthesize(text, params)
    assert result.sample_rate == 24000
    assert result.duration_seconds > 0.3
    assert _read_frames(result.audio_path) > 24000 // 2
    result.audio_path.unlink()
