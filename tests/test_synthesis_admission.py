"""Installed model admission uses synthetic files and fake loaders only."""

import io
import sys
import zipfile
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from synthesis import kokoro, service
from tests.test_kokoro_cache import backend
from tests.test_synthesis import write_voice_fixture


@pytest.fixture(autouse=True)
def clear_caches():
    kokoro.reset_caches()
    service._PIPER_VOICE_CACHE.clear()
    yield
    kokoro.reset_caches()
    service._PIPER_VOICE_CACHE.clear()


@pytest.mark.parametrize("kind", ["compressed", "too-many-rows", "forged-shape", "truncated"])
def test_style_metadata_refused_before_numpy_load(tmp_path, monkeypatch, kind):
    instance, _, pack = backend(tmp_path)
    if kind in {"compressed", "too-many-rows"}:
        rows = 4096 if kind == "compressed" else 513
        np.savez_compressed(pack, synthetic=np.zeros((rows, 1, 256), dtype="float32"))
    else:
        member = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            member,
            {
                "descr": "<f4",
                "fortran_order": False,
                "shape": (10**12, 1, 256) if kind == "forged-shape" else (512, 1, 256),
            },
        )
        with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("synthetic.npy", member.getvalue())
    calls = []
    original = np.load

    def load(*args, **kwargs):
        calls.append(True)
        # Avoid the unsafe allocation even on the old implementation.
        raise ValueError("allocation would start here")

    monkeypatch.setattr(np, "load", load)
    with pytest.raises(service.BackendUnavailableError):
        instance._voicepack(pack, "synthetic")
    assert calls == []
    assert not kokoro._VOICEPACK_CACHE
    monkeypatch.setattr(np, "load", original)
    np.savez(pack, synthetic=np.ones((512, 1, 256), dtype="float64"))
    assert instance._voicepack(pack, "synthetic").shape == (512, 1, 256)


@pytest.mark.parametrize("kind", ["session", "voicepack", "piper-model", "piper-config"])
def test_replaced_generation_is_not_returned_or_cached(tmp_path, monkeypatch, kind):
    instance, model, pack = backend(tmp_path)
    original_load = np.load
    calls = []
    cache: Any

    def replace(path):
        replacement = path.with_suffix(".replacement")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)

    if kind.startswith("piper"):
        model, config = write_voice_fixture(tmp_path, "synthetic")
        voice = service.build_voice_metadata(
            voice_name="synthetic", model_path=model, config_path=config
        )
        changed = model if kind == "piper-model" else config

        def load(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                replace(changed)
            return object()

        monkeypatch.setitem(
            sys.modules, "piper", SimpleNamespace(PiperVoice=SimpleNamespace(load=load))
        )

        def get():
            return service.load_piper_voice(voice)

        cache = service._PIPER_VOICE_CACHE
    else:

        def load(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                replace(model if kind == "session" else pack)
            return original_load(*args, **kwargs) if kind == "voicepack" else object()

        if kind == "session":
            monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=load))

            def get():
                return instance._session(model)

            cache = kokoro._SESSION_CACHE
        else:
            monkeypatch.setattr(np, "load", load)

            def get():
                return instance._voicepack(pack, "synthetic")

            cache = kokoro._VOICEPACK_CACHE
    with pytest.raises(service.BackendUnavailableError, match="changed"):
        get()
    assert not cache
    stable = get()
    assert get() is stable
    assert len(calls) == 2


def test_npz_raw_key_shadow_cannot_bypass_style_admission(tmp_path, monkeypatch):
    instance, _, pack = backend(tmp_path)
    safe = io.BytesIO()
    np.save(safe, np.ones((2, 1, 256), dtype="float32"))
    dangerous = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        dangerous, {"descr": "<f4", "fortran_order": False, "shape": (10**12, 1, 256)}
    )
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("synthetic.npy", safe.getvalue())
        # NpzFile resolves an exact raw member before the implicit .npy alias.
        archive.writestr("synthetic", dangerous.getvalue())
    calls = []

    def load(*args, **kwargs):
        calls.append(True)
        raise ValueError("allocation would start here")

    monkeypatch.setattr(np, "load", load)
    with pytest.raises(service.BackendUnavailableError):
        instance._voicepack(pack, "synthetic")
    assert calls == []
    assert not kokoro._VOICEPACK_CACHE


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
@pytest.mark.parametrize("key", ["synthetic", "synthetic.npy"])
def test_numeric_npy_versions_and_exact_filename_keys_remain_supported(tmp_path, version, key):
    instance, _, pack = backend(tmp_path)
    member = io.BytesIO()
    expected = np.ones((512, 1, 256), dtype="float32", order="F")
    np.lib.format.write_array(member, expected, version=version, allow_pickle=False)
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("synthetic.npy", member.getvalue())
    assert np.array_equal(instance._voicepack(pack, key), expected)


@pytest.mark.parametrize("version", [(1, 0), (2, 0), (3, 0)])
def test_style_header_size_is_checked_before_numpy_header_reader(tmp_path, monkeypatch, version):
    instance, _, pack = backend(tmp_path)
    length_bytes = 2 if version == (1, 0) else 4
    member = np.lib.format.magic(*version) + (10001).to_bytes(length_bytes, "little") + b" " * 10001
    with zipfile.ZipFile(pack, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("synthetic.npy", member)

    def unexpected(*args, **kwargs):
        pytest.fail("oversized header must be refused before NumPy reads it")

    monkeypatch.setattr(np.lib.format, "read_array_header_1_0", unexpected)
    monkeypatch.setattr(np.lib.format, "read_array_header_2_0", unexpected)
    monkeypatch.setattr(np, "load", unexpected)
    with pytest.raises(service.BackendUnavailableError, match="header"):
        instance._voicepack(pack, "synthetic")
    assert not kokoro._VOICEPACK_CACHE
