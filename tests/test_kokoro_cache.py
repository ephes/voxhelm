import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import numpy as np
import pytest

from synthesis import kokoro
from synthesis.service import BackendUnavailableError, SynthesizeParams


@pytest.fixture(autouse=True)
def caches():
    kokoro.reset_caches()
    yield
    kokoro.reset_caches()


def backend(tmp_path):
    model = tmp_path / "model.onnx"
    pack = tmp_path / "voices.npz"
    model.write_bytes(b"model-a")
    np.savez(pack, synthetic=np.ones((512, 1, 256), dtype="float32"))
    config = kokoro.KokoroModelConfig("synthetic", model, pack, "synthetic", "en-us")
    return kokoro.KokoroBackend(models={"synthetic": config}), model, pack


def test_synthesis_reloads_replaced_artifacts_and_reuses_unchanged(tmp_path, monkeypatch):
    instance, model, pack = backend(tmp_path)
    loaded, styles = [], []

    class Session:
        def __init__(self, path, **kwargs):
            loaded.append(Path(path).read_bytes())

        def run(self, _, inputs):
            styles.append(float(inputs["style"][0, 0]))
            return [np.zeros(1, dtype="float32")]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    monkeypatch.setattr(instance, "_phonemize_to_tokens", lambda *a: [43])

    def synthesize():
        result = instance.synthesize(
            "Synthetic.",
            SynthesizeParams(request_model="kokoro", voice="synthetic", language="en", speed=1.0),
        )
        result.audio_path.unlink()

    synthesize()
    synthesize()
    model.write_bytes(b"model-b-longer")
    replacement = tmp_path / "replacement.npz"
    np.savez(replacement, synthetic=np.full((512, 1, 256), 2, dtype="float32"))
    replacement.replace(pack)
    synthesize()
    assert loaded == [b"model-a", b"model-b-longer"]
    assert styles == [1.0, 1.0, 2.0]
    assert len(kokoro._SESSION_CACHE) == len(kokoro._VOICEPACK_CACHE) == 1


def test_synthesis_rejects_pair_replaced_between_artifact_loads(tmp_path, monkeypatch):
    instance, model, pack = backend(tmp_path)
    inferred = []

    class Session:
        def __init__(self, path, **kwargs):
            self.model = Path(path).read_bytes()

        def run(self, _, inputs):
            inferred.append((self.model, float(inputs["style"][0, 0])))
            return [np.zeros(1, dtype="float32")]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    monkeypatch.setattr(instance, "_phonemize_to_tokens", lambda *a: [43])
    params = SynthesizeParams(request_model="kokoro", voice="synthetic", language="en", speed=1.0)
    # Warm both caches so the replacement can happen after a real cache hit.
    instance.synthesize("Synthetic.", params).audio_path.unlink()
    inferred.clear()

    original_session = instance._session

    def session_then_replace(*args, **kwargs):
        session = original_session(*args, **kwargs)
        new_model = tmp_path / "replacement.onnx"
        new_model.write_bytes(b"model-b")
        new_model.replace(model)
        new_pack = tmp_path / "replacement.npz"
        np.savez(new_pack, synthetic=np.full((512, 1, 256), 2, dtype="float32"))
        new_pack.replace(pack)
        return session

    with monkeypatch.context() as patch:
        patch.setattr(instance, "_session", session_then_replace)
        with pytest.raises(BackendUnavailableError, match="artifacts changed.*retry"):
            instance.synthesize("Synthetic.", params).audio_path.unlink()
    assert inferred == []

    instance.synthesize("Synthetic.", params).audio_path.unlink()
    assert inferred == [(b"model-b", 2.0)]


@pytest.mark.parametrize("kind", ["session", "voicepack"])
def test_concurrent_first_use_loads_once(tmp_path, monkeypatch, kind):
    instance, model, pack = backend(tmp_path)
    started, release = Event(), Event()
    ready = Barrier(5)
    calls = []
    original_load = np.load

    def load(path, **kwargs):
        calls.append(path)
        started.set()
        assert release.wait(10)
        return original_load(path, **kwargs) if kind == "voicepack" else object()

    if kind == "session":
        monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=load))
    else:
        monkeypatch.setattr(np, "load", load)

    def get():
        ready.wait(10)
        return (
            instance._session(model)
            if kind == "session"
            else instance._voicepack(pack, "synthetic")
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(get) for _ in range(4)]
        ready.wait(10)
        try:
            assert started.wait(10)
        finally:
            release.set()
        results = [f.result(10) for f in futures]
    assert len(calls) == 1
    assert all(result is results[0] for result in results)


@pytest.mark.parametrize("failure", ["missing", "read", "success"])
def test_voice_archive_closes_on_all_paths(tmp_path, monkeypatch, failure):
    instance, _, pack = backend(tmp_path)
    closed = []

    class Archive:
        def __contains__(self, key):
            return failure != "missing"

        def keys(self):
            return ["synthetic"]

        def __getitem__(self, key):
            if failure == "read":
                raise RuntimeError("synthetic read failure")
            return np.ones((512, 1, 256))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

    monkeypatch.setattr(np, "load", lambda *a, **kw: Archive())
    if failure == "success":
        instance._voicepack(pack, "synthetic")
    else:
        with pytest.raises(RuntimeError):
            instance._voicepack(pack, "synthetic")
        assert not kokoro._VOICEPACK_CACHE
    assert closed == [True]


def test_caches_evict_old_entries_and_share_canonical_paths(tmp_path, monkeypatch):
    instance, model, pack = backend(tmp_path)
    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=lambda *a, **kw: object())
    )
    first = instance._session(model)
    (tmp_path / "nested").mkdir()
    assert instance._session(tmp_path / "nested" / ".." / model.name) is first
    for n in range(12):
        path = tmp_path / f"model-{n}.onnx"
        path.write_bytes(b"synthetic")
        instance._session(path)
        other = tmp_path / f"pack-{n}.npz"
        np.savez(other, synthetic=np.ones((512, 1, 256)))
        instance._voicepack(other, "synthetic")
    assert len(kokoro._SESSION_CACHE) == len(kokoro._VOICEPACK_CACHE) == 8
