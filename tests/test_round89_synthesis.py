"""Synthetic voice routing and descriptor admission regressions."""

import math
import os
from pathlib import Path

import pytest

from synthesis.service import (
    RoutingDetection,
    SynthesizeParams,
    apply_language_routing,
    discover_installed_voices,
)


@pytest.mark.parametrize(
    "language,confidence",
    [
        ("en", math.nan),
        ("en", math.inf),
        ("en", -math.inf),
        ("en", 1.01),
        ("en", -0.1),
        ("fr", 1.0),
        ("en", True),
        ("en", "0.99"),
        ("en", 10**400),
    ],
)
def test_routing_rejects_malformed_evidence(monkeypatch, settings, language, confidence):
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"en": "other", "fr": "other"}
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language",
        lambda _: RoutingDetection(language, confidence),
    )

    class Registry:
        def lookup(self, name):
            return object()

    monkeypatch.setattr("synthesis.service.build_voice_registry", Registry)
    params = SynthesizeParams(request_model="piper", voice="pinned", language="de", speed=1)
    assert apply_language_routing("A synthetic sentence long enough for routing.", params) is params


def test_routing_accepts_exact_floor(monkeypatch, settings):
    settings.VOXHELM_TTS_LANGUAGE_ROUTING = True
    settings.VOXHELM_TTS_LANGUAGE_VOICES = {"en": "other"}
    monkeypatch.setattr(
        "synthesis.service.detect_routing_language", lambda _: RoutingDetection("en", 0.90)
    )

    class Registry:
        def lookup(self, name):
            return object()

    monkeypatch.setattr("synthesis.service.build_voice_registry", Registry)
    params = SynthesizeParams(request_model="piper", voice="pinned", language="de", speed=1)
    routed = apply_language_routing("A synthetic sentence long enough for routing.", params)
    assert (routed.voice, routed.language, routed.request_model) == ("other", "en", "auto")


@pytest.mark.parametrize("kind", ["fifo", "symlink", "oversize", "directory", "bad_utf8"])
def test_discovery_isolates_unsafe_config(tmp_path, monkeypatch, kind):
    import synthesis.service as module

    good = "en_US-good-medium"
    bad = "en_US-bad-medium"
    for name in [good, bad]:
        (tmp_path / f"{name}.onnx").write_bytes(b"synthetic model, never loaded")
    (tmp_path / f"{good}.onnx.json").write_text('{"speaker_id_map":{"one":0}}')
    config = tmp_path / f"{bad}.onnx.json"
    if kind == "fifo":
        os.mkfifo(config)
    elif kind == "symlink":
        config.symlink_to(tmp_path / f"{good}.onnx.json")
    elif kind == "oversize":
        monkeypatch.setattr(module, "MAX_PIPER_CONFIG_BYTES", 64, raising=False)
        config.write_text('{"padding":"' + "x" * 80 + '"}')
    elif kind == "directory":
        config.mkdir()
    else:
        config.write_bytes(b"\xff")
    # Prevent the old blocking FIFO reader from hanging the regression itself.
    original = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == config and kind == "fifo":
            pytest.fail("FIFO must be admitted by nonblocking descriptor before reading")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    installed = discover_installed_voices(voice_dir=tmp_path, configured_voices=[bad, good])
    assert list(installed) == [good]
    assert installed[good].speakers == ("one",)


def test_growing_config_is_bounded_and_descriptor_closed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import synthesis.service as module

    name = "en_US-synthetic-medium"
    (tmp_path / f"{name}.onnx").write_bytes(b"model")
    (tmp_path / f"{name}.onnx.json").write_text('{"padding":"' + "x" * 80 + '"}')
    monkeypatch.setattr(module, "MAX_PIPER_CONFIG_BYTES", 64)
    real_fstat = os.fstat
    real_open = os.open
    descriptors = []

    def opened(path, flags):
        descriptor = real_open(path, flags)
        descriptors.append(descriptor)
        return descriptor

    def underreported_size(descriptor):
        metadata = real_fstat(descriptor)
        return SimpleNamespace(st_mode=metadata.st_mode, st_size=0)

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "fstat", underreported_size)
    assert discover_installed_voices(voice_dir=tmp_path, configured_voices=[name]) == {}
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        real_fstat(descriptors[0])
