from __future__ import annotations

import io
import json
import logging
import wave
from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from transcriptions.observability import emit_transcription_debug_log, summarize_audio_file
from transcriptions.service import TranscriptionResult


@pytest.mark.parametrize("contents", [b"RIFF", b"not a WAV"])
def test_invalid_wav_is_diagnostic_metadata(tmp_path, contents):
    path = tmp_path / "synthetic.wav"
    path.write_bytes(contents)

    summary = summarize_audio_file(path)

    assert summary["suffix"] == ".wav"
    assert summary["bytes"] == len(contents)
    assert "wav_error" in summary
    assert "frames" not in summary


def test_missing_audio_is_diagnostic_metadata(tmp_path):
    assert "stat_error" in summarize_audio_file(tmp_path / "missing.wav")


@pytest.mark.parametrize("debug_enabled", [False, True])
def test_completed_transcription_survives_truncated_diagnostic_wav(
    client, settings, monkeypatch, debug_enabled
):
    settings.VOXHELM_STT_DEBUG_LOGGING = debug_enabled
    paths: list[Path] = []

    def fake_transcribe(path, params):
        paths.append(path)
        path.write_bytes(b"RIFF")
        return TranscriptionResult(text="Synthetic success", language="en", segments=[])

    monkeypatch.setattr("transcriptions.views.transcribe_audio", fake_transcribe)
    # A real, synthetic upload is staged before fake inference truncates it.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav_file.writeframes(b"\x00\x00" * 16)
    if not debug_enabled:

        def unexpected_inspection(*args, **kwargs):
            pytest.fail("Disabled diagnostics must not inspect audio")

        # Patch only after fixture generation; neither operation is needed by fake inference.
        monkeypatch.setattr("transcriptions.observability.wave.open", unexpected_inspection)
        original_stat = Path.stat

        def guarded_stat(path, *args, **kwargs):
            if path in paths:
                pytest.fail("Disabled diagnostics must not stat audio")
            return original_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", guarded_stat)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"file": SimpleUploadedFile("sample.wav", buffer.getvalue()), "model": "whisper-1"},
        HTTP_AUTHORIZATION="Bearer test-token",
    )

    assert response.status_code == 200
    assert response.json() == {"text": "Synthetic success"}
    assert len(paths) == 1
    # open() avoids the stat guard and verifies the existing finally cleanup.
    with pytest.raises(FileNotFoundError):
        paths[0].open("rb")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("raw", [None, " \n ", "", "equal", "different"])
def test_real_debug_emitter_privacy_and_preview_bounds(settings, caplog, enabled, raw):
    settings.VOXHELM_STT_DEBUG_LOGGING = enabled
    transcript = "正規化🌍" * 100
    prompt = "文脈🌱" * 100
    raw_text = {"equal": f"  {transcript}  ", "different": "原文🔒\n" * 100}.get(raw, raw)
    result = TranscriptionResult(
        text=transcript, language="de", segments=[], backend_name="fake", model_name="resolved"
    )

    with caplog.at_level(logging.WARNING, logger="voxhelm.stt"):
        emit_transcription_debug_log(
            source="synthetic",
            audio_shape={"bytes": 32},
            request_model="requested",
            request_language=None,
            prompt=f"  {prompt}  ",
            result=result,
            duration_ms=7,
            raw_transcript=raw_text,
        )

    records = [record for record in caplog.records if record.name == "voxhelm.stt"]
    if not enabled:
        assert records == []
        assert prompt not in caplog.text
        assert transcript not in caplog.text
        return
    assert len(records) == 1
    message = records[0].getMessage()
    assert message.startswith("stt_debug ")
    assert "\n" not in message
    payload = json.loads(message.removeprefix("stt_debug "))
    assert payload == {
        "source": "synthetic",
        "audio_shape": {"bytes": 32},
        "backend": "fake",
        "duration_ms": 7,
        "language": {"requested": None, "resolved": "de"},
        "model": {"requested": "requested", "resolved": "resolved"},
        "prompt": prompt[:160] + "...",
        "transcript": transcript[:300] + "...",
        **(
            {"transcript_raw": raw_text.strip()[:300].rstrip() + "..."}
            if raw == "different"
            else {}
        ),
    }
