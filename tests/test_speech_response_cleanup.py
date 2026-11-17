"""Synthetic response construction, reader ownership and independent file cleanup."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from django.http import HttpResponse, StreamingHttpResponse
from django.http.response import HttpResponseBase
from django.test import AsyncRequestFactory, RequestFactory

from synthesis import views
from synthesis.service import ExportedAudio, SynthesisInputError
from tests.test_speech_bounds import SpeechResult
from transcriptions.views import ApiError

PRIVATE = "synthetic-private-cleanup-detail"


@pytest.mark.django_db
@pytest.mark.parametrize("asgi", [False, True])
@pytest.mark.parametrize("failed_deletes", [(), ("export",), ("source",), ("export", "source")])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "construction",
        "Content-Disposition",
        "X-Voxhelm-Backend",
        "X-Voxhelm-Voice",
        "X-Voxhelm-Language",
        "close",
        "backend",
        "api",
        "input",
    ],
)
def test_speech_cleanup_preserves_outcome(
    monkeypatch, tmp_path, caplog, asgi, failed_deletes, failure
):
    source = tmp_path / "source.wav"
    exported = tmp_path / "export.mp3"
    source.write_bytes(b"source")
    payload = b"synthetic-audio" * 10_000
    exported.write_bytes(payload)
    monkeypatch.setattr(views, "synthesize_text", lambda *args: SpeechResult(source))

    def export(*args, **kwargs):
        if failure == "backend":
            raise RuntimeError("primary backend failure")
        if failure == "api":
            raise ApiError("primary API failure", status=409)
        if failure == "input":
            raise SynthesisInputError("primary input failure")
        return ExportedAudio(exported, "mp3", "audio/mpeg")

    monkeypatch.setattr(views, "export_audio", export)
    attempted = []
    unlink = Path.unlink

    def delete(path, *args, **kwargs):
        if path in (source, exported):
            name = "source" if path == source else "export"
            attempted.append(name)
            if name in failed_deletes:
                raise PermissionError(PRIVATE)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", delete)
    handles = []
    closed = []
    reader_response = views.reader_response

    def build(request, handle, **kwargs):
        handles.append(handle)
        if failure == "close":
            close = handle.close

            def fail_close():
                closed.append(True)
                close()
                raise OSError(PRIVATE)

            monkeypatch.setattr(handle, "close", fail_close)
        if failure == "construction":
            raise OSError("primary response construction failure")
        return reader_response(request, handle, **kwargs)

    monkeypatch.setattr(views, "reader_response", build)
    set_header = HttpResponseBase.__setitem__

    def header(response, key, value):
        if key == failure or (failure == "close" and key == "X-Voxhelm-Voice"):
            raise OSError("primary header failure")
        set_header(response, key, value)

    monkeypatch.setattr(HttpResponseBase, "__setitem__", header)
    factory = AsyncRequestFactory() if asgi else RequestFactory()
    request = factory.post(
        "/v1/audio/speech",
        data={"model": "tts-1", "input": "Synthetic", "response_format": "mp3"},
        content_type="application/json",
        headers={"Authorization": "Bearer test-token"},
    )
    response = views.audio_speech(request)
    expected_deletes = (
        ["source"]
        if failure in ("backend", "api", "input")
        else [
            "export",
            "source",
        ]
    )
    assert attempted == expected_deletes
    assert source.exists() == ("source" in failed_deletes)
    if "export" in expected_deletes:
        assert exported.exists() == ("export" in failed_deletes)
    assert PRIVATE not in caplog.text
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == len(set(failed_deletes) & set(expected_deletes)) + (failure == "close")
    assert closed == ([True] if failure == "close" else [])
    if failure is None:
        assert isinstance(response, StreamingHttpResponse)
        assert response.status_code == 200
        assert response["Content-Length"] == str(len(payload))
        assert response["X-Voxhelm-Voice"] == "synthetic"
        assert not handles[0].closed

        async def consume():
            iterator = response.streaming_content
            assert isinstance(iterator, AsyncIterator)
            return b"".join([chunk async for chunk in iterator])

        try:
            if asgi:
                body = asyncio.run(consume())
            else:
                iterator = response.streaming_content
                assert isinstance(iterator, Iterator)
                body = b"".join(iterator)
            assert body == payload
        finally:
            response.close()
    else:
        assert isinstance(response, HttpResponse)
        assert response.status_code == {"api": 409, "input": 400}.get(failure, 500)
        error = json.loads(response.content)["error"]
        assert error["message"] == {
            "api": "primary API failure",
            "input": "primary input failure",
        }.get(failure, "Speech synthesis failed.")
    assert all(handle.closed for handle in handles)


def test_speech_passthrough_cancellation_closes_before_deleting(monkeypatch, tmp_path, caplog):
    source = tmp_path / "source.wav"
    source.write_bytes(b"synthetic")
    monkeypatch.setattr(views, "synthesize_text", lambda *args: SpeechResult(source))
    monkeypatch.setattr(
        views, "export_audio", lambda *args, **kwargs: ExportedAudio(source, "wav", "audio/wav")
    )
    handles = []
    attempted = []

    def cancel(request, handle, **kwargs):
        handles.append(handle)
        raise asyncio.CancelledError("primary cancellation")

    def delete(path, **kwargs):
        assert path == source
        assert handles[0].closed
        attempted.append(path)
        raise PermissionError(PRIVATE)

    monkeypatch.setattr(views, "reader_response", cancel)
    monkeypatch.setattr(Path, "unlink", delete)
    request = RequestFactory().post(
        "/v1/audio/speech",
        data={"model": "tts-1", "input": "Synthetic"},
        content_type="application/json",
        headers={"Authorization": "Bearer test-token"},
    )
    with pytest.raises(asyncio.CancelledError, match="primary cancellation"):
        views.audio_speech(request)
    assert attempted == [source]
    assert PRIVATE not in caplog.text
