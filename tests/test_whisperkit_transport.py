"""Loopback-only sidecars exercise whole HTTP exchanges without inference."""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from transcriptions import service


@pytest.fixture
def sidecar():
    stops = threading.Event()
    mode: dict[str, Any] = {
        "name": "ok",
        "received": threading.Event(),
        "body": b'{"text":"synthetic"}',
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.connection.settimeout(2)
            mode["received"].set()
            try:
                if mode["name"] == "upload":
                    while not stops.wait(0.05):
                        if not self.rfile.read(1024):
                            return
                    return
                self.rfile.read(int(self.headers["Content-Length"]))
                if mode["name"] == "headers":
                    stops.wait(3)
                    return
                self.send_response(503 if mode["name"] == "error" else 200)
                self.end_headers()
                body = mode["body"]
                if mode["name"] == "trickle":
                    for byte in body:
                        self.wfile.write(bytes([byte]))
                        self.wfile.flush()
                        if stops.wait(0.05):
                            break
                elif mode["name"] == "body":
                    self.wfile.write(b"{")
                    self.wfile.flush()
                    stops.wait(3)
                else:
                    self.wfile.write(body)
            except OSError, ValueError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/v1", mode
    stops.set()
    server.shutdown()
    server.server_close()
    thread.join(2)


def call(sidecar, tmp_path, **kwargs):
    path = tmp_path / "synthetic.wav"
    path.write_bytes(b"x" * kwargs.pop("audio_bytes", 16))
    return service.call_whisperkit_server(
        base_url=sidecar[0],
        audio_path=path,
        model_name="synthetic",
        language=None,
        prompt=None,
        timeout_seconds=kwargs.pop("timeout_seconds", 2),
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["ok", "error"])
def test_whisperkit_bounds_success_and_error_body(monkeypatch, sidecar, tmp_path, mode):
    monkeypatch.setattr(service, "WHISPERKIT_RESPONSE_MAX_BYTES", 1024, raising=False)
    monkeypatch.setattr(service, "WHISPERKIT_ERROR_MAX_BYTES", 128, raising=False)
    sidecar[1].update(name=mode, body=b"x" * 4096)
    with pytest.raises(RuntimeError, match="response byte limit"):
        call(sidecar, tmp_path)


@pytest.mark.parametrize(
    "raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":' + b"[" * 130 + b"0" + b"]" * 130 + b"}"]
)
def test_whisperkit_strict_json(sidecar, tmp_path, raw):
    sidecar[1]["body"] = raw
    with pytest.raises(RuntimeError, match="invalid JSON"):
        call(sidecar, tmp_path)


@pytest.mark.parametrize("phase", ["trickle", "upload"])
def test_whisperkit_deadline_covers_slow_progress(sidecar, tmp_path, phase):
    sidecar[1]["name"] = phase
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="elapsed deadline"):
        call(
            sidecar,
            tmp_path,
            timeout_seconds=0.2,
            audio_bytes=16 * 1024 * 1024 if phase == "upload" else 16,
        )
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize("phase", ["upload", "headers", "body"])
def test_whisperkit_backend_cancels_owned_exchange(sidecar, tmp_path, phase, monkeypatch):
    from lane_scheduler import InferenceCancelled

    sidecar[1]["name"] = phase
    path = tmp_path / "synthetic.wav"
    path.write_bytes(b"x" * (16 * 1024 * 1024 if phase == "upload" else 16))
    event = threading.Event()
    opened = []
    path_open = type(path).open

    def record_open(self, *args, **kwargs):
        handle = path_open(self, *args, **kwargs)
        if self == path:
            opened.append(handle)
        return handle

    monkeypatch.setattr(type(path), "open", record_open)

    def cancel():
        assert sidecar[1]["received"].wait(2)
        time.sleep(0.1)
        event.set()

    watcher = threading.Thread(target=cancel, daemon=True)
    watcher.start()
    backend = service.WhisperKitBackend(
        enabled=True,
        base_url=sidecar[0],
        model_name="synthetic",
        timeout_seconds=2,
    )
    started = time.monotonic()
    with pytest.raises(InferenceCancelled):
        backend.transcribe(
            path, service.TranscribeParams("whisperkit", None, None, cancel_event=event)
        )
    watcher.join(2)
    assert time.monotonic() - started < 1.2
    assert all(handle.closed for handle in opened)


def test_whisperkit_cancelled_before_start_never_calls_transport(monkeypatch, tmp_path):
    from lane_scheduler import InferenceCancelled

    event = threading.Event()
    event.set()

    def unexpected(**kwargs):
        pytest.fail("cancelled inference started HTTP work")

    monkeypatch.setattr(service, "call_whisperkit_server", unexpected)
    backend = service.WhisperKitBackend(
        enabled=True, base_url="http://localhost", model_name="x", timeout_seconds=1
    )
    with pytest.raises(InferenceCancelled):
        backend.transcribe(
            tmp_path / "absent.wav",
            service.TranscribeParams("auto", None, None, cancel_event=event),
        )


def test_whisperkit_http_error_omits_private_context(sidecar, tmp_path, caplog):
    import logging

    caplog.set_level(logging.DEBUG)
    sidecar[1].update(name="error", body=b"PRIVATE_PROMPT_SENTINEL")
    url = (
        sidecar[0].replace("http://", "http://user:PRIVATE_PASSWORD_SENTINEL@")
        + "?key=PRIVATE_QUERY_SENTINEL"
    )
    path = tmp_path / "PRIVATE_PATH_SENTINEL.wav"
    path.write_bytes(b"synthetic")
    backend = service.WhisperKitBackend(
        enabled=True, base_url=url, model_name="x", timeout_seconds=2
    )
    with pytest.raises(RuntimeError) as caught:
        backend.transcribe(
            path, service.TranscribeParams("whisperkit", "PRIVATE_PROMPT_SENTINEL", None)
        )
    assert "HTTP 503" in str(caught.value)
    public = str(caught.value) + caplog.text
    assert "PRIVATE_" not in public


def test_whisperkit_unreachable_omits_url_and_exception(monkeypatch, tmp_path, caplog):
    def unreachable(**kwargs):
        raise OSError("PRIVATE_PATH_SENTINEL")

    monkeypatch.setattr(service, "call_whisperkit_server", unreachable)
    backend = service.WhisperKitBackend(
        enabled=True,
        base_url="http://user:PRIVATE_PASSWORD_SENTINEL@localhost?key=PRIVATE_QUERY_SENTINEL",
        model_name="x",
        timeout_seconds=1,
    )
    with pytest.raises(service.BackendUnavailableError) as caught:
        backend.transcribe(tmp_path / "input", service.TranscribeParams("whisperkit", None, None))
    assert "not reachable" in str(caught.value)
    assert "PRIVATE_" not in str(caught.value) + caplog.text


def test_whisperkit_api_error_is_private(client, monkeypatch, sidecar, caplog):
    from django.core.files.uploadedfile import SimpleUploadedFile

    sidecar[1].update(name="error", body=b"PRIVATE_PROMPT_SENTINEL")
    backend = service.WhisperKitBackend(
        enabled=True,
        base_url=sidecar[0] + "?key=PRIVATE_QUERY_SENTINEL",
        model_name="x",
        timeout_seconds=2,
    )
    monkeypatch.setattr(
        service,
        "get_backend_services_for_model",
        lambda _: [service.BackendInvocation("whisperkit", backend)],
    )
    response = client.post(
        "/v1/audio/transcriptions",
        {
            "file": SimpleUploadedFile("synthetic.wav", b"synthetic", content_type="audio/wav"),
            "model": "whisper-1",
        },
        HTTP_AUTHORIZATION="Bearer test-token",
    )
    assert response.status_code == 500
    assert "HTTP 503" in response.json()["error"]["message"]
    assert "PRIVATE_" not in response.content.decode() + caplog.text


def test_whisperkit_late_success_after_cancellation_is_discarded(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from lane_scheduler import InferenceCancelled

    event = threading.Event()
    ownership = []

    @contextmanager
    def admit(*args, **kwargs):
        ownership.append("admitted")
        try:
            yield
        finally:
            ownership.append("released")

    def late_success(**kwargs):
        assert kwargs["cancel_event"] is event
        event.set()
        return {"text": "late synthetic result"}

    class ForbiddenFallback:
        def transcribe(self, *args):
            pytest.fail("cancellation ran a fallback backend")

    backend = service.WhisperKitBackend(
        enabled=True, base_url="http://localhost", model_name="x", timeout_seconds=1
    )
    monkeypatch.setattr(service, "call_whisperkit_server", late_success)
    monkeypatch.setattr(service, "admit_local_inference", admit)
    monkeypatch.setattr(
        service,
        "get_backend_services_for_params",
        lambda _: [
            service.BackendInvocation("whisperkit", backend),
            service.BackendInvocation("fallback", ForbiddenFallback()),
        ],
    )
    with pytest.raises(InferenceCancelled):
        service.transcribe_audio(
            tmp_path / "input.wav", service.TranscribeParams("auto", None, None, cancel_event=event)
        )
    assert ownership == ["admitted", "released"]
