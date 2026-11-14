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
