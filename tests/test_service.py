from __future__ import annotations

import threading
import time
from pathlib import Path

from transcriptions.service import TranscribeParams, TranscriptionResult, transcribe_audio


class SerialBackend:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def transcribe(self, audio_path: Path, params: TranscribeParams) -> TranscriptionResult:
        del audio_path, params
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return TranscriptionResult(text="ok", language="en", segments=[])


def test_transcribe_audio_serializes_backend_access(monkeypatch) -> None:
    backend = SerialBackend()
    monkeypatch.setattr("transcriptions.service.get_backend_service", lambda: backend)
    params = TranscribeParams(request_model="whisper-1", prompt=None, language=None)
    errors: list[Exception] = []

    def worker() -> None:
        try:
            transcribe_audio(Path("/tmp/sample.mp3"), params)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    second.start()
    first.join()
    second.join()

    assert errors == []
    assert backend.max_active == 1
