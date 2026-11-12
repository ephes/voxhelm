"""Wyoming connection lifecycle: inference ownership, admission, write deadline, shutdown."""

from __future__ import annotations

import asyncio
import threading
import time
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from wyoming.asr import Transcript
from wyoming.audio import AudioStart, AudioStop
from wyoming.error import Error
from wyoming.info import Describe
from wyoming.tts import Synthesize

import transcriptions.wyoming as wyoming_module
from tests.test_wyoming import (
    ONE_SECOND_PCM,
    ClosingWriter,
    decode_all_events,
    encode_events,
    limit_config,
    pcm_chunk,
)
from transcriptions.errors import ApiError
from transcriptions.wyoming import (
    WyomingAdmission,
    WyomingSttEventHandler,
    build_wyoming_info,
    run_wyoming_stt_server,
    shutdown_wyoming_server,
)

PRIVATE = "/private/synthetic-secret-path"


def make_handler(reader, writer, *, admission=None, config=None):
    resolved = config or limit_config()
    return WyomingSttEventHandler(
        resolved, build_wyoming_info(resolved), reader, writer, admission=admission
    )


def errors(writer) -> list[Error]:
    return [Error.from_event(e) for e in decode_all_events(writer) if Error.is_type(e.type)]


# --- Cancelled synthesis result ownership ----------------------------------------------


@pytest.mark.parametrize("cancel_first", [True, False])
def test_cancelled_synthesis_result_is_cleaned_up(monkeypatch, tmp_path, cancel_first) -> None:
    speech_path = tmp_path / "speech.wav"
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    cleaned: list[Path] = []

    def blocking_synthesize(text, params):
        entered.set()
        release.wait(5)
        speech_path.write_bytes(b"synthetic")
        return SimpleNamespace(audio_path=speech_path)

    def record_cleanup(*paths):
        cleaned.extend(paths)
        finished.set()

    monkeypatch.setattr(wyoming_module, "synthesize_text", blocking_synthesize)
    monkeypatch.setattr(wyoming_module, "cleanup_paths", record_cleanup)
    admission = WyomingAdmission()

    async def flow() -> None:
        handler = make_handler(asyncio.StreamReader(), ClosingWriter(), admission=admission)
        task = asyncio.create_task(handler.handle_event(Synthesize(text="Hello").event()))
        await asyncio.to_thread(entered.wait, 5)
        if cancel_first:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            release.set()
            await asyncio.to_thread(finished.wait, 5)
        else:
            release.set()
            # Block the loop (no await) until the worker has published its result, so
            # the request is cancelled before it can resume and take ownership.
            while admission.inference_threads:
                time.sleep(0.001)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(flow())
    assert cleaned == [speech_path]
    assert admission.inference_threads == 0


# --- Pre-thread inference admission ------------------------------------------------------


def test_synthesis_refused_before_thread_when_inference_capacity_is_full(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(wyoming_module, "synthesize_text", lambda text, params: calls.append(text))
    admission = WyomingAdmission(max_inference_threads=1, inference_threads=1)
    writer = ClosingWriter()

    async def flow() -> bool:
        handler = make_handler(asyncio.StreamReader(), writer, admission=admission)
        return await handler.handle_event(Synthesize(text="Hello").event())

    assert asyncio.run(flow()) is False
    assert calls == []
    assert [e.code for e in errors(writer)] == ["busy"]
    assert admission.inference_threads == 1


def test_transcription_refused_before_thread_when_inference_capacity_is_full(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(wyoming_module, "transcribe_audio", lambda *a: calls.append(a))
    admission = WyomingAdmission(max_inference_threads=1, inference_threads=1)
    writer = ClosingWriter()

    async def flow() -> None:
        handler = make_handler(asyncio.StreamReader(), writer, admission=admission)
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(pcm_chunk(ONE_SECOND_PCM[:3200]))
        assert await handler.handle_event(AudioStop().event()) is False

    asyncio.run(flow())
    assert calls == []
    assert [e.code for e in errors(writer)] == ["busy"]


# --- Active connection admission ---------------------------------------------------------


def test_connection_over_limit_is_refused_and_slot_released() -> None:
    admission = WyomingAdmission(max_connections=1)
    first_writer = ClosingWriter()
    second_writer = ClosingWriter()

    async def flow() -> None:
        first_reader = asyncio.StreamReader()
        first = asyncio.create_task(
            make_handler(first_reader, first_writer, admission=admission).run()
        )
        while admission.connections == 0:
            await asyncio.sleep(0)
        second_reader = asyncio.StreamReader()
        second_reader.feed_data(encode_events(Describe().event()))
        second_reader.feed_eof()
        await asyncio.wait_for(
            make_handler(second_reader, second_writer, admission=admission).run(), 5
        )
        first_reader.feed_eof()
        await asyncio.wait_for(first, 5)

    asyncio.run(flow())
    assert [e.code for e in errors(second_writer)] == ["server_busy"]
    assert len(decode_all_events(second_writer)) == 1
    assert second_writer.closed and first_writer.closed
    assert admission.connections == 0
    assert not admission.handlers


def test_config_reads_admission_limits(settings) -> None:
    settings.VOXHELM_WYOMING_MAX_CONNECTIONS = 3
    settings.VOXHELM_WYOMING_MAX_INFERENCE_THREADS = 4
    config = wyoming_module.get_wyoming_stt_config()
    assert (config.max_connections, config.max_inference_threads) == (3, 4)


@pytest.mark.parametrize(
    "name", ["VOXHELM_WYOMING_MAX_CONNECTIONS", "VOXHELM_WYOMING_MAX_INFERENCE_THREADS"]
)
def test_config_rejects_non_positive_admission_limits(settings, name) -> None:
    from django.core.exceptions import ImproperlyConfigured

    setattr(settings, name, 0)
    with pytest.raises(ImproperlyConfigured, match=name):
        wyoming_module.get_wyoming_stt_config()


# --- Outbound write deadline ---------------------------------------------------------------


class StalledTransport:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


class StalledWriter(ClosingWriter):
    """A client that never reads: drain never completes."""

    def __init__(self) -> None:
        super().__init__()
        self.transport = StalledTransport()

    async def drain(self) -> None:
        await asyncio.Event().wait()


def test_stalled_client_write_is_aborted_after_deadline(monkeypatch) -> None:
    monkeypatch.setattr(wyoming_module, "EVENT_WRITE_TIMEOUT_SECONDS", 0.05)
    writer = StalledWriter()
    admission = WyomingAdmission()

    async def flow() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(encode_events(Describe().event(), Describe().event()))
        await asyncio.wait_for(make_handler(reader, writer, admission=admission).run(), 5)

    asyncio.run(flow())
    assert writer.transport.aborted
    assert writer.closed
    assert admission.connections == 0


# --- Public failure diagnostics -------------------------------------------------------------


def test_synthesis_failure_does_not_expose_exception_text(monkeypatch) -> None:
    def fail(text, params):
        raise RuntimeError(f"ffmpeg failed reading {PRIVATE}")

    monkeypatch.setattr(wyoming_module, "synthesize_text", fail)
    writer = ClosingWriter()

    async def flow() -> None:
        await make_handler(asyncio.StreamReader(), writer).handle_event(
            Synthesize(text="Hello").event()
        )

    asyncio.run(flow())
    [error] = errors(writer)
    assert error.code == "synthesis_failed"
    assert PRIVATE not in error.text
    assert error.text == "Speech synthesis failed."


def test_transcription_failure_does_not_expose_exception_text(monkeypatch) -> None:
    def fail(*args):
        raise RuntimeError(f"model missing at {PRIVATE}")

    monkeypatch.setattr(wyoming_module, "transcribe_audio", fail)
    writer = ClosingWriter()

    async def flow() -> None:
        handler = make_handler(asyncio.StreamReader(), writer)
        await handler.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        await handler.handle_event(pcm_chunk(ONE_SECOND_PCM[:3200]))
        await handler.handle_event(AudioStop().event())

    asyncio.run(flow())
    [error] = errors(writer)
    assert error.code == "transcription_failed"
    assert PRIVATE not in error.text


def test_missing_voice_message_remains_client_facing(monkeypatch, tmp_path) -> None:
    from tests.test_synthesis import _piper_backend

    backend = _piper_backend(tmp_path)

    def resolve_missing(text, params):
        backend.resolve_voice(voice="synthetic-missing", language=None)

    monkeypatch.setattr(wyoming_module, "synthesize_text", resolve_missing)
    writer = ClosingWriter()

    async def flow() -> None:
        await make_handler(asyncio.StreamReader(), writer).handle_event(
            Synthesize(text="Hello").event()
        )

    asyncio.run(flow())
    [error] = errors(writer)
    assert error.code == "synthesis_failed"
    assert "Requested voice 'synthetic-missing' is not installed" in error.text


def test_api_errors_remain_client_facing(monkeypatch) -> None:
    def fail(text, params):
        raise ApiError("Synthetic request error.")

    monkeypatch.setattr(wyoming_module, "synthesize_text", fail)
    writer = ClosingWriter()

    async def flow() -> None:
        await make_handler(asyncio.StreamReader(), writer).handle_event(
            Synthesize(text="Hello").event()
        )

    asyncio.run(flow())
    assert [e.text for e in errors(writer)] == ["Synthetic request error."]


# --- Server cancellation and transport cleanup -------------------------------------------


class FakeServer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = False
        self.factory: Any = None

    async def start(self, factory) -> None:
        self.factory = factory
        self.started.set()

    async def stop(self) -> None:
        self.stopped = True


def test_cancelled_server_stops_listening_and_closes_handlers(monkeypatch) -> None:
    server = FakeServer()
    monkeypatch.setattr(wyoming_module.AsyncServer, "from_uri", staticmethod(lambda uri: server))
    writer = ClosingWriter()
    config = replace(limit_config(), host="127.0.0.1", port=0)

    async def flow() -> None:
        serving = asyncio.create_task(run_wyoming_stt_server(config=config))
        await asyncio.wait_for(server.started.wait(), 5)
        handler = server.factory(asyncio.StreamReader(), writer)
        handler_task = asyncio.create_task(handler.run())
        while not handler.admission.connections:
            await asyncio.sleep(0)
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        assert handler_task.done()

    asyncio.run(flow())
    assert server.stopped
    assert writer.closed


def test_shutdown_cancels_handler_waiting_for_input() -> None:
    server = FakeServer()
    admission = WyomingAdmission()
    writer = ClosingWriter()

    async def flow() -> None:
        handler = make_handler(asyncio.StreamReader(), writer, admission=admission)
        task = asyncio.create_task(handler.run())
        while not admission.connections:
            await asyncio.sleep(0)
        await asyncio.wait_for(shutdown_wyoming_server(cast(Any, server), admission), 5)
        assert task.cancelled() or task.done()

    asyncio.run(flow())
    assert server.stopped and writer.closed
    assert admission.connections == 0


def test_transport_closes_even_if_disconnect_hook_fails() -> None:
    class FailingDisconnect(WyomingSttEventHandler):
        async def disconnect(self) -> None:
            raise RuntimeError("synthetic disconnect failure")

    writer = ClosingWriter()
    config = limit_config()

    async def flow() -> None:
        reader = asyncio.StreamReader()
        reader.feed_eof()
        handler = FailingDisconnect(config, build_wyoming_info(config), reader, writer)
        with pytest.raises(RuntimeError):
            await handler.run()

    asyncio.run(flow())
    assert writer.closed


def test_normal_transcription_still_round_trips(monkeypatch) -> None:
    from transcriptions.service import TranscriptionResult

    monkeypatch.setattr(
        wyoming_module,
        "transcribe_audio",
        lambda path, params: TranscriptionResult(text="hallo", language="de", segments=[]),
    )
    writer = ClosingWriter()
    admission = WyomingAdmission()

    async def flow() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(
            encode_events(
                AudioStart(rate=16000, width=2, channels=1).event(),
                pcm_chunk(ONE_SECOND_PCM[:3200]),
                AudioStop().event(),
            )
        )
        reader.feed_eof()
        await asyncio.wait_for(make_handler(reader, writer, admission=admission).run(), 5)

    asyncio.run(flow())
    events = decode_all_events(writer)
    assert Transcript.from_event(events[-1]).text == "hallo"
    assert admission.inference_threads == 0 and admission.connections == 0


@pytest.mark.parametrize("client_state", ["idle", "inference", "not-reading"])
def test_real_tcp_shutdown_does_not_wait_for_clients(monkeypatch, tmp_path, client_state) -> None:
    """Shutdown against a real TCP server returns promptly and drops every client."""
    monkeypatch.setattr(wyoming_module, "EVENT_READ_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr(wyoming_module, "EVENT_WRITE_TIMEOUT_SECONDS", 60)
    monkeypatch.setattr(wyoming_module, "cleanup_paths", lambda *paths: None)
    entered = threading.Event()
    release = threading.Event()
    speech_path = tmp_path / "speech.wav"
    with wave.open(str(speech_path), "wb") as wav_file:
        wav_file.setframerate(16000)
        wav_file.setsampwidth(2)
        wav_file.setnchannels(1)
        wav_file.writeframes(b"\x01\x00" * 4_000_000)

    def blocking_synthesize(text, params):
        entered.set()
        if client_state == "not-reading":
            return SimpleNamespace(audio_path=speech_path)
        release.wait(10)
        raise RuntimeError("synthetic")

    monkeypatch.setattr(wyoming_module, "synthesize_text", blocking_synthesize)
    config = limit_config()
    admission = WyomingAdmission()

    async def flow() -> float:
        server = wyoming_module.AsyncServer.from_uri("tcp://127.0.0.1:0")
        await server.start(
            lambda r, w: WyomingSttEventHandler(
                config, build_wyoming_info(config), r, w, admission=admission
            )
        )
        port = cast(Any, server)._server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        while not admission.connections:
            await asyncio.sleep(0.01)
        if client_state != "idle":
            writer.write(encode_events(Synthesize(text="Hello").event()))
            await writer.drain()
            await asyncio.to_thread(entered.wait, 5)
        if client_state == "not-reading":
            # Let the server fill the socket buffers while this client reads nothing.
            await asyncio.sleep(0.3)
        started = time.monotonic()
        await asyncio.wait_for(shutdown_wyoming_server(server, admission), 5)
        elapsed = time.monotonic() - started
        if client_state != "not-reading":
            assert await asyncio.wait_for(reader.read(), 5) == b""
        writer.close()
        release.set()
        return elapsed

    try:
        elapsed = asyncio.run(flow())
    finally:
        release.set()
    assert elapsed < 2
    assert admission.connections == 0 and not admission.handlers


def test_connection_starting_during_shutdown_is_refused() -> None:
    admission = WyomingAdmission(closing=True)
    writer = ClosingWriter()

    async def flow() -> None:
        reader = asyncio.StreamReader()
        reader.feed_data(encode_events(Describe().event()))
        reader.feed_eof()
        await asyncio.wait_for(make_handler(reader, writer, admission=admission).run(), 5)

    asyncio.run(flow())
    assert [e.code for e in errors(writer)] == ["server_busy"]
    assert writer.closed and admission.connections == 0
