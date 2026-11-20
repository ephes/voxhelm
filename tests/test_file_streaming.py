"""Synthetic reader ownership failures without storage or HTTP transports."""

import asyncio
import io
from collections.abc import AsyncGenerator
from threading import Event

import pytest
from django.test import AsyncRequestFactory, RequestFactory

from transcriptions import file_streaming


@pytest.mark.parametrize("asgi", [False, True])
@pytest.mark.parametrize("stage", ["seek", "tell", "file_response", "stream_response", "header"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_response_construction_failure_closes_once(monkeypatch, asgi, stage, close_fails):
    if not asgi and stage in {"stream_response", "header"}:
        pytest.skip("ASGI-only construction stage")
    primary = OSError("synthetic primary")
    calls = []

    class Reader(io.BytesIO):
        def seek(self, *args):
            if stage == "seek":
                raise primary
            return super().seek(*args)

        def tell(self):
            if stage == "tell":
                raise primary
            return super().tell()

        def close(self):
            calls.append("close")
            super().close()
            if close_fails:
                raise OSError("private close diagnostic")

    reader = Reader(b"synthetic")

    def fail(*args, **kwargs):
        raise primary

    if stage == "file_response":
        monkeypatch.setattr(file_streaming, "FileResponse", fail)
    if stage == "stream_response":
        monkeypatch.setattr(file_streaming, "StreamingHttpResponse", fail)
    if stage == "header":
        original = file_streaming.StreamingHttpResponse.__setitem__

        def header(self, name, value):
            if name == "Content-Length":
                raise primary
            return original(self, name, value)

        monkeypatch.setattr(file_streaming.StreamingHttpResponse, "__setitem__", header)
    request = (AsyncRequestFactory() if asgi else RequestFactory()).get("/")
    with pytest.raises(OSError) as caught:
        file_streaming.reader_response(request, reader, content_type="application/octet-stream")
    assert caught.value is primary
    assert calls == ["close"]


@pytest.mark.parametrize("read_fails", [False, True])
@pytest.mark.parametrize("log_fails", [False, True])
def test_async_read_and_close_failure_preserve_primary(caplog, monkeypatch, read_fails, log_fails):
    primary = OSError("synthetic read")
    secondary = OSError("private close diagnostic")
    calls = []

    class Reader(io.BytesIO):
        def read(self, size=-1):
            if read_fails:
                raise primary
            return b""

        def close(self):
            calls.append("close")
            super().close()
            raise secondary

    if log_fails:

        def fail_warning(message):
            calls.append("warning")
            raise RuntimeError("synthetic logging failure")

        monkeypatch.setattr(file_streaming._LOGGER, "warning", fail_warning)

    stream = file_streaming.AsyncReaderStream(Reader())

    async def consume():
        return [chunk async for chunk in stream]

    with pytest.raises(OSError) as caught:
        asyncio.run(consume())
    assert caught.value is (primary if read_fails else secondary)
    stream.close()
    assert calls == (["close", "warning"] if read_fails and log_fails else ["close"])
    assert "private close diagnostic" not in caplog.text
    if read_fails and not log_fails:
        assert "Reader cleanup failed" in caplog.text


@pytest.mark.parametrize("read_fails", [False, True])
def test_cancelled_read_defers_failing_close_and_preserves_worker_error(caplog, read_fails):
    started, release, settled = Event(), Event(), Event()
    primary = OSError("synthetic read")
    calls = []
    worker_errors = []

    class Reader(io.BytesIO):
        def read(self, size=-1):
            started.set()
            assert release.wait(5)
            calls.append("read settled")
            if read_fails:
                raise primary
            return b""

        def close(self):
            calls.append("close")
            super().close()
            raise OSError("private close diagnostic")

    class Stream(file_streaming.AsyncReaderStream):
        def _read(self):
            try:
                return super()._read()
            except BaseException as exc:
                worker_errors.append(exc)
                raise
            finally:
                settled.set()

    stream = Stream(Reader())

    async def consume():
        iterator = aiter(stream)
        assert isinstance(iterator, AsyncGenerator)

        async def next_chunk():
            return await anext(iterator)

        task = asyncio.create_task(next_chunk())
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            stream.close()
            assert calls == []
        finally:
            release.set()
        assert await asyncio.to_thread(settled.wait, 5)
        await iterator.aclose()
        stream.close()

    asyncio.run(consume())
    assert calls == ["read settled", "close"]
    assert worker_errors == ([primary] if read_fails else [])
    assert "Reader cleanup failed" in caplog.text
    assert "private close diagnostic" not in caplog.text


def test_second_traversal_cannot_read_or_close_owned_reader():
    started, release = Event(), Event()
    calls = []

    class Reader(io.BytesIO):
        def read(self, size=-1):
            calls.append("read")
            if started.is_set():
                return b""
            started.set()
            assert release.wait(5)
            calls.append("read settled")
            return b""

        def close(self):
            calls.append("close")
            super().close()

    stream = file_streaming.AsyncReaderStream(Reader())

    async def consume():
        owner = asyncio.create_task(anext(aiter(stream), None))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            with pytest.raises(RuntimeError, match="already being consumed"):
                await anext(aiter(stream), None)
            assert calls == ["read"]
        finally:
            release.set()
            assert await owner is None
        stream.close()

    asyncio.run(consume())
    assert calls == ["read", "read settled", "close"]


def test_close_inside_caller_handled_exception_still_reports_close_failure(caplog):
    secondary = OSError("private close diagnostic")

    class Reader(io.BytesIO):
        def close(self):
            super().close()
            raise secondary

    stream = file_streaming.AsyncReaderStream(Reader())
    try:
        raise ValueError("caller is handling this")
    except ValueError:
        with pytest.raises(OSError) as caught:
            stream.close()
    assert caught.value is secondary
