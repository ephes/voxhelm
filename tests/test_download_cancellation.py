import asyncio
import io
from threading import Event
from types import SimpleNamespace

import pytest
from asgiref.sync import sync_to_async
from django.http import StreamingHttpResponse
from django.test import AsyncClient

from tests.test_job_downloads import artifact_fixture, artifact_url, install_store


@pytest.mark.django_db(transaction=True)
@pytest.mark.parametrize("read_fails", [False, True])
def test_disconnect_defers_close_until_active_read_settles(monkeypatch, read_fails):
    job, _ = artifact_fixture()
    started, release, closed = Event(), Event(), Event()
    events = []

    class Reader(io.BytesIO):
        def read(self, size=-1):
            assert size == 65536
            events.append("reading")
            started.set()
            assert release.wait(5)
            events.append("settled")
            if read_fails:
                raise OSError("synthetic read failure")
            return b"synthetic"

        def close(self):
            events.append("closed")
            closed.set()
            super().close()

    reader = Reader()
    install_store(monkeypatch, lambda _: SimpleNamespace(open_reader=lambda **kw: reader))

    async def request():
        response = await AsyncClient().get(
            artifact_url(job, False), headers={"Authorization": "Bearer test-token"}
        )
        assert isinstance(response, StreamingHttpResponse)
        iterator = response.streaming_content
        task = asyncio.create_task(anext(iterator))
        assert await asyncio.to_thread(started.wait, 5)
        try:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await sync_to_async(response.close)()
            assert events == ["reading"]
        finally:
            release.set()
        assert await asyncio.to_thread(closed.wait, 5)
        await iterator.aclose()
        await sync_to_async(response.close)()
        assert events == ["reading", "settled", "closed"]

    asyncio.run(request())


@pytest.mark.django_db(transaction=True)
def test_asgi_response_closed_before_iteration_closes_reader_once(monkeypatch):
    job, _ = artifact_fixture()
    calls = []

    class Reader(io.BytesIO):
        def close(self):
            calls.append(True)
            super().close()

    reader = Reader(b"synthetic")
    install_store(monkeypatch, lambda _: SimpleNamespace(open_reader=lambda **kw: reader))

    async def request():
        response = await AsyncClient().get(
            artifact_url(job, False), headers={"Authorization": "Bearer test-token"}
        )
        await sync_to_async(response.close)()
        await sync_to_async(response.close)()
        assert calls == [True]

    asyncio.run(request())
