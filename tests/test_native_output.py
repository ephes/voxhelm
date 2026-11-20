"""Stable descriptor generations are required before native bytes become evidence."""

import pytest

from jobs import remote_worker_cli as worker_cli
from tests.native_output_fakes import install_native_read_barrier
from tests.test_remote_worker_cli import (
    RecordingClient,
    base_claim,
    configure_filesystem_store,
    install_fake_download,
)
from tests.test_transcript_coverage import run_backend, seg
from tests.window_fakes import install_fake_window_runtime
from transcriptions import chapter_windows, windows
from transcriptions.native_output import read_native_json


@pytest.mark.parametrize("change", ["rewrite", "truncate", "grow", "short"])
def test_native_json_requires_stable_metadata_and_complete_bytes(tmp_path, monkeypatch, change):
    path = tmp_path / "out.json"
    path.write_bytes(b'{"text":"before"}')
    install_native_read_barrier(monkeypatch, change)
    with pytest.raises(RuntimeError, match="JSON output changed while being read"):
        read_native_json(path, limit=1024)


def test_stable_native_json_accepts_exact_limit(tmp_path):
    path = tmp_path / "out.json"
    raw = b'{"text":"before"}'
    path.write_bytes(raw)
    assert read_native_json(path, limit=len(raw)) == raw


def test_native_json_refuses_replaced_path_with_stable_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "current" / "out.json"
    path.parent.mkdir()
    path.write_bytes(b'{"text":"before"}')
    changed = install_native_read_barrier(monkeypatch, "replace")
    with pytest.raises(RuntimeError, match="JSON output changed while being read"):
        read_native_json(path, limit=1024)
    assert changed == [path]


@pytest.mark.parametrize("change", ["rewrite", "truncate"])
def test_backend_refuses_changing_json(tmp_path, monkeypatch, settings, change):
    changed = install_native_read_barrier(monkeypatch, change)
    with pytest.raises(RuntimeError, match="transcription produced invalid JSON output") as error:
        run_backend(
            monkeypatch, tmp_path, settings, media_seconds=10, segments=[seg(0, 10, text="before")]
        )
    assert "JSON output changed while being read" in str(error.value.__cause__)
    assert len(changed) == 1


@pytest.mark.parametrize("change", ["rewrite", "truncate"])
@pytest.mark.parametrize("consumer", ["window", "chapter"])
def test_window_consumers_refuse_changing_json(tmp_path, monkeypatch, settings, change, consumer):
    runtime = install_fake_window_runtime(settings, tmp_path, monkeypatch)
    runtime.tokens = [(" before", 1000, 1300)]
    runtime.chapter_segments = [(" before", 1000, 1300)]
    runtime.write_config()
    changed = install_native_read_barrier(monkeypatch, change)
    with pytest.raises(RuntimeError, match="JSON output changed while being read"):
        if consumer == "window":
            windows.transcribe_window(tmp_path / "audio", 1)
        else:
            executable, model = windows.resolve_window_runtime()
            chapter_windows.run_chapter_windows(
                tmp_path / "audio",
                [60000],
                executable=executable,
                model_path=model,
                duration_ms=100000,
            )
    assert len(changed) == 1
    assert list(runtime.temp_root.iterdir()) == []


@pytest.mark.parametrize("change", ["stable", "rewrite", "truncate"])
def test_remote_native_output_publication(tmp_path, monkeypatch, settings, change):
    configure_filesystem_store(settings, tmp_path / "artifacts")
    install_fake_download(monkeypatch, tmp_path)
    if change != "stable":
        install_native_read_barrier(monkeypatch, change)

    class NativeBackend:
        def transcribe(self, audio_path, params):
            return run_backend(
                monkeypatch,
                tmp_path,
                settings,
                media_seconds=10,
                segments=[seg(0, 10, text="before")],
            )[0]

    monkeypatch.setattr(worker_cli, "build_backend_service", lambda **_: NativeBackend())
    client = RecordingClient()
    if change == "stable":
        worker_cli.process_claim(base_claim(), config=client.config, client=client)
        assert client.completed is not None
        assert client.completed["result_text"] == "before"
    else:
        with pytest.raises(RuntimeError, match="transcription produced invalid JSON output"):
            worker_cli.process_claim(base_claim(), config=client.config, client=client)
        assert client.completed is None
        assert not list(settings.VOXHELM_ARTIFACT_ROOT.rglob("transcript.*"))
