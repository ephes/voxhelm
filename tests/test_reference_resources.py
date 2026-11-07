import importlib
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from jobs.media import DownloadedMedia
from jobs.models import Job
from transcriptions import known_speaker


@pytest.mark.parametrize(
    "start,end", [(0.0, 0.02), (0.01253125, 0.04378125), (3.222222, 3.444444), (4.9, 6), (6, 7)]
)
@pytest.mark.parametrize("rate", [16000, 44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("encoding", ["wav", "mp3", "m4a"])
def test_interval_samples_match_full_decode(
    tmp_path, settings, start, end, rate, channels, encoding
):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None, "Install ffmpeg to run synthetic audio codec regressions."
    settings.VOXHELM_FFMPEG_BIN = ffmpeg
    source = tmp_path / "synthetic.wav"
    pcm = (np.sin(np.arange(rate * 5) * 2 * np.pi * 440 / rate) * 10000).astype("<i2")
    with wave.open(str(source), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        if channels == 2:
            pcm = np.column_stack([pcm, pcm // 2]).astype("<i2")
        out.writeframes(pcm.tobytes())
    if encoding != "wav":
        encoded = source.with_suffix(f".{encoding}")
        codec = "libmp3lame" if encoding == "mp3" else "aac"
        subprocess.run(
            [ffmpeg, "-nostdin", "-y", "-i", str(source), "-c:a", codec, str(encoded)],
            check=True,
            capture_output=True,
            timeout=30,
        )
        source = encoded
    full = known_speaker.decode_mono_16k(source)
    interval = known_speaker.decode_mono_16k(source, start=start, end=end)
    np.testing.assert_array_equal(interval, known_speaker.slice_samples(full, 16000, start, end))


def test_long_source_range_bounds_pcm_materialization(tmp_path, monkeypatch):
    seen = []

    def run(argv, **kwargs):
        seen.append(argv)
        assert argv.index("-i") < argv.index("-ss")
        assert argv[argv.index("-ss") + 1] == "36000.0"
        assert argv[argv.index("-t") + 1] == "2.0"
        assert kwargs["timeout"] > 0
        with wave.open(argv[-1], "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(b"\0\0" * 32000)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(known_speaker.subprocess, "run", run)
    samples = known_speaker.decode_mono_16k(tmp_path / "ten-hours.wav", start=36000, end=36002)
    assert samples.shape == (32000,)
    assert samples.nbytes == 128000
    assert not Path(seen[0][-1]).parent.exists()


@pytest.mark.parametrize("failure", ["timeout", "decode"])
def test_interval_failure_cleans_tempfiles(tmp_path, monkeypatch, failure):
    outputs = []

    def run(argv, **kwargs):
        outputs.append(Path(argv[-1]))
        outputs[-1].write_bytes(b"partial")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 1)
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(known_speaker.subprocess, "run", run)
    with pytest.raises(known_speaker.KnownSpeakerError):
        known_speaker.decode_mono_16k(tmp_path / "source.wav", start=4, end=5)
    assert not outputs[0].parent.exists()


def collect(module, diarization, source_media=None):
    if module.__name__ == "jobs.services":
        job = Job(output_data={"diarization": diarization})
        if source_media is None:
            return module.build_known_speaker_references(job)
        return module.build_known_speaker_references(job, source_media=source_media)
    if source_media is None:
        return module.known_speaker_references_from_claim(diarization)
    return module.known_speaker_references_from_claim(diarization, source_media=source_media)


@pytest.mark.parametrize("module_name", ["jobs.services", "jobs.remote_worker_cli"])
@pytest.mark.parametrize("reuse_main", [False, True])
def test_reference_sources_are_job_scoped_snapshots(
    tmp_path, monkeypatch, settings, module_name, reuse_main
):
    settings.VOXHELM_ALLOWED_URL_HOSTS = ["media.example.com"]
    module = importlib.import_module(module_name)
    url = "https://media.example.com/input.wav"
    other = "https://media.example.com/other.wav"
    caller = tmp_path / "caller.wav"
    caller.write_bytes(b"caller")
    media = DownloadedMedia(caller, "audio/wav", url) if reuse_main else None
    downloaded: list[tuple[str, Path]] = []
    decoded = []

    def download(*, source_url):
        path = tmp_path / f"download-{len(downloaded)}.wav"
        path.write_bytes(source_url.encode())
        downloaded.append((source_url, path))
        return DownloadedMedia(path, "audio/wav", source_url)

    def decode(path, **kwargs):
        decoded.append((path, path.read_bytes(), kwargs))
        return np.zeros(32000, dtype="float32")

    monkeypatch.setattr(module, "download_allowed_media", download)
    monkeypatch.setattr(module, "decode_mono_16k", decode)
    diarization = {
        "known_speakers": [
            {
                "id": str(n),
                "name": f"Speaker {n}",
                "references": [
                    {
                        "kind": "source_range",
                        "audio": {"kind": "url", "url": source},
                        "start": n,
                        "end": n + 2,
                    }
                ],
            }
            for n, source in enumerate([url, url, other])
        ]
    }
    references = collect(module, diarization, media)
    assert len(references) == 3
    assert [entry[0] for entry in downloaded] == ([other] if reuse_main else [url, other])
    assert decoded[0][0] == decoded[1][0]
    assert decoded[0][1] == decoded[1][1]
    assert decoded[0][2] == {"start": 0.0, "end": 2.0}
    assert all(not path.exists() for _, path in downloaded)
    assert caller.read_bytes() == b"caller"
    # A second collection creates a new snapshot, never a global URL cache.
    collect(module, diarization, media)
    assert len(downloaded) == (2 if reuse_main else 4)


@pytest.mark.parametrize("module_name", ["jobs.services", "jobs.remote_worker_cli"])
def test_reference_partial_failure_cleans_owned_sources(
    tmp_path, monkeypatch, settings, module_name
):
    settings.VOXHELM_ALLOWED_URL_HOSTS = ["media.example.com"]
    module = importlib.import_module(module_name)
    path = tmp_path / "owned.wav"
    path.write_bytes(b"media")
    caller = tmp_path / "caller.wav"
    caller.write_bytes(b"caller")
    borrowed = DownloadedMedia(caller, "audio/wav", "https://media.example.com/caller.wav")
    monkeypatch.setattr(
        module, "download_allowed_media", lambda **kw: DownloadedMedia(path, "audio/wav")
    )

    def decode(*args, **kwargs):
        raise RuntimeError("synthetic decode failure")

    monkeypatch.setattr(module, "decode_mono_16k", decode)
    diarization = {
        "known_speakers": [
            {
                "id": "1",
                "name": "Synthetic",
                "references": [
                    {
                        "kind": "clip_artifact",
                        "audio": {"kind": "url", "url": "https://media.example.com/owned.wav"},
                    }
                ],
            }
        ]
    }
    with pytest.raises(RuntimeError, match="synthetic decode"):
        collect(module, diarization, borrowed)
    assert not path.exists()
    assert caller.exists()


@pytest.mark.parametrize("module_name", ["jobs.services", "jobs.remote_worker_cli"])
def test_reused_source_still_checks_allowlist(tmp_path, settings, module_name):
    from transcriptions.errors import ApiError

    settings.VOXHELM_ALLOWED_URL_HOSTS = []
    module = importlib.import_module(module_name)
    caller = tmp_path / "caller.wav"
    caller.write_bytes(b"caller")
    url = "https://denied.example/input.wav"
    media = DownloadedMedia(caller, "audio/wav", url)
    diarization = {
        "known_speakers": [
            {
                "id": "1",
                "name": "Synthetic",
                "references": [{"kind": "clip_artifact", "audio": {"kind": "url", "url": url}}],
            }
        ]
    }
    with pytest.raises(ApiError):
        collect(module, diarization, media)
    assert caller.read_bytes() == b"caller"
