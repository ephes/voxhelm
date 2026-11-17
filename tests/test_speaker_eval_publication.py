"""Frozen bundle and complete suite publication, using synthetic backends."""

import os
import time

import numpy as np
import pytest

from evals import speaker_naming_eval as ev
from transcriptions.service import TranscriptionResult, TranscriptionSegment


@pytest.fixture
def capture(tmp_path, monkeypatch):
    from transcriptions.diarization import PyannoteDiarizationBackend, SpeakerTurn

    (tmp_path / "audio.wav").write_bytes(b"synthetic audio")
    (tmp_path / "model.bin").write_bytes(b"synthetic model")
    manifest = tmp_path / "manifest.json"
    ev.write_json(
        manifest,
        {
            "audio": "audio.wav",
            "references": [{"id": "a", "name": "Alice", "start": 0, "end": 1}],
            "generation_files": {"asr": ["model.bin"], "diarization": ["model.bin"]},
        },
    )
    monkeypatch.setattr(
        "transcriptions.service.transcribe_audio",
        lambda *a, **k: TranscriptionResult("hi", "en", [TranscriptionSegment(0, 0, 1, "hi")]),
    )
    monkeypatch.setattr(
        PyannoteDiarizationBackend,
        "diarize_annotations",
        lambda *a, **k: {"regular": [SpeakerTurn(0, 1, "a")], "exclusive": []},
    )
    monkeypatch.setattr(
        "transcriptions.known_speaker.decode_mono_16k",
        lambda *a, **k: np.ones(16000, dtype=np.float32),
    )
    return manifest, tmp_path / "bundle"


@pytest.mark.parametrize("stage", ["asr", "reference", "manifest"])
def test_bundle_failure_never_publishes_and_retry_succeeds(capture, monkeypatch, stage):
    manifest, destination = capture
    with monkeypatch.context() as patch:

        def fail(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        if stage == "asr":
            patch.setattr("transcriptions.service.transcribe_audio", fail)
        elif stage == "reference":

            def decode(path, **kwargs):
                if "start" in kwargs:
                    fail()
                return np.ones(16000, dtype=np.float32)

            patch.setattr("transcriptions.known_speaker.decode_mono_16k", decode)
        else:
            patch.setattr(ev, "write_json", fail)
        with pytest.raises(RuntimeError):
            ev.create_bundle(manifest, destination)
    assert not destination.exists()
    ev.create_bundle(manifest, destination)
    assert (destination / "bundle.json").is_file()
    assert not list(destination.parent.glob(".bundle.stage-*"))


def test_existing_bundle_is_never_overwritten(capture):
    manifest, destination = capture
    destination.mkdir()
    (destination / "sentinel").write_bytes(b"old")
    with pytest.raises((ValueError, FileExistsError)):
        ev.create_bundle(manifest, destination)
    assert (destination / "sentinel").read_bytes() == b"old"


@pytest.mark.parametrize("change", ["manifest", "audio", "model", "settings"])
def test_frozen_identity_rejects_changed_sources_before_replay(
    capture, monkeypatch, settings, change
):
    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    if change == "manifest":
        value = ev.read_json(manifest)
        value["unreferenced_speakers"] = ["Guest"]
        ev.write_json(manifest, value)
    elif change == "settings":
        settings.VOXHELM_PYANNOTE_MODEL = "new-model"
    else:
        (manifest.parent / ("audio.wav" if change == "audio" else "model.bin")).write_bytes(b"new")
    with pytest.raises(ValueError, match="identity"):
        ev.verify_bundle_identity(manifest, destination)


def test_unchanged_identity_and_postprocess_variants_reuse(capture):
    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    ev.verify_bundle_identity(manifest, destination)
    # Postprocess configuration is intentionally absent from the frozen identity.
    assert ev.read_json(destination / "bundle.json")["version"] == 2


def test_long_job_audio_refuses_before_asr(capture, monkeypatch):
    from evals.speaker_inputs import MAX_AUDIO_SECONDS
    from transcriptions.diarization import PYANNOTE_MAX_FRAMES
    from transcriptions.known_speaker import KnownSpeakerError

    manifest, destination = capture
    assert MAX_AUDIO_SECONDS * 16000 == PYANNOTE_MAX_FRAMES
    allowances = []

    def decode(path, **kwargs):
        allowances.append(kwargs.get("max_frames"))
        raise KnownSpeakerError("over the decoded frame limit")

    monkeypatch.setattr("transcriptions.known_speaker.decode_mono_16k", decode)
    monkeypatch.setattr(
        "transcriptions.service.transcribe_audio", lambda *a, **k: pytest.fail("ASR started")
    )
    with pytest.raises(KnownSpeakerError):
        ev.create_bundle(manifest, destination)
    assert allowances == [PYANNOTE_MAX_FRAMES]
    assert not destination.exists()


def test_capture_code_identity_excludes_postprocess_and_covers_capture_imports(capture):
    import ast
    from pathlib import Path

    from evals.speaker_generations import CAPTURE_CODE

    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    identity = ev.read_json(destination / "bundle.json")["identity"]
    # Old/new postprocessing code must be able to replay the same frozen bundles.
    assert "known_speaker.py" not in identity["code"]
    assert set(identity["code"]) == set(CAPTURE_CODE)
    package = Path(ev.__file__).resolve().parent.parent / "transcriptions"
    pending, seen = ["service.py", "diarization.py"], set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(ast.parse((package / name).read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = "transcriptions" if node.level == 1 else node.module or ""
                if node.level == 1 and node.module:
                    base = "transcriptions." + node.module
                if base == "transcriptions":
                    # `from transcriptions import formats` imports submodules by name.
                    modules = [f"transcriptions.{alias.name}" for alias in node.names]
                else:
                    modules = [base]
            for module in modules:
                if module.startswith("transcriptions."):
                    pending.append(module.removeprefix("transcriptions.").split(".")[0] + ".py")
    assert seen <= set(CAPTURE_CODE)


@pytest.fixture
def suite_root(capture, monkeypatch):
    manifest, _ = capture
    root = manifest.parent
    (root / "manifests").mkdir()
    (root / "truth").mkdir()
    for episode in ("pp59", "pp77", "pp78", "pp80", "dc209", "dc210"):
        value = ev.read_json(manifest)
        value["audio"] = "../audio.wav"
        value["generation_files"] = {key: ["../model.bin"] for key in ("asr", "diarization")}
        path = root / "manifests" / f"{episode}.json"
        ev.write_json(path, value)
        ev.create_bundle(path, root / "bundles" / episode)
        if episode.startswith("pp"):
            ev.write_json(root / "truth" / f"{episode}.json", {"fps": 1, "frames": ["Alice"]})

    def replay(bundle, config, **kwargs):
        if kwargs.get("diagnostics_path"):
            ev.write_json(kwargs["diagnostics_path"], {})
        return {
            "summary": {},
            "segments": [{"start": 0, "end": 1, "speaker": "Alice", "speaker_uncertain": False}],
        }, 1

    monkeypatch.setattr(ev, "run_bundle", replay)
    return root, replay


def contents(path):
    return {p.name: p.read_bytes() for p in path.iterdir()}


@pytest.mark.parametrize("failure", ["episode-five", "score-write"])
def test_failed_suite_preserves_previous_complete_generation(suite_root, monkeypatch, failure):
    root, replay = suite_root
    ev.run_suite(root, {}, "baseline")
    output = root / "results" / "baseline"
    before = contents(output)
    if failure == "episode-five":

        def run(bundle, config, **kwargs):
            if bundle.name == "dc209":
                raise RuntimeError("late episode")
            return replay(bundle, config, **kwargs)

        monkeypatch.setattr(ev, "run_bundle", run)
    else:
        original = ev.write_json

        def write(path, value):
            if path.name == "dc209.score.json":
                raise RuntimeError("late write")
            original(path, value)

        monkeypatch.setattr(ev, "write_json", write)
    with pytest.raises(RuntimeError):
        ev.run_suite(root, {}, "baseline")
    assert contents(output) == before
    stages = list((root / "results").glob(".baseline.stage-*"))
    assert len(stages) == 1
    assert not (stages[0] / "complete.json").exists()


def test_same_tag_publishers_are_exclusive_and_completion_lists_exact_members(
    suite_root, monkeypatch
):
    root, replay = suite_root
    attempted = False

    def run(bundle, config, **kwargs):
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(ValueError, match="publisher"):
                ev.run_suite(root, {}, "baseline")
        return replay(bundle, config, **kwargs)

    monkeypatch.setattr(ev, "run_bundle", run)
    ev.run_suite(root, {}, "baseline")
    output = root / "results" / "baseline"
    complete = ev.read_json(output / "complete.json")
    assert set(complete["members"]) == set(contents(output)) - {"complete.json"}
    assert len(complete["episodes"]) == 6
    assert complete["config"]["embedding_model"]


@pytest.mark.parametrize(
    "tag", ["../manifests", "/tmp/escaped", "a/b", "a\\b", ".", "..", "a" * 65]
)
def test_invalid_suite_tag_refuses_before_any_work(tmp_path, monkeypatch, tag):
    monkeypatch.setattr(ev, "create_bundle", lambda *a: pytest.fail("backend"))
    with pytest.raises(ValueError, match="tag"):
        ev.run_suite(tmp_path, {}, tag)
    assert not (tmp_path / "results").exists()


def test_frozen_sample_replacement_refuses_before_embedding(capture, monkeypatch):
    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    np.save(destination / "audio.npy", np.zeros(16000, dtype=np.float32))
    monkeypatch.setattr(
        "transcriptions.known_speaker.get_known_speaker_backend",
        lambda *a: pytest.fail("changed frozen samples reached embedding"),
    )
    with pytest.raises(ValueError, match="bundle"):
        ev.run_bundle(destination, {})


@pytest.mark.parametrize("change", ["truth", "bundle"])
def test_variant_reuses_exact_baseline_inputs_before_replay(suite_root, monkeypatch, change):
    root, _ = suite_root
    ev.run_suite(root, {}, "baseline")
    if change == "truth":
        ev.write_json(root / "truth/pp59.json", {"fps": 1, "frames": ["Bob"]})
    else:
        path = root / "bundles/pp59/bundle.json"
        value = ev.read_json(path)
        value["result"]["text"] = "changed frozen ASR"
        ev.write_json(path, value)
    monkeypatch.setattr(ev, "run_bundle", lambda *a, **k: pytest.fail("stale baseline replayed"))
    with pytest.raises(ValueError, match="baseline"):
        ev.run_suite(root, {}, "variant")


def test_sealed_manifest_must_fit_control_limit_before_publication(capture, monkeypatch):
    import json

    from evals import speaker_inputs

    manifest, destination = capture
    ev.create_bundle(manifest, destination)
    unsealed = ev.read_json(destination / "bundle.json")
    del unsealed["sample_sha256"]
    size = len((json.dumps(unsealed, indent=2, ensure_ascii=False) + "\n").encode())
    monkeypatch.setattr(speaker_inputs, "JSON_MAX_BYTES", size)
    other = destination.with_name("other-bundle")
    with pytest.raises(ValueError, match="control"):
        ev.create_bundle(manifest, other)
    assert not other.exists()


@pytest.mark.parametrize("entry", ["symlink", "hardlink", "fifo", "directory", "foreign-owner"])
def test_publisher_refuses_unowned_or_aliased_lock(tmp_path, monkeypatch, entry):
    import os

    from evals.speaker_generations import publisher

    lock = tmp_path / ".baseline.lock"
    other = tmp_path / "other"
    other.write_bytes(b"unchanged")
    if entry == "symlink":
        lock.symlink_to(other)
    elif entry == "hardlink":
        os.link(other, lock)
    elif entry == "fifo":
        os.mkfifo(lock)
    elif entry == "directory":
        lock.mkdir()
    else:
        lock.touch()
        monkeypatch.setattr(os, "getuid", lambda: lock.stat().st_uid + 1)
    with pytest.raises(ValueError, match="publisher"):
        with publisher(tmp_path, "baseline"):
            pytest.fail("inadmissible lock acquired")
    assert other.read_bytes() == b"unchanged"


def test_publisher_rechecks_name_after_flock(tmp_path, monkeypatch):
    import fcntl

    from evals.speaker_generations import publisher

    flock = fcntl.flock

    def acquire_and_replace(fd, operation):
        flock(fd, operation)
        lock = tmp_path / ".baseline.lock"
        lock.unlink()
        lock.touch()

    monkeypatch.setattr(fcntl, "flock", acquire_and_replace)
    with pytest.raises(ValueError, match="publisher"):
        with publisher(tmp_path, "baseline"):
            pytest.fail("replaced lock admitted after flock")


def test_publisher_holds_and_checks_parent(tmp_path):
    from evals.speaker_generations import publisher

    parent = tmp_path / "results"
    with publisher(parent, "baseline") as verify:
        parent.rename(tmp_path / "old-results")
        parent.mkdir()
        (parent / ".baseline.lock").touch()
        with pytest.raises(ValueError, match="publisher"):
            verify()


def test_publisher_refuses_symlinked_parent_with_clear_message(tmp_path):
    from evals.speaker_generations import publisher

    (tmp_path / "real-results").mkdir()
    (tmp_path / "results").symlink_to("real-results", target_is_directory=True)
    with pytest.raises(ValueError, match="real directory, not a symlink"):
        with publisher(tmp_path / "results", "baseline"):
            raise AssertionError("must not admit a symlinked parent")
    assert not (tmp_path / "real-results" / ".baseline.lock").exists()


def test_bundle_refuses_replaced_lock_before_publication(capture, monkeypatch):
    manifest, destination = capture
    original = ev._capture_bundle

    def capture_and_replace(*args):
        original(*args)
        lock = destination.parent / f".{destination.name}.lock"
        lock.unlink()
        lock.touch()

    monkeypatch.setattr(ev, "_capture_bundle", capture_and_replace)
    with pytest.raises(ValueError, match="publisher"):
        ev.create_bundle(manifest, destination)
    assert not destination.exists()
    assert not list(destination.parent.glob(".bundle.stage-*"))


def test_suite_rechecks_lock_immediately_before_pointer_publication(suite_root, monkeypatch):
    from pathlib import Path

    root, _ = suite_root
    ev.run_suite(root, {}, "baseline")
    output = root / "results" / "baseline"
    previous = output.readlink()
    before = contents(output)
    symlink_to = Path.symlink_to

    def link_and_replace(self, *args, **kwargs):
        symlink_to(self, *args, **kwargs)
        if self.name.startswith(".baseline.pointer-"):
            lock = self.parent / ".baseline.lock"
            lock.unlink()
            lock.touch()

    monkeypatch.setattr(Path, "symlink_to", link_and_replace)
    with pytest.raises(ValueError, match="publisher"):
        ev.run_suite(root, {}, "baseline")
    assert output.readlink() == previous
    assert contents(output) == before
    assert not list(output.parent.glob(".baseline.pointer-*"))


def test_two_process_publishers_detect_replaced_lock_and_preserve_previous(suite_root):
    import selectors
    import subprocess
    import sys
    from pathlib import Path
    from textwrap import dedent

    from evals.speaker_generations import publisher

    root, _ = suite_root
    ev.run_suite(root, {}, "baseline")
    output = root / "results" / "baseline"
    previous = output.readlink()
    before = contents(output)
    script = dedent("""\
        import os
        import sys
        from pathlib import Path
        import django
        os.environ["DJANGO_SETTINGS_MODULE"] = "tests.settings"
        django.setup()
        from evals import speaker_naming_eval as ev
        waiting = True
        def run(bundle, config, **kwargs):
            global waiting
            if waiting:
                waiting = False
                print("ready", flush=True)
                assert input() == "continue"
            if kwargs.get("diagnostics_path"):
                ev.write_json(kwargs["diagnostics_path"], {})
            return {
                "summary": {},
                "segments": [{"start": 0, "end": 1,
                              "speaker": "Alice", "speaker_uncertain": False}],
            }, 1
        ev.run_bundle = run
        try:
            ev.run_suite(Path(sys.argv[1]), {}, "baseline")
        except ValueError as exc:
            print(str(exc), flush=True)
        else:
            print("published", flush=True)
    """)

    processes = []

    def receive(process):
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=15), "publisher did not reach barrier"
        return process.stdout.readline().strip()

    def release(process):
        process.stdin.write("continue\n")
        process.stdin.flush()

    def start():
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(root)],
            cwd=Path(ev.__file__).resolve().parent.parent,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        assert receive(process) == "ready"
        return process

    try:
        first = start()
        with pytest.raises(ValueError, match="publisher already active"):
            with publisher(output.parent, "baseline"):
                pytest.fail("same inode acquired twice")
        lock = output.parent / ".baseline.lock"
        lock.unlink()
        lock.touch()
        second = start()
        release(first)
        assert "publisher" in receive(first)
        assert output.readlink() == previous
        assert contents(output) == before
        release(second)
        assert receive(second) == "published"
        assert output.readlink() != previous
        ev.complete_generation(output.resolve())
        for process in processes:
            assert process.wait(timeout=15) == 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)


def test_suite_hashes_shared_unchanged_generations_once_per_run(suite_root, monkeypatch):
    import hashlib
    from collections import Counter
    from pathlib import Path
    from types import SimpleNamespace

    from evals import speaker_generations as generations

    root, _ = suite_root
    reads: Counter[bytes] = Counter()
    sha256 = hashlib.sha256

    class CountedDigest:
        def __init__(self):
            self.digest = sha256()

        def update(self, data):
            reads[data] += 1
            self.digest.update(data)

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(
        generations,
        "hashlib",
        SimpleNamespace(sha256=lambda *a: sha256(*a) if a else CountedDigest()),
    )
    code = Path(ev.__file__).resolve().parent.parent / "transcriptions"
    shared = [root / "model.bin", root / "audio.wav", *(code / n for n in generations.CAPTURE_CODE)]
    ev.run_suite(root, {}, "baseline")
    assert {path.name: reads[path.read_bytes()] for path in shared} == {
        path.name: 1 for path in shared
    }
    # A subsequent invocation starts with a fresh cache, even in the same process.
    ev.run_suite(root, {}, "baseline")
    assert all(reads[path.read_bytes()] == 2 for path in shared)


@pytest.mark.parametrize("filename", ["model.bin", "audio.wav"])
@pytest.mark.parametrize("change", ["rewrite", "replace"])
def test_suite_rechecks_cached_generations_after_replay(suite_root, monkeypatch, filename, change):
    root, replay = suite_root
    ev.run_suite(root, {}, "baseline")
    output = root / "results" / "baseline"
    previous = output.readlink()
    before = contents(output)
    changed = False

    def run(*args, **kwargs):
        nonlocal changed
        if not changed:
            changed = True
            path = root / filename
            data = b"X" * path.stat().st_size  # Same-size rewrites must invalidate too.
            if change == "rewrite":
                path.write_bytes(data)
            else:
                replacement = path.with_suffix(".new")
                replacement.write_bytes(data)
                replacement.replace(path)
        return replay(*args, **kwargs)

    monkeypatch.setattr(ev, "run_bundle", run)
    with pytest.raises(ValueError, match="identity"):
        ev.run_suite(root, {}, "baseline")
    assert output.readlink() == previous
    assert contents(output) == before


def test_suite_rechecks_cached_generation_before_replay(suite_root, monkeypatch):
    root, _ = suite_root
    verify = ev.verify_bundle_identity
    calls = 0

    def verify_and_change(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 7:  # All six preflight checks have already warmed the cache.
            (root / "model.bin").write_bytes(b"changed model")
        return verify(*args, **kwargs)

    monkeypatch.setattr(ev, "verify_bundle_identity", verify_and_change)
    monkeypatch.setattr(ev, "run_bundle", lambda *a, **k: pytest.fail("stale model replayed"))
    with pytest.raises(ValueError, match="identity"):
        ev.run_suite(root, {}, "baseline")
    assert not (root / "results" / "baseline").exists()


@pytest.mark.parametrize("change", ["rewrite", "replace", "symlink"])
def test_fingerprint_cache_refuses_change_during_reuse(tmp_path, monkeypatch, change):
    import hashlib
    from pathlib import Path

    from evals.speaker_generations import FingerprintCache, file_digest

    path = tmp_path / "model.bin"
    path.write_bytes(b"old model")
    fingerprints: FingerprintCache = {}
    assert file_digest(path, fingerprints) == hashlib.sha256(b"old model").hexdigest()
    original_stat = Path.stat
    cached_ctime = os.stat(path).st_ctime_ns

    def stat_and_change(self, *args, **kwargs):
        if self == path:
            if change == "rewrite":
                # Same size, same inode: keep rewriting until ctime visibly
                # advances, so the test does not depend on timestamp granularity.
                deadline = time.monotonic() + 2
                while True:
                    path.write_bytes(b"new model")
                    if os.stat(path).st_ctime_ns != cached_ctime:
                        break
                    if time.monotonic() > deadline:
                        raise AssertionError("ctime did not advance")
                    time.sleep(0.001)
            elif change == "replace":
                replacement = tmp_path / "replacement.bin"
                replacement.write_bytes(b"new model")
                replacement.replace(path)
            else:
                # Even a link back to the opened inode loses named authority.
                path.rename(tmp_path / "original.bin")
                path.symlink_to("original.bin")
        return original_stat(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", stat_and_change)
        with pytest.raises(ValueError, match="identity.changed"):
            file_digest(path, fingerprints)
    if change == "symlink":
        with pytest.raises(ValueError, match="identity file"):
            file_digest(path, fingerprints)
    else:
        # A changed generation is rehashed rather than permanently poisoned.
        assert file_digest(path, fingerprints) == hashlib.sha256(b"new model").hexdigest()
