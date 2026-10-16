"""The German drift check of ``ad-analysis-v1``: the frozen function-word rule, the turbo
retry's acceptance, the splice, per-window records, no silent skip, cancellation, the
identity, remote capability and the records' retention."""

from __future__ import annotations

import json
import re
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from django.utils import timezone

from jobs import remote_worker_cli as worker_cli
from jobs.models import Job, JobArtifact
from jobs.retention import prune_job_artifacts
from lane_scheduler import InferenceCancelled
from tests.test_ad_analysis_profile import (  # noqa: F401  (fixture)
    FIXTURE,
    complete_profile_job,
    delete_job,
    drift_identity,
    parakeet_cli,
    run_local_profile_job,
    submit_remote_profile_job,
    words_manifest_entry,
    worker_identity,
)
from tests.test_remote_workers import producer_headers
from transcriptions import drift, windows
from transcriptions.drift import (
    DRIFT_RULE,
    ENGLISH_FUNCTION_WORDS,
    FUNCTION_WORDS_SHA256,
    GERMAN_FUNCTION_WORDS,
    count_function_words,
    function_word_tokens,
    run_drift_check,
    splice_words,
)
from transcriptions.profiles import AD_ANALYSIS_V1, parakeet_payload_words
from transcriptions.service import BackendUnavailableError, ParakeetBackend
from transcriptions.windows import WindowTranscription

AUDIO = Path("/nonexistent/episode.mp3")


@pytest.fixture
def runtime(parakeet_cli):  # noqa: F811
    """The fake Parakeet CLI with the fake window runtime (``runtime.window``)."""
    return parakeet_cli


# --- the research tokenizer and rule, verbatim (lib.py, splice.py) ----------------------

_EN = set(
    """the and you that this with for have what are but they just like it is of to we our your be been there their which would could should
about from not do does did it's i'm don't can if or at on by my me he she them these those because when where how who then than really very
going know think yeah get got gonna you're we're they're that's there's let's isn't doesn't didn't can't won't i've i'll you've into out
some any more most other our us its his her has had were said say one""".split()  # noqa: E501
)
_DE = set(
    """der die das und ist nicht ich wir sie es ein eine einen einem einer dem den des mit auf für von zu auch aber dass wie wenn noch nur schon
dann oder sich mal ja sind haben wird werden kann bei aus nach jetzt hier mehr sehr immer gibt genau halt eben doch du ihr uns mir mich dir dich
wo weil über vor durch gerade eigentlich irgendwie hat habe bin bist war waren wurde würde können müssen muss soll diese dieser dieses
kein keine nichts etwas viel viele ganz gut also man um""".split()  # noqa: E501
)
_EN -= _DE


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9äöüß' ]", " ", s.lower().replace("’", "'"))


def lib_toks(text: str) -> list[str]:
    return [w.strip("'") if w not in _EN else w for w in _norm(text).split()]


def lib_counts(words: list[str]) -> tuple[int, int]:
    return sum(w in _EN for w in words), sum(w in _DE for w in words)


def splice_py(pw: list[dict[str, Any]], ret: dict[int, list[dict[str, Any]]]) -> list[Any]:
    """``splice.py``'s word list; ``ret`` is keyed by window start t in seconds."""
    ts = sorted(ret)
    new = [w for w in pw if int(w["s"] // 30.0) * 30 not in ts] + [w for t in ts for w in ret[t]]
    new.sort(key=lambda w: w["s"])
    return new


# --- helpers ------------------------------------------------------------------------------


def word(text: str, start: float, end: float | None = None, c: float = 0.9) -> dict[str, Any]:
    return {"w": text, "s": start, "e": start + 0.2 if end is None else end, "c": c}


def words_at(start: float, text: str, step: float = 0.5) -> list[dict[str, Any]]:
    return [word(token, start + index * step) for index, token in enumerate(text.split())]


DRIFTED = "the thing is that you know we have a lot of"  # EN: the is that you know we have of
GERMAN = "und dann ist das ja auch nicht so schlimm"


class FakeRetry:
    """A fake turbo retry: answers per window start, records its calls."""

    def __init__(self, answers: dict[float, tuple[str, float]] | None = None) -> None:
        self.answers = answers or {}
        self.calls: list[float] = []
        self.events: list[threading.Event | None] = []
        self.on_call: Any = None

    def __call__(
        self, audio_path: Path, t: float, *, cancel_event: threading.Event | None = None
    ) -> WindowTranscription:
        self.calls.append(t)
        self.events.append(cancel_event)
        if self.on_call is not None:
            self.on_call(t)
        language, probability = self.answers.get(t, ("de", 0.999))
        return WindowTranscription(
            t=t,
            clip_start=max(0.0, t - 1),
            language=language,
            language_probability=probability,
            words=[
                {"w": "Retry", "s": round(t + 0.25, 3), "e": round(t + 0.5, 3)},
                {"w": "Wort.", "s": round(t + 29.5, 3), "e": round(t + 29.9, 3)},
            ],
            identity={},
        )


# --- the rule -------------------------------------------------------------------------------


def test_function_word_lists_and_tokenizer_are_the_research_ones():
    assert ENGLISH_FUNCTION_WORDS == _EN
    assert GERMAN_FUNCTION_WORDS == _DE
    assert "these" in ENGLISH_FUNCTION_WORDS  # the frozen rule keeps it
    assert "die" not in ENGLISH_FUNCTION_WORDS and "also" not in ENGLISH_FUNCTION_WORDS
    # Pins the lists: a change is a new profile.
    assert FUNCTION_WORDS_SHA256 == (
        "5dbea279a0c276d25d5f7606093a73448cd7682f211807e1e95a896e1bce2432"
    )
    for text in [
        "Hi, it's THE thing — don’t 'they' Über-Straße these",
        "You're we're 'the' l'amour so, also: die Leute",
        "Ä Ö Ü ß é 42% the/and",
    ]:
        assert function_word_tokens(text) == lib_toks(text)
        assert count_function_words(text.split()) == lib_counts(lib_toks(text))


@pytest.mark.parametrize(
    ("english", "german", "flagged"),
    [
        (4, 36, True),  # share exactly 0.10
        (4, 37, False),  # share just below 0.10
        (5, 45, True),
        (5, 46, False),
        (3, 0, False),  # fewer than 4 English function words
        (4, 0, True),
        (0, 0, False),
    ],
)
def test_flag_rule_at_the_threshold_edges(english: int, german: int, flagged: bool):
    assert DRIFT_RULE.flags(english, german) is flagged
    # The research expression with the frozen "4,0.1".
    assert (english >= 4.0 and english / max(1, english + german) >= 0.1) is flagged


@pytest.mark.parametrize(("german", "flagged"), [(36, True), (37, False)])
def test_flag_rule_on_window_text_at_the_share_edge(german: int, flagged: bool):
    text = " ".join(["the", "and", "you", "that"] + ["und"] * german)
    retry = FakeRetry()

    check = run_drift_check(AUDIO, words_at(0.0, text, step=0.1), language="de", retry=retry)

    assert (check.record["windows"][0]["en"], check.record["windows"][0]["de"]) == (4, german)
    assert check.record["windows"][0]["flagged"] is flagged
    assert retry.calls == ([0.0] if flagged else [])


# --- splice and acceptance --------------------------------------------------------------------


def test_splice_replaces_exactly_the_words_starting_in_the_window():
    words = [
        word("vorher", 29.999),
        word("the", 30.0),
        *words_at(31.0, DRIFTED),
        word("end", 59.999),
        word("danach", 60.0),
    ]
    retry = FakeRetry()

    check = run_drift_check(AUDIO, words, language="de", retry=retry)

    assert retry.calls == [30.0]
    assert [(w["w"], w["s"]) for w in check.words] == [
        ("vorher", 29.999),
        ("Retry", 30.25),
        ("Wort.", 59.5),
        ("danach", 60.0),
    ]
    # Retry words carry no Parakeet confidence; Parakeet words keep theirs.
    assert [w["c"] for w in check.words] == [0.9, None, None, 0.9]
    retry_words = [dict(w) for w in check.words[1:3]]
    assert check.words == splice_py(words, {30: retry_words})
    assert splice_words(words, {1: retry_words}, 30.0) == check.words


def test_splice_keeps_order_across_several_windows_like_splice_py():
    words = words_at(0.0, GERMAN) + words_at(30.0, DRIFTED) + words_at(60.0, DRIFTED)
    words += words_at(90.0, GERMAN)
    retry = FakeRetry()

    check = run_drift_check(AUDIO, words, language="de", retry=retry)

    assert retry.calls == [30.0, 60.0]
    replacements = {t: [dict(w) for w in check.words if t <= w["s"] < t + 30] for t in (30, 60)}
    assert check.words == splice_py(words, replacements)
    assert [w["w"] for w in check.words if 30 <= w["s"] < 90] == ["Retry", "Wort."] * 2


@pytest.mark.parametrize(
    ("answer", "accepted"),
    [
        (("de", 0.9), True),  # p exactly at the threshold
        (("de", 0.899999), False),
        (("en", 0.999), False),  # genuinely English: Parakeet stays
        (("nl", 0.95), False),
    ],
)
def test_retry_is_accepted_only_in_the_episode_language_with_p_at_least_0_9(
    answer: tuple[str, float], accepted: bool
):
    words = words_at(0.0, GERMAN) + words_at(30.0, DRIFTED)
    retry = FakeRetry({30.0: answer})

    check = run_drift_check(AUDIO, words, language="de", retry=retry)

    record = check.record["windows"][1]
    assert record == {
        "t": 30.0,
        "en": 8,
        "de": 0,
        "flagged": True,
        "words_before": 11,
        "words_after": 2 if accepted else 11,
        "language": answer[0],
        "p": answer[1],
        "accepted": accepted,
        "retry_words": 2,
        "retry_seconds": record["retry_seconds"],
    }
    assert record["retry_seconds"] >= 0
    if accepted:
        assert [w["w"] for w in check.words[-2:]] == ["Retry", "Wort."]
    else:
        assert check.words == words
    assert check.record["summary"] == {
        "windows": 2,
        "flagged": 1,
        "accepted": int(accepted),
        "rejected": int(not accepted),
        "retry_seconds": check.record["summary"]["retry_seconds"],
    }


def test_every_window_gets_a_record_and_unflagged_ones_have_no_retry_fields():
    words = words_at(0.0, GERMAN) + words_at(65.0, DRIFTED)  # no words in [30, 60)
    check = run_drift_check(AUDIO, words, language="de", retry=FakeRetry())

    assert check.record["rule"] == "fw-en4-share10-v1"
    assert check.record["checked"] is True
    assert check.record["language"] == "de"
    assert [r["t"] for r in check.record["windows"]] == [0.0, 60.0]
    assert check.record["windows"][0] == {
        "t": 0.0,
        "en": 0,
        "de": 7,
        "flagged": False,
        "words_before": 9,
        "words_after": 9,
    }


@pytest.mark.parametrize("language", ["en", None, "fr"])
def test_non_german_jobs_are_untouched(language: str | None):
    words = words_at(0.0, DRIFTED) + words_at(30.0, DRIFTED)
    retry = FakeRetry()

    check = run_drift_check(AUDIO, words, language=language, retry=retry)

    assert retry.calls == []
    assert check.words == words
    assert check.record["checked"] is False
    assert check.record["language"] == language
    assert (
        "is not checked" in check.record["reason"] or "no language_hint" in (check.record["reason"])
    )
    assert check.record["windows"] == []
    assert check.record["summary"]["flagged"] == 0


def test_a_failing_retry_fails_the_check_instead_of_skipping_the_window():
    retry = FakeRetry()

    def fail(t: float) -> None:
        raise RuntimeError("whisper.cpp window transcription failed: exit status 3")

    retry.on_call = fail
    with pytest.raises(RuntimeError, match="window transcription failed"):
        run_drift_check(AUDIO, words_at(0.0, DRIFTED), language="de", retry=retry)


def test_cancellation_before_and_between_windows():
    words = words_at(0.0, DRIFTED) + words_at(30.0, DRIFTED)
    cancel = threading.Event()
    cancel.set()
    retry = FakeRetry()
    with pytest.raises(InferenceCancelled):
        run_drift_check(AUDIO, words, language="de", retry=retry, cancel_event=cancel)
    assert retry.calls == []

    cancel = threading.Event()
    retry = FakeRetry()
    retry.on_call = lambda t: cancel.set()  # cancelled while the first window runs
    with pytest.raises(InferenceCancelled):
        run_drift_check(AUDIO, words, language="de", retry=retry, cancel_event=cancel)
    assert retry.calls == [0.0]
    assert retry.events == [cancel]  # the retry itself gets the event


# --- through the profile (fake Parakeet CLI, real transcribe_window, fake whisper-cli) ---------


def fixture_words() -> list[dict[str, Any]]:
    return parakeet_payload_words(json.loads(FIXTURE.read_text(encoding="utf-8")))


def expected_flags(words: list[dict[str, Any]]) -> list[float]:
    grid: dict[int, list[str]] = {}
    for w in words:
        grid.setdefault(int(w["s"] // 30.0), []).append(w["w"])
    return [
        index * 30.0
        for index, ws in sorted(grid.items())
        if DRIFT_RULE.flags(*lib_counts(lib_toks(" ".join(ws))))
    ]


def words_document(client, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.get(payload["result"]["artifacts"]["words"], **producer_headers())
    assert response.status_code == 200
    return json.loads(response.content)


@pytest.mark.django_db
def test_local_german_job_splices_accepted_retries_and_returns_the_records(
    client, settings, runtime
):
    window = runtime.window
    window.set_language("default", "de", 0.999549)
    flagged = expected_flags(fixture_words())
    assert len(flagged) >= 2
    # One flagged window answers English: it keeps Parakeet's words.
    rejected_t = flagged[1]
    window.set_language(f"{rejected_t - 1:.3f}", "en", 0.97)

    payload = run_local_profile_job(client, settings)

    assert payload["state"] == "succeeded"
    assert window.clip_starts() == [f"{max(0.0, t - 1):.3f}" for t in flagged]
    document = words_document(client, payload)
    records = document["drift"]["windows"]
    assert [r["t"] for r in records if r["flagged"]] == flagged
    assert {r["t"]: r["accepted"] for r in records if r["flagged"]} == {
        t: t != rejected_t for t in flagged
    }
    by_window: dict[float, list[str]] = {}
    for w in document["words"]:
        by_window.setdefault(int(w["s"] // 30.0) * 30.0, []).append(w["w"])
    parakeet_by_window: dict[float, list[str]] = {}
    for w in fixture_words():
        parakeet_by_window.setdefault(int(w["s"] // 30.0) * 30.0, []).append(w["w"])
    for t in flagged:
        expected = parakeet_by_window[t] if t == rejected_t else ["Das", "ist", "gut."]
        assert by_window[t] == expected
    for record in records:
        assert record["words_after"] == len(by_window.get(record["t"], []))
    assert document["drift"]["summary"]["accepted"] == len(flagged) - 1
    assert document["drift"]["summary"]["rejected"] == 1
    # The transcript artifacts stay Parakeet's sentences.
    transcript = client.get(payload["result"]["artifacts"]["json"], **producer_headers())
    assert b"Django Chat" in transcript.content
    assert list(window.temp_root.iterdir()) == []


@pytest.mark.django_db
def test_profile_identity_records_the_drift_rule_and_the_pinned_retry_model(
    client, settings, runtime
):
    payload = run_local_profile_job(client, settings)

    identity = payload["result"]["metadata"]["profile"]
    assert identity["drift"] == {
        "rule": "fw-en4-share10-v1",
        "languages": ["de"],
        "window_seconds": 30.0,
        "min_english": 4,
        "min_english_share": 0.1,
        "function_words_sha256": FUNCTION_WORDS_SHA256,
        "accept_language": "episode",
        "accept_probability": 0.9,
        "retry": {
            "engine": "whisper.cpp",
            "cli": "whisper-cli",
            "cli_version": "1.9.5",
            "model": "ggml-large-v3-turbo",
            "model_sha256": windows.WINDOW_MODEL_SHA256,
            "settings": windows.WINDOW_SETTINGS.record(),
        },
    }
    assert words_document(client, payload)["identity"] == identity


@pytest.mark.django_db
def test_local_english_job_runs_no_retry_but_records_why(client, settings, runtime):
    payload = run_local_profile_job(client, settings, language_hint="en")

    assert payload["state"] == "succeeded"
    assert runtime.window.clip_starts() == []
    document = words_document(client, payload)
    assert document["words"] == fixture_words()
    assert document["drift"]["checked"] is False
    assert "is not checked" in document["drift"]["reason"]
    assert document["drift"]["windows"] == []
    # The identity names the retry model all the same: it is part of the profile.
    assert document["identity"]["drift"]["retry"]["model_sha256"] == windows.WINDOW_MODEL_SHA256


@pytest.mark.django_db
@pytest.mark.parametrize("language_hint", ["de", "en"])
@pytest.mark.parametrize("breakage", ["missing", "other-hash", "no-binary"])
def test_missing_window_runtime_fails_the_job_before_parakeet_runs(
    client, settings, runtime, language_hint: str, breakage: str
):
    window = runtime.window
    if breakage == "missing":
        window.model.unlink()
    elif breakage == "other-hash":
        window.model.write_bytes(b"another turbo")
    else:
        settings.VOXHELM_WHISPERCPP_BIN = str(window.root / "no-such-whisper-cli")

    payload = run_local_profile_job(client, settings, language_hint=language_hint)

    assert payload["state"] == "failed"
    with pytest.raises(FileNotFoundError):
        runtime.args()  # Parakeet never ran
    assert window.clip_starts() == []
    assert not JobArtifact.objects.filter(kind=JobArtifact.Kind.WORDS).exists()
    assert runtime.other_engine_calls == []


@pytest.mark.django_db
def test_window_runtime_lost_after_parakeet_fails_the_job(
    client, settings, runtime, monkeypatch: pytest.MonkeyPatch
):
    runtime.window.set_language("default", "de", 0.999549)
    original_run = ParakeetBackend.run

    def run_then_lose_the_model(self, audio_path, params):
        payload = original_run(self, audio_path, params)
        runtime.window.model.unlink()
        return payload

    monkeypatch.setattr(ParakeetBackend, "run", run_then_lose_the_model)

    payload = run_local_profile_job(client, settings)

    assert payload["state"] == "failed"
    assert "missing" in payload["error"]["message"]
    assert not JobArtifact.objects.filter(kind=JobArtifact.Kind.WORDS).exists()


def test_profile_run_honours_cancellation_in_the_drift_check(
    runtime, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from transcriptions.profiles import PROFILES, run_profile_transcription

    runtime.window.set_language("default", "de", 0.999549)
    cancel = threading.Event()
    real_retry = drift.transcribe_window

    def cancel_during_first_window(audio_path, t, *, cancel_event=None):
        result = real_retry(audio_path, t, cancel_event=cancel_event)
        cancel.set()  # the caller cancels while the first window runs
        return result

    monkeypatch.setattr(drift, "transcribe_window", cancel_during_first_window)
    audio = tmp_path / "episode.wav"
    audio.write_bytes(b"RIFF")

    with pytest.raises(InferenceCancelled):
        run_profile_transcription(
            PROFILES[AD_ANALYSIS_V1], audio, language_hint="de", cancel_event=cancel
        )
    assert len(runtime.window.clip_starts()) == 1
    assert list(runtime.window.temp_root.iterdir()) == []

    # Cancelled before it starts: neither Parakeet nor a retry runs.
    with pytest.raises(InferenceCancelled):
        run_profile_transcription(
            PROFILES[AD_ANALYSIS_V1], audio, language_hint="de", cancel_event=cancel
        )
    assert len(runtime.window.clip_starts()) == 1


# --- records come back with the words and go with them -------------------------------------


@pytest.mark.django_db
def test_producer_delete_removes_the_words_with_their_records(client, settings, runtime):
    runtime.window.set_language("default", "de", 0.999549)
    payload = run_local_profile_job(client, settings)
    assert words_document(client, payload)["drift"]["summary"]["flagged"] > 0
    job = Job.objects.get(id=payload["id"])

    assert delete_job(client, job).status_code == 204

    assert not Job.objects.filter(id=job.id).exists()
    assert not JobArtifact.objects.exists()
    root = Path(settings.VOXHELM_ARTIFACT_ROOT)
    assert [path for path in root.rglob("*") if path.is_file()] == []


@pytest.mark.django_db
def test_backstop_removes_the_words_with_their_records(client, settings, runtime):
    runtime.window.set_language("default", "de", 0.999549)
    payload = run_local_profile_job(client, settings)
    assert words_document(client, payload)["drift"]["windows"]
    Job.objects.filter(id=payload["id"]).update(
        finished_at=timezone.now() - timedelta(days=7, minutes=1)
    )

    prune_job_artifacts()

    assert not Job.objects.filter(id=payload["id"]).exists()
    assert not JobArtifact.objects.exists()
    root = Path(settings.VOXHELM_ARTIFACT_ROOT)
    assert [path for path in root.rglob("*") if path.is_file()] == []


# --- remote workers -------------------------------------------------------------------------


def test_worker_advertises_the_profile_only_while_the_window_runtime_resolves(runtime, settings):
    settings.VOXHELM_DIARIZATION_BACKEND = "none"
    settings.VOXHELM_WHISPERKIT_ENABLED = False
    capabilities = worker_cli.build_capabilities()
    assert capabilities["profiles"] == [AD_ANALYSIS_V1]

    runtime.window.model.rename(runtime.window.model.with_suffix(".moved"))
    current = worker_cli.with_current_stt_runtimes(capabilities)
    assert current["profiles"] == []
    assert "parakeet" in current["backends"]  # Parakeet alone is not the profile

    runtime.window.model.with_suffix(".moved").rename(runtime.window.model)
    assert worker_cli.with_current_stt_runtimes(capabilities)["profiles"] == [AD_ANALYSIS_V1]


def test_worker_profile_claim_without_the_window_runtime_is_retryable(runtime):
    runtime.window.model.unlink()
    claim = {"id": "job-1", "profile": AD_ANALYSIS_V1, "language": "de"}

    with pytest.raises(BackendUnavailableError):
        worker_cli.transcribe_claim_profile(claim=claim, audio_path=Path("/tmp/a.mp3"))
    with pytest.raises(FileNotFoundError):
        runtime.args()
    assert worker_cli.retryable_worker_error(BackendUnavailableError("gone"))


@pytest.mark.django_db
def test_remote_completion_keeps_the_drift_identity(client, settings):
    job = submit_remote_profile_job(client, settings)

    response = complete_profile_job(
        client,
        settings,
        job,
        artifacts=lambda job_id, attempt: [words_manifest_entry(job_id, attempt)],
        metadata=worker_identity(drift=drift_identity(cli_version="1.9.6")),
    )

    assert response.status_code == 200
    identity = response.json()["result"]["metadata"]["profile"]
    assert identity["drift"]["retry"]["cli_version"] == "1.9.6"
    assert identity["drift"]["retry"]["model_sha256"] == windows.WINDOW_MODEL_SHA256
    assert identity["drift"]["rule"] == "fw-en4-share10-v1"


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("drift_value", "message"),
    [
        (None, "drift does not match"),
        ({**drift_identity(), "min_english": 6}, "drift does not match"),
        ({**drift_identity(), "function_words_sha256": "0" * 64}, "drift does not match"),
        (drift_identity(model_sha256="0" * 64), "pinned window model"),
        (drift_identity(model="ggml-large-v3"), "pinned window model"),
        (
            drift_identity(settings={**windows.WINDOW_SETTINGS.record(), "best_of": 5}),
            "pinned window model",
        ),
    ],
)
def test_remote_completion_with_another_drift_check_is_refused(
    client, settings, drift_value: Any, message: str
):
    job = submit_remote_profile_job(client, settings)
    identity = worker_identity()
    if drift_value is None:
        identity.pop("drift")
    else:
        identity["drift"] = drift_value

    response = complete_profile_job(
        client,
        settings,
        job,
        artifacts=lambda job_id, attempt: [words_manifest_entry(job_id, attempt)],
        metadata=identity,
    )

    assert response.status_code == 400
    assert message in response.json()["error"]["message"]
