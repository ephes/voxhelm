from __future__ import annotations

import importlib
import subprocess
import tempfile
import wave
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from django.conf import settings

from lane_scheduler import LANE_NON_INTERACTIVE, admit_local_inference
from transcriptions.service import TranscriptionResult


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


class DiarizationError(RuntimeError):
    """Base error for requested diarization failures."""


class DiarizationBackendUnavailableError(DiarizationError):
    """Raised when the configured diarization backend cannot be used."""


class DiarizationConfigurationError(DiarizationError):
    """Raised when diarization settings are incomplete or invalid."""


class DiarizationBackendProtocol(Protocol):
    def diarize(self, audio_path: Path) -> list[SpeakerTurn]: ...


class UnavailableDiarizationBackend:
    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        del audio_path
        raise DiarizationBackendUnavailableError(
            "Diarization was requested, but no diarization backend is configured. "
            "Set VOXHELM_DIARIZATION_BACKEND=pyannote and configure the backend first."
        )


def load_audio_for_pyannote(audio_path: Path) -> dict[str, Any]:
    """Decode audio without relying on pyannote's torchcodec path.

    pyannote.audio 4 imports torchcodec, but the current Python 3.14 stack can
    resolve a torch/torchcodec combination where built-in decoding warns or
    fails. Supplying an in-memory waveform keeps model execution independent of
    that optional decoder path.
    """
    numpy = importlib.import_module("numpy")
    torch = importlib.import_module("torch")

    with tempfile.TemporaryDirectory(prefix="voxhelm-pyannote-") as temp_dir:
        wav_path = Path(temp_dir) / "audio.wav"
        try:
            subprocess.run(
                [
                    settings.VOXHELM_FFMPEG_BIN,
                    "-nostdin",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    str(wav_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DiarizationError("Could not decode audio for pyannote diarization.") from exc

        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

    if channels != 1 or sample_width != 2:
        raise DiarizationError("Decoded diarization audio was not 16-bit mono PCM WAV.")

    samples = numpy.frombuffer(frames, dtype=numpy.dtype("<i2")).astype("float32") / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


class PyannoteDiarizationBackend:
    def __init__(self, *, model_name: str, auth_token: str) -> None:
        self.model_name = model_name
        self.auth_token = auth_token
        self._pipeline: Any | None = None

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        pipeline = self._load_pipeline()
        diarization_result = pipeline(load_audio_for_pyannote(audio_path))
        annotation = extract_pyannote_annotation(diarization_result)
        turns: list[SpeakerTurn] = []
        try:
            iterator = annotation.itertracks(yield_label=True)
        except AttributeError as exc:
            raise DiarizationError(
                "pyannote.audio returned an unexpected diarization result."
            ) from exc
        for segment, _track, label in iterator:
            turns.append(
                SpeakerTurn(
                    start=float(segment.start),
                    end=float(segment.end),
                    speaker=str(label),
                )
            )
        return turns

    def _load_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        if not self.auth_token:
            raise DiarizationConfigurationError(
                "VOXHELM_HUGGINGFACE_TOKEN must be set for "
                "VOXHELM_DIARIZATION_BACKEND=pyannote."
            )

        try:
            pyannote_audio = importlib.import_module("pyannote.audio")
        except ModuleNotFoundError as exc:
            missing = exc.name or "pyannote.audio"
            raise DiarizationBackendUnavailableError(
                f"{missing} is not installed. Install a Python 3.14-compatible "
                "pyannote.audio stack before enabling VOXHELM_DIARIZATION_BACKEND=pyannote."
            ) from exc

        pipeline_cls = getattr(pyannote_audio, "Pipeline", None)
        if pipeline_cls is None:
            raise DiarizationBackendUnavailableError(
                "pyannote.audio does not expose Pipeline; check the installed version."
            )

        try:
            self._pipeline = pipeline_cls.from_pretrained(
                self.model_name,
                token=self.auth_token,
            )
        except TypeError:
            self._pipeline = pipeline_cls.from_pretrained(
                self.model_name,
                use_auth_token=self.auth_token,
            )
        if self._pipeline is None:
            raise DiarizationBackendUnavailableError(
                f"pyannote.audio could not load diarization model '{self.model_name}'."
            )
        return self._pipeline


_DIARIZATION_LOCK = Lock()


def extract_pyannote_annotation(diarization_result: Any) -> Any:
    if hasattr(diarization_result, "itertracks"):
        return diarization_result
    annotation = getattr(diarization_result, "speaker_diarization", None)
    if annotation is not None:
        return annotation
    annotation = getattr(diarization_result, "exclusive_speaker_diarization", None)
    if annotation is not None:
        return annotation
    raise DiarizationError("pyannote.audio returned an unexpected diarization result.")


def diarize_audio(audio_path: Path) -> list[SpeakerTurn]:
    with admit_local_inference(LANE_NON_INTERACTIVE):
        with _DIARIZATION_LOCK:
            return get_diarization_backend_service().diarize(audio_path)


def get_diarization_backend_service() -> DiarizationBackendProtocol:
    return build_diarization_backend_service(
        backend_name=settings.VOXHELM_DIARIZATION_BACKEND,
    )


def build_diarization_backend_service(*, backend_name: str) -> DiarizationBackendProtocol:
    normalized = backend_name.strip().lower()
    if normalized in {"", "none"}:
        return UnavailableDiarizationBackend()
    if normalized == "pyannote":
        return get_pyannote_diarization_backend(
            model_name=settings.VOXHELM_PYANNOTE_MODEL,
            auth_token=settings.VOXHELM_HUGGINGFACE_TOKEN,
        )
    raise DiarizationConfigurationError(
        f"Unsupported diarization backend '{backend_name}'. Supported values: none, pyannote."
    )


@lru_cache(maxsize=4)
def get_pyannote_diarization_backend(
    *,
    model_name: str,
    auth_token: str,
) -> PyannoteDiarizationBackend:
    return PyannoteDiarizationBackend(model_name=model_name, auth_token=auth_token)


def apply_speaker_labels(
    result: TranscriptionResult,
    turns: list[SpeakerTurn],
) -> TranscriptionResult:
    normalized_turns = normalize_speaker_turns(turns)
    if not normalized_turns:
        raise DiarizationError("Diarization backend returned no usable speaker turns.")

    labeled_segments = []
    assigned_count = 0
    for segment in result.segments:
        speaker = choose_speaker_for_segment(
            segment_start=segment.start,
            segment_end=segment.end,
            turns=normalized_turns,
        )
        if speaker is not None:
            assigned_count += 1
        labeled_segments.append(replace(segment, speaker=speaker or segment.speaker))
    if assigned_count == 0:
        raise DiarizationError("Diarization speaker turns did not overlap transcript segments.")
    return replace(result, segments=labeled_segments)


def normalize_speaker_turns(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
    labels_by_backend_label: dict[str, str] = {}
    normalized_turns: list[SpeakerTurn] = []
    for turn in sorted(turns, key=lambda item: (item.start, item.end, item.speaker)):
        backend_label = turn.speaker.strip()
        if not backend_label:
            continue
        start = max(float(turn.start), 0.0)
        end = max(float(turn.end), 0.0)
        if end <= start:
            continue
        if backend_label not in labels_by_backend_label:
            labels_by_backend_label[backend_label] = (
                f"Speaker {len(labels_by_backend_label) + 1}"
            )
        normalized_turns.append(
            SpeakerTurn(
                start=start,
                end=end,
                speaker=labels_by_backend_label[backend_label],
            )
        )
    return normalized_turns


def choose_speaker_for_segment(
    *,
    segment_start: float,
    segment_end: float,
    turns: list[SpeakerTurn],
) -> str | None:
    start = min(segment_start, segment_end)
    end = max(segment_start, segment_end)
    if end <= start:
        return None

    candidates: list[tuple[float, float, float, str, int]] = []
    for index, turn in enumerate(turns):
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap <= 0:
            continue
        candidates.append((overlap, turn.start, turn.end, turn.speaker, index))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    return candidates[0][3]
