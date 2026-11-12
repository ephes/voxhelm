from __future__ import annotations

import importlib
import math
import subprocess
import tempfile
import wave
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from django.conf import settings

from lane_scheduler import LANE_NON_INTERACTIVE, admit_local_inference
from transcriptions.audio_decode import AudioDecodeError, run_audio_decode
from transcriptions.service import TranscriptionResult

# Two hours of 16 kHz PCM; bound the compressed-to-decoded expansion.
PYANNOTE_MAX_FRAMES = 2 * 60 * 60 * 16000


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class DiarizationParams:
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None

    def pyannote_kwargs(self) -> dict[str, int]:
        kwargs = {}
        if self.num_speakers is not None:
            kwargs["num_speakers"] = self.num_speakers
        if self.min_speakers is not None:
            kwargs["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            kwargs["max_speakers"] = self.max_speakers
        return kwargs


class DiarizationError(RuntimeError):
    """Base error for requested diarization failures."""


class DiarizationBackendUnavailableError(DiarizationError):
    """Raised when the configured diarization backend cannot be used."""


class DiarizationConfigurationError(DiarizationError):
    """Raised when diarization settings are incomplete or invalid."""


class DiarizationBackendProtocol(Protocol):
    def diarize(
        self,
        audio_path: Path,
        params: DiarizationParams | None = None,
    ) -> list[SpeakerTurn]: ...


class UnavailableDiarizationBackend:
    def diarize(
        self,
        audio_path: Path,
        params: DiarizationParams | None = None,
    ) -> list[SpeakerTurn]:
        del audio_path, params
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
            run_audio_decode(
                [
                    settings.VOXHELM_FFMPEG_BIN,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostats",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-t",
                    str((PYANNOTE_MAX_FRAMES + 1) / 16000),
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    str(wav_path),
                ],
                timeout=settings.VOXHELM_AUDIO_DECODE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise DiarizationError(
                "Audio decode for pyannote diarization exceeded the configured decode deadline."
            ) from exc
        except (OSError, subprocess.CalledProcessError, AudioDecodeError) as exc:
            raise DiarizationError("Could not decode audio for pyannote diarization.") from exc

        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                if (wav_file.getnchannels(), wav_file.getsampwidth(), sample_rate) != (1, 2, 16000):
                    raise DiarizationError(
                        "Decoded diarization audio was not 16 kHz 16-bit mono PCM WAV."
                    )
                frame_count = wav_file.getnframes()
                if frame_count > PYANNOTE_MAX_FRAMES:
                    raise DiarizationError("Diarization audio exceeded the decoded frame limit.")
                frames = wav_file.readframes(frame_count)
                if len(frames) != frame_count * 2:
                    raise DiarizationError("Decoded diarization audio contained truncated PCM.")
        except (OSError, EOFError, wave.Error) as exc:
            raise DiarizationError("Could not read decoded diarization audio.") from exc

    samples = numpy.frombuffer(frames, dtype=numpy.dtype("<i2")).astype("float32") / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sample_rate}


class PyannoteDiarizationBackend:
    def __init__(self, *, model_name: str, auth_token: str, device_name: str = "auto") -> None:
        self.model_name = model_name
        self.auth_token = auth_token
        self.device_name = device_name
        self._pipeline: Any | None = None

    def diarize(
        self,
        audio_path: Path,
        params: DiarizationParams | None = None,
    ) -> list[SpeakerTurn]:
        pipeline = self._load_pipeline()
        diarization_params = params or DiarizationParams()
        diarization_result = pipeline(
            load_audio_for_pyannote(audio_path),
            **diarization_params.pyannote_kwargs(),
        )
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
                "VOXHELM_HUGGINGFACE_TOKEN must be set for VOXHELM_DIARIZATION_BACKEND=pyannote."
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
            candidate = pipeline_cls.from_pretrained(
                self.model_name,
                token=self.auth_token,
            )
        except TypeError:
            candidate = pipeline_cls.from_pretrained(
                self.model_name,
                use_auth_token=self.auth_token,
            )
        if candidate is None:
            raise DiarizationBackendUnavailableError(
                f"pyannote.audio could not load diarization model '{self.model_name}'."
            )
        candidate.to(resolve_pyannote_device(self.device_name))
        self._pipeline = candidate
        return candidate


_DIARIZATION_LOCK = Lock()


_AUTO_DEVICE_NAMES = frozenset({"", "auto"})
_EXPLICIT_DEVICE_NAMES = frozenset({"cpu", "mps", "cuda"})


def _torch_device_available(torch: Any, device_name: str) -> bool:
    if device_name == "cuda":
        return bool(torch.cuda.is_available())
    if device_name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        return mps_backend is not None and bool(mps_backend.is_available())
    return device_name == "cpu"


def resolve_pyannote_device(device_name: str) -> Any:
    torch = importlib.import_module("torch")
    normalized = device_name.strip().lower()

    if normalized in _AUTO_DEVICE_NAMES:
        for candidate in ("mps", "cuda"):
            if _torch_device_available(torch, candidate):
                return torch.device(candidate)
        return torch.device("cpu")

    if normalized not in _EXPLICIT_DEVICE_NAMES:
        supported = ", ".join(["auto", *sorted(_EXPLICIT_DEVICE_NAMES)])
        raise DiarizationConfigurationError(
            f"Unsupported pyannote device '{device_name}'. Supported values: {supported}."
        )

    if not _torch_device_available(torch, normalized):
        raise DiarizationBackendUnavailableError(
            f"pyannote device '{normalized}' is not available on this host."
        )
    return torch.device(normalized)


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


def diarize_audio(
    audio_path: Path,
    params: DiarizationParams | None = None,
) -> list[SpeakerTurn]:
    with admit_local_inference(LANE_NON_INTERACTIVE):
        with _DIARIZATION_LOCK:
            return get_diarization_backend_service().diarize(audio_path, params)


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
            device_name=settings.VOXHELM_PYANNOTE_DEVICE,
        )
    raise DiarizationConfigurationError(
        f"Unsupported diarization backend '{backend_name}'. Supported values: none, pyannote."
    )


@lru_cache(maxsize=4)
def get_pyannote_diarization_backend(
    *,
    model_name: str,
    auth_token: str,
    device_name: str,
) -> PyannoteDiarizationBackend:
    return PyannoteDiarizationBackend(
        model_name=model_name,
        auth_token=auth_token,
        device_name=device_name,
    )


def apply_speaker_labels(
    result: TranscriptionResult,
    turns: list[SpeakerTurn],
) -> TranscriptionResult:
    normalized_turns = normalize_speaker_turns(turns)
    if not normalized_turns:
        raise DiarizationError("Diarization backend returned no usable speaker turns.")

    labeled_segments = []
    assigned_count = 0
    turn_index = SpeakerTurnIndex(normalized_turns)
    for segment in result.segments:
        speaker = choose_speaker_for_segment(
            segment_start=segment.start,
            segment_end=segment.end,
            turns=turn_index,
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
    # Non-finite times are outside the turn domain: NaN breaks ordering and overlap
    # arithmetic, and an infinite bound would claim every segment. Drop them first.
    finite_turns = [turn for turn in turns if math.isfinite(turn.start) and math.isfinite(turn.end)]
    for turn in sorted(finite_turns, key=lambda item: (item.start, item.end, item.speaker)):
        backend_label = turn.speaker.strip()
        if not backend_label:
            continue
        start = max(float(turn.start), 0.0)
        end = max(float(turn.end), 0.0)
        if end <= start:
            continue
        if backend_label not in labels_by_backend_label:
            labels_by_backend_label[backend_label] = f"Speaker {len(labels_by_backend_label) + 1}"
        normalized_turns.append(
            SpeakerTurn(
                start=start,
                end=end,
                speaker=labels_by_backend_label[backend_label],
            )
        )
    return normalized_turns


class SpeakerTurnIndex:
    """Overlap queries over speaker turns without scanning every turn per segment.

    A max-end segment tree over the turns ordered by start answers "which turns
    overlap ``[start, end)``" in O((k + 1) log n) for k overlapping turns, so
    aligning S segments costs O((S + overlaps) log n) instead of O(S * n).
    Results are the original turn indices in ascending order, so callers see the
    same candidates in the same order as a full scan.
    """

    def __init__(self, turns: Sequence[SpeakerTurn]) -> None:
        self.turns = list(turns)
        self._order = sorted(range(len(self.turns)), key=lambda i: self.turns[i].start)
        self._starts = [self.turns[i].start for i in self._order]
        size = 1
        while size < len(self._order):
            size *= 2
        self._size = size
        tree = [-math.inf] * (2 * size)
        for position, index in enumerate(self._order):
            tree[size + position] = self.turns[index].end
        for node in range(size - 1, 0, -1):
            tree[node] = max(tree[2 * node], tree[2 * node + 1])
        self._tree = tree

    def overlapping(self, start: float, end: float) -> list[int]:
        """Indices of turns with ``turn.start < end`` and ``turn.end > start``."""
        limit = bisect_left(self._starts, end)
        found: list[int] = []
        stack = [(1, 0, self._size)]
        while stack:
            node, low, high = stack.pop()
            if low >= limit or self._tree[node] <= start:
                continue
            if high - low == 1:
                found.append(self._order[low])
                continue
            middle = (low + high) // 2
            stack.append((2 * node + 1, middle, high))
            stack.append((2 * node, low, middle))
        found.sort()
        return found


def overlapping_turns(
    turns: Sequence[SpeakerTurn] | SpeakerTurnIndex, start: float, end: float
) -> list[tuple[int, SpeakerTurn]]:
    """``(index, turn)`` pairs that may overlap ``[start, end)``, in turn order."""
    if isinstance(turns, SpeakerTurnIndex):
        return [(index, turns.turns[index]) for index in turns.overlapping(start, end)]
    return list(enumerate(turns))


def choose_speaker_for_segment(
    *,
    segment_start: float,
    segment_end: float,
    turns: Sequence[SpeakerTurn] | SpeakerTurnIndex,
) -> str | None:
    if not (math.isfinite(segment_start) and math.isfinite(segment_end)):
        # A segment without a finite time cannot be aligned to any turn.
        return None
    start = min(segment_start, segment_end)
    end = max(segment_start, segment_end)
    if end <= start:
        return None

    candidates: list[tuple[float, float, float, str, int]] = []
    for index, turn in overlapping_turns(turns, start, end):
        overlap = min(end, turn.end) - max(start, turn.start)
        if overlap <= 0:
            continue
        candidates.append((overlap, turn.start, turn.end, turn.speaker, index))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3], item[4]))
    return candidates[0][3]
