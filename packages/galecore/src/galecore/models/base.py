"""Model interfaces. Every model sits behind one of these (CLAUDE.md).

Implementations: ``stub`` (deterministic, no weights, runs on a laptop) and ``real``
(weights from ``GALE_MODEL_DIR`` only, never downloaded at runtime).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from galecore.audio import FloatArray
from galecore.schemas import (
    AsrSegment,
    CallScorecard,
    Category,
    Channel,
    FusedEvent,
    Signal,
    WindowJudgement,
)


@dataclass(frozen=True, slots=True)
class RawWord:
    word: str
    start: float
    end: float
    conf: float


@dataclass(frozen=True, slots=True)
class RawSegment:
    """ASR output *before* redaction. Lives only inside asr-worker memory (D-01); never on
    the bus, never persisted, never logged."""

    channel: Channel
    ts_start: float
    ts_end: float
    text: str
    words: list[RawWord]
    is_final: bool
    avg_conf: float
    model_version: str


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Mono 16 kHz audio for one channel, positioned in the call."""

    call_id: str
    channel: Channel
    samples: FloatArray
    offset_s: float = 0.0
    sample_rate: int = 16000

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class TextClfOutput:
    scores: dict[Category, float]
    severity: int  # 0 none … 3 severe
    aggression: float = 0.0  # auxiliary head (D-06)
    model_version: str = ""


@dataclass(frozen=True, slots=True)
class TextContext:
    """Redacted, normalised utterance plus its preceding turns (docs/03 §6)."""

    current: str
    speaker: Channel
    previous: list[tuple[Channel, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WindowInput:
    call_id: str
    segments: list[AsrSegment]
    signals: list[Signal]
    provisional: list[FusedEvent]
    queue: str | None = None


class VADModel(ABC):
    """Voice activity: speech regions (seconds, absolute call time) in one channel."""

    @abstractmethod
    def speech_regions(self, chunk: AudioChunk) -> list[tuple[float, float]]: ...


class ASRBackend(ABC):
    name: str

    @abstractmethod
    def transcribe(self, chunk: AudioChunk) -> list[RawSegment]: ...


class ShoutDetector(ABC):
    @abstractmethod
    def detect(self, chunk: AudioChunk) -> list[Signal]: ...


class SERModel(ABC):
    @abstractmethod
    def detect(self, chunk: AudioChunk) -> list[Signal]: ...


class AudioToxModel(ABC):
    @abstractmethod
    def detect(self, chunk: AudioChunk) -> list[Signal]: ...


class KWSModel(ABC):
    @abstractmethod
    def detect(self, chunk: AudioChunk) -> list[Signal]: ...


class TextClassifier(ABC):
    @abstractmethod
    def classify(self, ctx: TextContext) -> TextClfOutput: ...


class RegisterDetector(ABC):
    """Stateful per (call, speaker): tracks আপনি/তুমি/তুই level through the call."""

    @abstractmethod
    def observe(self, segment: AsrSegment, negative_context: bool = False) -> list[Signal]:
        """``negative_context``: the utterance also carries abuse (lexicon / text-clf)."""


class LLMJudge(ABC):
    @abstractmethod
    def judge_window(self, window: WindowInput) -> WindowJudgement: ...

    @abstractmethod
    def judge_call(
        self, call_id: str, segments: list[AsrSegment], events: list[FusedEvent]
    ) -> CallScorecard: ...
