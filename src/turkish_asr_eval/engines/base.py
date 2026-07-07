from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ASREngine(ABC):
    """Common interface implemented by all ASR backends."""

    def __init__(self, model: str, **options: Any) -> None:
        self.model = model
        self.options = options

    @abstractmethod
    def load(self) -> None:
        """Load model resources."""

    @abstractmethod
    def transcribe(self, audio: Any) -> str:
        """Return a transcription for a single audio input."""
