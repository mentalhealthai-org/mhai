"""Base class for text evaluators."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ModelBase(ABC):
    """
    Base class for text evaluators.

    Manages default parameters and requires subclasses to implement:
      - _load_model()
      - evaluate(text)
    """

    default_model_name: str
    default_temperature: float = 0.5
    default_output_max_length: int = 500

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        output_max_length: Optional[int] = None,
        api_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name or self.default_model_name
        self.temperature = (
            temperature
            if temperature is not None
            else self.default_temperature
        )
        self.output_max_length = (
            output_max_length
            if output_max_length is not None
            else self.default_output_max_length
        )
        self.api_params = api_params or {}
        self._model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        """Load and return the underlying pipeline or model."""
        ...

    @abstractmethod
    def evaluate(self, text: str) -> Any:
        """Run inference on `text` and return the result."""
        ...
