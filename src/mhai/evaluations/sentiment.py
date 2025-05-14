"""
src/mhai/evaluations/sentiment.py.

Sentiment evaluation module.

Defines:
- ModelBase: abstract base for evaluators
- get_sentiment_pipeline: factory for HF sentiment pipeline
- SentimentEvaluator: binary sentiment evaluator
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from transformers import pipeline  # type: ignore[attr-defined]


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


def get_sentiment_pipeline(model_name: str, **kwargs: Any) -> Any:
    """
    Return a Hugging Face sentiment-analysis pipeline.

    Defaults to top_k=1; accepts extra kwargs like device.
    """
    params: Dict[str, Any] = {'top_k': 1}
    params.update(kwargs)
    return pipeline(
        task='sentiment-analysis',
        model=model_name,
        **params,
    )


class SentimentEvaluator(ModelBase):
    """Binary sentiment evaluator using SST-2 by default."""

    default_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    default_temperature = 0.0
    default_output_max_length = 2

    def _load_model(self) -> Any:
        """Instantiate the sentiment pipeline."""
        return get_sentiment_pipeline(self.model_name, **self.api_params)

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Evaluate sentiment.

        Returns a dict with:
        - label: 'POSITIVE' or 'NEGATIVE'
        - score: confidence score
        """
        raw = self._model(text)
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]
        result = raw[0] if isinstance(raw, list) else raw
        return {'label': result['label'], 'score': result['score']}
