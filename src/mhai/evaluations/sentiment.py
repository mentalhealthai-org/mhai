# src/mhai/evaluations/sentiment.py
"""
Sentiment evaluation using a binary classifier by default.

This module defines:
- `ModelBase`: abstract base for evaluators
- `get_sentiment_pipeline`: factory for HF sentiment-analysis pipeline
- `SentimentEvaluator`: wrapper that defaults to SST-2 binary sentiment model
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import transformers


class ModelBase(ABC):
    """
    Base class for any textual evaluator.

    Manages default parameters and enforces interface.
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
        # Lazy-load the underlying pipeline
        self._model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        """
        Instantiate and return the underlying HF pipeline or model.
        """

        ...

    @abstractmethod
    def evaluate(self, text: str) -> Any:
        """
        Run inference on `text` and return the result.
        """

        ...


def get_sentiment_pipeline(model_name: str, **kwargs: Any) -> Any:
    """
    Factory for Hugging Face sentiment-analysis pipeline.

    Defaults to using a binary classifier (SST-2) and top_k=1.
    Additional kwargs are forwarded to the pipeline constructor.
    """

    params: Dict[str, Any] = {'top_k': 1}
    params.update(kwargs)

    return transformers.pipeline(
        task='sentiment-analysis',
        model=model_name,
        **params,
    )


class SentimentEvaluator(ModelBase):
    default_model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    default_temperature = 0.0
    default_output_max_length = 2

    def _load_model(self) -> Any:
        return get_sentiment_pipeline(self.model_name, **self.api_params)

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Perform sentiment classification on `text`, returning:
          {'label': 'POSITIVE'|'NEGATIVE', 'score': float}.

        Handles both flat and nested-list outputs from HF.
        """

        raw = self._model(text)

        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]

        first = raw[0] if isinstance(raw, list) else raw

        return {'label': first['label'], 'score': first['score']}
