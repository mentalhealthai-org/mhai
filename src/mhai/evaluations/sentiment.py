"""
Sentiment evaluation module.

Defines:
- ModelBase: abstract base for evaluators
- get_sentiment_pipeline: factory for HF sentiment pipeline
- SentimentEvaluator: binary sentiment evaluator
"""

from typing import Any

from transformers import pipeline  # type: ignore[attr-defined]

from .base import ModelBase


def get_sentiment_pipeline(model_name: str, **kwargs: Any) -> Any:
    """
    Return a Hugging Face sentiment-analysis pipeline.

    Defaults to top_k=1; accepts extra kwargs like device.
    """
    params: dict[str, Any] = {'top_k': 1}
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

    def evaluate(self, text: str) -> dict[str, Any]:
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
