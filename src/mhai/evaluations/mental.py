"""
Mental evaluation module.

Defines:
- get_mental_pipeline: factory for HF mental pipeline
- MentalEvaluator: wrapper that returns all detected emotions and scores
"""

from typing import Any

from transformers import pipeline  # type: ignore[attr-defined]

from .base import ModelBase


def get_mental_pipeline(model_name: str, **kwargs: Any) -> Any:
    """Load a Hugging Face mental-state classification pipeline."""
    params: dict[str, Any] = {'top_k': 5}
    params.update(kwargs)
    return pipeline(
        task='text-classification',
        model=model_name,
        top_k=None,  # returns all labels
        **kwargs,
    )


class MentalEvaluator(ModelBase):
    """Detects emotions/mental states from text (GoEmotions model)."""

    default_model_name = 'SamLowe/roberta-base-go_emotions'
    default_temperature = 0.0
    default_output_max_length = 6

    def _load_model(self) -> Any:
        return get_mental_pipeline(self.model_name, **self.api_params)

    def evaluate(self, text: str) -> dict[str, float]:
        """
        Run mental-state detection on `text`.

        Returns a mapping of label→score.
        """
        raw = self._model(text)

        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            raw = raw[0]

        # best = max(raw, key=lambda x: x["score"])

        return {entry['label']: entry['score'] for entry in raw}
