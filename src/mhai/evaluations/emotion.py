"""
Emotion evaluation module.

Defines:
- get_emotion_pipeline: factory for HF emotion-classification pipeline
- EmotionEvaluator: wrapper that returns all detected emotions and scores
"""

from typing import Any

from transformers import pipeline  # type: ignore[attr-defined]

from .sentiment import ModelBase


def get_emotion_pipeline(model_name: str, **kwargs: Any) -> Any:
    """
    Return a Hugging Face emotion-classification pipeline.

    By default uses top_k=None to return all scores.
    Additional kwargs (device, etc.) are forwarded.
    """
    params: dict[str, Any] = {'top_k': None}
    params.update(kwargs)
    return pipeline(
        task='text-classification',
        model=model_name,
        **params,
    )


class EmotionEvaluator(ModelBase):
    """
    Emotion detection evaluator.

    Default model: j-hartmann/emotion-english-distilroberta-base
    """

    default_model_name = 'j-hartmann/emotion-english-distilroberta-base'
    default_temperature = 0.0
    default_output_max_length = 6

    def _load_model(self) -> Any:
        """Instantiate the emotion pipeline."""
        return get_emotion_pipeline(self.model_name, **self.api_params)

    def evaluate(self, text: str) -> Any:
        """Evaluate text emotions and return list of label-score dicts."""
        return self._model(text)
