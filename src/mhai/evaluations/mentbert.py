"""
MentBERT Evaluation Module.

Detects mental health conditions from text using MentBERT.
"""

import os

from typing import Any

from transformers import pipeline  # type: ignore[attr-defined]

from .base import ModelBase

access_token = os.getenv('HUGGINGFACE_TOKEN')


def get_mentbert_pipeline(model_name: str, **kwargs: Any) -> Any:
    """Load a Hugging Face MentBERT pipeline for text classification."""
    return pipeline(
        task='text-classification',
        model=model_name,
        token=access_token,
        top_k=None,  # returns all labels
        **kwargs,
    )


class MentBERTMentalHealthEvaluator(ModelBase):
    """
    Detect mental health conditions using MentBERT.

    Conditions may include depression, anxiety, PTSD, and related risks.
    """

    default_model_name = 'mental/mental-bert-base-uncased'
    default_temperature = 0.0
    default_output_max_length = 8

    def _load_model(self) -> Any:
        return get_mentbert_pipeline(self.model_name, **self.api_params)

    def evaluate(self, text: str) -> dict[str, float]:
        """Run mental health classification on `text`."""
        raw = self._model(text)

        # Support single-item output format (list of dicts)
        if (
            isinstance(raw, list)
            and len(raw) == 1
            and isinstance(raw[0], list)
        ):
            raw = raw[0]

        return {entry['label']: round(entry['score'], 4) for entry in raw}
