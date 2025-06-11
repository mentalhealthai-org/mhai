"""Mapping from mentBERT to core mental health categories."""

from __future__ import annotations

from typing import ClassVar, Dict

from mhai.evaluations.mental import MentalEvaluator


class MentBERTClassifier(MentalEvaluator):
    """
    Map mentBERT labels to core mental health categories.

    MentBERT is a mental health specific transformer model that can detect
    multiple mental health conditions from text. This class takes the output of
    the mentBERT model and maps its labels to a smaller set of core mental
    health categories.
    """

    MENTBERT_TO_CORE: ClassVar[Dict[str, str]] = {
        'Anxiety': 'anxiety',  # Classic anxiety symptoms
        'Depression': 'depression',  # Sadness, low energy
        'Schizophrenia': 'psychosis',  # Delusions, hallucinations
        'Borderline': 'other',  # Personality instability
        'Asperger': 'other',  # On the autism spectrum
        'Bipolar': 'other',  # Mood swings
        'OCD': 'anxiety',  # Often grouped with anxiety spectrum
        'PTSD': 'anxiety',  # Related to trauma and fear
        'ADHD': 'other',  # Attention and hyperactivity
        'Autism': 'other',  # Developmental spectrum
        'None': 'none',  # No apparent mental condition
    }

    def __init__(self) -> None:
        super().__init__(
            api_params={'device': 0}, model_name='reab5555/mentBERT'
        )

    def map_to_core_categories(
        self, raw_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Map mentBERT labels to broader core categories and aggregates scores.

        Returns a dictionary sorted by score descending.
        """
        output: dict[str, float] = {}
        for label, score in raw_scores.items():
            core = self.MENTBERT_TO_CORE.get(label, 'unknown')
            output[core] = output.get(core, 0.0) + score
        return dict(sorted(output.items(), key=lambda x: x[1], reverse=True))
