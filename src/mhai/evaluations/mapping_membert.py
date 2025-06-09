from typing import Dict

from mhai.evaluations.mental import MentalEvaluator


class MentBERTClassifier(MentalEvaluator):
    """
    Specialized evaluator using the mentBERT model, with mapping from specific
    mental health labels to broader core categories.
    """

    MENTBERT_TO_CORE: Dict[str, str] = {
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
        Maps mentBERT labels to broader core categories and aggregates scores.
        Returns a dictionary sorted by score descending.
        """
        output: Dict[str, float] = {}
        for label, score in raw_scores.items():
            core = self.MENTBERT_TO_CORE.get(label, 'unknown')
            output[core] = output.get(core, 0.0) + score
        return dict(sorted(output.items(), key=lambda x: x[1], reverse=True))
