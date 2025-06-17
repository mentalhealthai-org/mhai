"""Test suite for the mapping_membert module."""

import unittest

from typing import Dict, List

from mhai.evaluations.mapping_membert import (
    MentBERTClassifier,
)

mental_health_test_texts: Dict[str, List[str]] = {
    'anxiety': [
        'I struggle to breathe just thinking about my daily tasks.',
        (
            'I feel like something bad is going to happen, '
            'even when everything seems fine.'
        ),
        'My heart races every time I need to leave the house.',
        'I stay up all night worrying about things that might never happen.',
        'I avoid crowded places because I panic just thinking about them.',
    ],
    'depression': [
        'Nothing makes sense anymore; everything feels colorless.',
        'I do not even want to get out of bed.',
        'I have lost interest in everything that used to bring me joy.',
    ],
    'psychosis': [
        'I hear voices telling me I am in danger, even when I am alone.',
        'I think I am being followed, but no one believes me.',
        'Time feels like it stops sometimes, like I am stuck in a loop.',
        ('I am certain I can communicate through signals others cannot see.'),
        ('I see things that others say are not real, but I know they are.'),
    ],
    'other': [
        'Even when I am surrounded by people, I feel completely alone.',
    ],
}


class TestMentBERTClassifier(unittest.TestCase):
    """Test suite for the MentBERTClassifier class."""

    @classmethod
    def setUpClass(cls):
        """Test that the classifier can be initialized."""
        cls.classifier = MentBERTClassifier()

    def test_category_predictions(self):
        """Test that the category predictions are correct."""
        for category, texts in mental_health_test_texts.items():
            for text in texts:
                with self.subTest(category=category, text=text):
                    raw = self.classifier.evaluate(text)
                    core = self.classifier.map_to_core_categories(raw)
                    top_label, top_score = max(
                        core.items(), key=lambda x: x[1]
                    )
                    self.assertEqual(
                        top_label,
                        category,
                        msg=f"""Expected '{category}'
                        but got '{top_label}' for: '{text}'""",
                    )
