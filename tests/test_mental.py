"""Test suite for the MentalEvaluator class."""

import importlib.util
import unittest

from mhai.evaluations.mental import MentalEvaluator


@unittest.skipUnless(
    any(
        importlib.util.find_spec(pkg)
        for pkg in ('torch', 'tensorflow', 'flax')
    ),
    'Transformers backend required (PyTorch, TensorFlow or Flax)',
)
class TestMentalEvaluator(unittest.TestCase):
    """Test suite for the MentalEvaluator class."""

    def test_load_model_mental(self):
        """Ensure that _load_model() returns a callable model."""
        evaluator = MentalEvaluator(api_params={'device': 0})
        model = evaluator._load_model()
        self.assertTrue(callable(model))
        self.assertTrue(hasattr(model, '__call__'))

    def test_evaluate_mental_returns_scores(self):
        """Verify that evaluate() returns a mapping of label to score."""
        evaluator = MentalEvaluator(api_params={'device': 0})
        result = evaluator.evaluate(
            "I've been very overwhelmed and anxious at work."
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

        for label, score in result.items():
            with self.subTest(label=label):
                self.assertIsInstance(label, str)
                self.assertIsInstance(score, float)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
