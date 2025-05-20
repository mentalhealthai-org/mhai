"""Test suite for the EmotionEvaluator class."""

import importlib.util
import unittest

from mhai.evaluations.emotion import EmotionEvaluator

EXPECTED_OUTPUT = [
    {'label': 'joy', 'score': 0.9771687984466553},
    {'label': 'surprise', 'score': 0.008528684265911579},
    {'label': 'neutral', 'score': 0.005764591973274946},
    {'label': 'anger', 'score': 0.004419785924255848},
    {'label': 'sadness', 'score': 0.002092393347993493},
    {'label': 'disgust', 'score': 0.001611992483958602},
    {'label': 'fear', 'score': 0.0004138524236623198},
]


@unittest.skipUnless(
    any(
        importlib.util.find_spec(pkg)
        for pkg in ('torch', 'tensorflow', 'flax')
    ),
    'Transformers backend required (PyTorch, TensorFlow or Flax)',
)
class TestEmotionEvaluator(unittest.TestCase):
    """Test suite for the EmotionEvaluator class."""

    def test_load_model_emotion(self):
        """Ensure that _load_model() returns a callable model."""
        evaluator = EmotionEvaluator(api_params={'device': 0})
        model = evaluator._load_model()
        self.assertTrue(callable(model))
        self.assertTrue(hasattr(model, '__call__'))

    def test_evaluate_emotion_output_format(self):
        """Verify that evaluate() returns a list of label-score dicts."""
        evaluator = EmotionEvaluator(api_params={'device': 0})
        output = evaluator.evaluate('I love this!')

        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

        if isinstance(output[0], list):  # unwrap if nested
            output = output[0]

        for expected, actual in zip(EXPECTED_OUTPUT, output):
            with self.subTest(label=expected['label']):
                self.assertEqual(actual['label'], expected['label'])
                self.assertAlmostEqual(
                    actual['score'], expected['score'], places=2
                )
