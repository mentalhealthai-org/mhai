"""Test suite for the SentimentEvaluator model."""

import importlib.util

import pytest

from mhai.evaluations.sentiment import SentimentEvaluator

# Skip all tests if no supported backend is available
_has_backend = any(
    importlib.util.find_spec(pkg) for pkg in ('torch', 'tensorflow', 'flax')
)

pytestmark = pytest.mark.skipif(
    not _has_backend,
    reason='Requires a real model inference',
)


def test_positive_sentiment() -> None:
    """Test the SentimentEvaluator for positive sentiment."""
    e = SentimentEvaluator(api_params={'device': 0})
    out = e.evaluate('I love this!')
    assert out['label'] == 'POSITIVE'
    assert out['score'] > 0.9


def test_negative_sentiment() -> None:
    """Test the SentimentEvaluator for negative sentiment."""
    e = SentimentEvaluator(api_params={'device': 0})
    out = e.evaluate('I hate that!')
    assert out['label'] == 'NEGATIVE'
    assert out['score'] > 0.9
