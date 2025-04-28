"""Test suite for the TwitterExtractor class."""

import unittest

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from mhai.sns.twitter import Twitter


class TestTwitterExtractor(unittest.TestCase):
    """Unit tests for the TwitterExtractor class."""

    def setUp(self):
        """Set up a Twitter instance for testing."""
        self.token = 'fake_token'
        self.username = 'esloch'
        self.mock_client = MagicMock()

        self.twitter = Twitter(client=self.mock_client, username=self.username)

    def test_connect_method_creates_instance(self):
        """Verify that the connect class method creates a Twitter instance."""
        with patch('tweepy.Client') as MockClient:
            instance = Twitter.connect(
                token=self.token, username=self.username
            )
            self.assertIsInstance(instance, Twitter)
            MockClient.assert_called_with(bearer_token=self.token)

    def test_get_user_id_caches_result(self):
        """Verify that get_user_id caches the user ID retrieved."""
        mock_user = MagicMock()
        mock_user.data.id = 123456
        self.mock_client.get_user.return_value = mock_user

        user_id = self.twitter.get_user_id()
        self.assertEqual(user_id, 123456)

        # second call should use cached value, no additional API call
        user_id_again = self.twitter.get_user_id()
        self.assertEqual(user_id_again, 123456)
        self.mock_client.get_user.assert_called_once_with(
            username=self.username
        )

    def test_get_posts_returns_dataframe(self):
        """Test that get_posts returns a DataFrame."""
        self.twitter.user_id = 123456

        # Create fake tweet data
        mock_tweet = MagicMock()
        mock_tweet.id = '111'
        mock_tweet.created_at = datetime(2023, 12, 25, 10, 30)
        mock_tweet.text = 'This is a mock tweet'
        mock_tweet.public_metrics = {
            'like_count': 10,
            'retweet_count': 2,
            'reply_count': 1,
            'quote_count': 0,
        }

        page = MagicMock()
        page.data = [mock_tweet]

        with patch('tweepy.Paginator') as MockPaginator:
            MockPaginator.return_value = [page]
            df = self.twitter.get_posts('2023-12-01', '2023-12-31')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn('text', df.columns)
        self.assertEqual(df.loc[0, 'likes'], 10)
        self.assertEqual(df.loc[0, 'text'], 'This is a mock tweet')

    def test_get_posts_handles_empty_data(self):
        """Verify that get_posts returns an empty DataFrame."""
        self.twitter.user_id = 123456

        empty_page = MagicMock()
        empty_page.data = []

        with patch('tweepy.Paginator') as MockPaginator:
            MockPaginator.return_value = [empty_page]
            df = self.twitter.get_posts('2025-01-01', '2025-01-31')

        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_get_posts_handles_api_error(self):
        """Verify that get_posts handles API errors gracefully."""
        self.twitter.user_id = 123456

        with patch('tweepy.Paginator', side_effect=Exception('API error')):
            with self.assertRaises(Exception) as context:
                self.twitter.get_posts('2023-01-01', '2023-12-31')
            self.assertIn('API error', str(context.exception))


if __name__ == '__main__':
    unittest.main()
