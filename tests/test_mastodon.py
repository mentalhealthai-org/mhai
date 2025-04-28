"""Test suite for the MastodonExtractor class."""

import unittest

import pytest

from mhai.sns.mastodon import MastodonExtractor


@pytest.mark.skip_on_ci
class TestMastodonExtractor(unittest.TestCase):
    """Test suite for the MastodonExtractor class."""

    @classmethod
    def setUpClass(cls):
        """Connect to Mastodon using env-based config."""
        cls.client = MastodonExtractor.connect()
        cls.test_handle = 'Gargron@mastodon.social'
        cls.test_hashtag = 'mastodon'

    def test_get_me(self):
        """Retrieve authenticated user information."""
        me = self.client.get_me()
        self.assertIn('username', me)

    def test_get_my_statuses(self):
        """Fetch recent statuses from authenticated user."""
        df = self.client.get_my_statuses(limit=3)
        self.assertFalse(df.empty)

    def test_get_user_statuses(self):
        """Fetch recent public statuses from a known user."""
        df = self.client.get_user_statuses(self.test_handle, limit=3)
        self.assertFalse(df.empty)

    def test_get_public_timeline(self):
        """Retrieve posts from the public timeline."""
        df = self.client.get_public_timeline(limit=3)
        self.assertFalse(df.empty)

    def test_get_hashtag_timeline(self):
        """Retrieve posts from a hashtag timeline."""
        df = self.client.get_hashtag_timeline(self.test_hashtag, limit=3)
        self.assertFalse(df.empty)

    def test_get_followers(self):
        """Fetch followers of authenticated user."""
        followers = self.client.get_followers()
        self.assertIsInstance(followers, list)

    def test_get_following(self):
        """Fetch accounts followed by authenticated user."""
        following = self.client.get_following()
        self.assertIsInstance(following, list)

    def test_get_notifications(self):
        """Retrieve notifications of authenticated user."""
        notifications = self.client.get_notifications(limit=3)
        self.assertIsInstance(notifications, list)
