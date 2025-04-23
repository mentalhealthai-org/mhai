"""Mastodon extractor module (SNS wrapper)."""

import os

from typing import Any

import pandas as pd

from dotenv import load_dotenv
from mastodon import Mastodon, MastodonNotFoundError

load_dotenv(dotenv_path='.envs/.env')


class SocialMediaExtractorBase:
    """Base class for social media extractors."""

    def __init__(self) -> None:
        pass


class MastodonExtractor(SocialMediaExtractorBase):
    """Mastodon extractor class."""

    def __init__(self, client: Mastodon) -> None:
        """Initialize Mastodon extractor."""
        super().__init__()
        self.client = client

    @classmethod
    def connect(cls) -> 'MastodonExtractor':
        """Connect to Mastodon using env-based config."""
        access_token = os.getenv('MASTODON_TOKEN')
        instance_url = os.getenv(
            'MASTODON_INSTANCE', 'https://mastodon.social'
        )

        if not access_token:
            raise RuntimeError(
                'MASTODON_TOKEN environment variable is not set.'
            )

        client = Mastodon(access_token=access_token, api_base_url=instance_url)
        return cls(client=client)

    def get_me(self) -> dict[str, Any]:
        """Return authenticated user metadata."""
        return dict(self.client.me())

    def get_user(self, handle: str) -> dict[str, Any]:
        """Return metadata for the given user handle."""
        try:
            return dict(self.client.account_lookup(handle))
        except MastodonNotFoundError:
            raise ValueError(f"User '{handle}' not found on instance.")

    def get_my_statuses(self, limit: int = 40) -> pd.DataFrame:
        """Return authenticated user's statuses as a DataFrame."""
        user = self.get_me()
        statuses = self.client.account_statuses(user['id'], limit=limit)
        return self._to_dataframe(statuses)

    def get_user_statuses(self, handle: str, limit: int = 40) -> pd.DataFrame:
        """Return public statuses from a given handle as a DataFrame."""
        user = self.get_user(handle)
        statuses = self.client.account_statuses(user['id'], limit=limit)
        return self._to_dataframe(statuses)

    def get_public_timeline(self, limit: int = 40) -> pd.DataFrame:
        """Return public timeline posts as a DataFrame."""
        statuses = self.client.timeline_public(limit=limit)
        return self._to_dataframe(statuses)

    def get_hashtag_timeline(
        self, hashtag: str, limit: int = 40
    ) -> pd.DataFrame:
        """Return hashtag timeline posts as a DataFrame."""
        statuses = self.client.timeline_hashtag(hashtag, limit=limit)
        return self._to_dataframe(statuses)

    def get_followers(self) -> list[dict[str, Any]]:
        """Return list of followers of the authenticated user."""
        user = self.get_me()
        return [dict(f) for f in self.client.account_followers(user['id'])]

    def get_following(self) -> list[dict[str, Any]]:
        """Return list of followed accounts."""
        user = self.get_me()
        return [dict(f) for f in self.client.account_following(user['id'])]

    def get_notifications(self, limit: int = 40) -> list[dict[str, Any]]:
        """Return recent notifications."""
        return [dict(n) for n in self.client.notifications(limit=limit)]

    def _to_dataframe(self, statuses: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert status list to DataFrame."""
        return pd.DataFrame(
            [
                {
                    'id': s['id'],
                    'created_at': s['created_at'],
                    'content': s['content'],
                    'replies': s['replies_count'],
                    'boosts': s['reblogs_count'],
                    'likes': s['favourites_count'],
                    'url': s.get('url', ''),
                }
                for s in statuses
            ]
        )
