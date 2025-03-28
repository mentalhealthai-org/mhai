"""sns (social networking service) module."""

import time

from datetime import datetime
from typing import Any, Optional

import pandas as pd
import tweepy


class SocialMediaExtractorBase:
    """Base class for social media extractors."""

    def __init__(self) -> None:
        pass


class Twitter(SocialMediaExtractorBase):
    """Twitter extractor class (app-only Bearer Token support)."""

    def __init__(
        self,
        client: tweepy.Client,
        username: str,
    ) -> None:
        """Initialize the Twitter extractor."""
        super().__init__()
        self.client: tweepy.Client = client
        self.username: str = username
        self.user_id: Optional[int] = None

    @classmethod
    def connect(
        cls,
        token: str,
        username: str,
    ) -> 'Twitter':
        """Connect to the Twitter API using an app-only Bearer Token."""
        if not token:
            raise ValueError('Bearer token is required.')
        client = tweepy.Client(bearer_token=token)
        return cls(client=client, username=username)

    def get_user_id(self) -> int:
        """Retrieve the user ID from the provided username."""
        if self.user_id is not None:
            return self.user_id

        user = self.client.get_user(username=self.username)
        self.user_id = user.data.id
        return self.user_id

    def get_posts(
        self, from_: str, to: str, max_results: int = 3200
    ) -> pd.DataFrame:
        """Retrieve tweets from the user between two dates."""
        user_id = self.get_user_id()

        start_time = datetime.strptime(from_, '%Y-%m-%d').isoformat() + 'Z'
        end_time = datetime.strptime(to, '%Y-%m-%d').isoformat() + 'Z'

        all_tweets: list[dict[str, Any]] = []

        try:
            paginator = tweepy.Paginator(
                self.client.get_users_tweets,
                id=user_id,
                start_time=start_time,
                end_time=end_time,
                tweet_fields=['created_at', 'text', 'public_metrics'],
                max_results=100,
                limit=max_results // 100,
            )

            for page in paginator:
                if page.data:
                    for tweet in page.data:
                        metrics = tweet.public_metrics
                        all_tweets.append(
                            {
                                'id': tweet.id,
                                'created_at': tweet.created_at,
                                'text': tweet.text,
                                'likes': metrics['like_count'],
                                'retweets': metrics['retweet_count'],
                                'replies': metrics['reply_count'],
                                'quotes': metrics['quote_count'],
                            }
                        )
                time.sleep(1)

        except tweepy.TooManyRequests:
            print('Rate limit hit. Consider exponential backoff.')
        except Exception as e:
            print(f'Unexpected error: {e}')
            raise

        return pd.DataFrame(
            all_tweets,
            columns=[
                'id',
                'created_at',
                'text',
                'likes',
                'retweets',
                'replies',
                'quotes',
            ],
        )
