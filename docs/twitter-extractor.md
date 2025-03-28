### `docs/twitter-extractor.md`

# 🐦 Twitter Extractor

The `Twitter` class is a flexible and robust extractor for fetching tweets using the Twitter (X.com) API v2 via [Tweepy](https://docs.tweepy.org/en/stable/).
This module uses **app-only access (Bearer Token)** and supports pagination, date filtering, and basic error handling.

---

## ⚠️ Important Note on OAuth2

Tweepy does **not** currently support OAuth 2.0 user access tokens (e.g., PKCE flow) for authenticated requests like `.get_me()` or `.get_users_tweets()`.
Only **Bearer Tokens** (app-only) are supported in this implementation.

To access user-specific endpoints, use `requests` manually with an OAuth2 access token.

---

## 🚀 Quick Start

### 🔹 App-only Mode (Bearer Token)

```python
from sns import Twitter

client = Twitter.connect(
    token="YOUR_BEARER_TOKEN",
    username="esloch"
)

df = client.get_posts("2022-01-01", "2022-12-31")
print(df.head())
```

---

## 🔧 Class: `Twitter`

### `Twitter.connect(...)`

Connects to the Twitter API using a Bearer Token.

**Parameters:**

| Parameter   | Type            | Description                        |
|-------------|------------------|------------------------------------|
| `token`     | `str`           | App-only Bearer Token              |
| `username`  | `str`           | Twitter username (without @)       |

---

### `get_posts(from_, to, max_results=3200)`

Fetches all tweets from the given user in a date range.

**Parameters:**

| Parameter     | Type     | Description                                 |
|----------------|----------|---------------------------------------------|
| `from_`        | `str`    | Start date in format `"YYYY-MM-DD"`         |
| `to`           | `str`    | End date in format `"YYYY-MM-DD"`           |
| `max_results`  | `int`    | Max tweets to fetch (default: 3200)         |

**Returns:**

A `pandas.DataFrame` with:
- `id`
- `created_at`
- `text`
- `likes`
- `retweets`
- `replies`
- `quotes`

---

## 📌 Features

- ✅ App-only authentication (Bearer Token)
- ✅ Pagination using `tweepy.Paginator`
- ✅ Date range filtering
- ✅ Graceful handling of rate limits and API errors
- ✅ Caching of user ID after first request

---

## ⚠️ Limitations

### 🔹 Twitter API Behavior

| Feature                      | App-only (Bearer Token) |
|-----------------------------|--------------------------|
| Read public tweets           | ✅ Yes                   |
| Access user timeline         | ✅ Yes                   |
| Get current authenticated user | ❌ No                  |
| Post / like / retweet        | ❌ No                   |
| Tweet history limit          | ~3200 tweets per user   |

---

### 🔹 Rate Limits

- Up to 900 requests per 15 minutes (free tier)
- Each page = up to 100 tweets
- A `TooManyRequests` error will print a warning — consider adding sleep/backoff logic

---

## 🧪 Testing

You can test the module with:

```bash
makim tests.unit
```

The test suite mocks the Twitter API and includes:

- Instance creation
- User ID caching
- Tweet fetching (with and without results)
- Error handling

---

## 📚 References

- [Twitter API v2 Overview](https://developer.twitter.com/en/docs/twitter-api)
- [Tweepy Documentation](https://docs.tweepy.org/en/stable/)
- [get_users_tweets endpoint](https://developer.twitter.com/en/docs/twitter-api/tweets/timelines/api-reference/get-users-id-tweets)
