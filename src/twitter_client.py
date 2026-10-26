"""
Twitter API v2 wrapper with rate limit handling, retry logic, and media upload.
Uses tweepy with OAuth 1.0a authentication.
"""
import os
import time
import requests
import mimetypes
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from urllib.parse import urlparse

import tweepy
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

from .config import get_config


# Custom exceptions
class TwitterError(Exception):
    """Base Twitter API error."""
    pass

class RateLimitError(TwitterError):
    """Rate limit exceeded."""
    def __init__(self, reset_time: Optional[int] = None):
        self.reset_time = reset_time
        super().__init__(f"Rate limit exceeded. Reset at {reset_time}")

class AuthError(TwitterError):
    """Authentication error."""
    pass

class MediaError(TwitterError):
    """Media upload error."""
    pass

class NotFoundError(TwitterError):
    """Resource not found."""
    pass


@dataclass
class TweetData:
    """Structured tweet data from API."""
    id: str
    text: str
    author_id: str
    created_at: str
    media_urls: List[str]
    media_types: List[str]  # photo, video, animated_gif
    referenced_tweets: List[Dict]  # retweets, replies, quotes
    conversation_id: str
    in_reply_to_user_id: Optional[str] = None


@dataclass
class UserData:
    """Structured user data."""
    id: str
    username: str
    name: str
    description: str
    public_metrics: Dict


class TwitterClient:
    """
    Twitter API v2 client wrapper.
    Handles OAuth 1.0a auth, rate limits, retries, and media uploads.
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self._client: Optional[tweepy.Client] = None
        self._user_id: Optional[str] = None
        self._rate_limit_remaining: Dict[str, int] = {}
        self._rate_limit_reset: Dict[str, int] = {}

    @property
    def client(self) -> tweepy.Client:
        """Lazy-initialize tweepy client."""
        if self._client is None:
            creds = self.config.twitter_credentials
            self._client = tweepy.Client(
                consumer_key=creds["api_key"],
                consumer_secret=creds["api_secret"],
                access_token=creds["access_token"],
                access_token_secret=creds["access_token_secret"],
                wait_on_rate_limit=False,  # We handle rate limits ourselves
            )
        return self._client

    def _update_rate_limits(self, response: tweepy.Response):
        """Extract and store rate limit headers from response."""
        if hasattr(response, "headers"):
            headers = response.headers
            for key in ["x-rate-limit-remaining", "x-rate-limit-reset", "x-rate-limit-limit"]:
                if key in headers:
                    endpoint = key.replace("x-rate-limit-", "")
                    self._rate_limit_remaining[endpoint] = int(headers[key])
            if "x-rate-limit-reset" in headers:
                self._rate_limit_reset["default"] = int(headers["x-rate-limit-reset"])

    def _check_rate_limit(self, endpoint: str = "default") -> bool:
        """Check if we have rate limit remaining."""
        remaining = self._rate_limit_remaining.get(endpoint, 1)
        if remaining <= 0:
            reset_time = self._rate_limit_reset.get(endpoint, 0)
            wait_time = max(0, reset_time - int(time.time())) + 1
            logger.warning(f"Rate limit reached for {endpoint}, waiting {wait_time}s")
            time.sleep(wait_time)
            return False
        return True

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((RateLimitError, requests.RequestException, tweepy.TweepyException)),
        reraise=True
    )
    def _make_request(self, func, *args, **kwargs):
        """Execute API request with retry logic and rate limit handling."""
        try:
            response = func(*args, **kwargs)
            self._update_rate_limits(response)
            return response
        except tweepy.TooManyRequests as e:
            reset_time = int(e.response.headers.get("x-rate-limit-reset", time.time() + 900))
            raise RateLimitError(reset_time)
        except tweepy.Unauthorized as e:
            raise AuthError(f"Authentication failed: {e}")
        except tweepy.NotFound as e:
            raise NotFoundError(f"Resource not found: {e}")
        except tweepy.TweepyException as e:
            if "media" in str(e).lower():
                raise MediaError(f"Media error: {e}")
            raise TwitterError(f"Twitter API error: {e}")

    # ==================== User Methods ====================

    def get_user_id(self, username: str) -> str:
        """Get user ID from username (cached)."""
        if username == self.config.twitter.username if hasattr(self.config.twitter, 'username') else None:
            if self._user_id:
                return self._user_id

        response = self._make_request(
            self.client.get_user,
            username=username,
            user_fields=["id", "username", "name", "description", "public_metrics"]
        )
        if response.data:
            user_id = response.data.id
            if username.lower() == "universcule":  # Our bot account
                self._user_id = user_id
            return user_id
        raise NotFoundError(f"User @{username} not found")

    def get_me(self) -> UserData:
        """Get authenticated user info."""
        response = self._make_request(
            self.client.get_me,
            user_fields=["id", "username", "name", "description", "public_metrics"]
        )
        if response.data:
            return UserData(
                id=response.data.id,
                username=response.data.username,
                name=response.data.name,
                description=response.data.description or "",
                public_metrics=response.data.public_metrics or {}
            )
        raise AuthError("Could not get authenticated user")

    # ==================== Tweet Methods ====================

    def get_user_tweets(
        self,
        username: str,
        since_id: Optional[str] = None,
        max_results: int = 10,
        exclude_retweets: bool = True,
        exclude_replies: bool = True
    ) -> List[TweetData]:
        """
        Get recent tweets from a user.
        Returns list of TweetData with media info.
        """
        user_id = self.get_user_id(username)

        exclude = []
        if exclude_retweets:
            exclude.append("retweets")
        if exclude_replies:
            exclude.append("replies")

        response = self._make_request(
            self.client.get_users_tweets,
            id=user_id,
            since_id=since_id,
            max_results=min(max_results, 100),
            exclude=exclude if exclude else None,
            tweet_fields=["created_at", "author_id", "conversation_id", "in_reply_to_user_id", "referenced_tweets", "entities", "attachments"],
            expansions=["attachments.media_keys", "referenced_tweets.id"],
            media_fields=["url", "preview_image_url", "type", "width", "height", "duration_ms", "variants"]
        )

        if not response.data:
            return []

        # Build media lookup
        media_map = {}
        if response.includes and "media" in response.includes:
            for media in response.includes["media"]:
                media_map[media.media_key] = {
                    "url": media.url,
                    "type": media.type,
                    "preview_url": getattr(media, "preview_image_url", None)
                }

        tweets = []
        for tweet in response.data:
            # Skip retweets/quotes if they're in referenced_tweets
            is_retweet = False
            is_quote = False
            if tweet.referenced_tweets:
                for ref in tweet.referenced_tweets:
                    if ref.type == "retweeted":
                        is_retweet = True
                    elif ref.type == "quoted":
                        is_quote = True

            if is_retweet and exclude_retweets:
                continue

            # Extract media
            media_urls = []
            media_types = []
            if tweet.attachments and "media_keys" in tweet.attachments:
                for key in tweet.attachments["media_keys"]:
                    if key in media_map:
                        m = media_map[key]
                        media_urls.append(m["url"])
                        media_types.append(m["type"])

            tweets.append(TweetData(
                id=tweet.id,
                text=tweet.text,
                author_id=tweet.author_id,
                created_at=tweet.created_at.isoformat() if tweet.created_at else "",
                media_urls=media_urls,
                media_types=media_types,
                referenced_tweets=tweet.referenced_tweets or [],
                conversation_id=tweet.conversation_id,
                in_reply_to_user_id=tweet.in_reply_to_user_id
            ))

        return tweets

    def get_users_mentions(
        self,
        user_id: str,
        since_id: Optional[str] = None,
        max_results: int = 20
    ) -> List[TweetData]:
        """Get mentions of the authenticated user."""
        response = self._make_request(
            self.client.get_users_mentions,
            id=user_id,
            since_id=since_id,
            max_results=min(max_results, 100),
            tweet_fields=["created_at", "author_id", "conversation_id", "in_reply_to_user_id", "referenced_tweets", "entities"],
            expansions=["attachments.media_keys", "referenced_tweets.id"],
            media_fields=["url", "preview_image_url", "type"]
        )

        if not response.data:
            return []

        media_map = {}
        if response.includes and "media" in response.includes:
            for media in response.includes["media"]:
                media_map[media.media_key] = {
                    "url": media.url,
                    "type": media.type
                }

        tweets = []
        for tweet in response.data:
            media_urls = []
            media_types = []
            if tweet.attachments and "media_keys" in tweet.attachments:
                for key in tweet.attachments["media_keys"]:
                    if key in media_map:
                        m = media_map[key]
                        media_urls.append(m["url"])
                        media_types.append(m["type"])

            tweets.append(TweetData(
                id=tweet.id,
                text=tweet.text,
                author_id=tweet.author_id,
                created_at=tweet.created_at.isoformat() if tweet.created_at else "",
                media_urls=media_urls,
                media_types=media_types,
                referenced_tweets=tweet.referenced_tweets or [],
                conversation_id=tweet.conversation_id,
                in_reply_to_user_id=tweet.in_reply_to_user_id
            ))

        return tweets

    def get_tweet(self, tweet_id: str) -> Optional[TweetData]:
        """Get a single tweet by ID."""
        response = self._make_request(
            self.client.get_tweet,
            id=tweet_id,
            tweet_fields=["created_at", "author_id", "conversation_id", "in_reply_to_user_id", "referenced_tweets", "entities", "attachments"],
            expansions=["attachments.media_keys"],
            media_fields=["url", "preview_image_url", "type"]
        )

        if not response.data:
            return None

        tweet = response.data
        media_urls = []
        media_types = []
        if response.includes and "media" in response.includes:
            for media in response.includes["media"]:
                media_urls.append(media.url)
                media_types.append(media.type)

        return TweetData(
            id=tweet.id,
            text=tweet.text,
            author_id=tweet.author_id,
            created_at=tweet.created_at.isoformat() if tweet.created_at else "",
            media_urls=media_urls,
            media_types=media_types,
            referenced_tweets=tweet.referenced_tweets or [],
            conversation_id=tweet.conversation_id,
            in_reply_to_user_id=tweet.in_reply_to_user_id
        )

    # ==================== Posting Methods ====================

    def create_tweet(
        self,
        text: str,
        media_ids: Optional[List[str]] = None,
        in_reply_to_tweet_id: Optional[str] = None,
        quote_tweet_id: Optional[str] = None
    ) -> str:
        """
        Post a tweet.
        Returns the created tweet ID.
        """
        # Validate length
        if len(text) > self.config.length.max_chars:
            raise ValueError(f"Tweet exceeds {self.config.length.max_chars} characters: {len(text)}")

        kwargs = {"text": text}
        if media_ids:
            kwargs["media_ids"] = media_ids
        if in_reply_to_tweet_id:
            kwargs["in_reply_to_tweet_id"] = in_reply_to_tweet_id
        if quote_tweet_id:
            kwargs["quote_tweet_id"] = quote_tweet_id

        response = self._make_request(self.client.create_tweet, **kwargs)
        if response.data:
            return response.data["id"]
        raise TwitterError("Failed to create tweet: no data returned")

    def delete_tweet(self, tweet_id: str) -> bool:
        """Delete a tweet."""
        response = self._make_request(self.client.delete_tweet, id=tweet_id)
        return response.data.get("deleted", False) if response.data else False

    # ==================== Media Methods ====================

    def download_media(self, media_url: str, timeout: int = 30) -> tuple[bytes, str, str]:
        """
        Download media from Twitter URL.
        Returns (content_bytes, media_type, file_extension).
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(media_url, headers=headers, timeout=timeout, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            media_type = self._guess_media_type(content_type, media_url)
            extension = self._guess_extension(content_type, media_url)

            content = response.content

            # Size check
            max_size = self.config.media.max_size_mb * 1024 * 1024
            if len(content) > max_size:
                raise MediaError(f"Media too large: {len(content)} bytes > {max_size} bytes")

            return content, media_type, extension

        except requests.RequestException as e:
            raise MediaError(f"Failed to download media: {e}")

    def _guess_media_type(self, content_type: str, url: str) -> str:
        """Guess media type from content-type or URL."""
        if content_type:
            if content_type.startswith("image/"):
                return "image"
            elif content_type.startswith("video/"):
                return "video"
            elif content_type == "application/octet-stream":
                pass  # Check URL
        # Fallback to URL
        parsed = urlparse(url)
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
            return "image"
        elif any(path.endswith(ext) for ext in [".mp4", ".mov", ".webm", ".avi"]):
            return "video"
        return "unknown"

    def _guess_extension(self, content_type: str, url: str) -> str:
        """Guess file extension."""
        if content_type:
            ext = mimetypes.guess_extension(content_type)
            if ext:
                return ext
        # Fallback to URL
        parsed = urlparse(url)
        path = parsed.path
        ext = os.path.splitext(path)[1]
        if ext:
            return ext
        return ".bin"

    def upload_media(
        self,
        media_source: Union[str, bytes, Path],
        media_type: str = "auto",
        timeout: int = 60
    ) -> str:
        """
        Upload media to Twitter.
        media_source: URL, local file path, or bytes
        media_type: "image", "video", "gif", or "auto"
        Returns media_id string.
        """
        # Determine media content and type
        if isinstance(media_source, (str, Path)) and str(media_source).startswith(("http://", "https://")):
            # Download from URL
            content, detected_type, ext = self.download_media(str(media_source), timeout=timeout)
            if media_type == "auto":
                media_type = detected_type
        elif isinstance(media_source, (str, Path)):
            # Local file
            path = Path(media_source)
            content = path.read_bytes()
            if media_type == "auto":
                media_type = self._guess_media_type("", str(path))
        elif isinstance(media_source, bytes):
            content = media_source
            if media_type == "auto":
                media_type = "image"  # Default
        else:
            raise MediaError(f"Unsupported media source type: {type(media_source)}")

        # Upload based on type and size
        max_chunked = 5 * 1024 * 1024  # 5MB
        if len(content) > max_chunked and media_type in ("video", "gif"):
            return self._upload_chunked(content, media_type, timeout)
        else:
            return self._upload_simple(content, media_type, timeout)

    def _upload_simple(self, content: bytes, media_type: str, timeout: int) -> str:
        """Simple media upload (images, small videos/GIFs)."""
        # Determine media category for Twitter API
        category_map = {
            "image": "tweet_image",
            "video": "tweet_video",
            "gif": "tweet_gif",
        }
        category = category_map.get(media_type, "tweet_image")

        # Use tweepy's media upload (v1.1 endpoint)
        api = tweepy.API(
            tweepy.OAuth1UserHandler(
                self.config.twitter_credentials["api_key"],
                self.config.twitter_credentials["api_secret"],
                self.config.twitter_credentials["access_token"],
                self.config.twitter_credentials["access_token_secret"]
            )
        )

        try:
            media = api.media_upload(
                filename="media",
                file=content,
                media_category=category
            )
            return media.media_id_string
        except Exception as e:
            raise MediaError(f"Media upload failed: {e}")

    def _upload_chunked(self, content: bytes, media_type: str, timeout: int) -> str:
        """Chunked media upload for large videos."""
        api = tweepy.API(
            tweepy.OAuth1UserHandler(
                self.config.twitter_credentials["api_key"],
                self.config.twitter_credentials["api_secret"],
                self.config.twitter_credentials["access_token"],
                self.config.twitter_credentials["access_token_secret"]
            )
        )

        category_map = {
            "video": "tweet_video",
            "gif": "tweet_gif",
        }
        category = category_map.get(media_type, "tweet_video")

        try:
            # Use chunked upload
            media = api.media_upload(
                filename="video.mp4",
                file=content,
                media_category=category,
                chunked=True
            )
            return media.media_id_string
        except Exception as e:
            raise MediaError(f"Chunked media upload failed: {e}")

    def upload_multiple_media(self, media_sources: List[Union[str, bytes, Path]]) -> List[str]:
        """Upload multiple media files. Returns list of media_ids."""
        media_ids = []
        for source in media_sources:
            try:
                media_id = self.upload_media(source)
                media_ids.append(media_id)
                logger.info(f"Uploaded media: {media_id}")
            except MediaError as e:
                logger.error(f"Failed to upload media: {e}")
                # Continue with other media
        return media_ids

    # ==================== Engagement Methods ====================

    def like_tweet(self, tweet_id: str) -> bool:
        """Like a tweet."""
        try:
            response = self._make_request(self.client.like, tweet_id=tweet_id)
            return response.data.get("liked", False) if response.data else False
        except TwitterError:
            return False

    def retweet(self, tweet_id: str) -> bool:
        """Retweet a tweet."""
        try:
            response = self._make_request(self.client.retweet, tweet_id=tweet_id)
            return response.data.get("retweeted", False) if response.data else False
        except TwitterError:
            return False

    # ==================== Utility ====================

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        return {
            "remaining": dict(self._rate_limit_remaining),
            "reset": dict(self._rate_limit_reset)
        }