"""Source monitoring for the six Barça accounts.

Reading is done through the authenticated Playwright/GraphQL session. The
public X API is not used: its read endpoints are rate-limited and the write
endpoint is paywalled. RSS is kept as a fallback for feeds that expose one.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger

from .config import get_config
from .database import Database
from .timeutil import clean_tweet_text, iso_utc, now_utc, parse_dt
from .x_reader import ReadTweet, XReader


@dataclass
class SourceTweet:
    source_username: str
    tweet_id: str
    text: str
    created_at: str
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    author_id: str = ""
    source_type: str = "twitter"
    language: str = "unknown"
    url: str = ""
    reply_count: int = 0

    @property
    def created_at_dt(self) -> Optional[datetime]:
        return parse_dt(self.created_at)


class SourceMonitor:
    def __init__(self, database: Database, config=None, reader: Optional[XReader] = None):
        self.db = database
        self.config = config or get_config()
        self.sources = list(self.config.sources.accounts)
        self.rss_feeds = dict(self.config.sources.rss_feeds)
        self.lookback_minutes = int(self.config.polling.lookback_minutes)
        self.max_results = int(self.config.polling.max_results_per_source)
        self.max_candidates = int(self.config.polling.max_candidates_per_cycle)
        self.reader = reader or XReader(self.config)
        self.last_errors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def poll_all_sources(self) -> List[SourceTweet]:
        try:
            return asyncio.run(self.poll_all_sources_async())
        except RuntimeError as exc:
            # A caller already inside an event loop cannot use asyncio.run.
            logger.error("Polling impossible dans une boucle asyncio active: {}", exc)
            return []

    async def poll_all_sources_async(self) -> List[SourceTweet]:
        logger.info("Polling de {} sources Barça...", len(self.sources))
        results = await self.reader.fetch_profiles(self.sources)
        self.last_errors = dict(getattr(self.reader, "last_errors", {}))
        cutoff = now_utc() - timedelta(minutes=self.lookback_minutes)
        all_new: List[SourceTweet] = []

        for username in self.sources:
            tweets = results.get(username, [])
            if not tweets:
                # A failed X read can be supplemented by an official RSS feed.
                if self.last_errors.get(username) or self.last_errors.get("__all__"):
                    rss = self._poll_rss_source(username)
                    if rss:
                        all_new.extend(rss)
                continue
            try:
                new = self._select_new(username, tweets, cutoff)
                if new:
                    logger.info("@{}: {} nouveau(x) tweet(s)", username, len(new))
                    all_new.extend(new)
                else:
                    logger.debug("@{}: aucun nouveau tweet", username)
            except Exception as exc:
                self.last_errors[username] = str(exc)[:300]
                logger.error("Traitement de @{} impossible: {}", username, exc)

        # Keep the freshest candidates, then restore chronological order so
        # the queue reads naturally (oldest of the new batch first).
        all_new.sort(key=lambda t: (t.created_at_dt or now_utc()))
        if len(all_new) > self.max_candidates:
            all_new = all_new[-self.max_candidates:]
        return all_new

    # ------------------------------------------------------------------
    # Selection / checkpoints
    # ------------------------------------------------------------------
    def _select_new(self, username: str, tweets: List[ReadTweet], cutoff: datetime) -> List[SourceTweet]:
        originals = [
            t for t in tweets
            if not t.is_retweet
            and not t.reply_to
            and t.author_username.lower() == username.lower()
            and t.id
        ]
        if not originals:
            return []
        originals.sort(key=lambda t: t.created_at, reverse=True)
        last_id = self.db.get_last_tweet_id(username)
        # The freshness window is authoritative, including after a restart:
        # never enqueue an older source post merely because its id is new to
        # this database.
        fresh = [t for t in originals if t.created_at >= cutoff]
        if last_id:
            try:
                last_int = int(last_id)
                new = [t for t in fresh if int(t.id) > last_int]
            except (TypeError, ValueError):
                new = fresh
        else:
            new = fresh

        # Advance the checkpoint even when all fetched tweets are old, so a
        # cold start does not replay the account's entire history.
        newest = originals[0]
        if not last_id or (new and int(newest.id) > int(new[-1].id)):
            self.db.update_last_tweet_id(username, newest.id)

        if not new:
            return []
        # Deduplicate IDs just in case a pinned tweet was returned twice.
        seen = set()
        selected: List[ReadTweet] = []
        for tweet in sorted(new, key=lambda t: t.created_at):
            if tweet.id in seen:
                continue
            seen.add(tweet.id)
            if not tweet.text.strip() and not tweet.media_urls:
                continue
            selected.append(tweet)
        return [self._to_source(username, tweet) for tweet in selected]

    @staticmethod
    def _to_source(username: str, tweet: ReadTweet) -> SourceTweet:
        return SourceTweet(
            source_username=username,
            tweet_id=str(tweet.id),
            text=tweet.text.strip(),
            created_at=iso_utc(tweet.created_at),
            media_urls=list(tweet.media_urls),
            media_types=list(tweet.media_types),
            author_id=tweet.author_id,
            source_type="twitter",
            language=tweet.lang or "unknown",
            url=tweet.url,
            reply_count=tweet.reply_count,
        )

    # ------------------------------------------------------------------
    # RSS fallback
    # ------------------------------------------------------------------
    def _poll_rss_source(self, username: str) -> List[SourceTweet]:
        feed_url = self.rss_feeds.get(username)
        if not feed_url:
            return []
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser absent: RSS impossible pour @{}", username)
            return []
        try:
            feed = feedparser.parse(feed_url)
        except Exception as exc:
            self.last_errors[username] = f"RSS: {exc}"[:300]
            logger.warning("RSS @{} illisible: {}", username, exc)
            return []
        if not getattr(feed, "entries", None):
            return []

        last_id = self.db.get_last_rss_entry_id(username)
        cutoff = now_utc() - timedelta(minutes=self.lookback_minutes)
        new: List[SourceTweet] = []
        newest_id = None
        for entry in feed.entries:
            entry_id = str(entry.get("id") or entry.get("link") or "")
            if not entry_id:
                entry_id = hashlib.sha256(
                    (entry.get("title", "") + entry.get("link", "")).encode("utf-8")
                ).hexdigest()[:20]
            newest_id = newest_id or entry_id
            if last_id and entry_id == last_id:
                break
            published = parse_dt(entry.get("published") or entry.get("updated"))
            # RSS follows the same strict freshness rule as X timelines.
            if published and published < cutoff:
                continue
            title = clean_tweet_text(entry.get("title", ""))
            summary = clean_tweet_text(
                html.unescape(entry.get("summary", entry.get("description", "")))
            )
            text = f"{title}. {summary}".strip(" .")
            if not text:
                continue
            media_urls, media_types = self._rss_media(entry)
            new.append(SourceTweet(
                source_username=username,
                tweet_id=entry_id,
                text=text[:1000],
                created_at=iso_utc(published or now_utc()),
                media_urls=media_urls,
                media_types=media_types,
                author_id=username,
                source_type="rss",
                language="unknown",
                url=entry.get("link", ""),
            ))
        if newest_id:
            self.db.update_last_rss_entry_id(username, newest_id)
        new.sort(key=lambda t: t.created_at_dt or now_utc())
        return new[-self.max_candidates:]

    @staticmethod
    def _rss_media(entry: Dict) -> tuple[List[str], List[str]]:
        urls: List[str] = []
        types: List[str] = []
        candidates = []
        for key in ("media_content", "media_thumbnail"):
            for item in entry.get(key, []) or []:
                if isinstance(item, dict) and item.get("url"):
                    candidates.append(item["url"])
        content = ""
        if entry.get("content"):
            content = entry["content"][0].get("value", "")
        summary = entry.get("summary", entry.get("description", ""))
        for raw in re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content + summary, re.I):
            candidates.append(html.unescape(raw))
        for url in candidates:
            if not url or url.startswith("data:"):
                continue
            lower = url.lower().split("?")[0]
            if not re.search(r"\.(?:jpe?g|png|webp|gif)$", lower):
                continue
            if url not in urls:
                urls.append(url)
                types.append("photo")
        return urls[:4], types[:4]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_source_stats(self) -> Dict[str, Dict]:
        checkpoints = {row["source_username"]: row for row in self.db.get_all_checkpoints()}
        out: Dict[str, Dict] = {}
        for username in self.sources:
            row = checkpoints.get(username, {})
            out[username] = {
                "last_tweet_id": row.get("last_tweet_id") or self.db.get_last_tweet_id(username),
                "last_rss_entry_id": row.get("last_rss_entry_id") or self.db.get_last_rss_entry_id(username),
                "last_checked": row.get("last_checked"),
                "has_rss": username in self.rss_feeds,
                "rss_url": self.rss_feeds.get(username),
                "last_error": self.last_errors.get(username),
            }
        return out
