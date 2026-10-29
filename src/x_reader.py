"""Authenticated, read-only X reader powered by Playwright + GraphQL.

It uses a normal browser session (cookies) instead of the paid X API. The
active session is exported from Firefox in ``x_session.py``. All reads are
public profile timelines and public conversation pages.
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

from loguru import logger

from .config import get_config
from .timeutil import clean_tweet_text, now_utc, parse_dt
from .x_session import PROJECT_ROOT, XSessionError, ensure_reader_state, load_state

_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['fr-FR', 'fr', 'en-US']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = window.chrome || {runtime: {}};
"""


@dataclass
class ReadTweet:
    id: str
    author_username: str
    author_id: str
    text: str
    created_at: datetime
    lang: str = "unknown"
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)
    reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    is_retweet: bool = False
    is_quote: bool = False
    reply_count: int = 0
    like_count: int = 0
    url: str = ""

    def as_dict(self) -> Dict:
        return {
            "id": self.id,
            "author_username": self.author_username,
            "author_id": self.author_id,
            "text": self.text,
            "created_at": self.created_at,
            "lang": self.lang,
            "media_urls": self.media_urls,
            "media_types": self.media_types,
            "reply_to": self.reply_to,
            "conversation_id": self.conversation_id,
            "is_retweet": self.is_retweet,
            "is_quote": self.is_quote,
            "reply_count": self.reply_count,
            "like_count": self.like_count,
            "url": self.url,
        }


class XReader:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.timeout_seconds = max(15, int(self.config.x_reader.timeout_seconds))
        self.headless = bool(self.config.x_reader.headless)
        state_file = Path(self.config.x_reader.state_file)
        self.state_path = state_file if state_file.is_absolute() else PROJECT_ROOT / state_file
        self.last_errors: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Browser plumbing
    # ------------------------------------------------------------------
    def _state(self) -> Path:
        if not self.state_path.exists():
            return ensure_reader_state()
        try:
            load_state(self.state_path)
        except XSessionError:
            logger.warning("Session lecteur expirée ou invalide, tentative de réimport Firefox")
            return ensure_reader_state(force=True)
        return self.state_path

    async def _new_context(self, playwright):
        state = self._state()
        browser = await playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
        context = await browser.new_context(
            storage_state=str(state),
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="fr-FR",
            timezone_id="Europe/Madrid",
        )
        await context.add_init_script(_STEALTH_JS)
        return browser, context

    @staticmethod
    def _attach_capture(page, wanted: Sequence[str]) -> List[Tuple[str, Dict]]:
        captured: List[Tuple[str, Dict]] = []

        async def _handler(response):
            url = response.url
            if "/graphql/" not in url:
                return
            if not any(op in url for op in wanted):
                return
            try:
                data = await response.json()
            except Exception:
                return
            op = url.split("/graphql/")[1].split("?")[0].split("/")[-1]
            captured.append((op, data))

        def _on_response(response):
            asyncio.create_task(_handler(response))

        page.on("response", _on_response)
        return captured

    async def _wait_capture(self, page, captured, seconds: Optional[int] = None) -> None:
        limit = (seconds or self.timeout_seconds) * 2
        for _ in range(limit):
            if captured:
                break
            await page.wait_for_timeout(500)
        # Let a second response (e.g. cursor/module) arrive.
        await page.wait_for_timeout(1500)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _author(tweet: Dict) -> Tuple[str, str]:
        result = ((tweet.get("core") or {}).get("user_results") or {}).get("result") or {}
        core = result.get("core") or {}
        username = core.get("screen_name") or (result.get("legacy") or {}).get("screen_name") or ""
        author_id = result.get("rest_id") or (result.get("legacy") or {}).get("id_str") or ""
        if not username:
            # Some legacy tweet payloads carry the author in a user result
            # under a slightly different wrapper.
            username = (tweet.get("user") or {}).get("screen_name", "")
        return str(username), str(author_id)

    @classmethod
    def _parse_tweet(cls, tweet: Dict) -> Optional[ReadTweet]:
        if not isinstance(tweet, dict):
            return None
        legacy = tweet.get("legacy")
        if not isinstance(legacy, dict):
            return None
        text = legacy.get("full_text") or ""
        note = tweet.get("note_tweet") or {}
        note_text = (
            ((note.get("note_tweet_results") or {}).get("result") or {}).get("text")
            if isinstance(note, dict) else None
        )
        if note_text:
            text = note_text
        created = parse_dt(legacy.get("created_at"))
        if not text or not created:
            return None
        author, author_id = cls._author(tweet)
        if not author:
            return None
        tweet_id = str(tweet.get("rest_id") or legacy.get("id_str") or "")
        if not tweet_id:
            return None
        media_urls: List[str] = []
        media_types: List[str] = []
        entities = legacy.get("extended_entities") or legacy.get("entities") or {}
        for media in (entities.get("media") or []):
            if not isinstance(media, dict):
                continue
            media_type = str(media.get("type") or "photo")
            if media_type == "video" or media_type == "animated_gif":
                variants = ((media.get("video_info") or {}).get("variants") or [])
                mp4 = [v for v in variants if isinstance(v, dict) and v.get("url")
                       and str(v.get("content_type", "")).lower() == "video/mp4"]
                if mp4:
                    chosen = max(mp4, key=lambda v: int(v.get("bitrate") or 0))
                    url = str(chosen["url"])
                    if url not in media_urls:
                        media_urls.append(url)
                        media_types.append("video")
                continue
            url = media.get("media_url_https") or media.get("media_url")
            if not url:
                continue
            url = str(url)
            if "?" not in url:
                url += "?name=orig"
            if url not in media_urls:
                media_urls.append(url)
                media_types.append("photo")
        return ReadTweet(
            id=tweet_id,
            author_username=author,
            author_id=author_id,
            text=clean_tweet_text(text),
            created_at=created,
            lang=str(legacy.get("lang") or "unknown"),
            media_urls=media_urls,
            media_types=media_types,
            reply_to=(str(legacy["in_reply_to_status_id_str"])
                      if legacy.get("in_reply_to_status_id_str") else None),
            conversation_id=(str(legacy["conversation_id_str"])
                             if legacy.get("conversation_id_str") else None),
            is_retweet=bool(legacy.get("retweeted_status_result")),
            is_quote=any(
                ref.get("type") == "quoted" 
                for ref in (tweet.get("referenced_tweets") or [])
            ),
            reply_count=int(legacy.get("reply_count") or 0),
            like_count=int(legacy.get("favorite_count") or 0),
            url=f"https://x.com/{author}/status/{tweet_id}",
        )

    @classmethod
    def _collect_tweets(cls, obj) -> List[ReadTweet]:
        found: List[ReadTweet] = []
        seen: set[str] = set()

        def walk(value):
            if isinstance(value, dict):
                legacy = value.get("legacy")
                if isinstance(legacy, dict) and (legacy.get("full_text") or value.get("note_tweet")):
                    parsed = cls._parse_tweet(value)
                    if parsed and parsed.id not in seen:
                        seen.add(parsed.id)
                        found.append(parsed)
                    return  # do not descend into quoted tweets
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(obj)
        return found

    @classmethod
    def parse_user_timeline(cls, data: Dict) -> List[ReadTweet]:
        if not isinstance(data, dict):
            return []
        return cls._collect_tweets(data)

    @classmethod
    def parse_tweet_detail(cls, data: Dict) -> List[ReadTweet]:
        if not isinstance(data, dict):
            return []
        return cls._collect_tweets(data)

    @staticmethod
    async def _dom_tweets(page, focal_id: Optional[str] = None) -> List[ReadTweet]:
        """Small DOM fallback when X changes a GraphQL operation name."""
        out: List[ReadTweet] = []
        try:
            articles = page.locator('article[data-testid="tweet"]')
            count = await articles.count()
        except Exception:
            return out
        for i in range(count):
            article = articles.nth(i)
            try:
                text = await article.locator('[data-testid="tweetText"]').inner_text()
            except Exception:
                text = ""
            try:
                href = await article.locator('a[href*="/status/"]').first.get_attribute("href") or ""
            except Exception:
                href = ""
            match = re.search(r"/status/(\d+)", href)
            tweet_id = match.group(1) if match else ""
            author_match = re.search(r"x\.com/([^/]+)/status", href)
            author = unquote(author_match.group(1)) if author_match else ""
            if not tweet_id or not author:
                continue
            try:
                created_raw = await article.locator("time[datetime]").first.get_attribute("datetime")
            except Exception:
                created_raw = None
            created = parse_dt(created_raw) or now_utc()
            media_urls = []
            media_types = []
            try:
                for img in await article.locator('img[src*="pbs.twimg.com/media"]').all():
                    src = await img.get_attribute("src")
                    if src:
                        src = src.split("&")[0]
                        if "?" not in src:
                            src += "?name=orig"
                        if src not in media_urls:
                            media_urls.append(src)
                            media_types.append("photo")
                for video in await article.locator('video').all():
                    src = await video.get_attribute("src")
                    if src and src not in media_urls and src.startswith("http"):
                        media_urls.append(src)
                        media_types.append("video")
            except Exception:
                pass
            if not text.strip() and not media_urls:
                continue
            out.append(ReadTweet(
                id=tweet_id,
                author_username=author,
                author_id="",
                text=clean_tweet_text(text),
                created_at=created,
                media_urls=media_urls,
                media_types=media_types,
                reply_to=focal_id if (focal_id and tweet_id != focal_id) else None,
                conversation_id=focal_id or tweet_id,
                url=f"https://x.com/{author}/status/{tweet_id}",
            ))
        return out

    # ------------------------------------------------------------------
    # Public reads
    # ------------------------------------------------------------------
    async def fetch_profiles(self, usernames: Iterable[str]) -> Dict[str, List[ReadTweet]]:
        names = [str(u).lstrip("@") for u in usernames if u]
        self.last_errors = {}
        results: Dict[str, List[ReadTweet]] = {name: [] for name in names}
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = None
        try:
            browser, context = await self._new_context(playwright)
            for username in names:
                page = await context.new_page()
                page.set_default_timeout(self.timeout_seconds * 1000)
                captured = self._attach_capture(
                    page, ("UserOriginalsTimeline", "UserTweets", "UserTweetsAndReplies")
                )
                page_failed = False
                # Retry une fois si X est lent (timeout ou erreur passagere)
                for attempt in range(2):
                    try:
                        await page.goto(
                            f"https://x.com/{username}",
                            wait_until="domcontentloaded",
                            timeout=self.timeout_seconds * 1000,
                        )
                        break  # Chargement reussi, on sort de la boucle retry
                    except Exception as exc:
                        if attempt == 0:
                            logger.warning(
                                "Lecture de @{} tentative {} echouee, retry: {}",
                                username, attempt + 1, exc,
                            )
                            # Nouvelle page pour le retry (l'ancienne peut etre invalide)
                            await page.close()
                            page = await context.new_page()
                            page.set_default_timeout(self.timeout_seconds * 1000)
                            captured = self._attach_capture(
                                page, ("UserOriginalsTimeline", "UserTweets", "UserTweetsAndReplies")
                            )
                        else:
                            self.last_errors[username] = str(exc)[:300]
                            logger.warning("Lecture de @{} impossible: {}", username, exc)
                            page_failed = True
                            break
                if page_failed:
                    await page.close()
                    # Continue with the other sources. SourceMonitor can use
                    # its RSS fallback for this account when available.
                    continue
                try:
                    await self._wait_capture(page, captured)
                    tweets: List[ReadTweet] = []
                    for op, data in captured:
                        if "UserTweetsAndReplies" in op:
                            continue
                        tweets.extend(self.parse_user_timeline(data))
                    if not tweets:
                        tweets = await self._dom_tweets(page)
                        if tweets:
                            logger.debug("Lecture DOM de secours pour @{}", username)
                    # Keep only original posts by the profile owner.
                    filtered = []
                    seen = set()
                    for tweet in tweets:
                        if tweet.id in seen:
                            continue
                        if tweet.author_username.lower() != username.lower():
                            continue
                        if tweet.is_retweet or tweet.reply_to:
                            continue
                        seen.add(tweet.id)
                        filtered.append(tweet)
                    filtered.sort(key=lambda t: t.created_at, reverse=True)
                    results[username] = filtered[: self.config.polling.max_results_per_source]
                    logger.debug("@{}: {} tweets originaux lus", username, len(results[username]))
                except Exception as exc:
                    self.last_errors[username] = str(exc)[:300]
                    logger.warning("Lecture de @{} impossible: {}", username, exc)
                finally:
                    await page.close()

        except Exception as exc:
            self.last_errors["__all__"] = str(exc)[:500]
            logger.error("Session lecteur Playwright impossible: {}", exc)
        finally:
            if browser:
                await browser.close()
            await playwright.stop()
        return results

    async def fetch_details(self, tweet_ids: Iterable[str], click_more: bool = True) -> Dict[str, List[ReadTweet]]:
        ids = [str(i) for i in tweet_ids if i]
        self.last_errors = {}
        results: Dict[str, List[ReadTweet]] = {i: [] for i in ids}
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = None
        try:
            browser, context = await self._new_context(playwright)
            for tweet_id in ids:
                page = await context.new_page()
                page.set_default_timeout(self.timeout_seconds * 1000)
                captured = self._attach_capture(page, ("TweetDetail",))
                # Retry une fois si X est lent
                for attempt in range(2):
                    try:
                        await page.goto(
                            f"https://x.com/i/status/{tweet_id}",
                            wait_until="domcontentloaded",
                            timeout=self.timeout_seconds * 1000,
                        )
                        break  # Chargement reussi
                    except Exception as exc:
                        if attempt == 0:
                            logger.warning(
                                "Lecture du fil {} tentative {} echouee, retry: {}",
                                tweet_id, attempt + 1, exc,
                            )
                            await page.close()
                            page = await context.new_page()
                            page.set_default_timeout(self.timeout_seconds * 1000)
                            captured = self._attach_capture(page, ("TweetDetail",))
                        else:
                            self.last_errors[tweet_id] = str(exc)[:300]
                            logger.warning("Lecture du fil {} impossible: {}", tweet_id, exc)
                            raise
                try:
                    await self._wait_capture(page, captured, seconds=8)
                    if click_more:
                        for label in ("Afficher plus de reponses", "Show more replies"):
                            try:
                                more = page.get_by_text(re.compile(label, re.IGNORECASE))
                                if await more.count():
                                    await more.first.click(timeout=3000)
                                    await page.wait_for_timeout(2500)
                                    break
                            except Exception:
                                pass
                    tweets: List[ReadTweet] = []
                    for _, data in captured:
                        tweets.extend(self.parse_tweet_detail(data))
                    if len(tweets) <= 1:
                        dom = await self._dom_tweets(page, focal_id=tweet_id)
                        if len(dom) > len(tweets):
                            tweets = dom
                    # Deduplicate, focal first.
                    unique: Dict[str, ReadTweet] = {}
                    for tweet in tweets:
                        unique[tweet.id] = tweet
                    results[tweet_id] = list(unique.values())
                except Exception as exc:
                    self.last_errors[tweet_id] = str(exc)[:300]
                    logger.warning("Lecture du fil {} impossible: {}", tweet_id, exc)
                finally:
                    await page.close()

        except Exception as exc:
            self.last_errors["__all__"] = str(exc)[:500]
            logger.error("Session lecteur Playwright impossible: {}", exc)
        finally:
            if browser:
                await browser.close()
            await playwright.stop()
        return results

    async def check_auth(self) -> str:
        """Return the connected reader handle (used by setup/status)."""
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = None
        try:
            browser, context = await self._new_context(playwright)
            page = await context.new_page()
            await page.goto("https://x.com/home", wait_until="domcontentloaded",
                            timeout=self.timeout_seconds * 1000)
            await page.wait_for_timeout(5000)
            account = page.locator('[data-testid="SideNav_AccountSwitcher_Button"]')
            if await account.count():
                return (await account.first.inner_text()).strip()
            return (await page.title()).strip()
        finally:
            if browser:
                await browser.close()
            await playwright.stop()
