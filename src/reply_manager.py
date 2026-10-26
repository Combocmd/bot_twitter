"""Comment reply pipeline.

Comments are read from the authenticated ``TweetDetail`` GraphQL page of our
own recent posts. Only direct replies written by other accounts are eligible.
Replies are filtered for spam/bots, scored for relevance, generated in French
by Ollama, then scheduled 20–30 minutes later.
"""
from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger

from .config import get_config
from .database import Database
from .filter import BarcaFilter
from .rewriter import TweetRewriter
from .timeutil import (
    clean_tweet_text,
    is_active_hours,
    iso_utc,
    local_now,
    next_active_start,
    now_utc,
    parse_dt,
)
from .web_client import XWebAuthError, XWebError, XWebUncertainError
from .x_reader import ReadTweet, XReader


@dataclass
class ReplyCandidate:
    our_tweet_id: str
    comment_id: str
    comment_author_id: str
    comment_author_username: str
    comment_text: str
    our_tweet_text: str
    relevance_score: float
    is_bot: bool = False
    language: str = "unknown"


class ReplyManager:
    def __init__(
        self,
        database: Database,
        web_client,
        rewriter: TweetRewriter,
        config=None,
        reader: Optional[XReader] = None,
        filter_: Optional[BarcaFilter] = None,
    ):
        self.db = database
        self.web = web_client
        self.rewriter = rewriter
        self.config = config or get_config()
        self.cfg = self.config.replies
        self.reader = reader or XReader(self.config)
        self.filter = filter_ or BarcaFilter(self.config)
        self._last_check = 0.0
        self._auth_failed = False

    # ------------------------------------------------------------------
    # Bot / spam detection
    # ------------------------------------------------------------------
    @staticmethod
    def _is_bot_comment(comment: ReadTweet) -> bool:
        username = (comment.author_username or "").lower()
        text = clean_tweet_text(comment.text).lower()
        normalized = re.sub(r"\s+", " ", text)
        if not username:
            return True
        if username in {"universcule"}:
            return True
        if re.search(r"(?:^|[_-])(bot|automated|spam|promo|crypto)(?:$|[_-])", username):
            return True
        if username.endswith("bot") and len(username) <= 20:
            return True
        if re.search(r"https?://|www\.", normalized):
            return True
        spam_phrases = (
            "follow for follow", "f4f", "follow me", "abonne-toi", "abonne toi",
            "check my profile", "visit my", "click link", "clique ici", "dm me",
            "mp moi", "telegram", "whatsapp", "discord", "earn money", "gagne de l'argent",
            "crypto", "investment", "investis", "giveaway", "concours", "airdrop",
            "claim your", "verify account", "vérifie ton compte", "lien en bio",
            "link in bio", "onlyfans", "promo", "sponsorisé", "parrainage",
        )
        if any(phrase in normalized for phrase in spam_phrases):
            return True
        if normalized.count("@") > 3 or normalized.count("#") > 5:
            return True
        if re.search(r"(.)\1{7,}", normalized):
            return True
        # A long all-caps message is almost always spam or shouting.
        letters = [c for c in normalized if c.isalpha()]
        if len(letters) > 25 and sum(1 for c in letters if c.isupper()) / len(letters) > 0.8:
            return True
        # No letters and no useful words: emoji-only reactions.
        if not re.search(r"[a-zA-ZÀ-ÿ0-9]", normalized):
            return True
        return False

    # ------------------------------------------------------------------
    # Comment discovery
    # ------------------------------------------------------------------
    def check_comments(self) -> List[ReplyCandidate]:
        if not self.cfg.enabled or self._auth_failed:
            return []
        tweets = self.db.get_tweets_to_check(
            limit=self.cfg.max_tweets_to_check,
            max_age_hours=self.cfg.max_age_hours,
        )
        tweets = [
            t for t in tweets
            if t.get("tweet_id") and str(t.get("tweet_id")).isdigit()
        ]
        if not tweets:
            return []
        ids = [str(t["tweet_id"]) for t in tweets]
        logger.info("Vérification des commentaires de {} post(s)", len(ids))
        try:
            details = asyncio.run(self.reader.fetch_details(ids))
        except Exception as exc:
            logger.error("Lecture des fils impossible: {}", exc)
            return []

        candidates: List[ReplyCandidate] = []
        authors_seen: set[str] = set()
        replies_sent = self.db.get_today_replies_sent(self.config.scheduling.timezone)
        remaining = max(0, self.cfg.max_replies_per_day - replies_sent - self.db.get_pending_replies_count())
        if remaining <= 0:
            logger.debug("Budget de réponses déjà réservé; pas de nouvelle détection")
            return []
        for row in tweets:
            our_id = str(row["tweet_id"])
            comments = details.get(our_id, [])
            our_posted = parse_dt(row.get("posted_at") or row.get("created_at"))
            if not comments:
                self.db.mark_tweet_checked(our_id)
                continue
            our_text = row.get("rewritten_text") or row.get("source_text") or ""
            for comment in comments:
                if str(comment.id) == our_id:
                    continue
                if comment.author_username.lower() == self.config.x_web.expected_handle.lower():
                    continue
                if comment.reply_to and str(comment.reply_to) != our_id:
                    # A reply to another comment, not directly to our post.
                    continue
                if comment.conversation_id and str(comment.conversation_id) != our_id:
                    continue
                if our_posted and comment.created_at < our_posted - timedelta(minutes=2):
                    continue
                if self.db.has_replied_to_comment(our_id, comment.id):
                    continue
                if self.db.has_queued_reply(comment.id):
                    continue
                if self.db.count_replies_to_author_today(
                    comment.author_username, self.config.scheduling.timezone
                ) >= 1:
                    continue
                if self.db.has_pending_reply_to_author(comment.author_username):
                    continue
                if comment.author_username.lower() in authors_seen:
                    continue
                if self._is_bot_comment(comment):
                    logger.debug("Commentaire ignoré (bot/spam) @{}", comment.author_username)
                    continue
                score = self.filter.comment_relevance(comment.text, our_text, comment.author_username)
                if score < self.cfg.relevance_threshold:
                    logger.debug(
                        "Commentaire @{} ignoré (pertinence {:.2f} < {:.2f})",
                        comment.author_username, score, self.cfg.relevance_threshold,
                    )
                    continue
                candidate = ReplyCandidate(
                    our_tweet_id=our_id,
                    comment_id=str(comment.id),
                    comment_author_id=str(comment.author_id or ""),
                    comment_author_username=comment.author_username,
                    comment_text=comment.text,
                    our_tweet_text=our_text,
                    relevance_score=score,
                    is_bot=False,
                    language=comment.lang,
                )
                candidates.append(candidate)
                authors_seen.add(comment.author_username.lower())
            self.db.mark_tweet_checked(our_id)
        # Highest relevance first, then newest.
        candidates.sort(key=lambda c: c.relevance_score, reverse=True)
        logger.info("{} commentaire(s) pertinent(s) retenu(s)", len(candidates))
        return candidates

    def schedule_reply(self, candidate: ReplyCandidate) -> Optional[int]:
        if self.db.has_queued_reply(candidate.comment_id):
            return None
        if self.db.has_pending_reply_to_author(candidate.comment_author_username):
            return None
        sent = self.db.get_today_replies_sent(self.config.scheduling.timezone)
        if sent + self.db.get_pending_replies_count() >= self.cfg.max_replies_per_day:
            logger.debug("Plafond de réponses atteint; commentaire non programmé")
            return None
        delay = random.randint(self.cfg.min_delay_minutes, self.cfg.max_delay_minutes)
        scheduled = now_utc() + timedelta(minutes=delay)
        if not is_active_hours(
            scheduled,
            start_hour=self.config.scheduling.active_hours_start,
            end_hour=self.config.scheduling.active_hours_end,
            tz_name=self.config.scheduling.timezone,
        ):
            scheduled = next_active_start(
                scheduled,
                start_hour=self.config.scheduling.active_hours_start,
                tz_name=self.config.scheduling.timezone,
            ) + timedelta(minutes=random.randint(1, 8))
        reply_text = self.rewriter.generate_reply(
            candidate.comment_text,
            candidate.our_tweet_text,
            candidate.comment_author_username,
        )
        queue_id = self.db.enqueue_reply(
            our_tweet_id=candidate.our_tweet_id,
            comment_id=candidate.comment_id,
            comment_author_id=candidate.comment_author_id,
            comment_author_username=candidate.comment_author_username,
            comment_text=candidate.comment_text,
            reply_text=reply_text,
            scheduled_for=scheduled,
        )
        if queue_id:
            logger.info(
                "Réponse programmée pour @{} dans {} min (pertinence {:.2f})",
                candidate.comment_author_username, delay, candidate.relevance_score,
            )
        return queue_id or None

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------
    def _budget_left(self) -> int:
        used = self.db.get_today_posted_count(self.config.scheduling.timezone)
        used += self.db.get_today_replies_sent(self.config.scheduling.timezone)
        return max(0, self.config.scheduling.max_tweets_per_day - used)

    def send_due_replies(self) -> List[Dict]:
        if not self.cfg.enabled or self._auth_failed:
            return []
        if not is_active_hours(
            now_utc(),
            start_hour=self.config.scheduling.active_hours_start,
            end_hour=self.config.scheduling.active_hours_end,
            tz_name=self.config.scheduling.timezone,
        ):
            self._postpone_due_replies_to_morning()
            return []
        replies_sent = self.db.get_today_replies_sent(self.config.scheduling.timezone)
        if replies_sent >= self.cfg.max_replies_per_day or self._budget_left() <= 0:
            self._postpone_due_replies_to_morning(force_budget=True)
            return []

        # One reply per scheduler call is enough; the main loop calls us often.
        row = self.db.claim_due_reply(now_utc())
        if not row:
            return []
        queue_id = int(row["id"])
        try:
            reply_text = row.get("reply_text") or self.rewriter.generate_reply(
                row.get("comment_text", ""), "", row.get("comment_author_username", "")
            )
            reply_text = clean_tweet_text(reply_text)
            if not reply_text:
                raise XWebError("Réponse générée vide")
            reply_id = self.web.reply_to(str(row["comment_id"]), reply_text)
        except XWebAuthError as exc:
            logger.critical("Session @UniversCule invalide; réponses en pause: {}", exc)
            self._auth_failed = True
            self.db.set_reply_failed(queue_id, f"auth: {exc}")
            return [{"queue_id": queue_id, "status": "failed", "error": str(exc)}]
        except XWebUncertainError as exc:
            logger.error("Réponse #{} incertaine; aucun nouvel essai: {}", queue_id, exc)
            self.db.set_reply_failed(queue_id, f"incertain: {exc}")
            self.db.update_daily_stats(errors_count=1)
            return [{"queue_id": queue_id, "status": "uncertain", "error": str(exc)}]
        except XWebError as exc:
            retries = int(row.get("retries") or 0)
            if retries < 2:
                self.db.set_reply_retry(queue_id, now_utc() + timedelta(minutes=10), str(exc))
                logger.warning("Échec réponse #{} (essai {}): {}", queue_id, retries + 1, exc)
            else:
                self.db.set_reply_failed(queue_id, str(exc))
                self.db.update_daily_stats(errors_count=1)
            return [{"queue_id": queue_id, "status": "retry" if retries < 2 else "failed",
                     "error": str(exc)}]
        except Exception as exc:
            self.db.set_reply_failed(queue_id, str(exc))
            self.db.update_daily_stats(errors_count=1)
            logger.exception("Erreur réponse #{}: {}", queue_id, exc)
            return [{"queue_id": queue_id, "status": "failed", "error": str(exc)}]
        else:
            self.db.mark_reply_sent(queue_id, reply_id, reply_text, now_utc())
            self.db.update_daily_stats(replies_sent=1)
            logger.info("Réponse envoyée: https://x.com/UniversCule/status/{}", reply_id)
            return [{"queue_id": queue_id, "status": "sent", "reply_id": reply_id}]

    def _postpone_due_replies_to_morning(self, force_budget: bool = False) -> None:
        due = self.db.get_due_replies(limit=20)
        if not due:
            return
        target = next_active_start(
            now_utc(),
            start_hour=self.config.scheduling.active_hours_start,
            tz_name=self.config.scheduling.timezone,
        )
        target += timedelta(minutes=random.randint(2, 12))
        for row in due:
            with self.db.transaction() as conn:
                conn.execute(
                    "UPDATE reply_queue SET scheduled_for=?, updated_at=? WHERE id=? AND status='pending'",
                    (iso_utc(target), iso_utc(), row["id"]),
                )
        if force_budget:
            logger.warning("Budget quotidien atteint: réponses reportées à demain")
        else:
            logger.debug("Hors heures actives: réponses reportées à {}", target)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def process_cycle(self) -> Dict:
        if not self.cfg.enabled:
            return {"checked": 0, "candidates": 0, "sent": 0}
        result = {"checked": 0, "candidates": 0, "sent": 0}
        now = time.time()
        if now - self._last_check >= self.cfg.check_interval_minutes * 60:
            try:
                candidates = self.check_comments()
                result["checked"] = len(self.db.get_tweets_to_check(
                    limit=self.cfg.max_tweets_to_check,
                    max_age_hours=self.cfg.max_age_hours,
                ))
                result["candidates"] = len(candidates)
                for candidate in candidates:
                    self.schedule_reply(candidate)
            except Exception as exc:
                logger.exception("Cycle de détection des commentaires: {}", exc)
            self._last_check = now
        try:
            sent = self.send_due_replies()
            result["sent"] = sum(1 for item in sent if item.get("status") == "sent")
        except Exception as exc:
            logger.exception("Cycle d'envoi des réponses: {}", exc)
        return result

    def process_due_replies(self) -> List[Dict]:
        return self.send_due_replies()

    def get_status(self) -> Dict:
        return {
            "enabled": self.cfg.enabled,
            "pending": self.db.get_pending_replies_count(),
            "sent_today": self.db.get_today_replies_sent(self.config.scheduling.timezone),
            "max_per_day": self.cfg.max_replies_per_day,
            "delay_minutes": f"{self.cfg.min_delay_minutes}–{self.cfg.max_delay_minutes}",
            "relevance_threshold": self.cfg.relevance_threshold,
        }
