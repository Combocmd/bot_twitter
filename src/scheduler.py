"""Persistent, human-like posting queue.

The account's hard cap is 50 create-tweet calls per local day. Original posts
are capped by ``original_tweets_per_day``; the remaining slots are reserved for
replies. Each claim is atomic and a post is never retried after an uncertain
result, which prevents duplicate tweets after a crash or a slow X response.
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from .config import get_config
from .database import Database
from .timeutil import (
    end_of_local_day_utc,
    get_zone,
    is_active_hours,
    iso_utc,
    local_now,
    next_active_start,
    now_utc,
    parse_dt,
    start_of_local_day_utc,
    weighted_length,
)
from .web_client import XWebAuthError, XWebError, XWebUncertainError


class PostScheduler:
    def __init__(self, database: Database, web_client, config=None):
        self.db = database
        self.web = web_client
        self.config = config or get_config()
        self.cfg = self.config.scheduling
        self._running = False
        self._auth_failed = False
        self._limit_until: Optional[datetime] = None
        self.scheduler = BackgroundScheduler(timezone=self.cfg.timezone)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        recovered = self.db.recover_in_progress()
        if recovered.get("posting") or recovered.get("replies"):
            logger.warning("Éléments interrompus marqués incertains: {}", recovered)
        self.scheduler.add_job(
            self.process_due_tweets,
            IntervalTrigger(seconds=5),
            id="process_tweets",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=45,
        )
        self.scheduler.add_job(
            self._daily_cleanup,
            CronTrigger(hour=3, minute=0),
            id="daily_cleanup",
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.start()
        self._running = True
        logger.info("Planificateur démarré ({}, max {}/jour)",
                    self.cfg.active_hours_label, self.cfg.max_tweets_per_day)

    def stop(self) -> None:
        if self._running:
            try:
                self.scheduler.shutdown(wait=True)
            except Exception as exc:
                logger.warning("Arrêt du planificateur: {}", exc)
            self._running = False
            logger.info("Planificateur arrêté")

    # ------------------------------------------------------------------
    # Time / budget helpers
    # ------------------------------------------------------------------
    def _now_local(self) -> datetime:
        return local_now(self.cfg.timezone)

    def _is_active(self, when: Optional[datetime] = None) -> bool:
        return is_active_hours(
            when or now_utc(),
            start_hour=self.cfg.active_hours_start,
            end_hour=self.cfg.active_hours_end,
            tz_name=self.cfg.timezone,
        )

    def _next_start(self, when: Optional[datetime] = None) -> datetime:
        return next_active_start(
            when or now_utc(),
            start_hour=self.cfg.active_hours_start,
            tz_name=self.cfg.timezone,
        )

    def _random_interval(self) -> timedelta:
        base = random.randint(self.cfg.min_interval_minutes, self.cfg.max_interval_minutes)
        jitter = random.randint(-self.cfg.jitter_minutes, self.cfg.jitter_minutes)
        minutes = max(self.cfg.min_interval_minutes, base + jitter)
        return timedelta(minutes=minutes)

    def _posted_today(self) -> int:
        return self.db.get_today_posted_count(self.cfg.timezone)

    def _replies_sent_today(self) -> int:
        return self.db.get_today_replies_sent(self.cfg.timezone)

    def _pending_replies(self) -> int:
        return self.db.get_pending_replies_count()

    @property
    def publishing_paused(self) -> bool:
        return self._auth_failed

    def available_original_slots(self) -> int:
        """How many more original posts may be scheduled today."""
        posted = self.db.get_posted_count_for_local_day(tz_name=self.cfg.timezone)
        queued = self.db.count_queue_for_local_day(tz_name=self.cfg.timezone)
        replies_sent = self._replies_sent_today()
        pending_replies = min(
            self._pending_replies(),
            max(0, self.config.replies.max_replies_per_day - replies_sent),
        )
        # Reserve actual pending reply slots so the total never passes 50.
        max_originals = min(
            self.cfg.original_tweets_per_day,
            self.cfg.max_tweets_per_day - replies_sent - pending_replies,
        )
        return max(0, max_originals - posted - queued)

    def _last_scheduled_for_day(self, day: date) -> Optional[datetime]:
        zone = get_zone(self.cfg.timezone)
        last = None
        for row in self.db.get_pending_queue():
            dt = parse_dt(row.get("scheduled_for"))
            if dt and dt.astimezone(zone).date() == day:
                last = max(last, dt) if last else dt
        return last

    def _last_posted_for_day(self, day: date) -> Optional[datetime]:
        last = None
        for row in self.db.get_posted_tweets_for_local_day(day, self.cfg.timezone):
            dt = parse_dt(row.get("posted_at"))
            if dt:
                last = max(last, dt) if last else dt
        return last

    def _schedule_slot_for_day(self, day: date, now: datetime) -> Optional[datetime]:
        zone = get_zone(self.cfg.timezone)
        start_local = datetime.combine(day, datetime.min.time(), tzinfo=zone).replace(
            hour=self.cfg.active_hours_start
        )
        if self.cfg.active_hours_end >= 24:
            end_local = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=zone)
        else:
            end_local = start_local.replace(hour=self.cfg.active_hours_end)
        start_utc = start_local.astimezone(now.tzinfo)
        end_utc = end_local.astimezone(now.tzinfo)
        today = now.astimezone(zone).date()

        queued = self.db.count_queue_for_local_day(day, self.cfg.timezone)
        if day == today:
            posted = self.db.get_posted_count_for_local_day(day, self.cfg.timezone)
            replies_sent = self._replies_sent_today()
            pending_replies = min(
                self._pending_replies(),
                max(0, self.config.replies.max_replies_per_day - replies_sent),
            )
            max_originals = min(
                self.cfg.original_tweets_per_day,
                self.cfg.max_tweets_per_day - replies_sent - pending_replies,
            )
            if posted + queued >= max_originals:
                return None
        elif queued >= self.cfg.original_tweets_per_day:
            return None

        last = self._last_scheduled_for_day(day)
        if day == today:
            posted_last = self._last_posted_for_day(day)
            if posted_last and (last is None or posted_last > last):
                last = posted_last
        if self.cfg.min_interval_minutes == 0 and self.cfg.max_interval_minutes == 0:
            # Explicit immediate-publication mode. The daily hard cap remains
            # enforced by the budget checks above.
            candidate = now if day == today else start_utc
        elif last is not None:
            candidate = last + self._random_interval()
        else:
            first = start_utc + timedelta(minutes=random.randint(2, 10))
            candidate = max(first, now + timedelta(minutes=2)) if day == today else first

        if candidate < start_utc:
            candidate = start_utc + timedelta(minutes=1)
        if candidate >= end_utc:
            return None
        if not (self.cfg.min_interval_minutes == 0 and self.cfg.max_interval_minutes == 0) and candidate <= now + timedelta(seconds=20):
            candidate = now + timedelta(minutes=1)
        if candidate >= end_utc:
            return None
        return candidate.astimezone(now.tzinfo)

    def _get_next_schedule_slot(self) -> datetime:
        now = now_utc()
        zone = get_zone(self.cfg.timezone)
        today = now.astimezone(zone).date()
        for offset in range(0, 15):
            day = today + timedelta(days=offset)
            candidate = self._schedule_slot_for_day(day, now)
            if candidate is not None:
                return candidate
        # The queue is full for two weeks; keep the tweet rather than dropping
        # it, and schedule it at the next active start.
        return self._next_start() + timedelta(days=15, minutes=5)

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------
    def enqueue_tweet(
        self,
        text: str,
        media_ids: Optional[List[str]] = None,
        priority: int = 0,
        source_tweet_id: Optional[str] = None,
        source_username: Optional[str] = None,
        media_paths: Optional[List[str]] = None,
    ) -> Dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Texte vide")
        length = weighted_length(text)
        if length > self.config.length.max_chars:
            raise ValueError(
                f"Tweet trop long: {length}/{self.config.length.max_chars}"
            )
        paths = media_paths if media_paths is not None else media_ids
        scheduled_for = self._get_next_schedule_slot()
        queue_id = self.db.enqueue_tweet(
            text=text,
            media_ids=paths,
            media_paths=paths,
            priority=priority,
            scheduled_for=scheduled_for,
            source_tweet_id=source_tweet_id,
            source_username=source_username,
        )
        posted_row_id = self.db.add_posted_tweet(
            source_tweet_id=source_tweet_id or f"queue-{queue_id}",
            source_username=source_username or "UniversCule",
            source_text=text,
            rewritten_text=text,
            media_ids=paths,
            media_paths=paths,
            status="pending",
            queue_id=queue_id,
        )
        logger.info(
            "Tweet en file #{} pour {} ({} / {} originaux aujourd'hui)",
            queue_id,
            scheduled_for.astimezone(get_zone(self.cfg.timezone)).strftime("%d/%m %H:%M"),
            self._posted_today(),
            self.cfg.original_tweets_per_day,
        )
        return {
            "queue_id": queue_id,
            "posted_row_id": posted_row_id,
            "scheduled_for": scheduled_for,
        }

    # ------------------------------------------------------------------
    # Posting
    # ------------------------------------------------------------------
    def process_due_tweets(self) -> List[Dict]:
        if self._auth_failed or not self._is_active():
            return []
        if self._limit_until and now_utc() < self._limit_until:
            return []
        total_today = self._posted_today() + self._replies_sent_today()
        if total_today >= self.cfg.max_tweets_per_day:
            logger.warning("Plafond dur de {} posts/jour atteint", self.cfg.max_tweets_per_day)
            return []
        # Always enforce a safety floor between posts: a burst of a few tweets
        # within a minute triggers X's anti-spam/daily limits. Live coverage
        # stays possible, but never faster than one post per minute.
        last_posted = self._last_posted_for_day(self._now_local().date())
        if last_posted is not None:
            min_gap = max(
                timedelta(minutes=self.cfg.min_interval_minutes),
                timedelta(seconds=60),
            )
            if now_utc() - last_posted < min_gap:
                return []

        row = self.db.claim_due_tweet(now_utc())
        if not row:
            return []
        result = self._post_queued_tweet(row)
        return [result] if result else []

    def _limit_reset_at(self) -> datetime:
        """Next local midnight (+5 min), when X's daily counter resets."""
        tomorrow = self._now_local().date() + timedelta(days=1)
        return start_of_local_day_utc(tomorrow, self.cfg.timezone) + timedelta(minutes=5)

    def _post_queued_tweet(self, row: Dict) -> Optional[Dict]:
        queue_id = int(row["id"])
        text = row.get("text") or ""
        raw_paths = row.get("media_paths") or row.get("media_ids")
        paths: List[str] = []
        if raw_paths:
            try:
                parsed = json.loads(raw_paths) if isinstance(raw_paths, str) else raw_paths
                if isinstance(parsed, list):
                    paths = [str(p) for p in parsed if p]
            except Exception:
                paths = []
        logger.info("Publication du tweet en file #{}: {}", queue_id, text[:70])
        try:
            tweet_id = self.web.post_tweet(text, media_paths=paths)
        except XWebAuthError as exc:
            logger.critical("Session @UniversCule invalide; file mise en pause: {}", exc)
            self._auth_failed = True
            self.db.set_queue_failed(queue_id, f"auth: {exc}")
            self.db.update_daily_stats(errors_count=1)
            return {"queue_id": queue_id, "status": "failed", "error": str(exc)}
        except XWebUncertainError as exc:
            # X may have accepted the post. Never retry automatically.
            logger.error("Résultat incertain pour #{}; aucun nouvel essai: {}", queue_id, exc)
            self.db.mark_queue_uncertain(queue_id, str(exc))
            self.db.update_daily_stats(errors_count=1)
            return {"queue_id": queue_id, "status": "uncertain", "error": str(exc)}
        except XWebError as exc:
            error_text = str(exc)
            if "status is a duplicate" in error_text.lower() or "(187)" in error_text:
                self.db.set_queue_failed(queue_id, error_text, status="duplicate")
                logger.warning("Tweet #{} ignoré: X signale un doublon", queue_id)
                return {"queue_id": queue_id, "status": "duplicate", "error": error_text}
            if "daily limit" in error_text.lower():
                until = self._limit_reset_at()
                self._limit_until = until
                self.db.defer_queue_item(queue_id, until, error_text)
                logger.warning(
                    "Limite de publication X atteinte; #{} conservé et reporté à {} "
                    "(aucun essai consommé): {}",
                    queue_id,
                    until.astimezone(get_zone(self.cfg.timezone)).strftime("%d/%m %H:%M"),
                    error_text,
                )
                return {"queue_id": queue_id, "status": "deferred", "error": error_text}
            retries = int(row.get("retries") or 0)
            if retries < 3:
                delay = timedelta(minutes=5 * (retries + 1))
                self.db.set_queue_retry(queue_id, now_utc() + delay, str(exc))
                logger.warning("Échec #{} (essai {}), nouvel essai dans {} min: {}",
                               queue_id, retries + 1, 5 * (retries + 1), exc)
                return {"queue_id": queue_id, "status": "retry", "error": str(exc)}
            self.db.set_queue_failed(queue_id, str(exc))
            self.db.update_daily_stats(errors_count=1)
            logger.error("Échec définitif #{}: {}", queue_id, exc)
            return {"queue_id": queue_id, "status": "failed", "error": str(exc)}
        except Exception as exc:
            retries = int(row.get("retries") or 0)
            if retries < 3:
                self.db.set_queue_retry(queue_id, now_utc() + timedelta(minutes=10), str(exc))
            else:
                self.db.set_queue_failed(queue_id, str(exc))
                self.db.update_daily_stats(errors_count=1)
            logger.exception("Erreur inattendue #{}: {}", queue_id, exc)
            return {"queue_id": queue_id, "status": "retry", "error": str(exc)}
        else:
            self.db.mark_queue_posted(queue_id, tweet_id, now_utc())
            self.db.update_daily_stats(tweets_posted=1)
            for path in paths:
                try:
                    from pathlib import Path
                    Path(path).unlink(missing_ok=True)
                    self.db.delete_cached_media_by_path(path)
                except OSError as exc:
                    logger.warning("Nettoyage média impossible {}: {}", path, exc)
            logger.info("Post publié: https://x.com/UniversCule/status/{}", tweet_id)
            return {"queue_id": queue_id, "status": "posted", "tweet_id": tweet_id}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_queue_status(self) -> List[Dict]:
        return self.db.get_queue_status()

    def get_today_stats(self) -> Dict:
        stats = self.db.get_today_stats()
        stats.update({
            "original_budget": self.cfg.original_tweets_per_day,
            "original_slots_left": self.available_original_slots(),
            "hard_cap": self.cfg.max_tweets_per_day,
            "active_hours": self.cfg.active_hours_label,
            "next_slot": (
                iso_utc(self._get_next_schedule_slot())
                if self.available_original_slots() > 0 else None
            ),
        })
        return stats

    def cancel_pending_tweets(self) -> int:
        count = self.db.cancel_pending()
        logger.info("{} tweets en attente annulés", count)
        return count

    def _daily_cleanup(self) -> None:
        logger.info("Nettoyage quotidien...")
        try:
            self.db.cleanup_old_data(days=30)
        except Exception as exc:
            logger.warning("Nettoyage impossible: {}", exc)
