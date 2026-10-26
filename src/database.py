"""Thread-safe SQLite persistence for the Barcelona bot.

The schema is intentionally migration-friendly because the original project
database already exists. All new timestamps are written as UTC ISO strings.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .config import get_config
from .timeutil import (
    end_of_local_day_utc,
    ensure_utc,
    iso_utc,
    local_date,
    local_day_for_utc,
    now_utc,
    parse_dt,
    start_of_local_day_utc,
)


class Database:
    """Thread-safe SQLite wrapper with one connection per thread."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = get_config().database.path
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------
    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def transaction(self):
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
        if column not in self._table_columns(conn, table):
            try:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            except sqlite3.OperationalError as exc:
                logger.debug("Migration {}.{}: {}", table, column, exc)

    def _init_db(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS posted_tweets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_tweet_id TEXT NOT NULL,
                    source_username TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    rewritten_text TEXT NOT NULL,
                    media_ids TEXT,
                    posted_at TEXT,
                    tweet_id TEXT,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    queue_id INTEGER,
                    created_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_posted_source_id ON posted_tweets(source_tweet_id);
                CREATE INDEX IF NOT EXISTS idx_posted_status ON posted_tweets(status);
                CREATE INDEX IF NOT EXISTS idx_posted_posted_at ON posted_tweets(posted_at);
                CREATE INDEX IF NOT EXISTS idx_posted_tweet_id ON posted_tweets(tweet_id);

                CREATE TABLE IF NOT EXISTS processed_sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_tweet_id TEXT NOT NULL UNIQUE,
                    source_username TEXT NOT NULL,
                    source_text_hash TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    content_embedding BLOB,
                    processed_at TEXT,
                    matched_existing INTEGER DEFAULT 0,
                    matched_tweet_id INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_processed_source_id ON processed_sources(source_tweet_id);
                CREATE INDEX IF NOT EXISTS idx_processed_hash ON processed_sources(source_text_hash);
                CREATE INDEX IF NOT EXISTS idx_processed_at ON processed_sources(processed_at);

                CREATE TABLE IF NOT EXISTS comments_replied (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    our_tweet_id TEXT NOT NULL,
                    comment_id TEXT NOT NULL,
                    comment_author_id TEXT NOT NULL,
                    comment_author_username TEXT,
                    comment_text TEXT NOT NULL,
                    reply_text TEXT NOT NULL,
                    replied_at TEXT,
                    UNIQUE(our_tweet_id, comment_id)
                );
                CREATE INDEX IF NOT EXISTS idx_comments_our_tweet ON comments_replied(our_tweet_id);
                CREATE INDEX IF NOT EXISTS idx_comments_comment_id ON comments_replied(comment_id);

                CREATE TABLE IF NOT EXISTS media_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_media_url TEXT NOT NULL UNIQUE,
                    local_path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    file_size INTEGER,
                    uploaded_media_id TEXT,
                    cached_at TEXT,
                    expires_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_media_url ON media_cache(source_media_url);
                CREATE INDEX IF NOT EXISTS idx_media_expires ON media_cache(expires_at);

                CREATE TABLE IF NOT EXISTS bot_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    tweets_posted INTEGER DEFAULT 0,
                    tweets_rejected INTEGER DEFAULT 0,
                    replies_sent INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    api_calls INTEGER DEFAULT 0,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS posting_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    media_ids TEXT,
                    priority INTEGER DEFAULT 0,
                    scheduled_for TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    retries INTEGER DEFAULT 0,
                    error_message TEXT,
                    source_tweet_id TEXT,
                    source_username TEXT,
                    tweet_id TEXT,
                    media_paths TEXT,
                    posted_at TEXT,
                    last_attempt_at TEXT,
                    updated_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_queue_scheduled ON posting_queue(scheduled_for, status);
                CREATE INDEX IF NOT EXISTS idx_queue_status ON posting_queue(status);

                CREATE TABLE IF NOT EXISTS reply_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    our_tweet_id TEXT NOT NULL,
                    comment_id TEXT NOT NULL,
                    comment_author_id TEXT NOT NULL,
                    comment_author_username TEXT,
                    comment_text TEXT NOT NULL,
                    reply_text TEXT,
                    scheduled_for TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    retries INTEGER DEFAULT 0,
                    sent_tweet_id TEXT,
                    error_message TEXT,
                    last_attempt_at TEXT,
                    updated_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_reply_queue_scheduled ON reply_queue(scheduled_for, status);
                CREATE INDEX IF NOT EXISTS idx_reply_queue_comment ON reply_queue(comment_id);

                CREATE TABLE IF NOT EXISTS source_checkpoints (
                    source_username TEXT PRIMARY KEY,
                    last_tweet_id TEXT,
                    last_rss_entry_id TEXT,
                    source_tweet_timestamp TEXT,
                    last_checked TEXT
                );

                CREATE TABLE IF NOT EXISTS reply_monitor_state (
                    tweet_id TEXT PRIMARY KEY,
                    last_checked_at TEXT,
                    last_seen_comment_id TEXT
                );
                """
            )
            # Migrations for the database created by the original version.
            for col, decl in [
                ("queue_id", "INTEGER"), ("created_at", "TEXT"),
            ]:
                self._ensure_column(conn, "posted_tweets", col, decl)
            for col, decl in [
                ("tweet_id", "TEXT"), ("media_paths", "TEXT"), ("posted_at", "TEXT"),
                ("last_attempt_at", "TEXT"), ("updated_at", "TEXT"),
            ]:
                self._ensure_column(conn, "posting_queue", col, decl)
            for col, decl in [
                ("sent_tweet_id", "TEXT"), ("error_message", "TEXT"),
                ("last_attempt_at", "TEXT"), ("updated_at", "TEXT"),
            ]:
                self._ensure_column(conn, "reply_queue", col, decl)
            for col, decl in [
                ("source_tweet_timestamp", "TEXT"),
            ]:
                self._ensure_column(conn, "source_checkpoints", col, decl)
            try:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_reply_queue_comment ON reply_queue(comment_id)"
                )
            except sqlite3.IntegrityError:
                logger.warning("Index unique reply_queue(comment_id) impossible: doublons existants")
            # Older rows may have naive timestamps; leave them readable.
            conn.execute(
                "UPDATE source_checkpoints SET last_checked=? "
                "WHERE last_checked IS NULL",
                (iso_utc(),),
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _text_hash(text: str) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").lower().strip())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _embedding_to_blob(embedding: np.ndarray) -> str:
        return base64.b64encode(np.asarray(embedding, dtype=np.float32).tobytes()).decode("ascii")

    @staticmethod
    def _blob_to_embedding(blob: str) -> np.ndarray:
        return np.frombuffer(base64.b64decode(blob), dtype=np.float32)

    # ------------------------------------------------------------------
    # Posted tweets
    # ------------------------------------------------------------------
    def add_posted_tweet(
        self,
        source_tweet_id: str,
        source_username: str,
        source_text: str,
        rewritten_text: str,
        media_ids: Optional[List[str]] = None,
        media_paths: Optional[List[str]] = None,
        status: str = "pending",
        queue_id: Optional[int] = None,
        posted_at: Optional[datetime] = None,
        tweet_id: Optional[str] = None,
    ) -> int:
        now = iso_utc()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO posted_tweets
                (source_tweet_id, source_username, source_text, rewritten_text,
                 media_ids, posted_at, tweet_id, status, queue_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_tweet_id, source_username, source_text, rewritten_text,
                    json.dumps(media_paths or media_ids) if (media_paths or media_ids) else None,
                    iso_utc(posted_at) if posted_at else None,
                    tweet_id, status, queue_id, now,
                ),
            )
            return int(cur.lastrowid)

    def update_posted_tweet(self, row_id: int, **kwargs) -> None:
        allowed = {
            "tweet_id", "status", "error_message", "media_ids", "posted_at", "queue_id",
        }
        fields = [k for k in kwargs if k in allowed]
        if not fields:
            return
        values = [kwargs[k] for k in fields]
        if "posted_at" in kwargs and kwargs["posted_at"] is not None:
            values[fields.index("posted_at")] = iso_utc(kwargs["posted_at"])
        values.append(row_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE posted_tweets SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                values,
            )

    def update_posted_tweet_by_queue_id(self, queue_id: int, **kwargs) -> None:
        allowed = {"tweet_id", "status", "error_message", "posted_at"}
        fields = [k for k in kwargs if k in allowed]
        if not fields:
            return
        values = [kwargs[k] for k in fields]
        if "posted_at" in kwargs and kwargs["posted_at"] is not None:
            values[fields.index("posted_at")] = iso_utc(kwargs["posted_at"])
        values.append(queue_id)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE posted_tweets SET {', '.join(f'{k}=?' for k in fields)} WHERE queue_id=?",
                values,
            )

    def get_posted_tweets(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict]:
        query = "SELECT * FROM posted_tweets WHERE 1=1"
        params: List[Any] = []
        if status:
            query += " AND status=?"
            params.append(status)
        if since:
            query += " AND posted_at>=?"
            params.append(iso_utc(since))
        query += " ORDER BY COALESCE(posted_at, created_at) DESC LIMIT ?"
        params.append(limit)
        with self.transaction() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def get_recent_posted_tweets(self, limit: int = 8, max_age_hours: int = 48) -> List[Dict]:
        cutoff = now_utc() - timedelta(hours=max_age_hours)
        rows = self.get_posted_tweets(limit=max(limit * 4, 40), status="posted")
        out = []
        for row in rows:
            created = parse_dt(row.get("posted_at") or row.get("created_at"))
            if created and created >= cutoff:
                out.append(row)
        return out[:limit]

    def get_posted_tweet_by_tweet_id(self, tweet_id: str) -> Optional[Dict]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM posted_tweets WHERE tweet_id=? LIMIT 1", (str(tweet_id),)
            ).fetchone()
            return dict(row) if row else None

    def get_today_posted_count(self, tz_name: str = "Europe/Madrid") -> int:
        start = start_of_local_day_utc(tz_name=tz_name)
        end = end_of_local_day_utc(tz_name=tz_name)
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT posted_at FROM posted_tweets WHERE status='posted' AND posted_at IS NOT NULL"
            ).fetchall()
        count = 0
        for row in rows:
            dt = parse_dt(row["posted_at"])
            if dt and start <= dt < end:
                count += 1
        return count

    def get_today_posted_tweets(self, tz_name: str = "Europe/Madrid") -> List[Dict]:
        return self.get_posted_tweets_for_local_day(local_date(tz_name), tz_name)

    def get_posted_tweets_for_local_day(
        self, day: Optional[date] = None, tz_name: str = "Europe/Madrid"
    ) -> List[Dict]:
        day = day or local_date(tz_name)
        start = start_of_local_day_utc(day, tz_name)
        end = end_of_local_day_utc(day, tz_name)
        rows = self.get_posted_tweets(limit=500, status="posted")
        out = []
        for row in rows:
            dt = parse_dt(row.get("posted_at"))
            if dt and start <= dt < end:
                out.append(row)
        return out

    def get_posted_count_for_local_day(
        self, day: Optional[date] = None, tz_name: str = "Europe/Madrid"
    ) -> int:
        return len(self.get_posted_tweets_for_local_day(day, tz_name))

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def add_processed_source(
        self,
        source_tweet_id: str,
        source_username: str,
        source_text: str,
        embedding: Optional[np.ndarray] = None,
        matched_existing: bool = False,
        matched_tweet_id: Optional[int] = None,
    ) -> int:
        text_hash = self._text_hash(source_text)
        emb_blob = self._embedding_to_blob(embedding) if embedding is not None else None
        now = iso_utc()
        with self.transaction() as conn:
            # Legacy databases declare matched_tweet_id as a foreign key to
            # posted_tweets.id, while deduplication supplies the id of the
            # matched processed_sources row. Preserve the dedup record without
            # violating the legacy constraint; the match reason is retained in
            # the processed row itself.
            if matched_tweet_id is not None:
                exists = conn.execute(
                    "SELECT 1 FROM posted_tweets WHERE id=?", (matched_tweet_id,)
                ).fetchone()
                if exists is None:
                    matched_tweet_id = None
            conn.execute(
                """
                INSERT INTO processed_sources
                (source_tweet_id, source_username, source_text_hash, source_text,
                 content_embedding, processed_at, matched_existing, matched_tweet_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_tweet_id) DO UPDATE SET
                    source_text_hash=excluded.source_text_hash,
                    source_text=excluded.source_text,
                    content_embedding=COALESCE(excluded.content_embedding, processed_sources.content_embedding),
                    processed_at=excluded.processed_at,
                    matched_existing=excluded.matched_existing,
                    matched_tweet_id=excluded.matched_tweet_id
                """,
                (
                    str(source_tweet_id), source_username, text_hash, source_text,
                    emb_blob, now, int(matched_existing), matched_tweet_id,
                ),
            )
            row = conn.execute(
                "SELECT id FROM processed_sources WHERE source_tweet_id=?", (str(source_tweet_id),)
            ).fetchone()
            return int(row[0]) if row else 0

    def find_similar_source(
        self,
        embedding: np.ndarray,
        threshold: float,
        time_window_hours: int = 48,
    ) -> Optional[Tuple[int, float, Dict]]:
        since = now_utc() - timedelta(hours=time_window_hours)
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT id, source_tweet_id, source_username, source_text,
                       content_embedding, processed_at
                FROM processed_sources
                WHERE content_embedding IS NOT NULL
                ORDER BY processed_at DESC LIMIT 2000
                """
            ).fetchall()
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            return None
        best: Optional[Tuple[int, float, Dict]] = None
        for row in rows:
            dt = parse_dt(row["processed_at"])
            if dt and dt < since:
                continue
            try:
                stored = self._blob_to_embedding(row["content_embedding"])
            except Exception:
                continue
            stored_norm = float(np.linalg.norm(stored))
            if stored_norm == 0 or stored.shape != embedding.shape:
                continue
            score = float(np.dot(embedding, stored) / (norm * stored_norm))
            if score >= threshold and (best is None or score > best[1]):
                best = (int(row["id"]), score, dict(row))
        return best

    def check_exact_hash(self, text: str) -> Optional[Dict]:
        text_hash = self._text_hash(text)
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM processed_sources WHERE source_text_hash=? ORDER BY processed_at DESC LIMIT 1",
                (text_hash,),
            ).fetchone()
            return dict(row) if row else None

    def get_recent_processed(self, limit: int = 500, time_window_hours: int = 48) -> List[Dict]:
        since = now_utc() - timedelta(hours=time_window_hours)
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM processed_sources ORDER BY processed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows if parse_dt(r["processed_at"]) and parse_dt(r["processed_at"]) >= since]

    def get_processed_count(self) -> int:
        with self.transaction() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM processed_sources").fetchone()[0])

    # ------------------------------------------------------------------
    # Comments / replies
    # ------------------------------------------------------------------
    def add_comment_replied(
        self,
        our_tweet_id: str,
        comment_id: str,
        comment_author_id: str,
        comment_author_username: str,
        comment_text: str,
        reply_text: str,
        replied_at: Optional[datetime] = None,
    ) -> int:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO comments_replied
                (our_tweet_id, comment_id, comment_author_id, comment_author_username,
                 comment_text, reply_text, replied_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(our_tweet_id, comment_id) DO UPDATE SET
                    reply_text=excluded.reply_text, replied_at=excluded.replied_at
                """,
                (
                    str(our_tweet_id), str(comment_id), str(comment_author_id),
                    comment_author_username, comment_text, reply_text,
                    iso_utc(replied_at),
                ),
            )
            row = conn.execute(
                "SELECT id FROM comments_replied WHERE our_tweet_id=? AND comment_id=?",
                (str(our_tweet_id), str(comment_id)),
            ).fetchone()
            return int(row[0]) if row else 0

    def has_replied_to_comment(self, our_tweet_id: str, comment_id: str) -> bool:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT 1 FROM comments_replied WHERE our_tweet_id=? AND comment_id=?",
                (str(our_tweet_id), str(comment_id)),
            ).fetchone()
            return row is not None

    def get_replies_for_tweet(self, our_tweet_id: str) -> List[Dict]:
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM comments_replied WHERE our_tweet_id=? ORDER BY replied_at DESC",
                (str(our_tweet_id),),
            ).fetchall()
            return [dict(r) for r in rows]

    def count_replies_to_author_today(self, username: str, tz_name: str = "Europe/Madrid") -> int:
        start = start_of_local_day_utc(tz_name=tz_name)
        end = end_of_local_day_utc(tz_name=tz_name)
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT replied_at FROM comments_replied WHERE lower(comment_author_username)=lower(?)",
                (username,),
            ).fetchall()
        return sum(
            1 for r in rows
            if (dt := parse_dt(r["replied_at"])) and start <= dt < end
        )

    def has_pending_reply_to_author(self, username: str) -> bool:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT 1 FROM reply_queue WHERE status IN ('pending','sending') "
                "AND lower(comment_author_username)=lower(?) LIMIT 1",
                (username,),
            ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Media cache
    # ------------------------------------------------------------------
    def cache_media(
        self,
        source_url: str,
        local_path: str,
        media_type: str,
        file_size: Optional[int] = None,
        uploaded_media_id: Optional[str] = None,
        expires_hours: int = 168,
    ) -> int:
        now = now_utc()
        expires_at = now + timedelta(hours=expires_hours)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO media_cache
                (source_media_url, local_path, media_type, file_size,
                 uploaded_media_id, cached_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_media_url) DO UPDATE SET
                    local_path=excluded.local_path,
                    media_type=excluded.media_type,
                    file_size=excluded.file_size,
                    cached_at=excluded.cached_at,
                    expires_at=excluded.expires_at
                """,
                (
                    source_url, local_path, media_type, file_size, uploaded_media_id,
                    iso_utc(now), iso_utc(expires_at),
                ),
            )
            row = conn.execute(
                "SELECT id FROM media_cache WHERE source_media_url=?", (source_url,)
            ).fetchone()
            return int(row[0]) if row else 0

    def get_cached_media(self, source_url: str) -> Optional[Dict]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM media_cache WHERE source_media_url=?", (source_url,)
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        expires = parse_dt(data.get("expires_at"))
        if expires and expires < now_utc():
            return None
        return data

    def delete_cached_media_by_path(self, local_path: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM media_cache WHERE local_path=?", (str(local_path),))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def update_daily_stats(self, **increments) -> None:
        allowed = {"tweets_posted", "tweets_rejected", "replies_sent", "errors_count", "api_calls"}
        fields = [k for k in allowed if increments.get(k)]
        if not fields:
            return
        today = local_date().isoformat()
        cols = ["date"] + fields
        values = [today] + [int(increments[k]) for k in fields]
        updates = ", ".join(f"{f}=bot_stats.{f}+excluded.{f}" for f in fields)
        sql = (
            f"INSERT INTO bot_stats ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)}) "
            f"ON CONFLICT(date) DO UPDATE SET {updates}, updated_at=CURRENT_TIMESTAMP"
        )
        with self.transaction() as conn:
            conn.execute(sql, values)

    def get_stats(self, days: int = 30) -> List[Dict]:
        since = (now_utc() - timedelta(days=days)).date().isoformat()
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM bot_stats WHERE date>=? ORDER BY date DESC", (since,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_today_stats(self) -> Dict:
        today = local_date().isoformat()
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM bot_stats WHERE date=?", (today,)).fetchone()
        data = dict(row) if row else {
            "tweets_posted": 0, "tweets_rejected": 0, "replies_sent": 0,
            "errors_count": 0, "api_calls": 0,
        }
        data["tweets_posted_actual"] = self.get_today_posted_count()
        data["replies_sent_actual"] = self.get_today_replies_sent()
        return data

    # ------------------------------------------------------------------
    # Posting queue
    # ------------------------------------------------------------------
    def enqueue_tweet(
        self,
        text: str,
        media_ids: Optional[List[str]] = None,
        priority: int = 0,
        scheduled_for: Optional[datetime] = None,
        source_tweet_id: Optional[str] = None,
        source_username: Optional[str] = None,
        media_paths: Optional[List[str]] = None,
    ) -> int:
        paths = media_paths if media_paths is not None else media_ids
        now = iso_utc()
        schedule = iso_utc(scheduled_for or now_utc())
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO posting_queue
                (text, media_ids, media_paths, priority, scheduled_for, status,
                 created_at, source_tweet_id, source_username, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    text, json.dumps(paths) if paths else None,
                    json.dumps(paths) if paths else None,
                    priority, schedule, now, source_tweet_id, source_username, now,
                ),
            )
            return int(cur.lastrowid)

    def get_due_tweets(self, limit: int = 10, now: Optional[datetime] = None) -> List[Dict]:
        current = ensure_utc(now or now_utc())
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM posting_queue WHERE status='pending' "
                "ORDER BY priority DESC, scheduled_for ASC LIMIT ?",
                (max(limit * 10, 100),),
            ).fetchall()
        due = []
        for row in rows:
            dt = parse_dt(row["scheduled_for"])
            if dt and dt <= current:
                due.append(dict(row))
                if len(due) >= limit:
                    break
        return due

    def claim_due_tweet(self, now: Optional[datetime] = None) -> Optional[Dict]:
        for row in self.get_due_tweets(limit=1, now=now):
            with self.transaction() as conn:
                cur = conn.execute(
                    "UPDATE posting_queue SET status='posting', last_attempt_at=?, updated_at=? "
                    "WHERE id=? AND status='pending'",
                    (iso_utc(now), iso_utc(now), row["id"]),
                )
                if cur.rowcount:
                    row = dict(row)
                    row["status"] = "posting"
                    return row
        return None

    def update_queue_status(
        self,
        queue_id: int,
        status: str,
        tweet_id: Optional[str] = None,
        error: Optional[str] = None,
        scheduled_for: Optional[datetime] = None,
    ) -> None:
        now = iso_utc()
        fields = ["status=?", "updated_at=?"]
        values: List[Any] = [status, now]
        if tweet_id is not None:
            fields.append("tweet_id=?")
            values.append(str(tweet_id))
        if error is not None:
            fields.append("error_message=?")
            values.append(error)
        if scheduled_for is not None:
            fields.append("scheduled_for=?")
            values.append(iso_utc(scheduled_for))
        if status == "posted":
            fields.append("posted_at=?")
            values.append(now)
        values.append(queue_id)
        with self.transaction() as conn:
            conn.execute(f"UPDATE posting_queue SET {', '.join(fields)} WHERE id=?", values)

    def mark_queue_posted(self, queue_id: int, tweet_id: str, posted_at: Optional[datetime] = None) -> None:
        when = iso_utc(posted_at or now_utc())
        with self.transaction() as conn:
            conn.execute(
                "UPDATE posting_queue SET status='posted', tweet_id=?, posted_at=?, updated_at=? WHERE id=?",
                (str(tweet_id), when, when, queue_id),
            )
            conn.execute(
                "UPDATE posted_tweets SET status='posted', tweet_id=?, posted_at=?, error_message=NULL "
                "WHERE queue_id=?",
                (str(tweet_id), when, queue_id),
            )

    def set_queue_retry(self, queue_id: int, scheduled_for: datetime, error: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE posting_queue SET status='pending', scheduled_for=?, error_message=?, "
                "retries=retries+1, updated_at=? WHERE id=?",
                (iso_utc(scheduled_for), error[:1000], iso_utc(), queue_id),
            )

    def defer_queue_item(self, queue_id: int, scheduled_for: datetime, error: str) -> None:
        """Postpone an item for a global limit (e.g. X daily cap) without
        consuming a retry attempt."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE posting_queue SET status='pending', scheduled_for=?, error_message=?, "
                "retries=0, updated_at=? WHERE id=?",
                (iso_utc(scheduled_for), error[:1000], iso_utc(), queue_id),
            )

    def set_queue_failed(self, queue_id: int, error: str, status: str = "failed") -> None:
        if status not in {"failed", "duplicate"}:
            raise ValueError(f"Statut de file invalide: {status}")
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE posting_queue SET status=?, error_message=?, updated_at=? WHERE id=?",
                (status, error[:1000], iso_utc(), queue_id),
            )
            conn.execute(
                "UPDATE posted_tweets SET status=?, error_message=? WHERE queue_id=?",
                (status, error[:1000], queue_id),
            )

    def mark_queue_uncertain(self, queue_id: int, error: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE posting_queue SET status='uncertain', error_message=?, updated_at=? WHERE id=?",
                (error[:1000], iso_utc(), queue_id),
            )
            conn.execute(
                "UPDATE posted_tweets SET status='uncertain', error_message=? WHERE queue_id=?",
                (error[:1000], queue_id),
            )

    def get_queue_status(self) -> List[Dict]:
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS count FROM posting_queue "
                "WHERE created_at>=? GROUP BY status",
                (iso_utc(now_utc() - timedelta(days=7)),),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_queue_item(self, queue_id: int) -> Optional[Dict]:
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM posting_queue WHERE id=?", (queue_id,)).fetchone()
            return dict(row) if row else None

    def get_pending_queue(self) -> List[Dict]:
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM posting_queue WHERE status='pending' ORDER BY scheduled_for ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_next_scheduled_for(self) -> Optional[datetime]:
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT scheduled_for FROM posting_queue "
                "WHERE status='pending' ORDER BY scheduled_for ASC LIMIT 1"
            ).fetchall()
        if not rows:
            return None
        return parse_dt(rows[0]["scheduled_for"])

    def count_queue_for_local_day(self, day: Optional[date] = None, tz_name: str = "Europe/Madrid") -> int:
        start = start_of_local_day_utc(day, tz_name)
        end = end_of_local_day_utc(day, tz_name)
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT scheduled_for FROM posting_queue WHERE status IN ('pending','posting')"
            ).fetchall()
        return sum(
            1 for r in rows
            if (dt := parse_dt(r["scheduled_for"])) and start <= dt < end
        )

    def get_original_budget_used(self, tz_name: str = "Europe/Madrid") -> int:
        return self.get_today_posted_count(tz_name) + self.count_queue_for_local_day(tz_name=tz_name)

    def recover_in_progress(self) -> Dict[str, int]:
        """Never retry a post automatically after a crash.

        A crash can happen after X accepted a post but before the local commit.
        Retrying would duplicate content, so those rows are marked uncertain.
        """
        with self.transaction() as conn:
            q = conn.execute(
                "UPDATE posting_queue SET status='uncertain', "
                "error_message='processus interrompu; non retenté pour éviter un doublon', updated_at=? "
                "WHERE status='posting'",
                (iso_utc(),),
            ).rowcount
            conn.execute(
                "UPDATE posted_tweets SET status='uncertain', "
                "error_message='processus interrompu; non retenté pour éviter un doublon' "
                "WHERE status='pending' AND queue_id IN "
                "(SELECT id FROM posting_queue WHERE status='uncertain')"
            )
            r = conn.execute(
                "UPDATE reply_queue SET status='failed', "
                "error_message='processus interrompu; non retenté pour éviter un doublon', updated_at=? "
                "WHERE status='sending'",
                (iso_utc(),),
            ).rowcount
        return {"posting": q, "replies": r}

    def cancel_pending(self) -> int:
        with self.transaction() as conn:
            now = iso_utc()
            cur = conn.execute(
                "UPDATE posting_queue SET status='cancelled', updated_at=? WHERE status='pending'",
                (now,),
            )
            conn.execute(
                "UPDATE posted_tweets SET status='cancelled', error_message='file nettoyée par l’utilisateur' "
                "WHERE status='pending' AND queue_id IN "
                "(SELECT id FROM posting_queue WHERE status='cancelled')"
            )
            return cur.rowcount

    def cancel_queue_item(self, queue_id: int, reason: str) -> bool:
        """Cancel one pending item (e.g. replaced by a same-event media version).

        Only a still-pending item can be cancelled; if it was already claimed
        for posting the caller must treat the new content as a duplicate.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT status FROM posting_queue WHERE id=?", (queue_id,)
            ).fetchone()
            if row is None or row["status"] != "pending":
                return False
            conn.execute(
                "UPDATE posting_queue SET status='cancelled', error_message=?, updated_at=? WHERE id=?",
                (reason[:1000], iso_utc(), queue_id),
            )
            conn.execute(
                "UPDATE posted_tweets SET status='cancelled', error_message=? "
                "WHERE queue_id=? AND status='pending'",
                (reason[:1000], queue_id),
            )
            return True

    # ------------------------------------------------------------------
    # Reply queue
    # ------------------------------------------------------------------
    def enqueue_reply(
        self,
        our_tweet_id: str,
        comment_id: str,
        comment_author_id: str,
        comment_author_username: str,
        comment_text: str,
        scheduled_for: datetime,
        reply_text: Optional[str] = None,
    ) -> int:
        now = iso_utc()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO reply_queue
                (our_tweet_id, comment_id, comment_author_id, comment_author_username,
                 comment_text, reply_text, scheduled_for, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    str(our_tweet_id), str(comment_id), str(comment_author_id),
                    comment_author_username, comment_text, reply_text,
                    iso_utc(scheduled_for), now, now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM reply_queue WHERE comment_id=?", (str(comment_id),)
            ).fetchone()
            return int(row[0]) if row else 0

    def has_queued_reply(self, comment_id: str) -> bool:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT 1 FROM reply_queue WHERE comment_id=?", (str(comment_id),)
            ).fetchone()
            return row is not None

    def get_due_replies(self, limit: int = 10, now: Optional[datetime] = None) -> List[Dict]:
        current = ensure_utc(now or now_utc())
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM reply_queue WHERE status='pending' ORDER BY scheduled_for ASC LIMIT ?",
                (max(limit * 10, 100),),
            ).fetchall()
        due = []
        for row in rows:
            dt = parse_dt(row["scheduled_for"])
            if dt and dt <= current:
                due.append(dict(row))
                if len(due) >= limit:
                    break
        return due

    def claim_due_reply(self, now: Optional[datetime] = None) -> Optional[Dict]:
        for row in self.get_due_replies(limit=1, now=now):
            with self.transaction() as conn:
                cur = conn.execute(
                    "UPDATE reply_queue SET status='sending', last_attempt_at=?, updated_at=? "
                    "WHERE id=? AND status='pending'",
                    (iso_utc(now), iso_utc(now), row["id"]),
                )
                if cur.rowcount:
                    row = dict(row)
                    row["status"] = "sending"
                    return row
        return None

    def mark_reply_sent(self, queue_id: int, reply_tweet_id: str, reply_text: str,
                        sent_at: Optional[datetime] = None) -> None:
        when = iso_utc(sent_at or now_utc())
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM reply_queue WHERE id=?", (queue_id,)).fetchone()
            if not row:
                return
            conn.execute(
                "UPDATE reply_queue SET status='sent', sent_tweet_id=?, reply_text=?, "
                "error_message=NULL, updated_at=? WHERE id=?",
                (str(reply_tweet_id), reply_text, when, queue_id),
            )
            conn.execute(
                """
                INSERT INTO comments_replied
                (our_tweet_id, comment_id, comment_author_id, comment_author_username,
                 comment_text, reply_text, replied_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(our_tweet_id, comment_id) DO UPDATE SET
                    reply_text=excluded.reply_text, replied_at=excluded.replied_at
                """,
                (
                    str(row["our_tweet_id"]), str(row["comment_id"]),
                    str(row["comment_author_id"]), row["comment_author_username"],
                    row["comment_text"], reply_text, when,
                ),
            )

    def set_reply_retry(self, queue_id: int, scheduled_for: datetime, error: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE reply_queue SET status='pending', scheduled_for=?, error_message=?, "
                "retries=retries+1, updated_at=? WHERE id=?",
                (iso_utc(scheduled_for), error[:1000], iso_utc(), queue_id),
            )

    def set_reply_failed(self, queue_id: int, error: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE reply_queue SET status='failed', error_message=?, updated_at=? WHERE id=?",
                (error[:1000], iso_utc(), queue_id),
            )

    def get_today_replies_sent(self, tz_name: str = "Europe/Madrid") -> int:
        start = start_of_local_day_utc(tz_name=tz_name)
        end = end_of_local_day_utc(tz_name=tz_name)
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT replied_at FROM comments_replied WHERE replied_at IS NOT NULL"
            ).fetchall()
        return sum(
            1 for r in rows
            if (dt := parse_dt(r["replied_at"])) and start <= dt < end
        )

    def get_pending_replies_count(self) -> int:
        with self.transaction() as conn:
            return int(conn.execute(
                "SELECT COUNT(*) FROM reply_queue WHERE status IN ('pending','sending')"
            ).fetchone()[0])

    # ------------------------------------------------------------------
    # Source checkpoints
    # ------------------------------------------------------------------
    def get_last_tweet_id(self, source_username: str) -> Optional[str]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT last_tweet_id FROM source_checkpoints WHERE source_username=?",
                (source_username,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def get_source_tweet_timestamp(self, source_username: str) -> Optional[str]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT source_tweet_timestamp FROM source_checkpoints WHERE source_username=?",
                (source_username,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def update_last_tweet_id(self, source_username: str, tweet_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO source_checkpoints (source_username, last_tweet_id, last_checked)
                VALUES (?, ?, ?)
                ON CONFLICT(source_username) DO UPDATE SET
                    last_tweet_id=excluded.last_tweet_id,
                    last_checked=excluded.last_checked
                """,
                (source_username, str(tweet_id), iso_utc()),
            )

    def get_last_rss_entry_id(self, source_username: str) -> Optional[str]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT last_rss_entry_id FROM source_checkpoints WHERE source_username=?",
                (source_username,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def update_last_rss_entry_id(self, source_username: str, entry_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO source_checkpoints (source_username, last_rss_entry_id, last_checked)
                VALUES (?, ?, ?)
                ON CONFLICT(source_username) DO UPDATE SET
                    last_rss_entry_id=excluded.last_rss_entry_id,
                    last_checked=excluded.last_checked
                """,
                (source_username, str(entry_id), iso_utc()),
            )

    def get_all_checkpoints(self) -> List[Dict]:
        with self.transaction() as conn:
            rows = conn.execute("SELECT * FROM source_checkpoints ORDER BY source_username").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Reply thread monitoring state
    # ------------------------------------------------------------------
    def mark_tweet_checked(self, tweet_id: str, last_comment_id: Optional[str] = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO reply_monitor_state (tweet_id, last_checked_at, last_seen_comment_id)
                VALUES (?, ?, ?)
                ON CONFLICT(tweet_id) DO UPDATE SET
                    last_checked_at=excluded.last_checked_at,
                    last_seen_comment_id=COALESCE(excluded.last_seen_comment_id, reply_monitor_state.last_seen_comment_id)
                """,
                (str(tweet_id), iso_utc(), last_comment_id),
            )

    def get_last_checked_at(self, tweet_id: str) -> Optional[datetime]:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT last_checked_at FROM reply_monitor_state WHERE tweet_id=?",
                (str(tweet_id),),
            ).fetchone()
        return parse_dt(row[0]) if row else None

    def get_tweets_to_check(self, limit: int = 8, max_age_hours: int = 48) -> List[Dict]:
        tweets = self.get_recent_posted_tweets(limit=max(limit * 5, 30), max_age_hours=max_age_hours)
        for tweet in tweets:
            tweet["last_checked_at"] = self.get_last_checked_at(tweet.get("tweet_id") or "")
        tweets.sort(key=lambda t: (t.get("last_checked_at") is not None,
                                   t.get("last_checked_at") or datetime.min.replace(tzinfo=now_utc().tzinfo)))
        return tweets[:limit]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def cleanup_old_data(self, days: int = 30) -> None:
        cutoff = iso_utc(now_utc() - timedelta(days=days))
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM posted_tweets WHERE posted_at IS NOT NULL AND posted_at<? "
                "AND status IN ('posted','failed','uncertain')", (cutoff,)
            )
            conn.execute("DELETE FROM processed_sources WHERE processed_at<?", (cutoff,))
            conn.execute("DELETE FROM comments_replied WHERE replied_at<?", (cutoff,))
            conn.execute("DELETE FROM media_cache WHERE expires_at<?", (iso_utc(),))
            conn.execute(
                "DELETE FROM posting_queue WHERE created_at<? AND status IN ('posted','failed','cancelled','uncertain')",
                (cutoff,),
            )
            conn.execute(
                "DELETE FROM reply_queue WHERE created_at<? AND status IN ('sent','failed','cancelled')",
                (cutoff,),
            )
            conn.execute("DELETE FROM bot_stats WHERE date<?", (cutoff[:10],))

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
