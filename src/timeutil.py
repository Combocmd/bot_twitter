"""Timezone, Twitter timestamp and 280-character helpers.

All timestamps stored by the bot are timezone-aware ISO strings in UTC.
The only local timezone used is the configured Europe/Madrid scheduling zone.
"""
from __future__ import annotations

import html
import re
import unicodedata
from datetime import date, datetime, time, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = timezone.utc
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_TCO_RE = re.compile(r"https?://t\.co/\w+", re.IGNORECASE)
_SPACE_RE = re.compile(r"[ \t\f\v]+")
_MENTION_PREFIX_RE = re.compile(r"^(?:@\w+[\s,:-]*)+", re.UNICODE)
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def get_zone(tz_name: str = "Europe/Madrid"):
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        # Windows without tzdata: fall back to a fixed CET offset. The
        # scheduling code still works, with a one-hour DST inaccuracy.
        return timezone(timedelta(hours=1), name="CET")


def now_utc() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_dt(value) -> Optional[datetime]:
    """Parse a datetime, SQLite timestamp or Twitter date string as UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    text = str(value).strip()
    if not text:
        return None
    # Twitter's "Sun Oct 25 10:57:44 +0000 2026"
    if re.match(r"^[A-Z][a-z]{2} [A-Z][a-z]{2} ", text):
        try:
            return ensure_utc(parsedate_to_datetime(text))
        except (TypeError, ValueError, OverflowError):
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return ensure_utc(parsed)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return ensure_utc(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


def iso_utc(value: Optional[datetime] = None) -> str:
    return ensure_utc(value or now_utc()).isoformat(timespec="seconds")


def local_now(tz_name: str = "Europe/Madrid") -> datetime:
    return now_utc().astimezone(get_zone(tz_name))


def local_date(tz_name: str = "Europe/Madrid") -> date:
    return local_now(tz_name).date()


def start_of_local_day_utc(day: Optional[date] = None, tz_name: str = "Europe/Madrid") -> datetime:
    zone = get_zone(tz_name)
    day = day or local_now(tz_name).date()
    return datetime.combine(day, time(0, 0), tzinfo=zone).astimezone(UTC)


def end_of_local_day_utc(day: Optional[date] = None, tz_name: str = "Europe/Madrid") -> datetime:
    return start_of_local_day_utc(day, tz_name) + timedelta(days=1)


def local_day_for_utc(value: datetime, tz_name: str = "Europe/Madrid") -> date:
    return ensure_utc(value).astimezone(get_zone(tz_name)).date()


def next_active_start(now: Optional[datetime] = None, start_hour: int = 8,
                      tz_name: str = "Europe/Madrid") -> datetime:
    """Return the next UTC datetime at the configured active-hour start."""
    zone = get_zone(tz_name)
    local = (now or now_utc()).astimezone(zone)
    candidate = local.replace(hour=start_hour, minute=0, second=0, microsecond=0)
    if candidate <= local:
        candidate += timedelta(days=1)
    return candidate.astimezone(UTC)


def is_active_hours(now: Optional[datetime] = None, start_hour: int = 8,
                    end_hour: int = 23, tz_name: str = "Europe/Madrid") -> bool:
    hour = (now or now_utc()).astimezone(get_zone(tz_name)).hour
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    # Overnight window (not used by default, but supported)
    return hour >= start_hour or hour < end_hour


def clean_tweet_text(text: str, strip_urls: bool = True) -> str:
    """Clean source or generated text while preserving meaningful content."""
    if not text:
        return ""
    text = html.unescape(str(text))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH_RE.sub("", text)
    if strip_urls:
        text = _TCO_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
    text = _MENTION_PREFIX_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove surrounding whitespace on each line, then overall.
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    return text


def remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text).strip()


def weighted_length(text: str) -> int:
    """Approximate X/Twitter weighted character count.

    URLs count as 23. Latin characters count as 1, CJK and emoji as 2.
    """
    if not text:
        return 0
    urls = _URL_RE.findall(text)
    length = 0
    if urls:
        text = _URL_RE.sub("", text)
        length += 23 * len(urls)
    for char in text:
        code = ord(char)
        if 0x0000 <= code <= 0x10FF:
            length += 1
        elif 0x1100 <= code <= 0x115F:
            length += 2
        elif 0x1160 <= code <= 0x200D:
            length += 1
        elif 0x200E <= code <= 0x2010:
            length += 1
        elif 0x2011 <= code <= 0x201F:
            length += 1
        elif 0x202A <= code <= 0x202E:
            length += 1
        elif 0x2030 <= code <= 0x203F:
            length += 1
        elif 0x2040 <= code <= 0x205F:
            length += 1
        elif 0x2060 <= code <= 0x206F:
            length += 1
        elif code >= 0x2100:
            length += 2
        else:
            length += 1
    return length


def truncate_weighted(text: str, max_chars: int = 280) -> str:
    """Truncate text without exceeding the weighted limit."""
    text = (text or "").strip()
    if weighted_length(text) <= max_chars:
        return text
    # Prefer cutting at a sentence/word boundary near the limit.
    for cut in range(len(text), 0, -1):
        suffix = "…" if cut < len(text) else ""
        candidate = text[:cut].rstrip() + suffix
        if weighted_length(candidate) <= max_chars:
            # Avoid cutting in the middle of a word when a nearby space exists.
            if cut < len(text) and " " in text[:cut][-30:]:
                candidate = text[:cut].rsplit(" ", 1)[0].rstrip() + "…"
                if weighted_length(candidate) > max_chars:
                    candidate = text[:cut].rstrip() + "…"
            return candidate
    return text[:max_chars]


def normalize_for_match(text: str) -> str:
    text = clean_tweet_text(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def truncate_preview(text: str, length: int = 100) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= length else text[: length - 1].rstrip() + "…"
