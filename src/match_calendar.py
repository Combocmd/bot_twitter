"""Cached FC Barcelona match calendar used by the editorial filter."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

from .timeutil import get_zone, now_utc, parse_dt
from .x_session import PROJECT_ROOT


class MatchCalendar:
    def __init__(self, config=None):
        self.config = config
        cfg = getattr(config, "match_calendar", None)
        self.enabled = bool(getattr(cfg, "enabled", True))
        self.refresh_hours = int(getattr(cfg, "refresh_hours", 6))
        cache = Path(getattr(cfg, "cache_file", "data/barca_matches.json"))
        self.cache_path = cache if cache.is_absolute() else PROJECT_ROOT / cache
        self.before_minutes = int(getattr(cfg, "before_minutes", 15))
        self.after_minutes = int(getattr(cfg, "after_minutes", 45))
        self.timezone = str(getattr(cfg, "timezone", "Europe/Madrid"))
        self.matches: List[Dict] = []
        self.last_refresh: datetime | None = None
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.matches = data.get("matches", []) if isinstance(data, dict) else []
            raw = data.get("updated_at") if isinstance(data, dict) else None
            self.last_refresh = parse_dt(raw)
        except (OSError, ValueError, TypeError):
            self.matches = []

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at": now_utc().isoformat(), "matches": self.matches}
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.cache_path)
        self.last_refresh = now_utc()

    def refresh_if_needed(self, force: bool = False) -> bool:
        if not self.enabled:
            return False
        if not force and self.last_refresh and now_utc() - self.last_refresh < timedelta(hours=self.refresh_hours):
            return False
        # ESPN exposes the Barcelona team schedule without credentials.
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/teams/83/schedule?limit=30"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            events = response.json().get("events", [])
            matches = []
            for event in events:
                start = parse_dt(event.get("date"))
                if not start:
                    continue
                competition = (event.get("competitions") or [{}])[0]
                competitors = competition.get("competitors") or []
                names = [str((item.get("team") or {}).get("displayName", "")) for item in competitors]
                home = next((n for n, item in zip(names, competitors) if item.get("home")), names[0] if names else "")
                away = next((n for n, item in zip(names, competitors) if not item.get("home")), names[1] if len(names) > 1 else "")
                opponent = away if "barcelona" in home.lower() else home
                matches.append({
                    "id": str(event.get("id") or start.isoformat()),
                    "start": start.isoformat(),
                    "name": event.get("name", f"{home} - {away}"),
                    "opponent": opponent,
                    "status": ((event.get("status") or {}).get("type") or {}).get("name", ""),
                })
            self.matches = matches
            self._save_cache()
            logger.info("Calendrier Barça actualisé: {} match(s)", len(matches))
            return True
        except Exception as exc:
            logger.warning("Calendrier indisponible, cache conservé: {}", exc)
            return False

    def match_in_progress(self, when: datetime | None = None) -> bool:
        if not self.enabled:
            return False
        current = (when or now_utc()).astimezone(get_zone(self.timezone))
        for match in self.matches:
            start = parse_dt(match.get("start"))
            if not start:
                continue
            local_start = start.astimezone(get_zone(self.timezone))
            begin = local_start - timedelta(minutes=self.before_minutes)
            end = local_start + timedelta(minutes=135 + self.after_minutes)
            if begin <= current <= end:
                return True
        return False

    def status(self) -> Dict:
        return {
            "enabled": self.enabled,
            "matches": len(self.matches),
            "cache": str(self.cache_path),
            "match_in_progress": self.match_in_progress(),
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
        }
