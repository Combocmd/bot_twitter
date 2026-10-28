"""Optimized Barça content filter with English support."""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from .config import get_config
from .timeutil import clean_tweet_text, normalize_for_match


@dataclass
class FilterResult:
    is_barca: bool
    matched_keywords: List[str] = field(default_factory=list)
    reason: str = ""
    language: str = "unknown"


class BarcaFilter:
    # Optimized patterns for fast matching
    BARCA_PATTERNS = [
        r"\b(fc\s*barcelona|barça|blaugrana|culé|camp nou|la masia|spotify camp nou|montjuic|fcbarcelona|barcelona fc)\b",
        r"\b(yamal|pedri|gavi|de jong|lewandowski|raphinha|koundé|torres|fati|araujo|cubarsi|balde|ter stegen|fermin|olmo|szczesny|pena|garcia|christensen|frenkie de jong|marc bernal|bernal|marc casado|casado|eric garcia|eric garcía|inaki pena|iñaki peña|dani olmo|pau cubarsi|ronald araujo|ansu fati|ferran torres|hector fort|fort|kounde|szczesny|lewandowski|raphinha|balde|terstegen|fermin lopez|lamine yamal)\b",
        r"\b(flick|hansi flick|laporta|joan laporta|deco)\b",
        r"\b(liga|la liga|laliga|champions|champions league|ucl|copa del rey|supercopa|clasico|clásico|el clasico|el clásico|derby|derby barcelona|derby madrid|el derbi)\b",
        r"\b(barcelona|barça|blaugrana|culé|culers|blaugranes|masia|la masia|camp nou|spotify camp nou|montjuic|estadi olimpic|estadi)\b",
    ]

    def __init__(self, config=None):
        self.config = config or get_config()
        self.sources = {s.lower().lstrip("@") for s in self.config.sources.accounts}
        self.barca_pattern = re.compile("|".join(self.BARCA_PATTERNS), re.IGNORECASE)
        self.football_context_words = {
            "match", "game", "transfer", "mercado", "liga", "champions",
            "goal", "player", "team", "club", "coach", "training", "substitution",
            "score", "win", "loss", "draw", "assist", "hat-trick", "penalty"
        }

    def is_barca_related(
        self,
        text: str,
        source_username: Optional[str] = None,
        language: Optional[str] = None
    ) -> FilterResult:
        """Fast Barça content detection with English support."""
        cleaned = clean_tweet_text(text)
        normalized = normalize_for_match(cleaned)

        if not normalized:
            return FilterResult(False, reason="texte vide", language=language or "unknown")

        # Check for explicit Barça signals (fast regex)
        matches = self._find_matches(normalized)
        if matches:
            return FilterResult(True, matched_keywords=matches, reason="signal Barça fort", language=language or "en")

        # Check if source is trusted
        source_is_barca = bool(source_username and source_username.lower().lstrip("@") in self.sources)

        # English content from trusted sources is accepted if it has football context
        if source_is_barca and self._has_football_context(normalized):
            return FilterResult(True, matched_keywords=["source"], reason="source Barça + contexte football", language=language or "en")

        return FilterResult(False, reason="aucun signal Barça suffisant", language=language or "unknown")

    def _find_matches(self, text: str) -> List[str]:
        """Find all matching Barça keywords in text."""
        matches = []
        for pattern in self.BARCA_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append(match.group(0).lower())
        return list(dict.fromkeys(matches))  # Deduplicate preserving order

    def _has_football_context(self, text: str) -> bool:
        """Fast football context detection."""
        lower_text = text.lower()
        return any(word in lower_text for word in self.football_context_words)

    def get_filter_stats(self, text: str) -> Dict:
        result = self.is_barca_related(text, source_username=None, language="en")
        return {
            "text_preview": text[:180],
            "is_barca": result.is_barca,
            "matched_keywords": ", ".join(result.matched_keywords[:12]),
            "reason": result.reason,
            "language": result.language,
        }