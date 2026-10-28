"""Optimized Barça content filter with English support and LLM-based quality scoring."""

import re
import ollama
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
    importance_score: float = 0.0  # 0-1, higher = more important


class BarcaFilter:
    # Optimized patterns for fast matching
    BARCA_PATTERNS = [
        r"\b(fc\s*barcelona|barça|blaugrana|culé|camp nou|la masia|spotify camp nou|montjuic|fcbarcelona|barcelona fc)\b",
        r"\b(yamal|pedri|gavi|de jong|lewandowski|raphinha|koundé|torres|fati|araujo|cubarsi|balde|ter stegen|fermin|olmo|szczesny|pena|garcia|christensen|frenkie de jong|marc bernal|bernal|marc casado|casado|eric garcia|eric garcía|inaki pena|iñaki peña|dani olmo|pau cubarsi|ronald araujo|ansu fati|ferran torres|hector fort|fort|kounde|szczesny|lewandowski|raphinha|balde|terstegen|fermin lopez|lamine yamal)\b",
        r"\b(flick|hansi flick|laporta|joan laporta|deco)\b",
        r"\b(liga|la liga|laliga|champions|champions league|ucl|copa del rey|supercopa|clasico|clásico|el clasico|el clásico|derby|derby barcelona|derby madrid|el derbi)\b",
        r"\b(barcelona|barça|blaugrana|culé|culers|blaugranes|masia|la masia|camp nou|spotify camp nou|montjuic|estadi olimpic|estadi)\b",
    ]

    # High-importance keywords (boost score)
    HIGH_IMPORTANCE = {
        "breaking", "just in", "urgent", "official", "exclusive", "confirmed",
        "signing", "transfer", "deal agreed", "medical", "contract", "extension",
        "injury", "injured", "sidelined", "out for", "recovery", "return",
        "lineup", "starting xi", "bench", "substitute", "captain",
        "debut", "first start", "milestone", "record", "hat-trick", "brace",
        "winner", "equaliser", "penalty", "red card", "yellow card",
        "suspended", "ban", "appeal", "disciplinary",
        "announcement", "press conference", "statement", "revealed",
        "here we go", "done deal", "agreed", "completed", "finalized",
        "loan", "buy option", "release clause", "bid", "offer", "negotiations",
        "president", "director", "sporting director", "board", "election"
    }

    # Low-importance keywords (reduce score)
    LOW_IMPORTANCE = {
        "opinion", "think", "believe", "feel", "hope", "wish", "dream",
        "meme", "funny", "lol", "lmao", "haha", "joke",
        "poll", "vote", "who would you", "rate", "rating",
        "reaction", "celebration", "emotional", "tears", "crying",
        "highlight", "skill", "dribble", "nutmeg", "assist",
        "what a", "magic",
        "visca", "forca", "catalunya", "independencia", "politics",
        "merch", "jersey", "kit", "store", "buy now", "discount",
        "youtube", "video", "watch", "stream", "live", "twitch",
        "podcast", "interview", "q&a", "ask me"
    }

    def __init__(self, config=None):
        self.config = config or get_config()
        self.sources = {s.lower().lstrip("@") for s in self.config.sources.accounts}
        self.barca_pattern = re.compile("|".join(self.BARCA_PATTERNS), re.IGNORECASE)
        self.football_context_words = {
            "match", "game", "transfer", "mercado", "liga", "champions",
            "goal", "player", "team", "club", "coach", "training", "substitution",
            "score", "win", "loss", "draw", "assist", "hat-trick", "penalty"
        }
        # Ollama client for LLM analysis
        self._ollama_client = ollama.Client(host=self.config.ollama.host)
        self._ollama_model = self.config.ollama.model

    def is_barca_related(
        self,
        text: str,
        source_username: Optional[str] = None,
        language: Optional[str] = None
    ) -> FilterResult:
        """Fast Barça content detection with English support and importance scoring."""
        cleaned = clean_tweet_text(text)
        normalized = normalize_for_match(cleaned)

        if not normalized:
            return FilterResult(False, reason="texte vide", language=language or "unknown")

        # Check for explicit Barça signals (fast regex)
        matches = self._find_matches(normalized)
        if matches:
            importance = self._calculate_importance(text, normalized, matches)
            # Apply importance threshold to filter out filler content
            if importance < 0.4:
                return FilterResult(False, reason=f"signal Barça mais faible importance ({importance:.2f})", language=language or "en", importance_score=importance)
            return FilterResult(True, matched_keywords=matches, reason="signal Barça fort", language=language or "en", importance_score=importance)

        # Check if source is trusted
        source_is_barca = bool(source_username and source_username.lower().lstrip("@") in self.sources)

        # English content from trusted sources is accepted if it has football context
        if source_is_barca and self._has_football_context(normalized):
            # For trusted sources, use LLM to assess importance
            importance = self._assess_importance_with_llm(text, source_username)
            # Only accept if importance is above threshold
            if importance >= 0.4:
                return FilterResult(True, matched_keywords=["source"], reason="source Barça + contexte football", language=language or "en", importance_score=importance)
            else:
                return FilterResult(False, reason=f"source Barça mais faible importance ({importance:.2f})", language=language or "en", importance_score=importance)

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

    def _calculate_importance(self, original_text: str, normalized_text: str, matches: List[str]) -> float:
        """Calculate importance score based on keyword analysis."""
        score = 0.5  # Base score

        lower_text = original_text.lower()

        # Boost for high-importance keywords
        high_hits = sum(1 for kw in self.HIGH_IMPORTANCE if kw in lower_text)
        score += min(high_hits * 0.15, 0.4)

        # Penalty for low-importance keywords
        low_hits = sum(1 for kw in self.LOW_IMPORTANCE if kw in lower_text)
        score -= min(low_hits * 0.15, 0.4)

        # Boost for multiple Barça keywords
        if len(matches) > 1:
            score += min(len(matches) * 0.05, 0.2)

        # Boost for alert words at start
        alert_words = {"breaking", "just in", "urgent", "official", "exclusive", "confirmed"}
        first_words = lower_text.split()[:3]
        if any(w in alert_words for w in first_words):
            score += 0.2

        return max(0.0, min(1.0, score))

    def _assess_importance_with_llm(self, text: str, source_username: str) -> float:
        """Use LLM to assess the importance/quality of a tweet."""
        try:
            prompt = (
                f"Analyze this FC Barcelona tweet from @{source_username} and rate its NEWS IMPORTANCE from 0.0 to 1.0.\n\n"
                f"Tweet: {text}\n\n"
                "IMPORTANT (high score 0.7-1.0): Official announcements, confirmed transfers, injuries, lineups, "
                "contract news, disciplinary actions, major milestones, exclusive reports.\n\n"
                "NOT IMPORTANT (low score 0.0-0.3): Opinions, reactions, memes, engagement bait, polls, "
                "generic comments, highlights/skills, merch promotion, political content.\n\n"
                "MODERATE (0.4-0.6): Rumors from reliable sources, pre-match analysis, post-match stats.\n\n"
                "Respond ONLY with a single number between 0.0 and 1.0 (e.g., 0.75)."
            )

            response = self._ollama_client.chat(
                model=self._ollama_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_predict": 10},
            )
            content = str(response.get("message", {}).get("content", "")).strip()
            # Extract first number
            import re
            nums = re.findall(r"0?\.\d+|1\.0|0|1", content)
            if nums:
                score = float(nums[0])
                return max(0.0, min(1.0, score))
        except Exception as e:
            # Silently fall back to keyword-based scoring
            pass

        # Fallback to keyword-based scoring
        return self._calculate_importance(text, text, self._find_matches(normalize_for_match(text)))

    def get_filter_stats(self, text: str) -> Dict:
        result = self.is_barca_related(text, source_username=None, language="en")
        return {
            "text_preview": text[:180],
            "is_barca": result.is_barca,
            "matched_keywords": ", ".join(result.matched_keywords[:12]),
            "reason": result.reason,
            "language": result.language,
            "importance_score": round(result.importance_score, 2),
        }

    def apply_vision_boost(self, filter_result: FilterResult, vision_boost: float, vision_players: List[str], vision_scene: str, vision_is_barca: bool = False, vision_confidence: float = 0.5) -> FilterResult:
        """Apply vision analysis boost to filter result with complementary logic."""
        new_importance = max(0.0, min(1.0, filter_result.importance_score + vision_boost))
        
        # COMPLEMENTARY LOGIC: Strong signal wins
        text_strong_accept = filter_result.importance_score >= 0.65
        text_strong_reject = filter_result.importance_score <= 0.25
        vision_strong_accept = vision_is_barca and vision_confidence >= 0.7 and vision_boost >= 0.3
        vision_strong_reject = not vision_is_barca and vision_confidence >= 0.7 and vision_boost <= -0.1
        
        # If text strongly accepts, keep it even if vision disagrees (unless vision very confidently rejects)
        if text_strong_accept and not (vision_strong_reject and vision_confidence >= 0.85):
            is_accepted = True
            decision = "texte fort"
        # If vision strongly accepts, rescue even if text was weak/rejected (high confidence vision wins)
        elif vision_strong_accept:
            is_accepted = True
            decision = "vision fort"
        # Normal threshold-based decision
        else:
            is_accepted = new_importance >= 0.4
            decision = "seuil normal"
        
        # Build enhanced reason
        vision_info = []
        if vision_players:
            vision_info.append(f"joueurs: {', '.join(vision_players[:3])}")
        if vision_scene and vision_scene != "unknown":
            vision_info.append(f"scène: {vision_scene}")
        if vision_is_barca:
            vision_info.append("barça:oui")
        else:
            vision_info.append("barça:non")
        vision_info.append(f"conf:{vision_confidence:.2f}")
        
        new_reason = f"{filter_result.reason} | vision: {', '.join(vision_info)} | décision: {decision}"
        
        if not is_accepted:
            new_reason += f" | score final: {new_importance:.2f}"
        
        return FilterResult(
            is_accepted,
            matched_keywords=filter_result.matched_keywords,
            reason=new_reason,
            language=filter_result.language,
            importance_score=new_importance
        )