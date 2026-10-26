"""Strict FC Barcelona content filter.

The original filter considered any occurrence of "barcelona", "liga" or
"champions" as Barça. This version uses strong identity/player/staff signals,
football context, source provenance and explicit false-positive patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger

from .config import get_config
from .timeutil import clean_tweet_text, normalize_for_match

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
except Exception:  # pragma: no cover - langdetect is optional at runtime
    detect = None


@dataclass
class FilterResult:
    is_barca: bool
    matched_keywords: List[str] = field(default_factory=list)
    matched_categories: Dict[str, List[str]] = field(default_factory=dict)
    reason: str = ""
    language: str = "unknown"
    entities: Dict[str, List[str]] = field(default_factory=dict)


class BarcaFilter:
    # Terms that unambiguously identify the club or its identity.
    _IDENTITY = [
        "fc barcelona", "fc barcelone", "f c barcelona", "barca", "barça",
        "blaugrana", "blaugranes", "cule", "cule", "culers", "camp nou",
        "spotify camp nou", "la masia", "masia", "estadi olimpic",
        "penya barcelonista", "barcauniversal", "fcblive", "forcadelsbarca",
    ]
    _FOOTBALL_CONTEXT = [
        "football", "foot", "soccer", "match", "game", "club", "team", "equipe",
        "transfert", "transfer", "mercato", "liga", "ligue", "champions", "copa",
        "supercopa", "supercoupe", "clasico", "clásico", "but", "goal", "entraineur",
        "coach", "joueur", "player", "saison", "season", "stade", "stadium",
        "supporter", "supporters", "fan", "fans", "victoire", "defaite", "nul",
        "blessure", "contrat", "contract", "signe", "signature", "officiel",
        "official", "vs", "contre", "derby", "titre", "title", "trophy", "trophee",
        "buteur", "scorer", "passeur", "assist", "milieu", "defenseur", "attaquant",
        "keeper", "goalkeeper", "gardien", "president", "presidente", "entrainement",
        "training", "vestiaire", "squad", "lineup", "composition", "banc", "bench",
        "lateral", "latéral", "defense", "défense", "attaque", "ailier", "piston",
        "central", "avant centre", "effectif", "blessure", "mercato", "renfort",
        "recrue", "prolongation", "contrat", "selection", "sélection", "coach",
    ]
    _OTHER_CLUB = [
        "real madrid", "atletico madrid", "atlético madrid", "psg", "paris saint germain",
        "bayern", "bayern munich", "borussia dortmund", "juventus", "inter milan",
        "ac milan", "napoli", "roma", "lazio", "manchester united", "manchester city",
        "liverpool", "chelsea", "arsenal", "tottenham", "sevilla", "valencia",
        "betis", "villarreal", "athletic bilbao", "girona", "osasuna", "getafe",
    ]
    _FALSE_POSITIVE = [
        "barcelona sc", "barcelona sporting", "barcelona city", "city of barcelona",
        "barcelona open", "barcelona airport", "aeroport de barcelone",
        "barcelona venezuela", "barcelona, venezuela", "barcelona beach",
        "barcelona metro", "barcelona hotel", "barcelona tourism", "barcelona marathon",
        "barcelona tennis", "barcelona eixample", "barcelona gothic",
        "barcelona province", "province de barcelone", "barcelone espagne",
        "barcelona spain", "barcelona, spain", "barcelona weather", "meteo barcelone",
    ]

    def __init__(self, config=None):
        self.config = config or get_config()
        self.keywords = self.config.barca_keywords
        self.blacklist = self.config.blacklist_keywords.keywords
        self.sources = {s.lower().lstrip("@") for s in self.config.sources.accounts}
        self._build()

    @staticmethod
    def _norm_keyword(keyword: str) -> str:
        normalized = normalize_for_match(keyword)
        return r"\s+".join(re.escape(part) for part in normalized.split())

    def _compile(self, keywords: List[str]) -> List[Tuple[str, re.Pattern]]:
        out = []
        seen = set()
        for keyword in keywords:
            normalized = normalize_for_match(keyword)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            out.append((keyword, re.compile(rf"\b{self._norm_keyword(keyword)}\b", re.IGNORECASE)))
        return out

    def _build(self) -> None:
        club = [k for k in self.keywords.club if normalize_for_match(k) not in {"barcelona", "catalunya", "catalogne"}]
        self.identity_patterns = self._compile(self._IDENTITY + club)
        self.player_patterns = self._compile(self.keywords.players)
        self.staff_patterns = self._compile(self.keywords.staff)
        self.competition_patterns = self._compile(self.keywords.competitions)
        self.context_patterns = self._compile(self._FOOTBALL_CONTEXT)
        self.other_club_patterns = self._compile(self._OTHER_CLUB)
        self.blacklist_patterns = self._compile(self.blacklist)
        self.false_positive_patterns = self._compile(self._FALSE_POSITIVE)
        self.bare_barcelona = re.compile(r"\bbarcelona\b", re.IGNORECASE)
        self.bare_barcelone = re.compile(r"\bbarcelone\b", re.IGNORECASE)

    @staticmethod
    def _hits(patterns: List[Tuple[str, re.Pattern]], text: str) -> List[str]:
        return [keyword for keyword, pattern in patterns if pattern.search(text)]

    def _language(self, text: str, source_lang: Optional[str] = None) -> str:
        cleaned = clean_tweet_text(text)
        if len(cleaned) >= 12 and detect is not None:
            try:
                return detect(cleaned)
            except Exception:
                pass
        return (source_lang or "unknown").lower()

    def is_barca_related(
        self,
        text: str,
        source_username: Optional[str] = None,
        source_is_barca: Optional[bool] = None,
        language: Optional[str] = None,
    ) -> FilterResult:
        original = clean_tweet_text(text)
        normalized = normalize_for_match(original)
        if not normalized:
            return FilterResult(False, reason="texte vide", language=self._language(original, language))

        identity = self._hits(self.identity_patterns, normalized)
        players = self._hits(self.player_patterns, normalized)
        staff = self._hits(self.staff_patterns, normalized)
        competitions = self._hits(self.competition_patterns, normalized)
        contexts = self._hits(self.context_patterns, normalized)
        other = self._hits(self.other_club_patterns, normalized)
        black = self._hits(self.blacklist_patterns, normalized)
        false_pos = self._hits(self.false_positive_patterns, normalized)
        bare_barca = bool(self.bare_barcelona.search(normalized) or self.bare_barcelone.search(normalized))

        source_is_barca = source_is_barca if source_is_barca is not None else bool(
            source_username and source_username.lower().lstrip("@") in self.sources
        )
        strong = identity + players + staff + (["barcelona"] if bare_barca and contexts else [])
        football_context = bool(contexts or competitions)

        categories = {
            "identity": identity,
            "players": players,
            "staff": staff,
            "competitions": competitions,
            "context": contexts,
            "other_clubs": other,
            "blacklist": black,
            "false_positive": false_pos,
        }
        matched = list(dict.fromkeys(strong + competitions + contexts))

        # A false-positive city/non-Barça use of "Barcelona" is rejected,
        # unless the club identity is explicit (FC Barcelona, Barça, ...).
        if false_pos and not identity:
            return FilterResult(False, matched, categories,
                                f"faux positif géographique/non-Barça: {false_pos[:3]}",
                                self._language(original, language))

        # Other clubs alone are not Barça news.
        if (other or black) and not strong:
            return FilterResult(False, matched, categories,
                                "autre club sans signal Barça explicite",
                                self._language(original, language))

        if strong:
            return FilterResult(True, matched, categories,
                                f"signal Barça fort: {(identity + players + staff)[:4]}",
                                self._language(original, language))

        # "Barcelona" plus football context is acceptable, but only with a
        # football match/transfer context.
        if bare_barca and football_context:
            return FilterResult(True, matched or ["barcelona"], categories,
                                "Barcelona + contexte football",
                                self._language(original, language))

        # A trusted source is not enough on its own: source accounts can also
        # publish live-match chatter, rivals or off-topic content. Only allow
        # an implicit Barça signal when the text still contains football
        # context; never accept an arbitrary sentence just because of its
        # author.
        if source_is_barca and not (other or black) and football_context:
            return FilterResult(True, matched, categories,
                                "source Barça + contexte football",
                                self._language(original, language))

        if competitions and not bare_barca and not strong:
            return FilterResult(False, matched, categories,
                                "compétition seule, sans Barça",
                                self._language(original, language))

        return FilterResult(False, matched, categories,
                            "aucun signal Barça suffisant",
                            self._language(original, language))

    def get_filter_stats(self, text: str) -> Dict:
        result = self.is_barca_related(text, source_is_barca=False)
        return {
            "text_preview": text[:180],
            "is_barca": result.is_barca,
            "matched_keywords": ", ".join(result.matched_keywords[:12]),
            "categories": {k: v for k, v in result.matched_categories.items() if v},
            "reason": result.reason,
            "language": result.language,
        }

    # ------------------------------------------------------------------
    # Comment relevance (used by ReplyManager)
    # ------------------------------------------------------------------
    def comment_relevance(
        self,
        comment_text: str,
        our_text: str,
        comment_author: str = "",
    ) -> float:
        comment = clean_tweet_text(comment_text)
        normalized = normalize_for_match(comment)
        if len(normalized) < max(2, int(self.config.replies.min_comment_length)):
            return 0.0
        score = 0.0
        identity = self._hits(self.identity_patterns, normalized)
        players = self._hits(self.player_patterns, normalized)
        staff = self._hits(self.staff_patterns, normalized)
        competitions = self._hits(self.competition_patterns, normalized)
        contexts = self._hits(self.context_patterns, normalized)
        if identity or players or staff:
            score += 0.55
        if competitions or contexts:
            score += 0.20
        roles = ("lateral", "defenseur", "defense", "attaquant", "milieu", "gardien",
                 "buteur", "ailier", "piston", "central", "avant centre", "coach",
                 "entraineur", "president", "recrue", "renfort", "effectif")
        if any(role in normalized for role in roles):
            score += 0.20
        if len(normalized) >= 15:
            score += 0.10
        if len(normalized) >= 40:
            score += 0.10
        # Shared vocabulary with our post is a useful relevance signal.
        our_normalized = normalize_for_match(our_text)
        common = {
            token for token in re.findall(r"[a-z0-9]{4,}", normalized)
            if token in set(re.findall(r"[a-z0-9]{4,}", our_normalized))
        }
        if len(common) >= 2:
            score += 0.30
        elif len(common) == 1:
            score += 0.15
        if "?" in comment:
            score += 0.10
        if any(word in normalized for word in ("merci", "bravo", "visca", "forca", "allez",
                                                "d accord", "accord", "exact", "oui", "non",
                                                "pense", "crois", "selon moi", "incroyable")):
            score += 0.10
        # Very short generic reactions are not worth an automated reply.
        if len(normalized.split()) <= 2 and not (identity or players or staff):
            score -= 0.25
        # Spam signals.
        spam = ("follow", "abonne", "dm", "crypto", "onlyfans", "promo", "giveaway",
                "concours", "telegram", "whatsapp", "lien en bio", "link in bio")
        if any(word in normalized for word in spam):
            score -= 0.7
        return max(0.0, min(1.0, score))
