"""Semantic and lexical deduplication.

Sentence-transformers is used when available. If the model cannot be loaded
(offline first run, incompatible torch build, ...), a token/sequence similarity
fallback keeps the bot functional.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from .config import get_config
from .database import Database
from .timeutil import normalize_for_match

try:
    import ollama
except Exception:  # pragma: no cover
    ollama = None

_MODEL = None
_MODEL_FAILED = False

_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
    "are", "was", "were", "be", "been", "this", "that", "it", "as", "at", "by",
    "from", "his", "her", "its", "their", "our", "le", "la", "les", "un", "une",
    "des", "du", "de", "et", "ou", "à", "au", "aux", "en", "dans", "sur", "pour",
    "avec", "est", "sont", "ce", "cette", "ces", "il", "elle", "ils", "elles",
    "el", "los", "las", "y", "o", "en", "del", "al", "se", "su", "sus", "que",
}


@dataclass
class DedupResult:
    is_duplicate: bool
    matched_tweet_id: Optional[int]
    similarity_score: float
    matched_text: Optional[str]
    matched_source: Optional[str]
    reason: str
    # Set when a pending queue item should be cancelled in favour of the new
    # tweet (same event, but the new one carries media).
    replace_queue_id: Optional[int] = None


def _get_model(model_name: str):
    global _MODEL, _MODEL_FAILED
    if _MODEL is not None or _MODEL_FAILED:
        return _MODEL
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Chargement du modèle d'embeddings {}", model_name)
        _MODEL = SentenceTransformer(model_name, device="cpu")
        logger.info("Modèle d'embeddings prêt")
    except Exception as exc:
        _MODEL_FAILED = True
        logger.warning("Embeddings indisponibles, fallback lexical actif: {}", exc)
        _MODEL = None
    return _MODEL


class ContentDeduplicator:
    def __init__(self, database: Database, config=None):
        self.db = database
        self.config = config or get_config()
        self.dedup_config = self.config.deduplication
        self.threshold = float(self.dedup_config.content_similarity_threshold)
        self.time_window_hours = int(self.dedup_config.time_window_hours)
        self.exact_hash_check = bool(self.dedup_config.exact_hash_check)
        self.model_name = self.config.embeddings.model_name
        self.model = _get_model(self.model_name)
        self.llm_model = self.config.ollama.model
        self._llm_client = None
        self._llm_failed = False

    @staticmethod
    def _normalize(text: str) -> str:
        return normalize_for_match(text)

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            t for t in re.findall(r"[a-z0-9]{3,}", ContentDeduplicator._normalize(text))
            if t not in _STOPWORDS
        }

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            # Keep the original wording for the multilingual model: stripping
            # punctuation/accents measurably hurts cross-lingual similarity.
            vector = self.model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )[0]
            return np.asarray(vector, dtype=np.float32)
        except Exception as exc:
            logger.warning("Échec embedding: {}", exc)
            return None

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _lexical_match(self, text: str) -> Optional[Tuple[int, float, Dict]]:
        tokens = self._tokens(text)
        normalized = self._normalize(text)
        best = None
        for row in self.db.get_recent_processed(limit=500, time_window_hours=self.time_window_hours):
            other = row.get("source_text") or ""
            score = max(
                self._jaccard(tokens, self._tokens(other)),
                SequenceMatcher(None, normalized, self._normalize(other)).ratio(),
            )
            if score >= 0.78 and (best is None or score > best[1]):
                best = (int(row["id"]), float(score), row)
        return best

    def _get_llm_client(self):
        if self._llm_failed or ollama is None:
            return None
        if self._llm_client is None:
            try:
                self._llm_client = ollama.Client(
                    host=self.config.ollama.host,
                    timeout=self.config.ollama.timeout_seconds,
                )
            except Exception as exc:
                self._llm_failed = True
                logger.warning("Vérification Ollama de doublons indisponible: {}", exc)
                return None
        return self._llm_client

    def _llm_same_story(self, text: str, other: str) -> Optional[bool]:
        """Ask the local LLM whether two source snippets describe one event."""
        client = self._get_llm_client()
        if client is None:
            return None
        prompt = (
            "Compare ces deux posts football. S'agit-il du même fait principal "
            "(même joueur concerné par la même absence, même transfert, même "
            "résultat, même annonce, ou même fait de match : même but, même "
            "changement, même carton), même si la formulation, la langue ou les "
            "détails secondaires diffèrent ? Deux posts sur un même match mais "
            "des faits différents ne sont PAS le même fait.\n"
            "Réponds uniquement par OUI ou NON.\n\n"
            f"POST A:\n{text[:700]}\n\nPOST B:\n{other[:700]}"
        )
        try:
            response = client.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Tu es un éditeur sportif francophone précis et concis."},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0, "num_predict": 8},
            )
            if isinstance(response, dict):
                content = response.get("message", {}).get("content", "")
            else:
                message = getattr(response, "message", None)
                content = getattr(message, "content", "") if message is not None else ""
            answer = (content or "").strip().upper()
            if "OUI" in answer and "NON" not in answer:
                return True
            if "NON" in answer:
                return False
        except Exception as exc:
            logger.debug("Vérification Ollama doublon impossible: {}", exc)
        return None

    def is_duplicate(self, text: str, source_tweet_id: str, source_username: str) -> DedupResult:
        if self.exact_hash_check:
            exact = self.db.check_exact_hash(text)
            if exact and str(exact.get("source_tweet_id")) != str(source_tweet_id):
                return DedupResult(
                    True, int(exact["id"]), 1.0, exact.get("source_text"),
                    exact.get("source_username"), "hash exact identique",
                )

        near_match: Optional[Tuple[int, float, Dict]] = None
        embedding = self._embed(text)
        if embedding is not None:
            # Scan once at a low floor so a borderline translation can be
            # confirmed by the local LLM instead of being silently missed.
            match = self.db.find_similar_source(
                embedding, min(0.30, self.threshold), self.time_window_hours
            )
            if match:
                row_id, score, row = match
                if score >= self.threshold:
                    return DedupResult(
                        True, row_id, score, row.get("source_text"),
                        row.get("source_username"),
                        f"similarité sémantique {score:.3f} ≥ {self.threshold:.2f}",
                    )
                near_match = match

        lexical = self._lexical_match(text)
        if lexical:
            row_id, score, row = lexical
            return DedupResult(
                True, row_id, score, row.get("source_text"),
                row.get("source_username"),
                f"similarité lexicale {score:.3f}",
            )

        if near_match:
            row_id, score, row = near_match
            other = row.get("source_text") or ""
            verdict = self._llm_same_story(text, other)
            if verdict is True:
                return DedupResult(
                    True, row_id, score, other, row.get("source_username"),
                    f"même info confirmée par Ollama (similarité {score:.3f})",
                )
            if verdict is None:
                # If Ollama is unavailable, only a strong shared-vocabulary
                # signal is accepted, to avoid merging unrelated news.
                common = self._tokens(text) & self._tokens(other)
                if len(common) >= 3:
                    return DedupResult(
                        True, row_id, score, other, row.get("source_username"),
                        f"vocabulaire partagé: {', '.join(sorted(common)[:5])}",
                    )

        return DedupResult(False, None, 0.0, None, None, "contenu nouveau")

    @staticmethod
    def _queue_has_media(raw) -> bool:
        if not raw:
            return False
        if isinstance(raw, list):
            return bool(raw)
        try:
            parsed = json.loads(raw)
            return bool(parsed)
        except Exception:
            return bool(str(raw).strip())

    def is_duplicate_event(
        self,
        rewritten_text: str,
        source_tweet_id: str,
        source_username: str,
        has_media: bool = False,
    ) -> DedupResult:
        """Second net: compare the REWRITTEN text against items already
        queued or recently posted. Two different sources can describe the
        same event with different words; the source-level dedup cannot catch
        those, this one can. If the pending counterpart has no media while
        the new tweet does, the pending item is replaced.
        """
        rewritten = (rewritten_text or "").strip()
        if not rewritten:
            return DedupResult(False, None, 0.0, None, None, "texte vide")

        candidates: List[Tuple[float, str, str, Optional[int], bool]] = []
        # (score, kind, text, queue_id, candidate_has_media)
        for row in self.db.get_pending_queue():
            if str(row.get("source_tweet_id") or "") == str(source_tweet_id):
                continue
            text = (row.get("text") or "").strip()
            if not text:
                continue
            score = self._event_score(rewritten, text)
            if score >= 0.30:
                candidates.append(
                    (score, "en file", text, int(row["id"]),
                     self._queue_has_media(row.get("media_paths") or row.get("media_ids")))
                )
        for row in self.db.get_recent_posted_tweets(
            limit=10, max_age_hours=self.time_window_hours
        ):
            if str(row.get("source_tweet_id") or "") == str(source_tweet_id):
                continue
            text = (row.get("rewritten_text") or row.get("source_text") or "").strip()
            if not text:
                continue
            score = self._event_score(rewritten, text)
            if score >= 0.30:
                candidates.append((score, "publié", text, None, False))

        if not candidates:
            return DedupResult(False, None, 0.0, None, None, "aucun événement proche")
        candidates.sort(key=lambda item: item[0], reverse=True)
        score, kind, text, queue_id, candidate_media = candidates[0]

        verdict = self._llm_same_story(rewritten, text)
        if verdict is not True:
            if verdict is None and score >= 0.45:
                # Ollama unavailable: only a strong overlap is trusted.
                pass
            else:
                return DedupResult(
                    False, None, score, text, None,
                    f"événement différent ({kind}, score {score:.2f})",
                )

        if kind == "en file" and queue_id is not None and has_media and not candidate_media:
            return DedupResult(
                True, queue_id, score, text, None,
                f"même événement déjà en file sans média (score {score:.2f}); "
                "la version avec média est prioritaire",
                replace_queue_id=queue_id,
            )
        return DedupResult(
            True, queue_id, score, text, None,
            f"même événement déjà {kind} (score {score:.2f})",
        )

    def _event_score(self, rewritten: str, other: str) -> float:
        return max(
            self._jaccard(self._tokens(rewritten), self._tokens(other)),
            SequenceMatcher(None, self._normalize(rewritten), self._normalize(other)).ratio(),
        )

    def store_processed(
        self,
        text: str,
        source_tweet_id: str,
        source_username: str,
        matched_existing: bool = False,
        matched_tweet_id: Optional[int] = None,
    ) -> int:
        embedding = self._embed(text) if not matched_existing else None
        return self.db.add_processed_source(
            source_tweet_id=source_tweet_id,
            source_username=source_username,
            source_text=text,
            embedding=embedding,
            matched_existing=matched_existing,
            matched_tweet_id=matched_tweet_id,
        )

    def get_cache_stats(self) -> Dict:
        return {
            "cached_embeddings": self.db.get_processed_count(),
            "threshold": self.threshold,
            "model": self.model_name,
            "model_loaded": self.model is not None,
            "time_window_hours": self.time_window_hours,
            "llm_verification": ollama is not None and not self._llm_failed,
        }
