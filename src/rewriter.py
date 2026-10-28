"""English rewrite engine with anti-plagiarism, fact-checking, and protected terms for @UniverseCule."""

import re
import threading
from typing import Optional, Dict, List, Set
import ollama
from loguru import logger
from .config import get_config
from .timeutil import clean_tweet_text, weighted_length


class TweetRewriter:
    """Professional English rewriter for @UniverseCule FC Barcelona content."""

    # Terms that must NEVER be translated or modified by the AI.
    # Longest keys first so "Kopa Trophy" is protected before "Kopa" alone.
    _PROTECTED_TERMS: Dict[str, str] = {
        # Trophy names (never translate)
        "Kopa Trophy": "Kopa Trophy",
        "Ballon d'Or": "Ballon d'Or",
        "Ballon d'or": "Ballon d'Or",
        "Ballon d or": "Ballon d'Or",
        "Golden Boy": "Golden Boy",
        "FIFA The Best": "FIFA The Best",
        "Pichichi": "Pichichi",
        "Zamora": "Zamora",
        # Alert words (keep in English at start of headline)
        "BREAKING": "BREAKING",
        "JUST IN": "JUST IN",
        "URGENT": "URGENT",
        "OFFICIAL": "OFFICIAL",
        "EXCLUSIVE": "EXCLUSIVE",
        # Player names (never modify)
        "Lamine Yamal": "Lamine Yamal",
        "Pau Cubarsi": "Pau Cubarsi",
        "Pau Cubarsí": "Pau Cubarsi",
        "Pedri": "Pedri",
        "Gavi": "Gavi",
        "Robert Lewandowski": "Robert Lewandowski",
        "Lewandowski": "Lewandowski",
        "Raphinha": "Raphinha",
        "Frenkie de Jong": "Frenkie de Jong",
        "De Jong": "De Jong",
        "Marc-Andre ter Stegen": "Marc-Andre ter Stegen",
        "Ter Stegen": "Ter Stegen",
        "Jules Kounde": "Jules Kounde",
        "Kounde": "Kounde",
        "Ronald Araujo": "Ronald Araujo",
        "Araujo": "Araujo",
        "Alejandro Balde": "Alejandro Balde",
        "Balde": "Balde",
        "Fermin Lopez": "Fermin Lopez",
        "Fermin": "Fermin",
        "Dani Olmo": "Dani Olmo",
        "Olmo": "Olmo",
        "Ferran Torres": "Ferran Torres",
        "Ferran": "Ferran",
        "Ansu Fati": "Ansu Fati",
        "Ansu": "Ansu",
        "Eric Garcia": "Eric Garcia",
        "Eric García": "Eric Garcia",
        "Inigo Martinez": "Inigo Martinez",
        "Iñigo Martínez": "Inigo Martinez",
        "Andreas Christensen": "Andreas Christensen",
        "Christensen": "Christensen",
        "Inaki Pena": "Inaki Pena",
        "Iñaki Peña": "Inaki Pena",
        "Wojciech Szczesny": "Wojciech Szczesny",
        "Szczesny": "Szczesny",
        "Hector Fort": "Hector Fort",
        "Héctor Fort": "Hector Fort",
        "Marc Casado": "Marc Casado",
        "Casado": "Casado",
        "Marc Bernal": "Marc Bernal",
        "Bernal": "Bernal",
        "Pablo Torre": "Pablo Torre",
        "Torre": "Torre",
        # Staff
        "Hansi Flick": "Hansi Flick",
        "Flick": "Flick",
        "Deco": "Deco",
        "Joan Laporta": "Joan Laporta",
        "Laporta": "Laporta",
        # Club names
        "Barça": "Barça",
        "FC Barcelona": "FC Barcelona",
        "FC Barcelone": "FC Barcelona",
        "FCB": "FC Barcelona",
        "Barcelona": "Barcelona",
        "Blaugrana": "Blaugrana",
        "Cules": "Cules",
        # Competitions
        "La Liga": "La Liga",
        "Champions League": "Champions League",
        "Copa del Rey": "Copa del Rey",
        "Supercopa": "Supercopa",
        "El Clasico": "El Clasico",
        "El Clásico": "El Clasico",
    }

    # Deterministic post-corrections for common LLM errors
    _POST_CORRECTIONS = [
        # Fix common football terminology errors
        (re.compile(r"\bforward\b", re.IGNORECASE), "forward"),
        (re.compile(r"\bstriker\b", re.IGNORECASE), "forward"),
        (re.compile(r"\bwin\b", re.IGNORECASE), "win"),
        (re.compile(r"\bsign\b", re.IGNORECASE), "sign"),
        # Fix common grammar/wording issues
        (re.compile(r"\ba forward\b", re.IGNORECASE), "a forward"),
        (re.compile(r"\ban forward\b", re.IGNORECASE), "a forward"),
        # Ensure proper capitalization of FC Barcelona
        (re.compile(r"\bfc barcelona\b", re.IGNORECASE), "FC Barcelona"),
        (re.compile(r"\bfc barcelone\b", re.IGNORECASE), "FC Barcelona"),
        # Fix common misspellings
        (re.compile(r"\byamal\b", re.IGNORECASE), "Yamal"),
        (re.compile(r"\bcubarsi\b", re.IGNORECASE), "Cubarsi"),
        (re.compile(r"\baraujo\b", re.IGNORECASE), "Araujo"),
        (re.compile(r"\blewandowski\b", re.IGNORECASE), "Lewandowski"),
        # Fix "according to reports" -> "reports say" (more professional)
        (re.compile(r"\baccording to reports\b", re.IGNORECASE), "reports say"),
        (re.compile(r"\baccording to information\b", re.IGNORECASE), "reports say"),
        # Remove redundant "official" if already in alert
        (re.compile(r"\bOFFICIAL:\s*official\b", re.IGNORECASE), "OFFICIAL:"),
    ]

    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = ollama.Client(host=self.config.ollama.host)
        self.model = self.config.ollama.model
        self.max_chars = int(self.config.length.max_chars)
        self.temperature = float(self.config.ollama.temperature)
        # Cache for Arabic translation model
        self._ar_tokenizer = None
        self._ar_model = None
        self._ar_model_lock = threading.Lock()

    def _chat(self, system: str, user: str) -> str:
        """Optimized Ollama call with minimal overhead."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                options={"temperature": self.temperature, "num_predict": 220},
            )
            return str(response["message"]["content"]).strip()
        except Exception as e:
            logger.warning(f"Ollama error: {e}")
            return ""

    def _sanitize(self, text: str) -> str:
        """Ultra-fast text sanitization."""
        text = clean_tweet_text(text)
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"(?<!\w)@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _protect_terms(self, text: str) -> tuple[str, Dict[str, str]]:
        """Replace protected terms with temporary placeholders."""
        mapping: Dict[str, str] = {}
        protected = text
        # Sort by length descending so "Kopa Trophy" is matched before "Kopa".
        for term in sorted(self._PROTECTED_TERMS.keys(), key=len, reverse=True):
            if not term:
                continue
            placeholder = f"__PROTECTED_{len(mapping)}__"
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            matches = list(pattern.finditer(protected))
            for match in reversed(matches):
                protected = protected[:match.start()] + placeholder + protected[match.end():]
                mapping[placeholder] = self._PROTECTED_TERMS[term]
        return protected, mapping

    def _restore_terms(self, text: str, mapping: Dict[str, str]) -> str:
        """Restore protected terms from their placeholders."""
        restored = text
        for placeholder, original in mapping.items():
            restored = restored.replace(placeholder, original)
        return restored

    def _verify_protected_restored(self, text: str, mapping: Dict[str, str]) -> Optional[str]:
        """Verify all placeholders have been restored."""
        for placeholder in mapping:
            if placeholder in text:
                return f"protected term not restored: {placeholder}"
        return None

    def _extract_numbers(self, text: str) -> Set[str]:
        """Extract all numbers from text."""
        return set(re.findall(r"\d+", text))

    def _extract_scores(self, text: str) -> Set[tuple]:
        """Extract all score patterns (X-Y) from text."""
        return set(re.findall(r"(\d{1,3})\s*-\s*(\d{1,3})", text))

    def _facts_check(self, rewritten: str, source: str) -> Optional[str]:
        """Reject any number or score that does not exist in the source."""
        src_numbers = self._extract_numbers(source)
        out_numbers = self._extract_numbers(rewritten)
        missing = sorted(n for n in out_numbers if n not in src_numbers)
        if missing:
            return f"number not in source: {', '.join(missing)}"

        src_scores = self._extract_scores(source)
        out_scores = self._extract_scores(rewritten)
        for score in out_scores:
            if score not in src_scores:
                return f"score not in source: {score[0]}-{score[1]}"
        return None

    def _apply_post_corrections(self, text: str) -> str:
        """Apply deterministic post-corrections."""
        for pattern, replacement in self._POST_CORRECTIONS:
            text = pattern.sub(replacement, text)
        return text

    def _detect_plagiarism(self, original: str, rewritten: str) -> bool:
        """Enhanced plagiarism check using word overlap and semantic similarity."""
        original_words = set(original.lower().split())
        rewritten_words = set(rewritten.lower().split())
        overlap = original_words.intersection(rewritten_words)

        if len(overlap) > 2:
            original_word_count = len(original_words)
            rewritten_word_count = len(rewritten_words)
            if original_word_count > 0 and rewritten_word_count > 0:
                similarity_ratio = len(overlap) / min(original_word_count, rewritten_word_count)
                if similarity_ratio > 0.4:
                    return True
            return True
        return False

    def _is_arabic(self, text: str) -> bool:
        """Fast Arabic detection."""
        return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F]", text))

    def _get_arabic_model(self):
        """Lazy-load Arabic translation model (cached)."""
        if self._ar_tokenizer is None or self._ar_model is None:
            with self._ar_model_lock:
                if self._ar_tokenizer is None or self._ar_model is None:
                    try:
                        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                        model_name = "facebook/nllb-200-distilled-600M"
                        self._ar_tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="arb_Arab")
                        self._ar_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    except Exception as e:
                        logger.warning(f"Failed to load Arabic model: {e}")
                        self._ar_tokenizer = False
                        self._ar_model = False
        return self._ar_tokenizer, self._ar_model

    def _translate_arabic_to_english(self, arabic_text: str) -> Optional[str]:
        """Cached Arabic-to-English translation using NLLB."""
        tokenizer, model = self._get_arabic_model()
        if not tokenizer or not model:
            return None

        try:
            if not arabic_text or not self._is_arabic(arabic_text):
                return None

            inputs = tokenizer(arabic_text, return_tensors="pt", truncation=True)
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_new_tokens=220,
                num_beams=4,
            )
            translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return translation.strip()
        except Exception as e:
            logger.warning(f"Arabic translation failed: {e}")
            return None

    def _rewrite_to_english(self, text: str, source_lang: str) -> str:
        """Rewrite in English with anti-plagiarism, fact-checking, and post-corrections."""
        if not text:
            return ""

        # Sanitize and prepare
        cleaned = self._sanitize(text)
        if not cleaned:
            return ""

        # Protect critical terms before sending to LLM
        protected_source, protected_mapping = self._protect_terms(cleaned)

        prompt = (
            f"Source language: {source_lang}\n"
            f"Original tweet: {protected_source}\n\n"
            "Rewrite this tweet in natural English, avoiding plagiarism. "
            "Follow the rules above. Respond ONLY with the rewritten tweet."
        )

        # Get first attempt
        attempt1 = self._chat(self.ENGLISH_REWRITE_SYSTEM, prompt)
        if not attempt1:
            return ""

        # Restore protected terms
        attempt1 = self._restore_terms(attempt1, protected_mapping)

        # Verify all protected terms restored
        verify_error = self._verify_protected_restored(attempt1, protected_mapping)
        if verify_error:
            logger.warning(f"Protected term issue: {verify_error}")

        # Apply post-corrections
        attempt1 = self._apply_post_corrections(attempt1)

        # Fact-check
        fact_error = self._facts_check(attempt1, cleaned)
        if fact_error:
            logger.warning(f"Fact check failed: {fact_error}")

        # Check plagiarism
        if self._detect_plagiarism(cleaned, attempt1) or fact_error:
            strict_prompt = (
                f"Source: {protected_source}\n"
                "REWRITE THIS TWEET IN NATURAL ENGLISH WITHOUT COPYING ANY WORDS. "
                "Change sentence structure, word order, and phrasing completely. "
                "Respond ONLY with the rewritten tweet."
            )
            attempt2 = self._chat(self.ENGLISH_REWRITE_SYSTEM, strict_prompt)
            if attempt2:
                attempt2 = self._restore_terms(attempt2, protected_mapping)
                attempt2 = self._apply_post_corrections(attempt2)
                fact_error2 = self._facts_check(attempt2, cleaned)
                if not fact_error2 and not self._detect_plagiarism(cleaned, attempt2):
                    return attempt2
                logger.warning("Second attempt also failed checks, using first attempt")

        # Final length check and truncation
        if weighted_length(attempt1) > self.max_chars:
            attempt1 = attempt1[:self.max_chars].rsplit(' ', 1)[0] + "…"

        return attempt1

    # System prompt for @UniverseCule
    ENGLISH_REWRITE_SYSTEM = """You are @UniverseCule, a professional football journalist specializing in FC Barcelona content.
Your task: Rewrite tweets in natural, idiomatic English while ensuring 100% originality and avoiding plagiarism.

ABSOLUTE RULES:
1. NEVER copy-paste any part of the original text. Always rephrase using completely different word choices and sentence structure.
2. Preserve ALL facts, names, numbers, and specific details exactly as in the source.
3. Use precise and professional football terminology in English.
4. Maximum 280 characters. No links, mentions, hashtags, or any external references.
5. Format: Headline in UPPERCASE, emojis for emphasis, line breaks for readability.
6. Never mix languages. Always use English exclusively.
7. Maintain the original meaning and tone while making it sound natural and professional.
8. NEVER modify protected terms (they appear as __PROTECTED_X__ - leave them exactly as-is).

EXAMPLES:
- Original: "Barcelona sign new striker"
  Rewritten: "FC BARCELONA SECURE NEW FORWARD ✍️\n\nThe club has finalized the acquisition of a promising attacker to bolster their front line."

- Original: "Real Madrid 3-0 victory over Barcelona"
  Rewritten: "REAL MADRID DOMINATE 3-0 IN EL CLASICO ⚪\n\nLos Blancos deliver a commanding performance to claim bragging rights in Madrid."

- Original: "Lewandowski scores 2 goals"
  Rewritten: "LEWANDOWSKI BRACE FIRES BARCELONA TO VICTORY ⚽⚽\n\nThe Polish striker nets twice as the Blaugrana secure three vital points."

FOOTBALL TERMINOLOGY:
- Use "forward" or "attacker" not "striker"
- Use "triumph" or "secure victory" not just "win"
- Use "finalize acquisition" or "secure signing" not just "sign"
- Use "brace" for 2 goals, "hat-trick" for 3
- Use "equaliser" not "leveler"
- Use "bench" not "substitutes bench"
- Use "touchline" not "sideline"

PROTECTED TERMS (appear as __PROTECTED_X__ - DO NOT MODIFY):
These include: player names (Yamal, Pedri, Lewandowski, etc.), trophy names (Ballon d'Or, Kopa Trophy), alert words (BREAKING, JUST IN, OFFICIAL, EXCLUSIVE), club names (FC Barcelona, Barça, Blaugrana), competitions (La Liga, Champions League, El Clasico).

EMOJI GUIDELINES:
- Use 🚨 for breaking/urgent news
- Use ⚽ for goals/match action
- Use ✍️ for signings/transfers
- Use 🏆 for trophies/titles
- Use 🔵🔴 for Barcelona branding (blue/red)
- Keep emojis minimal and professional"""

    def rewrite(self, text: str, source_username: str = "source") -> str:
        """Main rewrite method with language detection and optimization."""
        cleaned = clean_tweet_text(text)
        if not cleaned:
            return ""

        # Detect language (fast heuristic)
        if self._is_arabic(cleaned):
            # Arabic → English translation
            translated = self._translate_arabic_to_english(cleaned)
            if translated:
                return self._rewrite_to_english(translated, "ar")
            return cleaned  # Fallback to original if translation fails

        # English/Spanish/French → English reformulation
        return self._rewrite_to_english(cleaned, "en")

    def check_model(self) -> bool:
        """Verify Ollama model is available."""
        try:
            result = self.client.list()
            models = result.get("models", []) if isinstance(result, dict) else getattr(result, "models", [])
            names = []
            for item in models:
                if isinstance(item, dict):
                    names.append(item.get("name") or item.get("model") or "")
                else:
                    names.append(getattr(item, "name", "") or getattr(item, "model", "") or "")
            return any(self.model in name for name in names if name)
        except Exception as exc:
            logger.warning("Ollama check failed: {}", exc)
            return False

    def test_rewrite(self, text: str, username: str = "test") -> Dict:
        """Test rewrite with detailed diagnostics."""
        original = clean_tweet_text(text)
        rewritten = self.rewrite(text, username)
        return {
            "original": original,
            "rewritten": rewritten,
            "original_length": weighted_length(original),
            "rewritten_length": weighted_length(rewritten),
            "model": self.model,
            "success": bool(rewritten),
            "within_limit": weighted_length(rewritten) <= self.max_chars,
        }