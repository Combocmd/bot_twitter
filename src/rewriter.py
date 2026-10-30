"""English rewrite engine with anti-plagiarism, fact-checking, and protected terms for @UniverseCule."""

import os
import re
import threading
from dataclasses import dataclass
from typing import Optional, Dict, List, Set
import ollama
from loguru import logger
from .config import get_config
from .timeutil import clean_tweet_text, weighted_length

# Optional Gemini import (new google-genai SDK)
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None


@dataclass
class RewriteResult:
    """Result of a rewrite operation."""
    success: bool
    rewritten_text: str = ""
    error: str = ""


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

    # All known Barça players + staff for validation (must match _PROTECTED_TERMS)
    _VALID_NAMES = {
        # Players
        "Yamal", "Lamine Yamal", "Pedri", "Gavi", "De Jong", "Frenkie de Jong",
        "Lewandowski", "Robert Lewandowski", "Raphinha", "Kounde", "Jules Kounde",
        "Araujo", "Ronald Araujo", "Balde", "Alejandro Balde", "Ter Stegen",
        "Marc-Andre ter Stegen", "Fermin", "Fermin Lopez", "Olmo", "Dani Olmo",
        "Torres", "Ferran Torres", "Fati", "Ansu Fati", "Garcia", "Eric Garcia",
        "Martinez", "Inigo Martinez", "Christensen", "Andreas Christensen",
        "Pena", "Inaki Pena", "Szczesny", "Wojciech Szczesny", "Fort", "Hector Fort",
        "Casado", "Marc Casado", "Bernal", "Marc Bernal", "Torre", "Pablo Torre",
        "Cubarsi", "Pau Cubarsi",
        # Staff
        "Flick", "Hansi Flick", "Deco", "Laporta", "Joan Laporta",
    }

    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = ollama.Client(host=self.config.ollama.host)
        self.model = self.config.ollama.model
        self.max_chars = int(self.config.length.max_chars)
        self.temperature = float(self.config.ollama.temperature)
        
        # Gemini setup (priority provider)
        self._gemini_available = GEMINI_AVAILABLE
        self._gemini_client = None
        self._gemini_model = "gemini-flash-lite-latest"
        if self._gemini_available:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    self._gemini_client = genai.Client(api_key=api_key)
                    logger.info("Gemini client initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize Gemini client: {e}")
                    self._gemini_available = False
            else:
                logger.warning("GEMINI_API_KEY not found in environment, Gemini unavailable")
                self._gemini_available = False
        
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

    def _call_gemini_llm(self, system: str, user: str) -> str:
        """Call Gemini API with the given system and user prompts using new google-genai SDK.
        
        Includes 2 retries for 503 (High Demand) errors before falling back to Ollama.
        """
        if not self._gemini_available or not self._gemini_client:
            logger.debug("Gemini not available, skipping")
            return ""
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                # Combine system and user prompts for Gemini
                full_prompt = f"{system}\n\n{user}"
                
                response = self._gemini_client.models.generate_content(
                    model=self._gemini_model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=220,
                    )
                )
                
                if response.text:
                    result = response.text.strip()
                    logger.debug(f"Gemini call succeeded on attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Gemini returned empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Gemini error (attempt {attempt + 1}/{max_retries + 1}): {error_msg}")
                
                # Check for 503 (High Demand) - retry
                if "503" in error_msg or "UNAVAILABLE" in error_msg.upper() or "HIGH DEMAND" in error_msg.upper():
                    if attempt < max_retries:
                        import time
                        time.sleep(1.5 * (attempt + 1))  # Exponential backoff
                        continue
                
                # For other errors, don't retry
                break
        
        logger.warning("Gemini failed after retries, will fall back to Ollama")
        return ""

    def _llm_generate(self, system: str, user: str, context: str = "") -> str:
        """Generate text using Gemini (priority) with Ollama fallback.
        
        Args:
            system: System prompt
            user: User prompt
            context: Optional context string for logging (e.g., "first attempt", "strict retry", "concise retry")
            
        Returns:
            Generated text, or empty string if both providers fail.
        """
        # Try Gemini first
        if self._gemini_available and self._gemini_client:
            logger.debug(f"Attempting Gemini generation ({context})")
            result = self._call_gemini_llm(system, user)
            if result:
                logger.info(f"Gemini succeeded ({context})")
                return result
            logger.warning(f"Gemini returned empty, falling back to Ollama ({context})")
        else:
            logger.debug(f"Gemini unavailable, using Ollama directly ({context})")
        
        # Fallback to Ollama
        logger.debug(f"Attempting Ollama generation ({context})")
        result = self._chat(system, user)
        if result:
            logger.info(f"Ollama succeeded ({context})")
        else:
            logger.warning(f"Ollama also failed ({context})")
        return result

    def _sanitize(self, text: str) -> tuple[str, Optional[str]]:
        """Ultra-fast text sanitization. Returns (cleaned_text, source_mention)."""
        text = clean_tweet_text(text)
        text = re.sub(r"https?://\S+", "", text)
        
        # Extract source mention from end of tweet: (@user), [@user], via @user, etc.
        source_mention = None
        # Pattern for (@user), [@user], (@user) at end of text
        source_pattern = re.compile(r'\s*[\(\[]\s*@(\w+)\s*[\)\]]\s*$')
        match = source_pattern.search(text)
        if match:
            source_mention = match.group(1)
            text = text[:match.start()].rstrip()
        else:
            # Also check for "via @user" at end
            via_pattern = re.compile(r'\s+via\s+@(\w+)\s*$', re.IGNORECASE)
            match = via_pattern.search(text)
            if match:
                source_mention = match.group(1)
                text = text[:match.start()].rstrip()
        
        # Remove any remaining mentions (not the source one we extracted)
        text = re.sub(r"(?<!\w)@\w+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text, source_mention

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

    def _cleanup_protected_terms(self, text: str) -> str:
        """Remove any remaining protected term placeholders as a safety net.
        
        Placeholders have the format __PROTECTED_X__ where X is a number.
        This is a final safeguard in case restoration missed any.
        """
        # Match __PROTECTED_N__ where N is any number
        return re.sub(r'__PROTECTED_\d+__', '', text)

    def _extract_alert_keyword(self, text: str) -> tuple[str, str]:
        """Extract alert keyword from the beginning of text.
        
        Detects: JUST IN, BREAKING, OFFICIAL, OFFICIEL, URGENT (with or without 🚨 prefix).
        Returns: (cleaned_text, alert_keyword) where alert_keyword is empty if none found.
        """
        # Pattern: optional 🚨 followed by alert keyword at start
        keywords = ["JUST IN", "BREAKING", "OFFICIAL", "OFFICIEL", "URGENT"]
        pattern = r'^(?:🚨\s*)?(' + '|'.join(keywords) + r')[\s:]+'
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            kw = match.group(1).upper()
            # Normalize OFFICIEL to OFFICIAL
            if kw == "OFFICIEL":
                kw = "OFFICIAL"
            cleaned = text[match.end():].strip()
            return cleaned, kw
        return text, ""

    def _extract_numbers(self, text: str) -> Set[str]:
        """Extract all numbers from text."""
        return set(re.findall(r"\d+", text))

    def _extract_scores(self, text: str) -> Set[tuple]:
        """Extract all score patterns (X-Y) from text."""
        return set(re.findall(r"(\d{1,3})\s*-\s*(\d{1,3})", text))

    def _facts_check(self, rewritten: str, source: str, source_mention: Optional[str] = None) -> Optional[str]:
        """Reject any number or score that does not exist in the source."""
        src_numbers = self._extract_numbers(source)
        
        # Exclude numbers from source mention (e.g., @10JoseAlvarez -> "10" is not a fact)
        if source_mention:
            source_mention_numbers = self._extract_numbers(source_mention)
            src_numbers = src_numbers - source_mention_numbers
        
        out_numbers = self._extract_numbers(rewritten)
        # Also exclude numbers from source mention in the output (e.g., [@10JoseAlvarez])
        if source_mention:
            source_mention_numbers = self._extract_numbers(source_mention)
            out_numbers = out_numbers - source_mention_numbers
        
        missing = sorted(n for n in out_numbers if n not in src_numbers)
        if missing:
            return f"number not in source: {', '.join(missing)}"

        src_scores = self._extract_scores(source)
        out_scores = self._extract_scores(rewritten)
        for score in out_scores:
            if score not in src_scores:
                return f"score not in source: {score[0]}-{score[1]}"
        return None

    def _extract_names(self, text: str) -> Set[str]:
        """Extract all known player/staff names from text (case-insensitive)."""
        found = set()
        text_lower = text.lower()
        for name in self._VALID_NAMES:
            # Use word boundaries to avoid partial matches
            if re.search(rf"\b{re.escape(name.lower())}\b", text_lower):
                found.add(name)
        return found

    def _validate_names(self, rewritten: str, source: str, vision_context: Optional[Dict] = None, source_mention: Optional[str] = None, source_username: str = "source") -> Optional[str]:
        """Reject if any player/staff name from source is missing in rewritten, or if rewritten has names not in source/vision."""
        source_names = self._extract_names(source)
        
        # Add vision-detected names to allowed list
        allowed_names = set(source_names)
        if vision_context and vision_context.get("players_detected"):
            for vname in vision_context["players_detected"]:
                for valid_name in self._VALID_NAMES:
                    if valid_name.lower() in vname.lower() or vname.lower() in valid_name.lower():
                        allowed_names.add(valid_name)
        
        rewritten_names = self._extract_names(rewritten)
        
        # Check for missing source names (MUST be preserved)
        missing = source_names - rewritten_names
        if missing:
            return f"missing source names in rewrite: {', '.join(sorted(missing))}"
        
        # Check for missing source mention (only validate the handle that's actually used in suffix)
        # Priority: source_mention > source_username
        if source_mention and source_mention.strip():
            # source_mention is used in suffix
            expected_mention = f"[@{source_mention}]"
            if expected_mention not in rewritten:
                return f"missing source mention in rewrite: {expected_mention}"
            allowed_names.add(expected_mention)
        elif source_username and source_username.strip():
            # source_username is used in suffix (when no source_mention)
            expected_source = f"[@{source_username}]"
            if expected_source not in rewritten:
                return f"missing source mention in rewrite: {expected_source}"
            allowed_names.add(expected_source)
        
        # Check for hallucinated names (NOT in source or vision)
        hallucinated = rewritten_names - allowed_names
        if hallucinated:
            return f"hallucinated names not in source/vision: {', '.join(sorted(hallucinated))}"
        return None

    def _is_english_only(self, text: str) -> bool:
        """Check if text contains only English/Latin characters, numbers, punctuation, and common emojis."""
        # Allow: Latin letters, numbers, common punctuation, spaces, common emojis
        # Common emoji ranges
        emoji_ranges = [
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
            (0x1F680, 0x1F6FF),  # Transport and Map
            (0x1F1E0, 0x1F1FF),  # Flags
            (0x2600, 0x26FF),    # Misc symbols
            (0x2700, 0x27BF),    # Dingbats
            (0xFE00, 0xFE0F),    # Variation selectors
            (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
        ]
        
        def is_emoji(cp):
            for start, end in emoji_ranges:
                if start <= cp <= end:
                    return True
            return False
        
        for char in text:
            cp = ord(char)
            # Allow ASCII (Latin letters, numbers, punctuation, space)
            if cp <= 0x7F:
                continue
            # Allow common Latin extended (accents, etc.)
            if 0x80 <= cp <= 0x024F:
                continue
            # Allow emojis
            if is_emoji(cp):
                continue
            # If we get here, it's likely non-Latin (Chinese, Arabic, Cyrillic, etc.)
            return False
        return True

    def _fix_missing_names(self, text: str, source: str, vision_context: Optional[Dict] = None) -> str:
        """Try to fix rewrite by adding missing source names naturally."""
        source_names = self._extract_names(source)
        
        # Add vision-detected names to allowed list
        allowed_names = set(source_names)
        if vision_context and vision_context.get("players_detected"):
            for vname in vision_context["players_detected"]:
                for valid_name in self._VALID_NAMES:
                    if valid_name.lower() in vname.lower() or vname.lower() in valid_name.lower():
                        allowed_names.add(valid_name)
        
        rewritten_names = self._extract_names(text)
        missing = source_names - rewritten_names
        
        if not missing:
            return text
        
        # Try to prepend missing names in a natural way
        missing_list = sorted(missing)
        if len(missing_list) == 1:
            prefix = f"{missing_list[0]}: "
        else:
            prefix = f"{', '.join(missing_list)}: "
        
        # Check if adding prefix would exceed 280 chars
        from .timeutil import weighted_length
        if weighted_length(prefix + text) <= self.max_chars:
            return prefix + text
        
        # If too long, try to integrate more subtly
        # Just return original text if we can't fix it cleanly
        return text

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

    def _rewrite_to_english(self, text: str, source_lang: str, vision_context: Optional[Dict] = None, source_username: str = "source") -> str:
        """Rewrite in English with anti-plagiarism, fact-checking, and post-corrections."""
        if not text:
            return ""

        # Sanitize and prepare
        cleaned, source_mention = self._sanitize(text)
        if not cleaned:
            return ""

        # Extract alert keyword from the beginning of the text
        cleaned, alert_keyword = self._extract_alert_keyword(cleaned)

        # Build vision context string for prompt
        vision_prompt = ""
        required_names = []
        if vision_context:
            vision_parts = []
            if vision_context.get("players_detected"):
                vision_parts.append(f"Players visible: {', '.join(vision_context['players_detected'])}")
                # Add vision names to required names
                required_names.extend(vision_context['players_detected'])
            if vision_context.get("scene_type") and vision_context["scene_type"] != "unknown":
                vision_parts.append(f"Scene: {vision_context['scene_type']}")
            if vision_context.get("text_extracted"):
                vision_parts.append(f"Text in image/video: {vision_context['text_extracted'][:200]}")
            if vision_context.get("key_facts"):
                vision_parts.append(f"Key facts: {'; '.join(vision_context['key_facts'])}")
            if vision_parts:
                vision_prompt = "\n\nVISION CONTEXT (from media analysis):\n" + "\n".join(vision_parts)

        required_names = []
        if vision_context and vision_context.get("players_detected"):
            required_names.extend(vision_context["players_detected"])

        # Attribution logic: source_mention takes priority for suffix
        # If source_mention exists, it will be the suffix; source_username is NOT added to required_names
        # If source_mention is None, source_username is used as suffix (for Barça/mercato tweets)
        # Only add attribution source if one is actually provided
        attribution_source = source_mention if source_mention else source_username
        if attribution_source and attribution_source.strip():
            required_names.append(f"[@{attribution_source}]")

        source_names = self._extract_names(cleaned)
        for name in source_names:
            if name not in required_names:
                required_names.append(name)

        seen_lower = set()
        deduped_names = []
        for name in required_names:
            name_lower = name.lower()
            if name_lower not in seen_lower:
                seen_lower.add(name_lower)
                deduped_names.append(name)

        required_names = deduped_names
        required_names_str = ", ".join(required_names) if required_names else "none"

        # Protect critical terms before sending to LLM
        protected_source, protected_mapping = self._protect_terms(cleaned)

        prompt = (
            f"Source language: {source_lang}\n"
            f"Original tweet: {protected_source}"
            f"{vision_prompt}\n\n"
            f"REQUIRED NAMES (must ALL appear in rewrite): {required_names_str}\n\n"
            f"ALERT KEYWORD (must appear immediately after 🚨 in headline): {alert_keyword if alert_keyword else 'none'}\n\n"
            f"SOURCE HANDLE: The final 🔵🔴 suffix MUST use the EXACT handle from REQUIRED NAMES. "
            f"Do NOT invent or substitute any other handle (e.g., @_BarcaInfo, @FabrizioRomano).\n\n"
            "Rewrite this tweet in natural English, avoiding plagiarism. "
            "Only use vision context for player names or text visible in media. "
            "Do NOT add scene descriptions or inferred details not in the source text. "
            "CRITICAL: Every name in REQUIRED NAMES must appear in your output. Do not omit any. "
            "FORMAT: Start with 🚨. Use \\n\\n between headline and body, and before the final 🔵🔴 [@source] attribution. "
            "State each fact exactly once — no redundant repetition of names or information. "
            "IMPORTANT: The ALERT KEYWORD (if provided) must ONLY appear immediately after 🚨 in the headline. "
            "NEVER use alert keywords (JUST IN, BREAKING, OFFICIAL, URGENT) anywhere else in the tweet body. "
            "Follow the rules above. Respond ONLY with the rewritten tweet."
        )

        # Get first attempt
        attempt1 = self._llm_generate(self.ENGLISH_REWRITE_SYSTEM, prompt, "first attempt")
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
        fact_error = self._facts_check(attempt1, cleaned, source_mention)
        if fact_error:
            logger.warning(f"Fact check failed: {fact_error}")

        # Name validation (must preserve ALL source names, no hallucination)
        name_error = self._validate_names(attempt1, cleaned, vision_context, source_mention, source_username)
        if name_error:
            logger.warning(f"Name validation failed: {name_error}")

        # Check plagiarism
        if self._detect_plagiarism(cleaned, attempt1) or fact_error or name_error:
            strict_prompt = (
                f"Source: {protected_source}"
                f"{vision_prompt}\n\n"
                f"REQUIRED NAMES (must ALL appear in rewrite): {required_names_str}\n\n"
                f"SOURCE HANDLE: The final 🔵🔴 suffix MUST use the EXACT handle from REQUIRED NAMES. "
                f"Do NOT invent or substitute any other handle (e.g., @_BarcaInfo, @FabrizioRomano).\n\n"
                "REWRITE THIS TWEET IN NATURAL ENGLISH WITHOUT COPYING ANY WORDS. "
                "Change sentence structure, word order, and phrasing completely. "
                "CRITICAL: Every name in REQUIRED NAMES must appear in your output. Do not omit any. "
                "Do NOT add any player or staff names that are not in the source text or vision context. "
                "Only use vision context for player names or text visible in media. "
                "Do NOT add scene descriptions or inferred details not in the source text. "
                "FORMAT: Start with 🚨. Use \\n\\n between headline and body, and before the final 🔵🔴 [@source] attribution. "
                "State each fact exactly once — no redundant repetition of names or information. "
                "Respond ONLY with the rewritten tweet."
            )
            attempt2 = self._llm_generate(self.ENGLISH_REWRITE_SYSTEM, strict_prompt, "strict retry")
            if attempt2:
                attempt2 = self._restore_terms(attempt2, protected_mapping)
                attempt2 = self._apply_post_corrections(attempt2)
                fact_error2 = self._facts_check(attempt2, cleaned, source_mention)
                name_error2 = self._validate_names(attempt2, cleaned, vision_context, source_mention, source_username)
                if not fact_error2 and not name_error2 and not self._detect_plagiarism(cleaned, attempt2):
                    return attempt2
                logger.warning("Second attempt also failed checks, using first attempt")

        # Final length check and truncation for both attempts
        final_text = attempt1
        
        # If attempt2 was used and passed checks, use it
        if 'attempt2' in locals() and attempt2:
            final_text = attempt2
        
        # Ensure length compliance with retries
        final_text = self._enforce_length_limit(
            final_text, protected_source, vision_prompt, protected_mapping, 
            cleaned, vision_context, required_names_str,
            source_mention=source_mention, source_username=source_username
        )
        
        # Normalize output format (guarantees 🚨, \n\n sections, 🔵🔴 [@source] suffix)
        final_text = self._normalize_output_format(final_text, source_username, source_mention)
        
        # Check length after normalization (normalization shouldn't add much, but verify)
        from .timeutil import weighted_length
        if weighted_length(final_text) > self.max_chars:
            logger.warning(f"Text too long after format normalization ({weighted_length(final_text)} chars), attempting to fix")
            # Try to truncate while preserving format
            final_text = self._enforce_length_limit(
                final_text, protected_source, vision_prompt, protected_mapping,
                cleaned, vision_context, required_names_str,
                source_mention=source_mention, source_username=source_username
            )
            final_text = self._normalize_output_format(final_text, source_username, source_mention)
            if weighted_length(final_text) > self.max_chars:
                logger.error(f"Text still too long after normalization fix ({weighted_length(final_text)} chars)")
                return ""
        
        # Final name validation after length enforcement and normalization
        final_name_error = self._validate_names(final_text, cleaned, vision_context, source_mention, source_username)
        if final_name_error:
            logger.warning(f"Final name validation failed: {final_name_error}")
            # Try to fix by adding missing names
            fixed_text = self._fix_missing_names(final_text, cleaned, vision_context)
            if fixed_text and fixed_text != final_text:
                fixed_text = self._normalize_output_format(fixed_text, source_username, source_mention)
                fixed_error = self._validate_names(fixed_text, cleaned, vision_context, source_mention, source_username)
                if not fixed_error:
                    logger.info("Fixed missing names in rewrite")
                    return fixed_text
            # Fall back to attempt1 if it was valid
            if not name_error and attempt1:
                attempt1_norm = self._normalize_output_format(attempt1, source_username, source_mention)
                if weighted_length(attempt1_norm) <= self.max_chars:
                    return attempt1_norm
            return ""
        
        # Check for non-English text (Chinese, Arabic, etc.)
        if not self._is_english_only(final_text):
            logger.warning("Rewrite contains non-English characters, rejecting")
            # Fall back to attempt1 if it was valid and English-only, with length enforcement
            if not name_error and attempt1 and self._is_english_only(attempt1):
                attempt1_norm = self._normalize_output_format(
                    self._enforce_length_limit(
                        attempt1, protected_source, vision_prompt, protected_mapping,
                        cleaned, vision_context, required_names_str,
                        source_mention=source_mention, source_username=source_username
                    ),
                    source_username,
                    source_mention
                )
                if weighted_length(attempt1_norm) <= self.max_chars:
                    return attempt1_norm
            return ""
        
        return final_text

    def _enforce_length_limit(
        self, 
        text: str, 
        protected_source: str, 
        vision_prompt: str, 
        protected_mapping: Dict, 
        cleaned: str = "", 
        vision_context: Optional[Dict] = None, 
        required_names_str: str = "", 
        source_mention: Optional[str] = None, 
        source_username: str = "source",
        required_names: Optional[List[str]] = None
    ) -> str:
        """Garantit le respect des 280 caractères avec préservation de la source exclusive."""
        from .timeutil import weighted_length
        max_chars = self.max_chars

        if weighted_length(text) <= max_chars:
            return text

        logger.warning(f"Tweet trop long ({weighted_length(text)} chars), retentative concise...")
        
        # Prompt adapte selon la présence d'une source
        if source_mention:
            concise_prompt = (
                f"Source: {protected_source}"
                f"{vision_prompt}\n\n"
                f"REQUIRED NAMES: {required_names_str}\n\n"
                f"SOURCE HANDLE: The final 🔵🔴 suffix MUST use the EXACT handle from REQUIRED NAMES. "
                f"Do NOT invent or substitute any other handle.\n\n"
                "REWRITE IN NATURAL ENGLISH. MUST BE UNDER 280 CHARACTERS TOTAL.\n"
                "Structure: Headline starting with 🚨, body paragraph, ending with 🔵🔴 [@source].\n"
                "State each fact exactly ONCE. Do not repeat names or sentences.\n"
                "Respond ONLY with the rewritten tweet."
            )
        else:
            concise_prompt = (
                f"Source: {protected_source}"
                f"{vision_prompt}\n\n"
                f"REQUIRED NAMES: {required_names_str}\n\n"
                f"SOURCE HANDLE: The final 🔵🔴 suffix MUST use the EXACT handle from REQUIRED NAMES. "
                f"Do NOT invent or substitute any other handle.\n\n"
                "REWRITE IN NATURAL ENGLISH. MUST BE UNDER 280 CHARACTERS TOTAL.\n"
                "Structure: Headline starting with 🚨, body paragraph, ending with 🔵🔴.\n"
                "State each fact exactly ONCE. Do not repeat names or sentences.\n"
                "Respond ONLY with the rewritten tweet."
            )

        attempt3 = self._llm_generate(self.ENGLISH_REWRITE_SYSTEM, concise_prompt, "concise retry")
        if attempt3:
            attempt3 = self._restore_terms(attempt3, protected_mapping)
            attempt3 = self._apply_post_corrections(attempt3)
            attempt3_norm = self._normalize_output_format(attempt3, source_username, source_mention)
            if weighted_length(attempt3_norm) <= max_chars:
                name_err = self._validate_names(attempt3_norm, cleaned, vision_context, source_mention, source_username)
                if not name_err:
                    return attempt3_norm

        # Always add suffix (with source_mention or just 🔵🔴)
        if source_mention:
            handles = [h for h in re.findall(r'@?(\w+)', source_mention) if h.lower() != 'via']
            sources_str = " ".join([f"[@{h}]" for h in handles]) if handles else ""
            source_suffix = f"🔵🔴\n{sources_str}" if sources_str else "🔵🔴"
        else:
            source_suffix = "🔵🔴"

        suffix_len = weighted_length(source_suffix)
        available_len = max_chars - suffix_len

        if available_len < 30:
            return ""

        truncated = text[:available_len]
        last_para = truncated.rfind('\n\n')
        last_sentence = max(truncated.rfind('. '), truncated.rfind('! '), truncated.rfind('? '))

        if last_para > available_len * 0.4:
            body_part = truncated[:last_para].rstrip()
        elif last_sentence > available_len * 0.4:
            body_part = truncated[:last_sentence + 1].rstrip()
        else:
            body_part = truncated.rsplit(' ', 1)[0].rstrip() + "..."

        return body_part + source_suffix

    def _normalize_output_format(self, text: str, source_username: str, source_mention: Optional[str] = None) -> str:
        if not text:
            return text

        # Normalisation des sauts de ligne
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        # Détection des mots-clés d'urgence
        keywords_pattern = r'^(🚨\s*)?(JUST IN|BREAKING|OFFICIAL|OFFICIEL|URGENT)[\s:]*'
        match = re.match(keywords_pattern, text, flags=re.IGNORECASE)
        
        if match:
            kw = match.group(2).upper()
            if kw == "OFFICIEL":
                kw = "OFFICIAL"
            text_without_kw = text[match.end():].strip()
            text = f"🚨 {kw}: {text_without_kw}"
        elif not text.startswith('🚨'):
            text = '🚨 ' + text

        # Suppression de toute attribution/suffixe en fin de texte existant
        text = re.sub(r'\s*🔵🔴(\s*\n?\s*(\[@\w+\]\s*)+)?\s*$', '', text).rstrip()
        text = re.sub(r'\s*(\[@\w+\]\s*)+$', '', text).rstrip()

        # Construction du suffixe pour 1 ou plusieurs sources
        # Priority: source_mention (extracted from tweet text) > source_username (monitored account)
        if source_mention:
            # Nettoyage des mentions dans le corps
            text = re.sub(rf'\s*\[?@{re.escape(source_mention)}\]?\s*', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r' +', ' ', text).strip()
            
            # Extrait tous les handles (handles séparés par 'via', espace, virgule, etc.)
            handles = [h for h in re.findall(r'@?(\w+)', source_mention) if h.lower() != 'via']
            
            if handles:
                sources_str = " ".join([f"[@{h}]" for h in handles])
                expected_suffix = f"🔵🔴\n{sources_str}"
            else:
                expected_suffix = "🔵🔴"
        elif source_username and source_username.strip():
            # Use source_username (monitored account) when no source_mention in tweet
            expected_suffix = f"🔵🔴\n[@{source_username}]"
        else:
            expected_suffix = "🔵🔴"

        text = text.rstrip() + expected_suffix

        return re.sub(r'[ \t]+\n', '\n', text)

    def _reapply_suffix_format(self, text: str, source_username: str) -> str:
        """Re-apply suffix format after duplicate removal may have altered it."""
        expected_suffix = f'🔵🔴 [@{source_username}]'
        # Remove any trailing attribution for the monitored account only
        text = re.sub(rf'\s*🔵🔴\s*\[@{re.escape(source_username)}\]\s*$', '', text).rstrip()
        text = re.sub(rf'\s*\[@{re.escape(source_username)}\]\s*$', '', text).rstrip()
        # Ensure proper separation before suffix
        if not text.endswith('\n\n'):
            text = text + '\n\n'
        # Add correct suffix (without rstrip which would remove the \n\n)
        if not text.endswith(expected_suffix):
            text = text + expected_suffix
        return text

    def _remove_duplicate_facts(self, text: str, required_names: List[str]) -> str:
        """Remove duplicate facts/sentences while preserving all required names.
        
        Splits text into paragraphs, compares factual content, and removes
        redundant paragraphs that don't add new information.
        """
        if not text or '\n\n' not in text:
            return text
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        if len(paragraphs) <= 1:
            return text
        
        # Extract key facts from each paragraph (names + verbs + key nouns)
        def extract_facts(para: str) -> set:
            # Find all words that look like names (capitalized) and key football terms
            words = re.findall(r'\b[A-Z][a-z]+\b|\b(?:score|sign|win|loss|defeat|victory|goal|brace|hat-trick|injury|transfer|contract|deal|agree|extend|join|leave|move)\b', para, re.IGNORECASE)
            # Also include required names if present
            found_names = set()
            para_lower = para.lower()
            for name in required_names:
                if name.lower() in para_lower:
                    found_names.add(name.lower())
            return set(w.lower() for w in words) | found_names
        
        para_facts = [extract_facts(p) for p in paragraphs]
        
        # Keep paragraphs that add new facts
        kept_paragraphs = []
        seen_facts = set()
        
        for i, (para, facts) in enumerate(zip(paragraphs, para_facts)):
            # Always keep the first paragraph (headline)
            if i == 0:
                kept_paragraphs.append(para)
                seen_facts.update(facts)
                continue
            
            # For subsequent paragraphs, check if they add new facts
            new_facts = facts - seen_facts
            # A paragraph adds value if it has at least 2 new factual words or introduces a new required name
            new_names = new_facts & {n.lower() for n in required_names}
            
            if len(new_facts) >= 2 or new_names:
                kept_paragraphs.append(para)
                seen_facts.update(facts)
            else:
                logger.debug(f"Removed duplicate paragraph: {para[:80]}...")
        
        # Ensure all required names still appear in the result
        result = '\n\n'.join(kept_paragraphs)
        result_lower = result.lower()
        missing_names = [n for n in required_names if n.lower() not in result_lower]
        
        if missing_names:
            # If removing duplicates caused name loss, fall back to original
            logger.warning(f"Duplicate removal caused missing names: {missing_names}, reverting")
            return text
        
        return result

    # System prompt for @UniverseCule
    ENGLISH_REWRITE_SYSTEM = """You are @UniverseCule, a professional football journalist specializing in FC Barcelona content.
Your task: Rewrite tweets in natural, idiomatic English while ensuring 100% originality and avoiding plagiarism.

ABSOLUTE RULES:
1. NEVER copy-paste any part of the original text. Always rephrase using completely different word choices and sentence structure.
2. Preserve ALL facts, names, numbers, and specific details exactly as in the source.
3. YOU MUST INCLUDE EVERY player name and staff name from the source text in your rewrite. Do not omit any.
4. NEVER ADD player names, staff names, or any proper nouns that are NOT explicitly mentioned in the source text or vision context.
5. Use precise and professional football terminology in English.
6. **MAXIMUM 280 CHARACTERS TOTAL - THIS IS HARD LIMIT.** No links, hashtags, or any external references.
7. **FORMAT REQUIREMENTS:**
   - **Headline MUST start with 🚨** for breaking/urgent news - NO EXCEPTIONS
   - **Preserve alert keywords:** If source contains JUST IN, BREAKING, OFFICIAL, URGENT (or French OFFICIEL), keep the keyword in uppercase immediately after 🚨 (e.g., "🚨 JUST IN:", "🚨 BREAKING:", "🚨 OFFICIAL:")
   - Convert French "OFFICIEL" to English "OFFICIAL"
   - **Use line breaks (\\n\\n) between logical sections for readability - MANDATORY**
   - **Tweet MUST end with 🔵🔴 [@source_mention]** if source_mention exists, otherwise end with **🔵🔴** (always present, directly attached)
   - **SOURCE HANDLE RULE: The source handle in the final 🔵🔴 suffix MUST EXACTLY MATCH the handle provided in REQUIRED NAMES. NEVER invent, substitute, or modify the source handle. NEVER use handles like @_BarcaInfo, @FabrizioRomano, or any other aggregator unless explicitly provided as the source.**
   - **NO duplicate/redundant sentences** - each fact stated exactly once
   - No text after the emoji suffix
8. Never mix languages. Always use English exclusively.
9. Maintain the original meaning and tone while making it sound natural and professional.
10. NEVER modify protected terms (they appear as __PROTECTED_X__ - leave them exactly as-is).
11. Proper nouns (names, places, competitions) keep their standard capitalization.
12. **Sentence case only - NO UPPERCASE words (except preserved alert keywords).**

EXAMPLES:
- Original: "JUST IN: Barcelona sign new striker"
  Rewritten: "🚨 JUST IN: FC Barcelona secure new forward ✍️\n\nThe club has finalized the acquisition of a promising attacker to bolster their front line.🔵🔴 [@source]"

- Original: "BREAKING: Real Madrid 3-0 victory over Barcelona"
  Rewritten: "🚨 BREAKING: Real Madrid dominate 3-0 in El Clasico ⚪\n\nLos Blancos deliver a commanding performance to claim bragging rights in Madrid.🔵🔴 [@source]"

- Original: "OFFICIEL: Lewandowski scores 2 goals"
  Rewritten: "🚨 OFFICIAL: Lewandowski brace fires Barcelona to victory ⚽⚽\n\nThe Polish striker nets twice as the Blaugrana secure three vital points.🔵🔴 [@source]"

- Original: "Pedri, Gavi et De Jong dominent le milieu de terrain"
  Rewritten: "🚨 Pedri, Gavi and De Jong control midfield\n\nThe trio dictates play as Barcelona secure possession dominance.🔵🔴"

- Original: "Team rebounds after losses. Meeting went well. Errors acknowledged."
  Rewritten: "🚨 FC Barcelona aims to bounce back after two defeats.\n\nThe atmosphere improved after yesterday's meeting. Players acknowledged mistakes.\n\nTheir belief: we can win everything with the best squad.🔵🔴"

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
- **Headline MUST start with 🚨** for breaking/urgent news
- **Preserve alert keywords** (JUST IN, BREAKING, OFFICIAL, URGENT) immediately after 🚨
- **Tweet MUST end with 🔵🔴 [@source_mention]** if source_mention exists, otherwise end with **🔵🔴** (always present, directly attached)
- Use ⚽ for goals/match action (in headline or body)
- Use ✍️ for signings/transfers (in headline or body)
- Use 🏆 for trophies/titles (in headline or body)
- Keep emojis minimal and professional - max 3 per tweet
- **NO emojis after the 🔵🔴 suffix**

CRITICAL: Count characters before responding. If over 280, shorten aggressively. The source attribution [@source] at the end is ONLY added if source_mention exists. Line breaks (\\n\\n) between sections are mandatory. Headline MUST start with 🚨. No duplicate sentences."""

    def rewrite(self, text: str, source_username: str = "source", 
                 language: str = "en", vision_context: Optional[Dict] = None) -> RewriteResult:
        """Main rewrite method with language detection and optimization.
        
        Args:
            text: Source tweet text
            source_username: Source account username
            language: Detected language (en, es, fr, ar, etc.)
            vision_context: Optional dict with vision analysis results:
                - players_detected: List[str]
                - scene_type: str
                - text_extracted: str
                - key_facts: List[str]
        """
        cleaned = clean_tweet_text(text)
        if not cleaned:
            return RewriteResult(success=False, error="Empty text after cleaning")

        try:
            # Use provided language or detect (fast heuristic)
            if language == "ar" or self._is_arabic(cleaned):
                # Arabic → English translation
                translated = self._translate_arabic_to_english(cleaned)
                if translated:
                    rewritten = self._rewrite_to_english(translated, "ar", vision_context, source_username)
                else:
                    rewritten = cleaned  # Fallback to original if translation fails
            else:
                # English/Spanish/French → English reformulation
                rewritten = self._rewrite_to_english(cleaned, "en", vision_context, source_username)
            
            if not rewritten:
                return RewriteResult(success=False, error="Rewrite produced empty result")
            
            # Final cleanup: remove any remaining protected term placeholders
            rewritten = self._cleanup_protected_terms(rewritten)
            
            return RewriteResult(success=True, rewritten_text=rewritten)
        except Exception as e:
            logger.error(f"Rewrite failed: {e}")
            return RewriteResult(success=False, error=str(e))

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

    def test_rewrite(self, text: str, username: str = "test", language: str = "en", vision_context: Optional[Dict] = None) -> Dict:
        """Test rewrite with detailed diagnostics."""
        original = clean_tweet_text(text)
        rewritten = self.rewrite(text, username, language, vision_context)
        return {
            "original": original,
            "rewritten": rewritten,
            "original_length": weighted_length(original),
            "rewritten_length": weighted_length(rewritten),
            "model": self.model,
            "success": bool(rewritten),
            "within_limit": weighted_length(rewritten) <= self.max_chars,
        }