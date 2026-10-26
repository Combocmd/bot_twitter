"""French rewrite and reply generation with local Ollama.

The prompt is deliberately explicit about French football terminology because
small local models tend to translate "right-back" as "ailier droit" and
"reports" as "ragots". The output is always checked against X's weighted
280-character limit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

import ollama
from loguru import logger

from .config import get_config
from .timeutil import clean_tweet_text, truncate_weighted, weighted_length
from .translator import ArabicTranslator, is_arabic

try:
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
except Exception:  # pragma: no cover
    detect = None


@dataclass
class RewriteResult:
    original_text: str
    rewritten_text: str
    original_length: int
    rewritten_length: int
    language_detected: str
    model_used: str
    success: bool
    error: Optional[str] = None
    within_limit: bool = True


class TweetRewriter:
    SYSTEM_PROMPT = """Tu es @UniversCule, un média francophone passionné, 100% FC Barcelona.
Tu écris comme un journaliste sportif français professionnel : naturel, direct, vivant, jamais robotique.

RÈGLES ABSOLUES :
1. Réponds UNIQUEMENT avec le texte final du tweet, en FRANÇAIS.
2. Traduis toute information non francophone. Ne recopie jamais la langue source.
3. Garde les faits exacts. N'invente rien, ne rajoute pas de contexte absent.
4. Style : headline en MAJUSCULES, emojis pertinents, sauts de ligne pour la lisibilité.
5. Maximum 260 caractères. Pas de lien, pas de mention @ (sauf pour citer une source présente dans le tweet original), pas de markdown, pas de hashtag.
6. Vocabulaire football correct :
   - right-back = latéral droit ; left-back = latéral gauche
   - centre-back = défenseur central ; midfielder = milieu de terrain
   - winger = ailier ; striker = attaquant / buteur
   - reports / according to reports = selon / d'après les informations
   - loan = prêt ; signing = recrue ; transfer = transfert
   - squad list = liste des convoqués ; lineup = composition
   - fixture = rencontre ; kick-off = coup d'envoi
   - hat-trick = triplé ; brace = doublé
   - substitution = remplacement / changement ; enters (the pitch) = entre en jeu ; comes off = sort
   - "levels the match" / "levelle le match" = égalise
   - "But n°2" = deuxième but ; n'écris jamais "n°" dans le tweet final
   - El Clasico = le Clasico ; La Liga = la Liga
   - "boosted" (réseaux sociaux) = mis en avant / sponsorisé, selon le contexte
7. Ne traduis pas les noms propres, les surnoms de joueurs ni les noms de clubs.
8. N'invente AUCUN chiffre, score, statistique, date ou nom absent de la source.
9. Pas de point d'exclamation à répétition. Les emojis sont autorisés avec parcimonie.
10. Jamais d'espace après une apostrophe d'élision ("l'égalisation", pas "l' égalisation").
11. Ne mélange JAMAIS l'anglais et le français : "the", "goal", "substitution", "enters", "levels" sont interdits dans le tweet final.
12. Changements : écris une phrase française claire, jamais de flèches ni de pastilles de couleur.
    Format attendu : "Double changement pour le Barça à la 85e : Adeyemi et Gabriel Jesus entrent, Gordon et Raphinha sortent."
    Interdits dans le tweet final : ↑ ↓ ⬆ ⬇ 🟢 🔴 et toute notation brute de type "↑🟢 Joueur".
13. ANGLICISMES INTERDITS :
    - "website" → "site web"
    - "award" → "trophée" ou "prix"
    - "announcement" → "annonce"
    - "update" → "mise à jour"
    - "coming" → "arrive" ou "prochain"
    - "news" → "info" ou "nouvelle"
14. FORMAT MEDIA FRANÇAIS :
    - Début par des emojis pertinents (🚨 pour info urgente, 🌕 pour événement nocturne, ✈️ pour voyage, 💙🔴 pour Barça)
    - Headline en MAJUSCULES suivi d'un saut de ligne
    - Contenu en lettres normales
    - Si le tweet original cite une source (@username, "selon @xx", "Source: @xx"), conserve cette citation
    - Ne PAS ajouter de source si le tweet original n'en cite pas

Exemples de format attendu :
- "🚨🚨🚨 KOUNDÉ ET CHRISTENSEN : DES RETOURS ESPÉRÉS \n\nHansi Flick espère récupérer Jules Koundé pour affronter Alavés et Andreas Christensen pour le match contre l'Atlético de Madrid."
- "🌕🚨 LA DÉLÉGATION DU BARÇA VIENT D'ATTERRIR À LONDRES ! ✈️\n\nLes Blaugrana se dirigent désormais vers leur hôtel avant la cérémonie du Ballon d'Or ce soir."
- "🚨 LE BARÇA DÉVOILE UNE VIDÉO SPÉCIALE AVANT LA CÉRÉMONIE DU BALLON D'OR 🤩\n\n12 maillots, des joueuses et des joueurs mis à l'honneur… 💙🔴"
- Avec source citée DANS le tweet : "🚨 FERRAN TORRES 16E DU BALLON D'OR 🏆\n\nL'attaquant espagnol termine 16e du vote, selon @Barca_buzz."

Rappel : le Barça = bleu et rouge (💙🔴), toujours dans cet ordre !
"""

    REPLY_PROMPT = """Tu es @UniversCule, média francophone 100% FC Barcelona.
Tu réponds à un commentaire sous l'un de nos posts. Réponds en FRANÇAIS, en une phrase naturelle, respectueuse et utile.
Ne répète pas le commentaire, ne mets pas de lien, pas de hashtag, pas de mention @, pas de "en tant qu'IA".
Maximum 220 caractères. Si le commentaire est une simple réaction, reste bref et chaleureux. Si c'est une question, réponds précisément sans inventer.
"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.client = ollama.Client(host=self.config.ollama.host)
        self.model = self.config.ollama.model
        self.max_chars = int(self.config.length.max_chars)
        self.temperature = float(self.config.ollama.temperature)
        self._translator: Optional[ArabicTranslator] = None

    # ------------------------------------------------------------------
    # Ollama plumbing
    # ------------------------------------------------------------------
    def _chat(self, system: str, user: str, temperature: Optional[float] = None) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": 220,
            },
        )
        try:
            message = response["message"]["content"]
        except Exception:
            message = getattr(getattr(response, "message", None), "content", "")
        return str(message or "").strip()

    def _sanitize(self, text: str) -> str:
        text = clean_tweet_text(text)
        text = re.sub(r"^\s*(?:tweet|réponse|post)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*(?:bien sûr|voici|d'accord)[,:\s]+", "", text, flags=re.IGNORECASE)
        text = text.strip().strip('"').strip("'").strip()
        # Preserve newlines for media format: headline\ncontent\n[source]
        text = re.sub(r"[^\S\n]+", " ", text)  # Collapse horizontal whitespace only
        text = re.sub(r"\n{3,}", "\n\n", text)  # Limit consecutive newlines
        # Do not let the model insert source handles or links.
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"(?<!\w)@[A-Za-z0-9_]{2,15}", "", text)
        text = re.sub(r"#\w+", "", text, flags=re.UNICODE)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([.!?])([A-ZÀ-Ý])", r"\1 \2", text)
        # French typography: a space before : ; ! ? (but never inside a time
        # such as 21:30, where the colon sits between two digits).
        text = re.sub(r"(?<=[A-Za-zÀ-ÿ])\s*:", " :", text)
        text = re.sub(r"(?<=[A-Za-zÀ-ÿ0-9])\s*([;!?])", r" \1", text)
        return text

    def _looks_french(self, text: str) -> bool:
        if detect is None:
            return True
        cleaned = clean_tweet_text(text)
        if len(cleaned) < 25:
            return True
        try:
            lang = detect(cleaned)
        except Exception:
            return True
        if lang in {"fr", "unknown"}:
            return True
        # Short French sentences with foreign player names are frequently
        # misdetected (e.g. "Lewandowski signe 2 buts ce soir" -> Polish).
        # Accept when clear French markers are present; real English output is
        # still rejected by _ENGLISH_LEAKS below.
        markers = {m.lower() for m in self._FRENCH_MARKERS.findall(cleaned)}
        return len(markers) >= 2

    def _detect_source_citations(self, text: str) -> List[str]:
        """Detect @mentions that appear to be source citations."""
        # Find all @mentions in the text
        mentions = re.findall(r"@([A-Za-z0-9_]{2,15})", text)
        # Filter out common non-source mentions
        non_sources = {"barcelona", "fcb", "fcbarcelona", "universcule"}
        sources = [m for m in mentions if m.lower() not in non_sources]
        return list(set(sources))

    # Output fragments that were observed as broken/low-quality generations.
    _BAD_OUTPUTS = {
        "levelle", "leveling", "anxiety", "whistle", "passed'ici", "fermindinho",
        "the match", "in the match", "rush",
    }
    # Distinctive French function words, used to rescue a short French
    # sentence that langdetect misclassifies because of a foreign player name.
    _FRENCH_MARKERS = re.compile(
        r"\b(?:le|la|les|des|du|une|est|pour|avec|sur|dans|aux|ce|cette|ces|"
        r"ils|elles|ont|après|selon|plus|mais|et|à|au|en|par|son|sa|ses|"
        r"leur|leurs|nous|vous|je|tu|il|elle|ne|pas|très|bien|match|buts|"
        r"entrent|sort|changement|remplacement|égalise|victoire|défaite)\b",
        re.IGNORECASE,
    )
    # Raw substitution notation copied from the source (arrows/coloured dots)
    # is not publishable by a media account.
    _RAW_NOTATION = re.compile(r"[↑↓⬆⬇🟢🔴🟥]")
    # English words that must never survive into a French media tweet.
    # NB: "match", "score" and "chance" are valid French words and are NOT leaks.
    _ENGLISH_LEAKS = re.compile(
        r"\b(?:the|and|with|from|into|after|before|goal|shot|missed|"
        r"according|reports|substitution|enters|comes|off|level|levels|"
        r"half-time|full-time|kick-off|lineup|squad)\b",
        re.IGNORECASE,
    )
    # Deterministic repairs applied before validation.
    _GLOSSARY_FIXES = [
        (re.compile(r"\baccording\s+to\s+reports?\b", re.IGNORECASE), "selon les informations"),
        (re.compile(r"\b(?:levelle|levélle|levels?)\s+(?:le\s+|la\s+|the\s+)?(?:match|score|game)\b", re.IGNORECASE), "égalise"),
        (re.compile(r"\b(?:levelle|levélle)\b", re.IGNORECASE), "égalise"),
        (re.compile(r"\blevels\b", re.IGNORECASE), "égalise"),
        (re.compile(r"\blevel\b", re.IGNORECASE), "égalise"),
        (re.compile(r"\bdouble\s+substitution\b", re.IGNORECASE), "double changement"),
        (re.compile(r"\bsubstitutions\b", re.IGNORECASE), "remplacements"),
        (re.compile(r"\bsubstitution\b", re.IGNORECASE), "remplacement"),
        (re.compile(r"\benters\b", re.IGNORECASE), "entre en jeu"),
        (re.compile(r"\bcomes\s+off\b", re.IGNORECASE), "sort"),
        (re.compile(r"\bbut\s+n°\s*(\d+)\b", re.IGNORECASE), r"\1e but"),
        (re.compile(r"\bde\s+(Real\s+Madrid|FC\s+Barcelona|FCB)\b", re.IGNORECASE), r"du \1"),
    ]
    # Words that may legitimately become digits in a French rewrite; used to
    # normalize the SOURCE before the invented-number check.
    _WORD_NUMBERS = {
        "zero": "0", "nul": "0", "one": "1", "un": "1", "une": "1", "two": "2",
        "deux": "2", "three": "3", "trois": "3", "four": "4", "quatre": "4",
        "five": "5", "cinq": "5", "six": "6", "seven": "7", "sept": "7",
        "eight": "8", "huit": "8", "nine": "9", "neuf": "9", "ten": "10",
        "dix": "10", "first": "1", "premier": "1", "première": "1",
        "second": "2", "deuxième": "2", "third": "3", "troisième": "3",
        "fourth": "4", "fifth": "5", "brace": "2", "hat-trick": "3",
    }

    def _apply_glossary(self, text: str) -> str:
        for pattern, replacement in self._GLOSSARY_FIXES:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _numbers(text: str) -> set:
        return set(re.findall(r"\d+", text))

    # Facts expressed as ordinals in the source: the rewrite must keep the
    # same fact instead of substituting an invented one (e.g. turning
    # "second goal" into "égalisation", which adds information).
    _ORDINAL_FACTS = (
        (re.compile(r"\b(?:second|2nd)\s+goal\b", re.IGNORECASE),
         re.compile(r"\b(?:deuxième|2e|second)\s+but\b", re.IGNORECASE),
         "deuxième but"),
        (re.compile(r"\b(?:third|3rd)\s+goal\b", re.IGNORECASE),
         re.compile(r"\b(?:troisième|3e)\s+but\b", re.IGNORECASE),
         "troisième but"),
        (re.compile(r"\b(?:equaliser|equalizer|levels?\s+the\s+match)\b", re.IGNORECASE),
         re.compile(r"\bégalis", re.IGNORECASE),
         "égalisation"),
    )

    def _facts_check(self, rewritten: str, source: str) -> Optional[str]:
        """Reject any number or score that does not exist in the source."""
        src = source.lower()
        for word, digit in self._WORD_NUMBERS.items():
            src = re.sub(rf"\b{re.escape(word)}\b", digit, src)
        missing = sorted(
            n for n in self._numbers(rewritten)
            if n not in self._numbers(src)
        )
        if missing:
            return f"chiffre absent de la source: {', '.join(missing)}"
        normalize = lambda s: re.sub(r"[–—:/]", "-", s)
        out_scores = set(re.findall(r"\b(\d{1,3})\s*-\s*(\d{1,3})\b", normalize(rewritten)))
        if out_scores:
            src_scores = set(re.findall(r"(\d{1,3})\s*-\s*(\d{1,3})", normalize(src)))
            for score in sorted(out_scores):
                if score not in src_scores:
                    return f"score absent de la source: {score[0]}-{score[1]}"
        for source_pattern, expected, label in self._ORDINAL_FACTS:
            if source_pattern.search(source) and not expected.search(rewritten):
                return f"fait non conservé: {label} attendu"
        return None

    def _invalid(self, text: str, source: str = "") -> Optional[str]:
        if not text:
            return "réponse vide"
        if len(text) < 8:
            return "réponse trop courte"
        if re.search(
            r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af"
            r"\u0600-\u06ff\u0750-\u077f\u0590-\u05ff\u0400-\u04ff]",
            text,
        ):
            return "caractères non latins détectés"
        if not self._looks_french(text):
            return "langue non française"
        low = text.lower()
        for bad in self._BAD_OUTPUTS:
            if bad in low:
                return f"formulation suspecte: « {bad} »"
        leak = self._ENGLISH_LEAKS.search(text)
        if leak:
            return f"anglicisme/anglais détecté: « {leak.group(0)} »"
        # Angles courants à éviter
        for anglicism in ["website", "web site", "award", "announcement", "update", "coming", "news"]:
            if anglicism in low:
                return f"anglicisme détecté: « {anglicism} »"
        if self._RAW_NOTATION.search(text):
            return "notation brute de remplacement (flèches/pastilles)"
        if re.search(r"[A-Za-zÀ-ÿ](['’])\s", text):
            return "élision cassée (espace après apostrophe)"
        if re.search(r"[A-Za-zÀ-ÿ]'\s+[A-ZÀ-Ý0-9]", text):
            return "élision incorrecte devant majuscule ou chiffre"
        if re.search(r"[A-Za-zÀ-ÿ]'[0-9]", text):
            return "élision incorrecte devant un chiffre"
        if source:
            facts_error = self._facts_check(text, source)
            if facts_error:
                return facts_error
        return None

    # ------------------------------------------------------------------
    # Rewrite
    # ------------------------------------------------------------------
    def rewrite(self, text: str, source_username: str = "source", source_lang: str = "auto") -> RewriteResult:
        original = clean_tweet_text(text)
        if not original:
            return RewriteResult(
                text, "", 0, 0, source_lang, self.model, False, "texte source vide", False
            )
        model_source = original
        literal_translation: Optional[str] = None
        if (
            getattr(self.config.translation, "enabled", True)
            and is_arabic(original)
        ):
            if self._translator is None:
                self._translator = ArabicTranslator(self.config)
            literal_translation = self._translator.translate(original, source_lang)
            if literal_translation:
                model_source = literal_translation
                logger.debug("Traduction littérale arabe: {}", literal_translation[:160])
        if literal_translation:
            user_prompt = (
                f"Compte source : @{source_username}\n"
                "Traduction littérale fidèle d'un tweet arabe :\n"
                f"{literal_translation}\n\n"
                "Transforme cette traduction en tweet français naturel, en corrigeant "
                "les maladresses sans changer les faits, les noms ou les chiffres. "
                "N'ajoute aucune information absente. Écris le tweet final maintenant, "
                "sans préambule."
            )
        else:
            # Détecter les citations de source dans le tweet original
            source_citations = self._detect_source_citations(original)
            citation_info = ""
            if source_citations:
                citation_info = f"Citations de source présentes dans le tweet : {', '.join('@' + s for s in source_citations)}. Conserve-les dans ton tweet.\n"
            user_prompt = (
                f"Compte source : @{source_username}\n"
                f"Langue source : {source_lang or 'auto'}\n"
                f"{citation_info}"
                f"Information à transformer en tweet français :\n{original}\n\n"
                "Écris le tweet final maintenant, sans préambule."
            )
        last_error = None
        candidate = ""
        for attempt in range(2):
            try:
                candidate = self._apply_glossary(self._sanitize(self._chat(self.SYSTEM_PROMPT, user_prompt)))
                error = self._invalid(candidate, model_source)
                if error and attempt == 0:
                    last_error = error
                    user_prompt = (
                        f"Ta réponse précédente était invalide ({error}).\n"
                        "Rappels impératifs : 100% français (jamais 'the', 'goal', "
                        "'substitution', 'levels'...), vocabulaire football correct "
                        "(égaliser, remplacement, entre en jeu, sort), aucun chiffre "
                        "ni score absent de la source, jamais d'espace après une "
                        "apostrophe d'élision, aucune flèche ni pastille de couleur "
                        "(↑ ↓ 🟢 🔴 interdits : écris « X et Y entrent, Z et W sortent »), "
                        "et conserve le fait exact de la source (un « deuxième but » "
                        "n'est pas une « égalisation »).\n"
                        "Réécris cette information en français, en respectant le style "
                        f"et le maximum de 260 caractères :\n{model_source}\n"
                        "Réponds uniquement avec le tweet final."
                    )
                    continue
                if error:
                    last_error = error
                    break
                if weighted_length(candidate) > 260:
                    # Ask once for a shorter version.
                    try:
                        shortened = self._sanitize(self._chat(
                            self.SYSTEM_PROMPT,
                            f"Raccourcis ce tweet en français à 240 caractères maximum, sans perdre le fait principal :\n{candidate}\n"
                            "Réponds uniquement avec le tweet final.",
                        ))
                        if shortened and self._invalid(shortened, model_source) is None:
                            candidate = shortened
                    except Exception as exc:
                        logger.debug("Raccourcissement Ollama impossible: {}", exc)
                if weighted_length(candidate) > self.max_chars:
                    candidate = truncate_weighted(candidate, self.max_chars)
                # Restaurer les citations de source si présentes dans l'original
                if source_citations:
                    for cit in source_citations:
                        if f"@{cit}" not in candidate:
                            # Ajouter la citation si elle n'est pas déjà présente
                            candidate = candidate.rstrip() + f" @{cit}"
                            if weighted_length(candidate) > self.max_chars:
                                candidate = truncate_weighted(candidate, self.max_chars)
                                break
                # Cas spécial : Fabrizio Romano - si pas d'autre source, le citer lui
                elif source_username.lower() == "fabrizioromano" and not candidate.endswith("@FabrizioRomano"):
                    candidate = candidate.rstrip() + " @FabrizioRomano"
                    if weighted_length(candidate) > self.max_chars:
                        candidate = truncate_weighted(candidate, self.max_chars)
                return RewriteResult(
                    original_text=original,
                    rewritten_text=candidate,
                    original_length=weighted_length(original),
                    rewritten_length=weighted_length(candidate),
                    language_detected=self._detect_language(original, source_lang),
                    model_used=self.model,
                    success=True,
                    error=None,
                    within_limit=weighted_length(candidate) <= self.max_chars,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning("Ollama échec réécriture (tentative {}): {}", attempt + 1, exc)

        # French source fallback: post a clean version rather than dropping the
        # news entirely if Ollama is temporarily unavailable.
        lang = self._detect_language(original, source_lang)
        if lang == "fr" and self._invalid(original) is None:
            fallback = truncate_weighted(original, self.max_chars)
            # Restaurer les citations de source si présentes dans l'original
            source_citations = self._detect_source_citations(original)
            if source_citations:
                for cit in source_citations:
                    if f"@{cit}" not in fallback:
                        fallback = fallback.rstrip() + f" @{cit}"
                        if weighted_length(fallback) > self.max_chars:
                            fallback = truncate_weighted(fallback, self.max_chars)
                            break
            # Cas spécial : Fabrizio Romano - si pas d'autre source, le citer lui
            elif source_username.lower() == "fabrizioromano" and not fallback.endswith("@FabrizioRomano"):
                fallback = fallback.rstrip() + " @FabrizioRomano"
                if weighted_length(fallback) > self.max_chars:
                    fallback = truncate_weighted(fallback, self.max_chars)
            return RewriteResult(
                original, fallback, weighted_length(original), weighted_length(fallback),
                lang, "fallback-fr", True,
                f"Ollama indisponible ({last_error}); texte français conservé",
                weighted_length(fallback) <= self.max_chars,
            )
        return RewriteResult(
            original, "", weighted_length(original), 0, lang, self.model, False,
            f"réécriture impossible: {last_error}", False,
        )

    @staticmethod
    def _detect_language(text: str, fallback: str = "unknown") -> str:
        if fallback and fallback not in {"auto", "unknown"}:
            return fallback
        if detect is None or len(text) < 12:
            return fallback or "unknown"
        try:
            return detect(text)
        except Exception:
            return fallback or "unknown"

    def generate_reply(self, comment_text: str, our_tweet_text: str, author: str = "") -> str:
        comment = clean_tweet_text(comment_text)
        our = clean_tweet_text(our_tweet_text)
        user = (
            f"Notre post : {our}\n"
            f"Commentaire de @{author or 'un utilisateur'} : {comment}\n"
            "Écris notre réponse en français, une phrase, maximum 200 caractères."
        )
        try:
            reply = self._sanitize(self._chat(self.REPLY_PROMPT, user, temperature=0.65))
            if reply and self._invalid(reply) is None:
                return truncate_weighted(reply, min(220, self.max_chars))
        except Exception as exc:
            logger.warning("Ollama échec réponse: {}", exc)
        # Safe, human fallback; never empty.
        return truncate_weighted("Merci pour ton commentaire, Visca Barça 🔵🔴", 220)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def test_rewrite(self, text: str, username: str = "test") -> Dict:
        result = self.rewrite(text, username)
        return {
            "original": result.original_text,
            "rewritten": result.rewritten_text,
            "original_length": result.original_length,
            "rewritten_length": result.rewritten_length,
            "language": result.language_detected,
            "model": result.model_used,
            "success": result.success,
            "within_limit": result.within_limit,
            "error": result.error,
        }

    def check_model(self) -> bool:
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
            logger.warning("Vérification Ollama impossible: {}", exc)
            return False
