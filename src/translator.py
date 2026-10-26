"""Local, free Arabic-to-French translation support.

Qwen2.5:7b is good at French style but weak at reading Arabic names. For
Arabic sources we therefore run a dedicated local NLLB translator first, with
a football name/term glossary, then let Ollama turn the literal French into a
natural tweet. Everything stays on the machine after the one-time model
download.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from loguru import logger

from .config import get_config
from .timeutil import clean_tweet_text

_ARABIC_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f]")

# Arabic transliterations and recurring football terms. Longest keys are
# replaced first so "ريال مدريد" is not partially changed by "مدريد".
_GLOSSARY: Dict[str, str] = {
    "ريال مدريد": "Real Madrid",
    "برشلونة": "Barcelone",
    "بارسا": "Barça",
    "الكلاسيكو": "Clasico",
    "كلاسيكو": "Clasico",
    "مدريد": "Madrid",
    "الدوري الإسباني": "Liga",
    "الدوري": "Liga",
    "هداف": "meilleur buteur",
    "حارس": "gardien",
    "مدرب": "entraîneur",
    "رئيس": "président",
    "لاعب": "joueur",
    "إصابة": "blessure",
    "مباراة": "match",
    "هدف": "but",
    "انتقال": "transfert",
    "عقد": "contrat",
    "تجديد": "prolongation",
    "النادي": "club",
    "الفريق": "équipe",
    "اليوم": "aujourd'hui",
    "الليلة": "ce soir",
    "رسميا": "officiel",
    "عاجل": "urgent",
    "أُتخذ القرار": "La décision est prise",
    "اتخذ القرار": "La décision est prise",
    # Barcelona players / staff
    "لامين يامال": "Lamine Yamal",
    "لامين": "Lamine",
    "يامال": "Yamal",
    "رافينيا": "Raphinha",
    "رافينها": "Raphinha",
    "بيدري": "Pedri",
    "غافي": "Gavi",
    "فرينكي دي يونغ": "Frenkie de Jong",
    "دي يونغ": "de Jong",
    "روبرت ليفاندوفسكي": "Lewandowski",
    "ليفاندوفسكي": "Lewandowski",
    "تير شتيغن": "ter Stegen",
    "شتيغن": "ter Stegen",
    "كوندي": "Koundé",
    "أراوخو": "Araújo",
    "إينيغو مارتينيز": "Iñigo Martínez",
    "بالدي": "Balde",
    "داني أولمو": "Dani Olmo",
    "فيرمين": "Fermín",
    "أنسو فاتي": "Ansu Fati",
    "فيران توريس": "Ferran Torres",
    "إريك غارسيا": "Eric García",
    "كاسادو": "Casadó",
    "بابلو توري": "Pablo Torre",
    "هانسي فليك": "Hansi Flick",
    "فليك": "Flick",
    "لابورتا": "Laporta",
    "ديكو": "Deco",
    "تشافي": "Xavi",
    "ميسي": "Messi",
    "نيمار": "Neymar",
    "سواريز": "Suárez",
    "بيكيه": "Piqué",
    "جوردي ألبا": "Jordi Alba",
    "بوسكيتس": "Busquets",
    "إنييستا": "Iniesta",
    "بويول": "Puyol",
    "فالديس": "Valdés",
    "غوارديولا": "Guardiola",
    # Real Madrid and common rivals
    "كورتوا": "Courtois",
    "كورتويس": "Courtois",
    "لونين": "Lunin",
    "ميليتاو": "Militão",
    "روديغر": "Rüdiger",
    "ألابا": "Alaba",
    "كارفاخال": "Carvajal",
    "ميندي": "Mendy",
    "تشواميني": "Tchouaméni",
    "كامافينغا": "Camavinga",
    "فالفيردي": "Valverde",
    "بيلينغهام": "Bellingham",
    "فينيسيوس": "Vinícius",
    "رودريغو": "Rodrygo",
    "مبابي": "Mbappé",
    "بنزيمة": "Benzema",
    "مودريتش": "Modrić",
    "كروس": "Kroos",
    "أنشيلوتي": "Ancelotti",
    "بيريز": "Pérez",
    "لويس روخو": "Luis Rojo",
}

_LANG_CODES = {
    "ar": "arb_Arab",
    "arabic": "arb_Arab",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ca": "cat_Latn",
    "fr": "fra_Latn",
    "pt": "por_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "tr": "tur_Latn",
}

_SPLIT_RE = re.compile(r"[;:،|❌\n•]+")
_POST_FIXES = (
    ("Claxico", "Clasico"),
    ("Kurtoa", "Courtois"),
    ("Kurtua", "Courtois"),
    ("Courtoa", "Courtois"),
    ("Ravinia", "Raphinha"),
    ("Raphina", "Raphinha"),
    ("marqueur", "buteur"),
    ("harceleur", "buteur"),
    ("escargot", "buteur"),
    ("historique remarquable", "bilan remarquable"),
    ("passé remarquable", "bilan remarquable"),
    ("arrière droit", "latéral droit"),
    ("ailier droit", "latéral droit"),
    (" ragots", " informations"),
)


def is_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text or ""))


def _apply_glossary(text: str) -> str:
    for source, target in sorted(_GLOSSARY.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(
            rf"(?<![\u0600-\u06ff]){re.escape(source)}(?![\u0600-\u06ff])"
        )
        text = pattern.sub(target, text)
    return text


class ArabicTranslator:
    """Lazy NLLB translator (loaded only when an Arabic source appears)."""

    def __init__(self, config=None):
        self.config = config or get_config()
        cfg = getattr(self.config, "translation", None)
        self.model_name = getattr(cfg, "model_name", "facebook/nllb-200-distilled-600M")
        self.max_new_tokens = int(getattr(cfg, "max_new_tokens", 220))
        self.num_beams = int(getattr(cfg, "num_beams", 3))
        self._tokenizer = None
        self._model = None
        self._failed = False

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info("Chargement du traducteur local arabe→français {}", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, src_lang="arb_Arab"
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model.eval()
            logger.info("Traducteur local prêt")
            return True
        except Exception as exc:
            self._failed = True
            logger.warning("Traducteur arabe indisponible, retour à Ollama: {}", exc)
            return False

    def translate(self, text: str, source_lang: str = "ar") -> Optional[str]:
        text = clean_tweet_text(text)
        if not text or not is_arabic(text):
            return None
        if not self._load():
            return None
        src_code = _LANG_CODES.get((source_lang or "ar").lower(), "arb_Arab")
        try:
            import torch

            prepared = _apply_glossary(text)
            segments = [
                segment.strip(" .|❌•")
                for segment in _SPLIT_RE.split(prepared)
                if segment.strip(" .|❌•")
            ]
            if not segments:
                return None
            # A segment with no Arabic was already partly in Latin script
            # (e.g. a journalist name); keep it as-is.
            out: List[str] = []
            with torch.no_grad():
                for segment in segments:
                    if not is_arabic(segment):
                        out.append(segment)
                        continue
                    self._tokenizer.src_lang = src_code
                    inputs = self._tokenizer(
                        segment, return_tensors="pt", truncation=True, max_length=256
                    )
                    generated = self._model.generate(
                        **inputs,
                        forced_bos_token_id=self._tokenizer.convert_tokens_to_ids("fra_Latn"),
                        max_new_tokens=self.max_new_tokens,
                        num_beams=self.num_beams,
                    )
                    translated = self._tokenizer.batch_decode(
                        generated, skip_special_tokens=True
                    )[0]
                    if translated.strip():
                        out.append(translated.strip())
            result = clean_tweet_text(" ".join(out))
            for source, target in _POST_FIXES:
                result = result.replace(source, target)
            return result or None
        except Exception as exc:
            logger.warning("Traduction arabe impossible: {}", exc)
            return None


def translate_to_french(text: str, source_lang: str = "ar", config=None) -> Optional[str]:
    return ArabicTranslator(config).translate(text, source_lang)
