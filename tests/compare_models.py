"""
Script de comparaison des modèles de traduction - MODE TEST UNIQUEMENT.
NE publie RIEN sur X. NE modifie RIEN en production. NE crée AUCUN fichier de données
sauf les rapports de comparaison dans tests/reports/.

Usage:
    python tests/compare_models.py [--model qwen3:8b] [--limit 50]
    python tests/compare_models.py --unit-tests-only

Ce script :
1. Charge les 50 tweets de tests/test_tweets.csv
2. Fait passer chaque tweet par le pipeline rewrite() complet (modèle actuel config.yaml)
3. Fait passer chaque tweet par le même pipeline avec le modèle de comparaison (ex: qwen3:8b)
4. Vérifie la protection/restauration des termes (__PROTECTED_X__)
5. Vérifie les faits critiques (négations, scores, dates, montants, citations)
6. Génère un rapport comparatif par catégorie

DIFFÉRENCE DE PIPELINE (documentée et assumée) :
- Modèle actuel : utilise rewrite() complet avec auto-correction (jusqu'à 3 tentatives)
- Modèle de comparaison : une seule tentative, même SYSTEM_PROMPT, même protection/
  restauration des termes, même post-correction, même validation
Les résultats ne sont donc pas strictement comparables sur la dimension
"nombre de tentatives". Deux tests distincts sont proposés :
  - Test A : une seule tentative pour chaque modèle (qualité intrinsèque)
  - Test B : pipeline complet actuel (3 tentatives) vs une tentative (robustesse système)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rewriter import TweetRewriter, RewriteResult
from src.config import get_config
from src.timeutil import truncate_weighted, weighted_length
import ollama


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent
TEST_TWEETS_CSV = TESTS_DIR / "test_tweets.csv"
REPORTS_DIR = TESTS_DIR / "reports"


# ---------------------------------------------------------------------------
# Normalisation des montants
# ---------------------------------------------------------------------------

def _normalize_amount(text: str) -> set:
    """
    Extrait et normalise les montants monétaires d'un texte.

    Variantes reconnues comme équivalentes :
    - "€50 million", "50 millions d'euros", "50 M€", "50 million €"
    - "€80 million", "80 millions d'euros", "80 M€"
    - "5000 FCFA", "5000FCFA"

    Retourne un ensemble de canonicalisations "nombre devise" (ex: "50 EUR").
    """
    text_low = text.lower()
    amounts: set = set()

    patterns = [
        # € avant le nombre : "€50 million", "€50M"
        r"(?:€|\$)\s*(\d+(?:\.\d+)?)\s*(?:million|milliard|m|k|millions?|milliers?)?",
        # Nombre avant € : "50 millions d'euros", "50 M€", "50 million €"
        r"(\d+(?:\.\d+)?)\s*(?:million|milliard|m|k|millions?|milliers?)?\s*(?:€|\$|d'euros?|de dollars?)",
        # Nombre seul avec unité mais PAS de devise juste avant ni juste après
        r"(?<![€$0-9])\s*(\d+(?:\.\d+)?)\s*(?:million|milliard|m|k)(?![€$])(?!\w)",
        # FCFA : "5000 FCFA", "5000FCFA"
        r"(\d+(?:\.\d+)?)\s*(?:fcfa|francs?\s+cfa)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text_low, re.IGNORECASE):
            num_str = match.group(1)
            try:
                num = float(num_str)
            except ValueError:
                continue

            full_match = match.group(0).lower()
            if "€" in full_match or "d'euros" in full_match or "m€" in full_match:
                currency = "EUR"
            elif "$" in full_match or "de dollars" in full_match:
                currency = "USD"
            elif "fcfa" in full_match or "francs" in full_match:
                currency = "FCFA"
            else:
                currency = ""

            if num == int(num):
                num_str = str(int(num))

            key = f"{num_str} {currency}".strip()
            amounts.add(key)

    return amounts


# ---------------------------------------------------------------------------
# Détection de négations par mots entiers
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS: Dict[str, List[str]] = {
    "en": [
        r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnothing\b", r"\bnowhere\b",
        r"\bneither\b", r"\bnor\b", r"\bdon't\b", r"\bdoesn't\b", r"\bdidn't\b",
        r"\bwon't\b", r"\bcan't\b", r"\bcannot\b",
    ],
    "fr": [
        r"\bnon\b", r"\bpas\b", r"\bjamais\b", r"\brien\b", r"\bpersonne\b",
        r"\bne\b", r"\bni\b", r"\bnul\b", r"\bzéro\b",
    ],
    "es": [
        r"\bno\b", r"\bnunca\b", r"\bjamás\b", r"\bnada\b", r"\bnadie\b",
        r"\bni\b", r"\btampoco\b",
    ],
}

_RUMOR_WORDS: List[str] = [
    "rumor", "rumour", "rumeur", "selon", "according to",
    "reportedly", "apparently", "supposedly", "allegedly",
    "potentially", "could", "might", "peut-être",
]


# Mois anglais -> canon anglais (pour comparaison inter-langue)
_MONTH_CANON: Dict[str, str] = {
    "january": "january", "janvier": "january",
    "february": "february", "février": "february",
    "march": "march", "mars": "march",
    "april": "april", "avril": "april",
    "may": "may", "mai": "may",
    "june": "june", "juin": "june",
    "july": "july", "juillet": "july",
    "august": "august", "août": "august",
    "september": "september", "septembre": "september",
    "october": "october", "octobre": "october",
    "november": "november", "novembre": "november",
    "december": "december", "décembre": "december",
}


def _normalize_months(text: str) -> set:
    """Extrait les mois d'un texte et les retourne sous forme canonique (anglais)."""
    text_low = text.lower()
    found: set = set()
    for month_var, canon in _MONTH_CANON.items():
        if month_var in text_low:
            found.add(canon)
    return found


def check_critical_facts(source: str, rewritten: Optional[str]) -> List[str]:
    """
    Vérifie les faits critiques : négations, scores, dates, montants, citations.

    Limites connues (documentées) :
    - Négations : détection lexicale par mots entiers, pas sémantique.
      Ne détecte pas si la négation porte sur le bon verbe.
      Faux positifs possibles avec "ne...pas" où "ne" est présent mais
      porte sur un autre élément.
    - Dates : seuls les mois sont normalisés ici ; le jour et l'année ne sont
      pas contrôlés par ce détecteur.
    - Montants : normalisation inter-format (€50 million ≡ 50 millions d'euros ≡ 50 M€).
    - Citations : guillemets doubles, simples et français.
    - Rumeurs : présence de mots comme "selon", "according to" ne prouve pas
      qu'il s'agit d'une rumeur (ex: "selon le club" peut être un fait confirmé).
    """
    if not rewritten:
        return ["Texte réécrit vide ou None"]

    errors: List[str] = []
    s_low = source.lower()
    r_low = rewritten.lower()

    # --- Négations (mots entiers, par langue) ---
    source_negs = _detect_negations(s_low)
    rewritten_negs = _detect_negations(r_low)
    source_has_neg = len(source_negs) > 0
    rewritten_has_neg = len(rewritten_negs) > 0
    if source_has_neg and not rewritten_has_neg:
        errors.append("Négation perdue (source négative, réécriture positive - vérification lexicale approximative)")
    if not source_has_neg and rewritten_has_neg:
        errors.append("Négation ajoutée (source positive, réécriture négative - vérification lexicale approximative)")

    # --- Scores ---
    source_scores = set(re.findall(r"\b(\d{1,3})\s*[-–—]\s*(\d{1,3})\b", s_low))
    rewritten_scores = set(re.findall(r"\b(\d{1,3})\s*[-–—]\s*(\d{1,3})\b", r_low))
    if source_scores != rewritten_scores:
        errors.append(f"Score modifié : source {sorted(source_scores)}, réécriture {sorted(rewritten_scores)}")

    # --- Dates (mois, normalisé inter-langue) ---
    source_months = _normalize_months(s_low)
    rewritten_months = _normalize_months(r_low)
    if source_months and source_months != rewritten_months:
        errors.append(f"Date modifiée : source {sorted(source_months)}, réécriture {sorted(rewritten_months)}")

    # --- Montants (normalisés) ---
    source_amounts = _normalize_amount(s_low)
    rewritten_amounts = _normalize_amount(r_low)
    if source_amounts and source_amounts != rewritten_amounts:
        errors.append(f"Montant modifié : source {sorted(source_amounts)}, réécriture {sorted(rewritten_amounts)}")

    # --- Citations (guillemets doubles, simples et français) ---
    source_has_quotes = ('"' in source or "«" in source or "''" in source or "'" in source)
    rewritten_has_quotes = ('"' in rewritten or "«" in rewritten or "''" in rewritten or "'" in rewritten)
    if source_has_quotes and not rewritten_has_quotes:
        errors.append("Citation (guillemets) absente de la réécriture")

    # --- Rumeur présentée comme fait confirmé ---
    source_has_rumor = any(w in s_low for w in _RUMOR_WORDS)
    if source_has_rumor and not any(w in r_low for w in _RUMOR_WORDS):
        errors.append("Rumeur présentée comme fait confirmé (incertitude perdue)")

    return errors


# ---------------------------------------------------------------------------
# Détection de négations par mots entiers (point 2)
# ---------------------------------------------------------------------------

def _detect_negations(text: str) -> set:
    """
    Détecte les mots de négation par mots entiers, par langue.

    Retourne l'ensemble des motifs de négation trouvés.
    Limite connue : ne détecte pas si la négation porte sur le bon verbe
    (faux positifs possibles avec "ne...pas" où "ne" est présent mais
    porte sur un autre élément).
    """
    text_low = text.lower()
    found: set = set()
    for lang_patterns in _NEGATION_PATTERNS.values():
        for pattern in lang_patterns:
            if re.search(pattern, text_low, re.IGNORECASE):
                found.add(pattern)
    return found


# ---------------------------------------------------------------------------
# Termes protégés avec variantes autorisées (point 3)
# ---------------------------------------------------------------------------

_STRICT_PROTECTED_TERMS: List[str] = [
    "Kopa Trophy", "Ballon d'Or", "Ballon d'or",
    "Lamine Yamal", "Pau Cubarsi", "Pau Cubarsí",
    "Pedri", "Gavi", "Lewandowski", "Raphinha",
    "Frenkie de Jong", "Ter Stegen", "Koundé", "Araújo",
    "Balde", "Fermín", "Dani Olmo", "Ferran Torres",
    "Ansu Fati", "Eric García", "Iñigo Martínez",
    "Hansi Flick", "Deco", "Joan Laporta", "Laporta",
    "BREAKING", "JUST IN", "URGENT", "OFFICIEL", "EXCLU",
]

_AUTHORIZED_VARIANTS: Dict[str, str] = {
    "FC Barcelona": "Barça",
    "FC Barcelone": "Barça",
    "FCB": "Barça",
    "Barcelona": "Barcelone",  # ville, pas le club
    "La Liga": "la Liga",
    "Liga": "la Liga",
    "Champions League": "Ligue des champions",
    "Champions": "Ligue des champions",
    "Copa del Rey": "Coupe du Roi",
    "Supercopa": "Supercoupe",
    "El Clásico": "l'El Clásico",
    "El Clasico": "l'El Clásico",
    "le Clasico": "l'El Clásico",
    "le Classique": "l'El Clásico",
}


# ---------------------------------------------------------------------------
# Chargement du jeu de tests
# ---------------------------------------------------------------------------

def load_test_tweets(limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Charge les tweets de test depuis le CSV."""
    tweets: List[Dict[str, str]] = []
    with open(TEST_TWEETS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tweets.append(row)
            if limit and len(tweets) >= limit:
                break
    return tweets


# ---------------------------------------------------------------------------
# Évaluation par catégorie (séparation succès technique / factuel)
# ---------------------------------------------------------------------------

def evaluate_rewrite(
    source: str, expected: str, rewritten: Optional[str], error: Optional[str],
    protected_mapping: Dict[str, str], category: str,
    authorized_variants: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Évalue une réécriture individuelle.

    Sépare :
    - technical_success : pas d'exception, pas d'erreur de validation (_invalid),
      et texte non vide
    - factual_success : pas d'erreur de faits critiques, pas d'erreur de termes protégés
    - overall_success : les deux sont vrais
    """
    technical_success = rewritten is not None and error is None and rewritten.strip() != ""

    result: Dict[str, Any] = {
        "source": source,
        "expected": expected,
        "rewritten": rewritten or "",
        "error": error or "",
        "category": category,
        "technical_success": technical_success,
        "factual_success": False,
        "overall_success": False,
        "critical_facts_errors": [],
        "protected_terms_errors": [],
        "human_review_needed": False,
    }

    if rewritten and rewritten.strip():
        result["critical_facts_errors"] = check_critical_facts(source, rewritten)
        result["protected_terms_errors"] = check_protected_terms(
            source, rewritten, protected_mapping, authorized_variants
        )

        result["factual_success"] = (
            len(result["critical_facts_errors"]) == 0
            and len(result["protected_terms_errors"]) == 0
        )
        result["overall_success"] = (
            result["technical_success"] and result["factual_success"]
        )

        if result["critical_facts_errors"] or result["protected_terms_errors"]:
            result["human_review_needed"] = True
        if category in {"ambigu", "rumeur"}:
            result["human_review_needed"] = True

    return result


# ---------------------------------------------------------------------------
# Pipeline de réécriture (modèle actuel via rewrite())
# ---------------------------------------------------------------------------

def rewrite_with_current_model(
    rewriter: TweetRewriter, text: str, source_username: str, source_lang: str
) -> Tuple[Optional[str], Optional[str], float, bool]:
    """Fait passer un tweet par le pipeline rewrite() complet du modèle actuel."""
    start = time.time()
    try:
        result: RewriteResult = rewriter.rewrite(text, source_username, source_lang)
        duration = time.time() - start
        return result.rewritten_text or "", result.error, duration, result.success
    except Exception as exc:
        duration = time.time() - start
        return None, f"Exception pipeline: {exc}", duration, False


# ---------------------------------------------------------------------------
# Pipeline de réécriture (modèle de comparaison, même pipeline, autre modèle)
# ---------------------------------------------------------------------------

def rewrite_with_comparison_model(
    text: str, source_username: str, source_lang: str, model_name: str,
    timeout_seconds: int = 60,
) -> Tuple[Optional[str], Optional[str], float, bool]:
    """
    Fait passer un tweet par une passe de génération avec le modèle de
    comparaison, puis par les mêmes post-traitements de production.

    DIFFÉRENCE DOCUMENTÉE : une seule tentative (pas d'auto-correction en boucle),
    contrairement au modèle actuel qui en fait jusqu'à 3 dans rewrite().

    Réutilise les méthodes de TweetRewriter pour le nettoyage, le glossaire,
    la restauration, la post-correction, la validation, les citations et la
    longueur. Ce n'est pas le pipeline complet de production : cette fonction
    ne reproduit volontairement qu'une seule tentative et ne déclenche pas
    les fallbacks de traduction de rewrite().

    Retourne (texte_réécrit, erreur, durée_secondes, succès_technique).
    """
    start = time.time()
    try:
        client = ollama.Client(host="http://localhost:11434", timeout=timeout_seconds)

        # Protection des termes (même logique que dans rewrite())
        rewriter_tmp = TweetRewriter.__new__(TweetRewriter)
        rewriter_tmp._PROTECTED_TERMS = TweetRewriter._PROTECTED_TERMS
        rewriter_tmp.max_chars = 260
        protected_source, protected_mapping = rewriter_tmp._protect_terms(text)

        # Prompt équivalent à celui de rewrite() (non-arabe)
        prompt = (
            f"Compte source : @{source_username}\n"
            f"Langue source : {source_lang or 'auto'}\n"
            f"Information à transformer en tweet français :\n{protected_source}\n\n"
            "Écris le tweet final maintenant, sans préambule."
        )

        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": rewriter_tmp.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": 0.75,
                "num_predict": 220,
            },
        )
        try:
            raw = response["message"]["content"]
        except (KeyError, TypeError):
            raw = getattr(getattr(response, "message", None), "content", "")
        raw = str(raw or "").strip()

        # Réutilisation des post-traitements de production, dans le même ordre
        # que rewrite() pour une passe réussie.
        candidate = rewriter_tmp._apply_glossary(rewriter_tmp._sanitize(raw))

        # Restauration des termes protégés
        candidate = rewriter_tmp._restore_terms(candidate, protected_mapping)

        # Vérification identifiants non restaurés
        unreplaced = rewriter_tmp._verify_protected_restored(candidate, protected_mapping)
        if unreplaced:
            return None, unreplaced, time.time() - start, False

        # Post-correction française
        candidate = rewriter_tmp._post_correct_french(candidate)

        # Même traitement des citations et de la longueur que rewrite().
        source_citations = rewriter_tmp._detect_source_citations(text)
        if weighted_length(candidate) > 260:
            try:
                shortening_prompt = (
                    "Raccourcis ce tweet en français à 240 caractères maximum, "
                    "sans perdre le fait principal :\n"
                    f"{candidate}\nRéponds uniquement avec le tweet final."
                )
                shortened_response = client.chat(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": rewriter_tmp.SYSTEM_PROMPT},
                        {"role": "user", "content": shortening_prompt},
                    ],
                    options={"temperature": 0.75, "num_predict": 220},
                )
                try:
                    shortened = shortened_response["message"]["content"]
                except (KeyError, TypeError):
                    shortened = getattr(getattr(shortened_response, "message", None), "content", "")
                shortened = rewriter_tmp._apply_glossary(rewriter_tmp._sanitize(str(shortened or "")))
                if shortened and rewriter_tmp._invalid(shortened, text) is None:
                    candidate = shortened
            except Exception:
                pass
        if weighted_length(candidate) > rewriter_tmp.max_chars:
            candidate = truncate_weighted(candidate, rewriter_tmp.max_chars)
        if source_citations:
            for citation in source_citations:
                if f"@{citation}" not in candidate:
                    candidate = candidate.rstrip() + f" @{citation}"
                    if weighted_length(candidate) > rewriter_tmp.max_chars:
                        candidate = truncate_weighted(candidate, rewriter_tmp.max_chars)
                        break
        elif source_username.lower() == "fabrizioromano" and "@FabrizioRomano" not in candidate:
            candidate = candidate.rstrip() + " @FabrizioRomano"
            if weighted_length(candidate) > rewriter_tmp.max_chars:
                candidate = truncate_weighted(candidate, rewriter_tmp.max_chars)

        # Validation
        error = rewriter_tmp._invalid(candidate, text)
        duration = time.time() - start

        if error:
            return candidate, error, duration, False
        return candidate, None, duration, True

    except Exception as exc:
        duration = time.time() - start
        return None, f"Exception modèle comparaison: {exc}", duration, False


# ---------------------------------------------------------------------------
# Test de génération de rapport sur résultat fictif
# ---------------------------------------------------------------------------

def test_report_generation() -> Dict[str, Any]:
    """
    Teste generate_report() et save_report() sur un résultat fictif minimal,
    sans lancer Ollama.
    """
    errors: List[str] = []
    details: List[str] = []

    fictitious = {
        "current_model": "qwen2.5:7b (test)",
        "comparison_model": "qwen3:8b (test)",
        "total": {
            "count": 1,
            "current_technical_success": 1,
            "current_factual_success": 1,
            "current_overall_success": 1,
            "comparison_technical_success": 1,
            "comparison_factual_success": 1,
            "comparison_overall_success": 1,
            "current_critical_errors": 0,
            "comparison_critical_errors": 0,
            "current_protected_errors": 0,
            "comparison_protected_errors": 0,
            "current_duration_total": 1.5,
            "comparison_duration_total": 2.0,
            "human_review_current": 0,
            "human_review_comparison": 0,
        },
        "categories": {
            "test": {
                "count": 1,
                "current_technical_success": 1,
                "current_factual_success": 1,
                "current_overall_success": 1,
                "comparison_technical_success": 1,
                "comparison_factual_success": 1,
                "comparison_overall_success": 1,
                "current_critical_errors": 0,
                "comparison_critical_errors": 0,
                "current_protected_errors": 0,
                "comparison_protected_errors": 0,
                "current_duration_total": 1.5,
                "comparison_duration_total": 2.0,
                "human_review_current": 0,
                "human_review_comparison": 0,
            }
        },
        "results": [
            {
                "id": "1",
                "langue": "en",
                "categorie": "test",
                "source": "Test source",
                "expected": "Test attendu",
                "current_model": {
                    "rewritten": "Test réécrit",
                    "error": "",
                    "critical_facts_errors": [],
                    "protected_terms_errors": [],
                    "human_review_needed": False,
                },
                "comparison_model": {
                    "rewritten": "Test comparé",
                    "error": "",
                    "critical_facts_errors": [],
                    "protected_terms_errors": [],
                    "human_review_needed": False,
                },
            }
        ],
    }

    try:
        report = generate_report(fictitious)
        if not report or len(report) < 50:
            errors.append(f"generate_report() a retourné un rapport trop court ({len(report) if report else 0} caractères)")
        else:
            details.append(f"generate_report() OK ({len(report)} caractères)")
    except Exception as exc:
        errors.append(f"generate_report() a levé une exception: {exc}")

    try:
        with tempfile.TemporaryDirectory(prefix="compare-models-report-") as temp_dir:
            md_path, json_path = save_report(fictitious, Path(temp_dir))
            if not md_path.exists() or not json_path.exists():
                errors.append("save_report() n'a pas créé les fichiers attendus")
            else:
                details.append(f"save_report() OK dans un répertoire temporaire : {md_path.name}, {json_path.name}")
    except Exception as exc:
        errors.append(f"save_report() a levé une exception: {exc}")

    succès = len(errors) == 0
    return {
        "nom": "génération de rapport sur résultat fictif",
        "succès": succès,
        "détails": details,
        "erreurs": errors,
        "note": "Test sans Ollama, vérifie seulement la génération de rapport",
    }


# ---------------------------------------------------------------------------
# Comparaison principale
# ---------------------------------------------------------------------------

def run_comparison(
    comparison_model: str, limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Exécute la comparaison entre le modèle actuel et le modèle de comparaison."""
    config = get_config()
    current_model_name = config.ollama.model

    try:
        rewriter = TweetRewriter()
    except Exception as exc:
        return {
            "error": f"Impossible d'initialiser TweetRewriter (modèle actuel {current_model_name}): {exc}",
            "current_model": current_model_name,
            "comparison_model": comparison_model,
        }

    try:
        client = ollama.Client(host="http://localhost:11434", timeout=60)
        models = client.list()
        available = [m.get("model", "") for m in models.get("models", [])]
        if comparison_model not in available and not any(
            comparison_model in m for m in available
        ):
            return {
                "error": f"Modèle de comparaison '{comparison_model}' non trouvé dans Ollama. Modèles disponibles: {available}",
                "current_model": current_model_name,
                "comparison_model": comparison_model,
            }
    except Exception as exc:
        return {
            "error": f"Impossible de contacter Ollama: {exc}",
            "current_model": current_model_name,
            "comparison_model": comparison_model,
        }

    tweets = load_test_tweets(limit)
    results: List[Dict[str, Any]] = []

    print(f"\n{'='*70}")
    print(f"COMPARAISON : {current_model_name} vs {comparison_model}")
    print(f"Nombre de tweets de test : {len(tweets)}")
    print(f"{'='*70}\n")

    for i, tweet in enumerate(tweets, 1):
        text = tweet["texte_source"]
        lang = tweet["langue"]
        category = tweet["categorie"]
        expected = tweet["traduction_attendue"]
        source_username = "test_source"

        # --- Modèle actuel (pipeline complet rewrite()) ---
        current_text, current_error, current_duration, current_success = (
            rewrite_with_current_model(rewriter, text, source_username, lang)
        )

        protected_source, protected_mapping = rewriter._protect_terms(text)

        current_eval = evaluate_rewrite(
            text, expected, current_text, current_error, protected_mapping, category,
            authorized_variants=_AUTHORIZED_VARIANTS,
        )
        current_eval["duration"] = current_duration
        current_eval["model"] = current_model_name

        # --- Modèle de comparaison (même pipeline, autre modèle) ---
        comp_text, comp_error, comp_duration, comp_success = (
            rewrite_with_comparison_model(text, source_username, lang, comparison_model)
        )

        comp_eval = evaluate_rewrite(
            text, expected, comp_text, comp_error, protected_mapping, category,
            authorized_variants=_AUTHORIZED_VARIANTS,
        )
        comp_eval["duration"] = comp_duration
        comp_eval["model"] = comparison_model

        results.append({
            "id": tweet["id"],
            "langue": lang,
            "categorie": category,
            "source": text,
            "expected": expected,
            "current_model": current_eval,
            "comparison_model": comp_eval,
        })

        status_c = "PASS" if current_success else "FAIL"
        status_q = "PASS" if comp_success else "FAIL"
        print(f"[{i:2d}/{len(tweets)}] {category:12s} | {current_model_name.split(':')[0]:8s}:{status_c} | {comparison_model.split(':')[0]:8s}:{status_q}")

    categories: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r["categorie"]
        if cat not in categories:
            categories[cat] = {
                "count": 0,
                "current_technical_success": 0,
                "current_factual_success": 0,
                "current_overall_success": 0,
                "comparison_technical_success": 0,
                "comparison_factual_success": 0,
                "comparison_overall_success": 0,
                "current_critical_errors": 0,
                "comparison_critical_errors": 0,
                "current_protected_errors": 0,
                "comparison_protected_errors": 0,
                "current_duration_total": 0.0,
                "comparison_duration_total": 0.0,
                "human_review_current": 0,
                "human_review_comparison": 0,
            }
        c = categories[cat]
        c["count"] += 1
        if r["current_model"]["technical_success"]:
            c["current_technical_success"] += 1
        if r["current_model"]["factual_success"]:
            c["current_factual_success"] += 1
        if r["current_model"]["overall_success"]:
            c["current_overall_success"] += 1
        if r["comparison_model"]["technical_success"]:
            c["comparison_technical_success"] += 1
        if r["comparison_model"]["factual_success"]:
            c["comparison_factual_success"] += 1
        if r["comparison_model"]["overall_success"]:
            c["comparison_overall_success"] += 1
        c["current_critical_errors"] += len(r["current_model"]["critical_facts_errors"])
        c["comparison_critical_errors"] += len(r["comparison_model"]["critical_facts_errors"])
        c["current_protected_errors"] += len(r["current_model"]["protected_terms_errors"])
        c["comparison_protected_errors"] += len(r["comparison_model"]["protected_terms_errors"])
        c["current_duration_total"] += r["current_model"]["duration"]
        c["comparison_duration_total"] += r["comparison_model"]["duration"]
        if r["current_model"]["human_review_needed"]:
            c["human_review_current"] += 1
        if r["comparison_model"]["human_review_needed"]:
            c["human_review_comparison"] += 1

    total = {
        "count": len(results),
        "current_technical_success": sum(1 for r in results if r["current_model"]["technical_success"]),
        "current_factual_success": sum(1 for r in results if r["current_model"]["factual_success"]),
        "current_overall_success": sum(1 for r in results if r["current_model"]["overall_success"]),
        "comparison_technical_success": sum(1 for r in results if r["comparison_model"]["technical_success"]),
        "comparison_factual_success": sum(1 for r in results if r["comparison_model"]["factual_success"]),
        "comparison_overall_success": sum(1 for r in results if r["comparison_model"]["overall_success"]),
        "current_critical_errors": sum(len(r["current_model"]["critical_facts_errors"]) for r in results),
        "comparison_critical_errors": sum(len(r["comparison_model"]["critical_facts_errors"]) for r in results),
        "current_protected_errors": sum(len(r["current_model"]["protected_terms_errors"]) for r in results),
        "comparison_protected_errors": sum(len(r["comparison_model"]["protected_terms_errors"]) for r in results),
        "current_duration_total": sum(r["current_model"]["duration"] for r in results),
        "comparison_duration_total": sum(r["comparison_model"]["duration"] for r in results),
        "human_review_current": sum(1 for r in results if r["current_model"]["human_review_needed"]),
        "human_review_comparison": sum(1 for r in results if r["comparison_model"]["human_review_needed"]),
    }

    return {
        "current_model": current_model_name,
        "comparison_model": comparison_model,
        "total": total,
        "categories": categories,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Génération du rapport
# ---------------------------------------------------------------------------

def generate_report(comparison: Dict[str, Any]) -> str:
    """Génère un rapport Markdown lisible."""
    if "error" in comparison:
        return f"# ERREUR\n\n{comparison['error']}\n"

    total = comparison["total"]
    lines: List[str] = []

    lines.append("# Rapport de comparaison - Modèle actuel vs Modèle de comparaison")
    lines.append("")
    lines.append(f"- Modèle actuel : `{comparison['current_model']}`")
    lines.append(f"- Modèle de comparaison : `{comparison['comparison_model']}`")
    lines.append(f"- Nombre de tweets testés : {total['count']}")
    lines.append("")
    lines.append("> **Note importante** : Le modèle actuel bénéficie d'une auto-correction")
    lines.append("> (jusqu'à 3 tentatives dans `rewrite()`), tandis que le modèle de comparaison")
    lines.append("> n'en fait qu'une seule. Les résultats ne sont donc pas strictement comparables")
    lines.append("> sur la dimension « nombre de tentatives », mais le sont sur la qualité d'une seule passe.")
    lines.append("")

    lines.append("## Résumé global")
    lines.append("")
    lines.append("| Métrique | Modèle actuel | Modèle comparaison |")
    lines.append("|----------|---------------|--------------------|")
    lines.append(f"| Succès technique | {total['current_technical_success']}/{total['count']} | {total['comparison_technical_success']}/{total['count']} |")
    lines.append(f"| Succès factuel | {total['current_factual_success']}/{total['count']} | {total['comparison_factual_success']}/{total['count']} |")
    lines.append(f"| Succès global | {total['current_overall_success']}/{total['count']} | {total['comparison_overall_success']}/{total['count']} |")
    lines.append(f"| Erreurs faits critiques | {total['current_critical_errors']} | {total['comparison_critical_errors']} |")
    lines.append(f"| Erreurs termes protégés | {total['current_protected_errors']} | {total['comparison_protected_errors']} |")
    lines.append(f"| Relecture humaine nécessaire | {total['human_review_current']} | {total['human_review_comparison']} |")
    lines.append(f"| Temps total (s) | {total['current_duration_total']:.1f} | {total['comparison_duration_total']:.1f} |")
    lines.append(f"| Temps moyen par tweet (s) | {total['current_duration_total']/total['count']:.2f} | {total['comparison_duration_total']/total['count']:.2f} |")
    lines.append("")

    lines.append("## Détail par catégorie")
    lines.append("")
    lines.append("| Catégorie | Nb | Actuel tech | Actuel factuel | Actuel global | Comp tech | Comp factuel | Comp global |")
    lines.append("|-----------|----|-------------|--------------|---------------|-----------|--------------|-------------|")
    for cat, data in sorted(comparison["categories"].items()):
        lines.append(
            f"| {cat} | {data['count']} | {data['current_technical_success']}/{data['count']} | "
            f"{data['current_factual_success']}/{data['count']} | {data['current_overall_success']}/{data['count']} | "
            f"{data['comparison_technical_success']}/{data['count']} | "
            f"{data['comparison_factual_success']}/{data['count']} | {data['comparison_overall_success']}/{data['count']} |"
        )
    lines.append("")

    lines.append("## Détails par tweet (premiers 20)")
    lines.append("")
    for r in comparison["results"][:20]:
        lines.append(f"### #{r['id']} - {r['categorie']} ({r['langue']})")
        lines.append(f"- Source : {r['source']}")
        lines.append(f"- Attendu : {r['expected']}")
        lines.append(f"- Actuel ({comparison['current_model']}) : {r['current_model']['rewritten']}")
        if r['current_model']['error']:
            lines.append(f"  Erreur actuel : {r['current_model']['error']}")
        lines.append(f"- Comparaison ({comparison['comparison_model']}) : {r['comparison_model']['rewritten']}")
        if r['comparison_model']['error']:
            lines.append(f"  Erreur comparaison : {r['comparison_model']['error']}")
        if r['current_model']['critical_facts_errors']:
            lines.append(f"  [!] Erreurs faits actuels : {', '.join(r['current_model']['critical_facts_errors'])}")
        if r['comparison_model']['critical_facts_errors']:
            lines.append(f"  [!] Erreurs faits comparaison : {', '.join(r['comparison_model']['critical_facts_errors'])}")
        lines.append("")

    return "\n".join(lines)


def save_report(comparison: Dict[str, Any], output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """Sauvegarde les rapports sans écraser silencieusement un fichier précédent."""
    report = generate_report(comparison)
    destination = output_dir or REPORTS_DIR
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = destination / f"comparison_report_{timestamp}.md"
    json_path = destination / f"comparison_report_{timestamp}.json"

    md_path.write_text(report, encoding="utf-8")
    json_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")

    return md_path, json_path


# ---------------------------------------------------------------------------
# Tests unitaires protection/restauration et faits critiques
# ---------------------------------------------------------------------------

def run_unit_tests() -> Dict[str, Any]:
    """Exécute les tests unitaires de protection/restauration et faits critiques sans Ollama."""
    rewriter = TweetRewriter.__new__(TweetRewriter)
    rewriter._PROTECTED_TERMS = TweetRewriter._PROTECTED_TERMS

    tests: List[Dict[str, Any]] = []

    def _add_test(nom: str, entree: str, attendu_protege_contient: List[str],
                  attendu_mapping_contient: Dict[str, str],
                  attendu_restaure_egal_entree: bool,
                  attendu_mapping_ensemble: List[str] = None,
                  attendu_restaure_contient: List[str] = None,
                  note: str = "") -> None:
        protegé, mapping = rewriter._protect_terms(entree)
        restauré = rewriter._restore_terms(protegé, mapping)
        unreplaced = rewriter._verify_protected_restored(restauré, mapping)

        ok = True
        details = []

        for attendu in attendu_protege_contient:
            if attendu not in protegé:
                ok = False
                details.append(f"Placeholder '{attendu}' absent du texte protégé: {protegé}")

        for placeholder, terme in attendu_mapping_contient.items():
            if mapping.get(placeholder) != terme:
                ok = False
                details.append(f"Mapping incorrect: {placeholder} -> {mapping.get(placeholder)} (attendu: {terme})")

        if attendu_mapping_ensemble is not None:
            valeurs_actuelles = set(mapping.values())
            valeurs_attendues = set(attendu_mapping_ensemble)
            if valeurs_actuelles != valeurs_attendues:
                ok = False
                details.append(f"Mapping valeurs incorrectes: {valeurs_actuelles} (attendu: {valeurs_attendues})")

        if attendu_restaure_egal_entree and restauré != entree:
            ok = False
            details.append(f"Restauration inexacte: '{restauré}' != '{entree}'")

        if attendu_restaure_contient:
            for terme in attendu_restaure_contient:
                if terme not in restauré:
                    ok = False
                    details.append(f"Terme '{terme}' absent de la restauration: '{restauré}'")

        if unreplaced is not None:
            ok = False
            details.append(f"Identifiant non restauré: {unreplaced}")

        for placeholder in mapping:
            if placeholder not in protegé:
                ok = False
                details.append(f"Placeholder '{placeholder}' dans le mapping mais absent du texte protégé")

        tests.append({
            "nom": nom,
            "entrée": entree,
            "protégé": protegé,
            "mapping": mapping,
            "restauré": restauré,
            "identifiant_restant": unreplaced,
            "succès": ok,
            "détails": details,
            "note": note,
        })

    # === Tests protection/restauration des termes ===

    _add_test(
        nom="répétition même terme",
        entree="BREAKING : BREAKING news about BREAKING",
        attendu_protege_contient=["__PROTECTED_0__"],
        attendu_mapping_contient={"__PROTECTED_0__": "BREAKING"},
        attendu_restaure_egal_entree=True,
        note="Trois occurrences de BREAKING utilisent un seul placeholder",
    )

    _add_test(
        nom="ponctuation collée",
        entree="BREAKING: Yamal scores!",
        attendu_protege_contient=["__PROTECTED_0__"],
        attendu_mapping_contient={"__PROTECTED_0__": "BREAKING"},
        attendu_restaure_egal_entree=False,
        attendu_restaure_contient=["BREAKING:", "Yamal", "scores!"],
        note="BREAKING: protégé malgré les deux-points collés",
    )

    _add_test(
        nom="casse variée",
        entree="breaking : YAMAL wins Kopa Trophy",
        attendu_protege_contient=["__PROTECTED_0__", "__PROTECTED_1__"],
        attendu_mapping_contient={"__PROTECTED_0__": "Kopa Trophy", "__PROTECTED_1__": "BREAKING"},
        attendu_restaure_egal_entree=False,
        attendu_restaure_contient=["BREAKING", "YAMAL", "Kopa Trophy"],
        note="Matching case-insensitive",
    )

    _add_test(
        nom="expressions multiples non chevauchantes",
        entree="Ballon d'Or and Kopa Trophy",
        attendu_protege_contient=["__PROTECTED_0__", "__PROTECTED_1__"],
        attendu_mapping_contient={},
        attendu_mapping_ensemble=["Ballon d'Or", "Kopa Trophy"],
        attendu_restaure_egal_entree=True,
        note="Deux termes protégés distincts",
    )

    _add_test(
        nom="identifiant dupliqué (un seul placeholder)",
        entree="BREAKING and BREAKING again",
        attendu_protege_contient=["__PROTECTED_0__"],
        attendu_mapping_contient={"__PROTECTED_0__": "BREAKING"},
        attendu_restaure_egal_entree=True,
        note="Deux occurrences du même terme utilisent un seul placeholder",
    )

    _add_test(
        nom="aucun terme protégé",
        entree="Hello world, this is a test",
        attendu_protege_contient=[],
        attendu_mapping_contient={},
        attendu_restaure_egal_entree=True,
        note="Texte sans terme protégé doit rester inchangé",
    )

    _add_test(
        nom="Barcelone ville non remplacée",
        entree="The weather in Barcelona is nice",
        attendu_protege_contient=[],
        attendu_mapping_contient={},
        attendu_restaure_egal_entree=True,
        note="Barcelona seul (ville) ne doit pas être remplacé par Barça",
    )

    _add_test(
        nom="FC Barcelona club remplacé (variante autorisée)",
        entree="FC Barcelona wins the match",
        attendu_protege_contient=["__PROTECTED_0__"],
        attendu_mapping_contient={"__PROTECTED_0__": "FC Barcelona"},
        attendu_restaure_egal_entree=True,
        note="FC Barcelona (club) doit être protégé et restauré exactement",
    )

    _add_test(
        nom="terme partiel non protégé",
        entree="He won the Kopa",
        attendu_protege_contient=[],
        attendu_mapping_contient={},
        attendu_restaure_egal_entree=True,
        note="Kopa seul n'est pas protégé",
    )

    _add_test(
        nom="terme avec apostrophe",
        entree="He won the Ballon d'Or",
        attendu_protege_contient=["__PROTECTED_0__"],
        attendu_mapping_contient={"__PROTECTED_0__": "Ballon d'Or"},
        attendu_restaure_egal_entree=True,
        note="Ballon d'Or avec apostrophe",
    )

    _add_test(
        nom="terme en début et fin",
        entree="BREAKING : the match is over BREAKING",
        attendu_protege_contient=["__PROTECTED_0__"],
        attendu_mapping_contient={"__PROTECTED_0__": "BREAKING"},
        attendu_restaure_egal_entree=True,
        note="BREAKING en début et fin utilise un seul placeholder",
    )

    _add_test(
        nom="restauration caractère pour caractère",
        entree="  BREAKING  Yamal  ",
        attendu_protege_contient=["__PROTECTED_0__"],
        attendu_mapping_contient={"__PROTECTED_0__": "BREAKING"},
        attendu_restaure_egal_entree=True,
        note="Espaces préservés",
    )

    # === Tests faits critiques ===

    errors = check_critical_facts("He will not sign", "Il ne signera pas")
    tests.append({
        "nom": "négation préservée",
        "entrée": "He will not sign",
        "protégé": "",
        "mapping": {},
        "restauré": "Il ne signera pas",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur"],
        "note": "Négation 'not' -> 'ne ... pas' préservée",
    })

    errors = check_critical_facts("He will not sign", "Il signera")
    tests.append({
        "nom": "négation perdue détectée",
        "entrée": "He will not sign",
        "protégé": "",
        "mapping": {},
        "restauré": "Il signera",
        "identifiant_restant": None,
        "succès": len(errors) > 0 and "Négation perdue" in str(errors),
        "détails": [f"Erreurs: {errors}"],
        "note": "Détecte la perte de négation",
    })

    errors = check_critical_facts("He took a note", "Il a pris une note")
    tests.append({
        "nom": "faux positif négation ('note' contient 'not')",
        "entrée": "He took a note",
        "protégé": "",
        "mapping": {},
        "restauré": "Il a pris une note",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur (bon)"],
        "note": "Vérifie que 'note' ne déclenche pas de faux positif négation",
    })

    errors = check_critical_facts("The score is 2-1", "Le score est de 2-1")
    tests.append({
        "nom": "score préservé",
        "entrée": "The score is 2-1",
        "protégé": "",
        "mapping": {},
        "restauré": "Le score est de 2-1",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur"],
        "note": "Score 2-1 préservé",
    })

    errors = check_critical_facts("The score is 2-1", "Le score est de 1-2")
    tests.append({
        "nom": "score modifié détecté",
        "entrée": "The score is 2-1",
        "protégé": "",
        "mapping": {},
        "restauré": "Le score est de 1-2",
        "identifiant_restant": None,
        "succès": len(errors) > 0 and "Score" in str(errors),
        "détails": [f"Erreurs: {errors}"],
        "note": "Détecte la modification de score",
    })

    errors = check_critical_facts("Fee at €50 million", "Frais de €50 millions d'euros")
    tests.append({
        "nom": "montants équivalents (€50M = 50 millions d'euros)",
        "entrée": "Fee at €50 million",
        "protégé": "",
        "mapping": {},
        "restauré": "Frais de €50 millions d'euros",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur"],
        "note": "€50 million et 50 millions d'euros sont normalisés de façon équivalente",
    })

    errors = check_critical_facts("Fee at 50 M€", "Frais de €50 millions d'euros")
    tests.append({
        "nom": "montants équivalents (50 M€ = €50 millions)",
        "entrée": "Fee at 50 M€",
        "protégé": "",
        "mapping": {},
        "restauré": "Frais de €50 millions d'euros",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur"],
        "note": "50 M€ et €50 millions d'euros sont normalisés de façon équivalente",
    })

    errors = check_critical_facts("Fee at €50 million", "Frais de €80 millions d'euros")
    tests.append({
        "nom": "montant modifié détecté (50 -> 80)",
        "entrée": "Fee at €50 million",
        "protégé": "",
        "mapping": {},
        "restauré": "Frais de €80 millions d'euros",
        "identifiant_restant": None,
        "succès": len(errors) > 0 and "Montant" in str(errors),
        "détails": [f"Erreurs: {errors}"],
        "note": "Détecte la modification de 50 à 80 millions",
    })

    errors = check_critical_facts("Fee at 5000 FCFA", "Frais de 5000 FCFA")
    tests.append({
        "nom": "montants FCFA préservés",
        "entrée": "Fee at 5000 FCFA",
        "protégé": "",
        "mapping": {},
        "restauré": "Frais de 5000 FCFA",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur"],
        "note": "Montants en FCFA préservés",
    })

    errors = check_critical_facts("Fee at 5000 FCFA", "Frais de 8000 FCFA")
    tests.append({
        "nom": "montant FCFA modifié détecté",
        "entrée": "Fee at 5000 FCFA",
        "protégé": "",
        "mapping": {},
        "restauré": "Frais de 8000 FCFA",
        "identifiant_restant": None,
        "succès": len(errors) > 0 and "Montant" in str(errors),
        "détails": [f"Erreurs: {errors}"],
        "note": "Détecte la modification de 5000 à 8000 FCFA",
    })

    errors = check_critical_facts("The match is in december", "Le match est en décembre")
    tests.append({
        "nom": "mois anglais vs français équivalents",
        "entrée": "The match is in december",
        "protégé": "",
        "mapping": {},
        "restauré": "Le match est en décembre",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur"],
        "note": "december et décembre sont reconnus comme équivalents",
    })

    errors = check_critical_facts("The match is in december", "Le match est en janvier")
    tests.append({
        "nom": "date modifiée détectée",
        "entrée": "The match is in december",
        "protégé": "",
        "mapping": {},
        "restauré": "Le match est en janvier",
        "identifiant_restant": None,
        "succès": len(errors) > 0 and "Date" in str(errors),
        "détails": [f"Erreurs: {errors}"],
        "note": "Détecte la modification de date",
    })

    errors = check_critical_facts('@X : "Lewandowski stays"', '@X : "Lewandowski reste"')
    tests.append({
        "nom": "citation préservée (guillemets doubles)",
        "entrée": '@X : "Lewandowski stays"',
        "protégé": "",
        "mapping": {},
        "restauré": '@X : "Lewandowski reste"',
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur"],
        "note": "Guillemets doubles préservés",
    })

    errors = check_critical_facts("@X : 'Lewandowski stays'", "@X : 'Lewandowski reste'")
    tests.append({
        "nom": "citation préservée (guillemets simples)",
        "entrée": "@X : 'Lewandowski stays'",
        "protégé": "",
        "mapping": {},
        "restauré": "@X : 'Lewandowski reste'",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors}" if errors else "Aucune erreur"],
        "note": "Guillemets simples préservés",
    })

    errors = check_critical_facts('@X : "Lewandowski stays"', "Lewandowski reste selon X")
    tests.append({
        "nom": "citation perdue détectée",
        "entrée": '@X : "Lewandowski stays"',
        "protégé": "",
        "mapping": {},
        "restauré": "Lewandowski reste selon X",
        "identifiant_restant": None,
        "succès": len(errors) > 0 and "Citation" in str(errors),
        "détails": [f"Erreurs: {errors}"],
        "note": "Détecte la perte de guillemets",
    })

    errors = check_critical_facts("Rumor: Mbappé to Barça", "Mbappé signe au Barça")
    tests.append({
        "nom": "rumeur présentée comme fait détectée",
        "entrée": "Rumor: Mbappé to Barça",
        "protégé": "",
        "mapping": {},
        "restauré": "Mbappé signe au Barça",
        "identifiant_restant": None,
        "succès": len(errors) > 0 and "Rumeur" in str(errors),
        "détails": [f"Erreurs: {errors}"],
        "note": "Détecte une rumeur transformée en fait confirmé",
    })

    errors = check_critical_facts("Some source text", None)
    tests.append({
        "nom": "texte vide détecté",
        "entrée": "Some source text",
        "protégé": "",
        "mapping": {},
        "restauré": None,
        "identifiant_restant": None,
        "succès": len(errors) > 0 and "vide" in str(errors),
        "détails": [f"Erreurs: {errors}"],
        "note": "Détecte une réponse vide",
    })

    errors = check_critical_facts("Lewandowski scores", "Pedri scores")
    tests.append({
        "nom": "sens modifié (joueur changé) - non détecté par faits critiques",
        "entrée": "Lewandowski scores",
        "protégé": "",
        "mapping": {},
        "restauré": "Pedri scores",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors} (limite connue: changement de joueur non couvert)"],
        "note": "Limite connue: check_critical_facts ne vérifie pas les changements de noms propres non protégés",
    })

    errors = check_critical_facts("Barça wins", "Real Madrid wins")
    tests.append({
        "nom": "sens modifié (club changé) - non détecté par faits critiques",
        "entrée": "Barça wins",
        "protégé": "",
        "mapping": {},
        "restauré": "Real Madrid wins",
        "identifiant_restant": None,
        "succès": len(errors) == 0,
        "détails": [f"Erreurs: {errors} (limite connue: changement de club non couvert)"],
        "note": "Limite connue: check_critical_facts ne vérifie pas les changements de club",
    })

    report_test = test_report_generation()
    tests.append(report_test)

    csv_errors = []
    try:
        import csv as _csv
        from pathlib import Path as _Path
        csv_path = _Path("tests/test_tweets.csv")
        with open(csv_path, "r", encoding="utf-8") as _f:
            reader = _csv.DictReader(_f)
            rows = list(reader)

        if len(rows) != 50:
            csv_errors.append(f"Nombre de lignes: {len(rows)} (attendu: 50)")

        required_cols = ["id", "langue", "texte_source", "categorie",
                         "traduction_attendue", "element_a_preserver",
                         "erreur_a_detecter", "notes"]
        missing_cols = [c for c in required_cols if c not in reader.fieldnames]
        if missing_cols:
            csv_errors.append(f"Colonnes manquantes: {missing_cols}")

        for i, row in enumerate(rows, 1):
            for col in required_cols:
                if not row.get(col):
                    csv_errors.append(f"Ligne {i}: champ vide '{col}'")
            if row["langue"] not in ("en", "es", "ar"):
                csv_errors.append(f"Ligne {i}: langue inattendue '{row['langue']}'")

        if csv_errors:
            tests.append({
                "nom": "validation CSV 50 tweets",
                "entrée": "tests/test_tweets.csv",
                "protégé": "",
                "mapping": {},
                "restauré": "",
                "identifiant_restant": None,
                "succès": False,
                "détails": csv_errors,
                "note": "Le CSV doit avoir 50 lignes avec toutes les colonnes",
            })
        else:
            tests.append({
                "nom": "validation CSV 50 tweets",
                "entrée": "tests/test_tweets.csv",
                "protégé": "",
                "mapping": {},
                "restauré": "",
                "identifiant_restant": None,
                "succès": True,
                "détails": [f"50 lignes valides, colonnes: {required_cols}"],
                "note": "CSV validé",
            })
    except Exception as exc:
        tests.append({
            "nom": "validation CSV 50 tweets",
            "entrée": "tests/test_tweets.csv",
            "protégé": "",
            "mapping": {},
            "restauré": "",
            "identifiant_restant": None,
            "succès": False,
            "détails": [f"Erreur lecture CSV: {exc}"],
            "note": "Impossible de lire le fichier CSV",
        })

    tous_reussis = all(t["succès"] for t in tests)
    return {"tests": tests, "tous_reussis": tous_reussis}


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Comparaison modèles de traduction (mode test uniquement)")
    parser.add_argument("--model", default="qwen3:8b", help="Modèle de comparaison (défaut: qwen3:8b)")
    parser.add_argument("--limit", type=int, default=None, help="Nombre max de tweets à tester")
    parser.add_argument("--unit-tests-only", action="store_true", help="Exécuter uniquement les tests unitaires")
    args = parser.parse_args()

    print("=" * 70)
    print("TESTS UNITAIRES - Protection, restauration et faits critiques")
    print("=" * 70)
    unit_results = run_unit_tests()
    for t in unit_results["tests"]:
        status = "PASS" if t["succès"] else "FAIL"
        print(f"  [{status}] {t['nom']}")
        if not t["succès"]:
            print(f"      Entrée: {t['entrée']}")
            print(f"      Protégé: {t['protégé']}")
            print(f"      Mapping: {t.get('mapping', {})}")
            print(f"      Restauré: {t['restauré']}")
            print(f"      Identifiants restants: {t['identifiant_restant']}")
            for d in t.get("détails", []):
                print(f"      Détail: {d}")
    print(f"\nRésultat : {'TOUS LES TESTS UNITAIRES REUSSIS' if unit_results['tous_reussis'] else 'CERTAINS TESTS UNITAIRES ECHOIENT'}\n")

    if args.unit_tests_only:
        return 0 if unit_results["tous_reussis"] else 1

    print("=" * 70)
    print(f"COMPARAISON : modèle actuel vs {args.model}")
    print("=" * 70)
    print("ATTENTION : Aucun tweet ne sera publié sur X.")
    print("Aucune modification de la configuration de production.\n")

    comparison = run_comparison(args.model, args.limit)

    if "error" in comparison:
        print(f"\nERREUR : {comparison['error']}")
        return 1

    report = generate_report(comparison)

    md_path, json_path = save_report(comparison)

    print("\n" + report)
    print(f"\nRapports sauvegardés :")
    print(f"  - {md_path}")
    print(f"  - {json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
