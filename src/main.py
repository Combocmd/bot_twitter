"""Main orchestration and CLI for the @UniversCule Barcelona bot.

Posting and media upload use the free logged-in web session. Reads use the
separate Firefox session. The paid X API is never called by the normal loop.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime

# Windows consoles still default to cp1252 on some installations.  The bot
# handles French accents, arrows and emojis, so keep CLI diagnostics printable
# instead of letting Rich abort an otherwise successful cycle.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
from pathlib import Path
from typing import Dict, List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import Config, get_config, load_config
from .database import Database
from .dedup import ContentDeduplicator
from .filter import BarcaFilter
from .media import MediaDownloader
from .match_calendar import MatchCalendar
from .monitor import SourceMonitor, SourceTweet
from .reply_manager import ReplyManager
from .rewriter import TweetRewriter
from .scheduler import PostScheduler
from .timeutil import iso_utc, weighted_length
from .web_client import XWebAuthError, XWebClient, XWebError
from .x_reader import XReader
from .x_session import READER_STATE, WRITER_STATE, ensure_reader_state, state_summary

class SafeConsole(Console):
    """Console that never lets a closed/invalid terminal break the bot.

    When the launching terminal is closed (or output is redirected to an
    invalid handle on Windows), rich raises OSError on print. Logging remains
    the source of truth, so display errors must not abort a polling cycle.
    """

    def print(self, *args, **kwargs) -> None:  # type: ignore[override]
        try:
            super().print(*args, **kwargs)
        except (OSError, UnicodeError):
            pass


console = SafeConsole()
app = typer.Typer(
    name="barca-bot",
    help="Bot Twitter FC Barcelona pour @UniversCule (français, 100% gratuit)",
    add_completion=False,
)
_shutdown = False
_logging_configured = False
_INSTANCE_LOCK = None


def _pid_alive(pid: int) -> bool:
    """True if a process with this PID is still running (cross-platform)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class InstanceLock:
    def __init__(self, path: Path):
        self.path = path

    def running_pid(self) -> Optional[int]:
        """PID of the live process holding the lock, or None if free/stale."""
        if not self.path.exists():
            return None
        try:
            pid = int(self.path.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            return None
        return pid if pid and _pid_alive(pid) else None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        owner = self.running_pid()
        if owner:
            raise RuntimeError(
                f"Une autre instance du bot fonctionne déjà (PID {owner}, {self.path}). "
                "Arrêtez-la avant de relancer le bot."
            )
        # Stale lock left by a crash: the PID is gone, take over.
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            self._handle = self.path.open("x", encoding="utf-8")
            self._handle.write(str(os.getpid()))
            self._handle.close()
        except FileExistsError as exc:
            raise RuntimeError(
                f"Une autre instance du bot semble déjà fonctionner ({self.path}). "
                "Arrêtez-la avant de relancer le bot."
            ) from exc

    def release(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def signal_handler(signum, frame):
    global _shutdown
    console.print("\n[yellow]Signal d'arrêt reçu, arrêt propre...[/yellow]")
    _shutdown = True


try:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except (ValueError, OSError):
    # Not in the main thread (e.g. imported by a test runner).
    pass


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent.parent / path


def _setup_logging(config: Config) -> None:
    global _logging_configured
    if _logging_configured:
        return
    logger.remove()
    level = config.logging.level.upper()
    logger.add(sys.stderr, level=level, colorize=True)
    log_file = _project_path(config.logging.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_file),
        level=level,
        rotation=config.logging.max_bytes,
        retention=config.logging.backup_count,
        encoding="utf-8",
        enqueue=True,
        format=config.logging.format,
    )
    _logging_configured = True


class BotOrchestrator:
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else get_config()
        _setup_logging(self.config)
        self._init_components()

    def _init_components(self) -> None:
        console.print("[cyan]Initialisation des composants...[/cyan]")
        self.db = Database(str(_project_path(self.config.database.path)))
        self.web = XWebClient(self.config)
        self.reader = XReader(self.config)
        self.monitor = SourceMonitor(self.db, self.config, reader=self.reader)
        self.filter = BarcaFilter(self.config)
        self.dedup = ContentDeduplicator(self.db, self.config)
        self.rewriter = TweetRewriter(self.config)
        self.media = MediaDownloader(self.db, self.config)
        self.match_calendar = MatchCalendar(self.config)
        self.match_calendar.refresh_if_needed()
        self.scheduler = PostScheduler(self.db, self.web, self.config)
        self.reply_manager = ReplyManager(
            self.db, self.web, self.rewriter, self.config,
            reader=self.reader, filter_=self.filter,
        )
        console.print("[green]Composants prêts.[/green]")

    # ------------------------------------------------------------------
    # Cycle
    # ------------------------------------------------------------------
    def run_cycle(self, dry_run: bool = False) -> Dict:
        stats = {
            "sources_polled": 0,
            "tweets_found": 0,
            "barca_filtered": 0,
            "duplicates_rejected": 0,
            "rewritten": 0,
            "media_downloaded": 0,
            "enqueued": 0,
            "media_upgrades": 0,
            "errors": 0,
            "dry_run": dry_run,
        }
        if not dry_run and self.scheduler.publishing_paused:
            console.print("[red]Publication en pause: session @UniversCule invalide. Polling interrompu.[/red]")
            return stats
        if not dry_run and self.scheduler.available_original_slots() <= 0:
            console.print("[yellow]Budget d'originaux atteint pour aujourd'hui; polling reporté.[/yellow]")
            return stats
        try:
            console.print("[cyan]Polling des sources X...[/cyan]")
            self.match_calendar.refresh_if_needed()
            source_tweets = self.monitor.poll_all_sources()
            stats["sources_polled"] = len(self.config.sources.accounts)
            stats["tweets_found"] = len(source_tweets)
            stats["source_errors"] = dict(self.monitor.last_errors)
            if not source_tweets:
                console.print("[yellow]Aucun nouveau tweet source.[/yellow]")
                return stats

            for tweet in source_tweets:
                if _shutdown and not dry_run:
                    logger.info("Arrêt demandé: cycle interrompu avant le traitement suivant")
                    break
                try:
                    filter_result = self.filter.is_barca_related(
                        tweet.text,
                        source_username=tweet.source_username,
                        source_is_barca=True,
                        language=tweet.language,
                    )
                    if not filter_result.is_barca:
                        stats["barca_filtered"] += 1
                        logger.debug("Rejeté (pas Barça) @{}: {} — {}",
                                     tweet.source_username, tweet.tweet_id, filter_result.reason)
                        continue
                    in_match = self.match_calendar.match_in_progress()
                    if in_match and not tweet.media_urls:
                        logger.info("Rejeté pendant le match: tweet sans média @{} {}",
                                    tweet.source_username, tweet.tweet_id)
                        stats["barca_filtered"] += 1
                        continue
                    dedup_result = self.dedup.is_duplicate(
                        tweet.text, tweet.tweet_id, tweet.source_username
                    )
                    if dedup_result.is_duplicate:
                        stats["duplicates_rejected"] += 1
                        logger.info("Doublon @{} (score {:.3f}): {}",
                                    tweet.source_username, dedup_result.similarity_score,
                                    dedup_result.reason)
                        self.dedup.store_processed(
                            tweet.text, tweet.tweet_id, tweet.source_username,
                            matched_existing=True, matched_tweet_id=dedup_result.matched_tweet_id,
                        )
                        continue
                    rewrite_result = self.rewriter.rewrite(
                        tweet.text, tweet.source_username, filter_result.language
                    )
                    if not rewrite_result.success:
                        stats["errors"] += 1
                        logger.error("Échec réécriture @{}: {}", tweet.tweet_id,
                                     rewrite_result.error)
                        continue
                    if weighted_length(rewrite_result.rewritten_text) > self.config.length.max_chars:
                        stats["errors"] += 1
                        logger.error("Tweet réécrit trop long, ignoré: {}", tweet.tweet_id)
                        continue
                    stats["rewritten"] += 1

                    media_paths: List[str] = []
                    if tweet.media_urls and self.config.media.enabled:
                        try:
                            media_paths = self.media.download(
                                tweet.media_urls, tweet.tweet_id, tweet.media_types
                            )
                            stats["media_downloaded"] += len(media_paths)
                        except Exception as exc:
                            logger.warning("Médias non téléchargés pour {}: {}", tweet.tweet_id, exc)
                        if not media_paths:
                            # Never publish the text alone when the source had
                            # media: a media post without its image/video is
                            # misleading for a media account.
                            stats["errors"] += 1
                            logger.warning(
                                "Média annoncé mais non téléchargeable; tweet ignoré: @{} {}",
                                tweet.source_username, tweet.tweet_id,
                            )
                            continue

                    if dry_run:
                        console.print(
                            f"[green]DRY @{tweet.source_username}[/green] "
                            f"({len(media_paths)} image(s)) -> {rewrite_result.rewritten_text[:90]}"
                        )
                        continue

                    # Second net: compare the REWRITTEN text against items
                    # already queued or recently posted. Two sources can report
                    # the same event with different words; source-level dedup
                    # cannot catch those. A pending item without media is
                    # replaced by this version when the new tweet carries media.
                    event_result = self.dedup.is_duplicate_event(
                        rewrite_result.rewritten_text,
                        tweet.tweet_id, tweet.source_username,
                        has_media=bool(media_paths),
                    )
                    if event_result.is_duplicate:
                        if event_result.replace_queue_id:
                            replaced = self.db.cancel_queue_item(
                                event_result.replace_queue_id,
                                f"remplacé par la version avec média (tweet source {tweet.tweet_id})",
                            )
                            if not replaced:
                                logger.warning(
                                    "Remplacement impossible pour #{} (déjà parti), tweet ignoré: {}",
                                    event_result.replace_queue_id, tweet.tweet_id,
                                )
                                self.dedup.store_processed(
                                    tweet.text, tweet.tweet_id, tweet.source_username
                                )
                                continue
                            stats["media_upgrades"] = stats.get("media_upgrades", 0) + 1
                            logger.info(
                                "Tweet #{} remplacé par la version avec média @{}: {}",
                                event_result.replace_queue_id, tweet.source_username,
                                event_result.reason,
                            )
                        else:
                            stats["duplicates_rejected"] += 1
                            logger.info(
                                "Événement déjà couvert @{} (score {:.2f}): {}",
                                tweet.source_username, event_result.similarity_score,
                                event_result.reason,
                            )
                            self.dedup.store_processed(
                                tweet.text, tweet.tweet_id, tweet.source_username,
                                matched_existing=True,
                            )
                            continue

                    if _shutdown:
                        logger.info("Arrêt demandé: publication non ajoutée à la file")
                        break
                    queued = self.scheduler.enqueue_tweet(
                        text=rewrite_result.rewritten_text,
                        media_paths=media_paths,
                        source_tweet_id=tweet.tweet_id,
                        source_username=tweet.source_username,
                    )
                    stats["enqueued"] += 1
                    self.dedup.store_processed(
                        tweet.text, tweet.tweet_id, tweet.source_username
                    )
                    logger.info("En file #{}: @{} → {}",
                                queued["queue_id"], tweet.source_username,
                                rewrite_result.rewritten_text[:100])
                    console.print(
                        f"[green]@{tweet.source_username}[/green] → "
                        f"{rewrite_result.rewritten_text[:80]}"
                    )
                except Exception as exc:
                    stats["errors"] += 1
                    logger.exception("Erreur traitement @{} {}: {}",
                                     tweet.source_username, tweet.tweet_id, exc)
        except Exception as exc:
            stats["errors"] += 1
            logger.exception("Erreur du cycle de veille: {}", exc)
        if not dry_run and (stats["barca_filtered"] or stats["duplicates_rejected"] or stats["errors"]):
            self.db.update_daily_stats(
                tweets_rejected=stats["barca_filtered"] + stats["duplicates_rejected"],
                errors_count=stats["errors"],
            )
        return stats

    # ------------------------------------------------------------------
    # Daemon
    # ------------------------------------------------------------------
    def start(self) -> None:
        global _INSTANCE_LOCK
        _INSTANCE_LOCK = InstanceLock(_project_path("data/bot.lock"))
        _INSTANCE_LOCK.acquire()
        console.print(Panel.fit(
            "[bold blue]Bot @UniversCule — démarrage[/bold blue]\n"
            f"Sources : {len(self.config.sources.accounts)} comptes X\n"
            f"Calendrier : {'actif' if self.match_calendar.enabled else 'désactivé'}\n"
            f"Veille : toutes les {self.config.polling.interval_minutes} min\n"
            f"Publication : {self.config.scheduling.active_hours_label}, "
            f"{self.config.scheduling.original_tweets_per_day} originaux + "
            f"{self.config.replies.max_replies_per_day} réponses max/jour "
            f"(plafond dur {self.config.scheduling.max_tweets_per_day})\n"
            f"Réponses automatiques : {'activées' if self.config.replies.enabled else 'désactivées'}, "
            "réponses manuelles",
            border_style="blue",
        ))
        self.scheduler.start()
        console.print("[green]Bot démarré. Ctrl+C pour arrêter.[/green]\n")
        poll_interval = self.config.polling.interval_minutes * 60
        last_poll = 0.0
        last_reply = 0.0
        try:
            while not _shutdown:
                now = time.time()
                if now - last_poll >= poll_interval:
                    self.run_cycle()
                    last_poll = now
                if self.config.replies.enabled and now - last_reply >= 30:
                    try:
                        self.reply_manager.process_cycle()
                    except Exception as exc:
                        logger.exception("Cycle réponses: {}", exc)
                    last_reply = now
                time.sleep(10)
        except KeyboardInterrupt:
            pass
        finally:
            console.print("[yellow]Arrêt en cours...[/yellow]")
            self.scheduler.stop()
            self.db.close()
            if _INSTANCE_LOCK:
                _INSTANCE_LOCK.release()
            console.print("[green]Bot arrêté proprement.[/green]")

    def test_cycle(self) -> Dict:
        console.print("[cyan]Cycle de test (dry-run, aucune publication)...[/cyan]")
        return self.run_cycle(dry_run=True)

    def test_rewrite(self, text: str, username: str = "test") -> Dict:
        return self.rewriter.test_rewrite(text, username)

    def test_filter(self, text: str) -> Dict:
        return self.filter.get_filter_stats(text)

    def get_status(self) -> Dict:
        return {
            "queue": self.scheduler.get_queue_status(),
            "today_stats": self.scheduler.get_today_stats(),
            "dedup_cache": self.dedup.get_cache_stats(),
            "sources": self.monitor.get_source_stats(),
            "replies": self.reply_manager.get_status(),
            "sessions": {
                "reader": state_summary(READER_STATE),
                "writer": state_summary(WRITER_STATE),
            },
        }

    def get_stats(self, days: int = 30) -> List[Dict]:
        return self.db.get_stats(days)


# ==================== CLI ====================

@app.command()
def start(config: Optional[str] = typer.Option(None, "--config", "-c", help="Chemin vers config.yaml")):
    """Démarrer le bot en mode daemon."""
    # Fail fast: checking the lock before loading models avoids a 30 s wait
    # just to learn that the bot is already running.
    owner = InstanceLock(_project_path("data/bot.lock")).running_pid()
    if owner:
        console.print(f"[red]Le bot tourne déjà (PID {owner}).[/red]")
        console.print("[yellow]Utilisez arreter.bat pour l'arrêter.[/yellow]")
        raise typer.Exit(code=1)
    try:
        BotOrchestrator(config).start()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        console.print("[yellow]Arrêtez l'autre instance ou utilisez arreter.bat.[/yellow]")
        raise typer.Exit(code=1)


@app.command()
def test(config: Optional[str] = typer.Option(None, "--config", "-c", help="Chemin vers config.yaml")):
    """Exécuter un cycle de test (dry-run, ne poste rien)."""
    orchestrator = BotOrchestrator(config)
    stats = orchestrator.test_cycle()
    table = Table(title="Résultats du test")
    table.add_column("Métrique", style="cyan")
    table.add_column("Valeur", style="green")
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


@app.command()
def status(config: Optional[str] = typer.Option(None, "--config", "-c", help="Chemin vers config.yaml")):
    """Afficher le statut actuel du bot."""
    orchestrator = BotOrchestrator(config)
    data = orchestrator.get_status()
    console.print(Panel("[bold]Statut du bot @UniversCule[/bold]", border_style="blue"))

    queue = Table(title="File de publication (7 jours)")
    queue.add_column("Statut", style="cyan")
    queue.add_column("Nombre", style="green")
    for item in data["queue"]:
        queue.add_row(item["status"], str(item["count"]))
    console.print(queue)

    stats = Table(title="Stats du jour")
    stats.add_column("Métrique", style="cyan")
    stats.add_column("Valeur", style="green")
    for key, value in data["today_stats"].items():
        stats.add_row(key.replace("_", " ").title(), str(value))
    console.print(stats)

    replies = data["replies"]
    console.print(
        f"[cyan]Réponses:[/cyan] {replies['sent_today']}/{replies['max_per_day']} aujourd'hui, "
        f"{replies['pending']} en attente, délai {replies['delay_minutes']} min"
    )
    dedup = data["dedup_cache"]
    console.print(
        f"[cyan]Déduplication:[/cyan] {dedup['cached_embeddings']} sources, "
        f"seuil {dedup['threshold']:.2f}, modèle {'chargé' if dedup['model_loaded'] else 'fallback lexical'}"
    )
    for name, info in data["sessions"].items():
        if info.get("valid"):
            console.print(f"[cyan]Session {name}:[/cyan] valide ({info['cookies']} cookies)")
        else:
            console.print(f"[red]Session {name}:[/red] invalide — {info.get('error', 'absente')}")

    sources = Table(title="Sources")
    sources.add_column("Compte", style="cyan")
    sources.add_column("Dernier tweet", style="green")
    sources.add_column("RSS", style="blue")
    sources.add_column("Erreur", style="red")
    for name, info in data["sources"].items():
        sources.add_row(
            name, str(info.get("last_tweet_id") or "—"),
            "oui" if info.get("has_rss") else "non",
            (info.get("last_error") or "—")[:60],
        )
    console.print(sources)


@app.command()
def stats(
    days: int = typer.Option(30, "--days", "-d", help="Nombre de jours d'historique"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Afficher les statistiques historiques."""
    data = BotOrchestrator(config).get_stats(days)
    if not data:
        console.print("[yellow]Aucune statistique disponible.[/yellow]")
        return
    table = Table(title=f"Stats des {days} derniers jours")
    for column in ("Date", "Postés", "Rejetés", "Réponses", "Erreurs", "API"):
        table.add_column(column)
    for row in data:
        table.add_row(
            row["date"], str(row["tweets_posted"]), str(row["tweets_rejected"]),
            str(row["replies_sent"]), str(row["errors_count"]), str(row["api_calls"]),
        )
    console.print(table)


@app.command()
def test_rewrite(
    text: str = typer.Argument(..., help="Texte à réécrire"),
    username: str = typer.Option("test", "--user", "-u", help="Nom du compte source"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Tester la réécriture française sur un texte donné."""
    result = BotOrchestrator(config).test_rewrite(text, username)
    console.print(Panel(
        f"[bold]Original[/bold] ({result['original_length']} caractères)\n{result['original']}",
        border_style="yellow",
    ))
    console.print(Panel(
        f"[bold]Réécrit[/bold] ({result['rewritten_length']} caractères)\n{result['rewritten']}",
        border_style="green",
    ))
    console.print(
        f"Langue : {result['language']} | Modèle : {result['model']} | "
        f"Succès : {result['success']} | <=280 : {result['within_limit']}"
    )
    if result.get("error"):
        console.print(f"[red]{result['error']}[/red]")


@app.command()
def test_filter(
    text: str = typer.Argument(..., help="Texte à tester"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
):
    """Tester le filtre Barça sur un texte donné."""
    result = BotOrchestrator(config).test_filter(text)
    console.print(Panel(f"[bold]Texte[/bold]\n{result['text_preview']}", border_style="yellow"))
    table = Table(title="Résultat du filtre")
    table.add_column("Champ", style="cyan")
    table.add_column("Valeur", style="green")
    for key, value in result.items():
        if key != "text_preview":
            table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)


def _ollama_models(client) -> List[str]:
    try:
        result = client.list()
        models = result.get("models", []) if isinstance(result, dict) else getattr(result, "models", [])
        names = []
        for item in models:
            if isinstance(item, dict):
                names.append(item.get("name") or item.get("model") or "")
            else:
                names.append(getattr(item, "name", "") or getattr(item, "model", ""))
        return [n for n in names if n]
    except Exception:
        return []


@app.command()
def setup():
    """Vérifier la configuration, les sessions, Ollama et Playwright."""
    console.print(Panel("[bold]Vérification complète du bot[/bold]", border_style="blue"))
    checks: List[tuple[str, bool, str]] = []

    def check(name: str, fn):
        try:
            detail = fn()
            checks.append((name, True, str(detail or "")))
        except Exception as exc:
            checks.append((name, False, str(exc)[:300]))

    config: Optional[Config] = None

    def load_cfg():
        nonlocal config
        config = load_config()
        return f"{len(config.sources.accounts)} sources, plafond {config.scheduling.max_tweets_per_day}/jour"

    check("config.yaml", load_cfg)

    def env_check():
        if not Path(".env").exists():
            raise FileNotFoundError(".env absent")
        return "présent"

    check(".env", env_check)

    def deps():
        import apscheduler, feedparser, loguru, ollama, playwright, sentence_transformers, torch, tweepy, yaml  # noqa: F401
        return "tweepy, playwright, ollama, sentence-transformers, torch, APScheduler, feedparser"

    check("Dépendances Python", deps)

    def ollama_check():
        import ollama
        cfg = config or load_config()
        client = ollama.Client(host=cfg.ollama.host)
        names = _ollama_models(client)
        if not any(cfg.ollama.model in name for name in names):
            raise RuntimeError(f"modèle {cfg.ollama.model} absent de {names}")
        return f"{cfg.ollama.model} disponible"

    check("Ollama + qwen2.5:7b", ollama_check)

    def playwright_check():
        from playwright.sync_api import sync_playwright
        p = sync_playwright().start()
        try:
            path = p.chromium.executable_path
            if not path or not Path(path).exists():
                raise FileNotFoundError("Chromium Playwright absent")
            return path
        finally:
            p.stop()

    check("Chromium Playwright", playwright_check)

    def reader_state():
        path = ensure_reader_state()
        return path

    check("Session lecteur @XenoSama__", reader_state)

    def writer_state():
        from .x_session import ensure_writer_state
        return ensure_writer_state()

    check("Session @UniversCule", writer_state)

    def writer_auth():
        cfg = config or load_config()
        text = XWebClient(cfg).check_auth()
        if "UniversCule" not in text:
            raise RuntimeError(text)
        return text.replace("\n", " | ")

    check("Connexion web @UniversCule", writer_auth)

    def reader_auth():
        import asyncio
        cfg = config or load_config()
        text = asyncio.run(XReader(cfg).check_auth())
        if "XenoSama__" not in text:
            raise RuntimeError(text)
        return text.replace("\n", " | ")

    check("Connexion web lecteur", reader_auth)

    def db_check():
        cfg = config or load_config()
        db = Database(str(_project_path(cfg.database.path)))
        with db.transaction() as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"posted_tweets", "processed_sources", "comments_replied", "posting_queue", "reply_queue", "source_checkpoints"}
        missing = required - tables
        if missing:
            raise RuntimeError(f"tables manquantes: {missing}")
        db.close()
        return f"{len(tables)} tables"

    check("Base SQLite", db_check)

    table = Table(title="Vérifications")
    table.add_column("Contrôle", style="cyan")
    table.add_column("Statut", style="green")
    table.add_column("Détail", style="white")
    all_ok = True
    for name, ok, detail in checks:
        status = "OK" if ok else "ÉCHEC"
        style = "green" if ok else "red"
        table.add_row(name, f"[{style}]{status}[/{style}]", detail)
        all_ok = all_ok and ok
    console.print(table)
    if all_ok:
        console.print("\n[green]Tout est prêt. Lancez : python -m src.main start[/green]")
    else:
        console.print("\n[red]Corrigez les contrôles en échec avant de démarrer.[/red]")
        raise typer.Exit(code=1)


@app.command()
def version():
    """Afficher la version."""
    console.print(f"Bot @UniversCule v{__version__}")


if __name__ == "__main__":
    app()
