"""Free X publishing through the logged-in @UniverseCule web session.

The X API write endpoint now requires paid credits ("402 credits depleted").
This client uses Playwright and the normal web composer instead. It is slower
than an API call, but works with the account's existing browser session and
uploads images through the same UI.
"""
from __future__ import annotations

import os
import random
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loguru import logger
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from .config import get_config
from .timeutil import weighted_length
from .x_session import PROJECT_ROOT, XSessionError, load_state


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


class XWebError(RuntimeError):
    pass


class XWebAuthError(XWebError):
    pass


class XWebMediaError(XWebError):
    pass


class XWebUncertainError(XWebError):
    """The post may have been accepted but no tweet id was captured."""


_STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['fr-FR', 'fr', 'en-US']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = window.chrome || {runtime: {}};
"""


class XWebClient:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.timeout_ms = max(15000, int(self.config.x_web.timeout_seconds * 1000))
        self.headless = bool(self.config.x_web.headless)
        self.expected_handle = str(self.config.x_web.expected_handle or "UniverseCule").lstrip("@")
        state_file = Path(self.config.x_web.state_file)
        self.state_path = state_file if state_file.is_absolute() else PROJECT_ROOT / state_file
        self.profile_path = _project_path(getattr(self.config.x_web, "profile_dir", "data/x_universcule_profile"))
        self.profile_path.mkdir(parents=True, exist_ok=True)
        self._lock = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Browser plumbing
    # ------------------------------------------------------------------
    def _get_lock(self):
        if self._lock is None:
            import threading
            self._lock = threading.RLock()
        return self._lock

    @contextmanager
    def _session(self):
        lock = self._get_lock()
        with lock:
            if not self.state_path.exists():
                raise XWebAuthError(
                    f"Session @{self.expected_handle} introuvable: {self.state_path}. "
                    "Exportez la session du navigateur où le compte est connecté."
                )
            try:
                state = load_state(self.state_path)
            except XSessionError as exc:
                raise XWebAuthError(str(exc)) from exc
            playwright = sync_playwright().start()
            context = None
            try:
                context = playwright.chromium.launch_persistent_context(
                    str(self.profile_path),
                    headless=self.headless,
                    args=[
                        "--no-sandbox",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                    ],
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/131.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 900},
                    locale="fr-FR",
                    timezone_id="Europe/Madrid",
                )
                browser = context.browser
                if state.get("cookies"):
                    context.add_cookies(state["cookies"])
                context.add_init_script(_STEALTH_JS)
                page = context.pages[0] if context.pages else context.new_page()
                page.set_default_timeout(self.timeout_ms)
                yield page
            finally:
                try:
                    if context:
                        context.storage_state(path=str(self.state_path))
                        context.close()
                finally:
                    playwright.stop()

    def _verify_account(self, page) -> None:
        try:
            account = page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').first
            account.wait_for(state="visible", timeout=15000)
            text = account.inner_text()
        except Exception as exc:
            raise XWebAuthError(
                "Session web @UniverseCule invalide ou expirée. "
                "Reconnectez le compte et réexportez la session."
            ) from exc
        handle = f"@{self.expected_handle}".lower()
        if handle not in text.lower():
            raise XWebAuthError(
                f"Le navigateur est connecté à un autre compte que @{self.expected_handle}: "
                f"{text.strip()[:120]}"
            )

    @staticmethod
    def _visible(locator):
        return locator.locator(":visible")

    def _find_editor(self, page):
        deadline = time.time() + 30
        while time.time() < deadline:
            editors = page.locator('[data-testid="tweetTextarea_0"]:visible')
            if editors.count() == 0:
                editors = page.locator('[data-testid="tweetTextarea_0"]')
            if editors.count():
                try:
                    if editors.first.is_visible():
                        return editors.first
                except Exception:
                    pass
            page.wait_for_timeout(500)
        # Last attempt: open the explicit composer route.
        try:
            page.goto("https://x.com/compose/post", wait_until="domcontentloaded",
                      timeout=self.timeout_ms)
            page.wait_for_timeout(3000)
            editors = page.locator('[data-testid="tweetTextarea_0"]:visible')
            if editors.count():
                return editors.first
        except Exception:
            pass
        raise XWebError("Éditeur de post X introuvable")

    def _find_post_button(self, page, reply: bool = False):
        selectors = ['[data-testid="tweetButtonInline"]:visible']
        if reply:
            selectors = ['[data-testid="tweetButtonInline"]:visible', '[data-testid="tweetButton"]:visible']
        else:
            selectors += ['[data-testid="tweetButton"]:visible']
        deadline = time.time() + 20
        while time.time() < deadline:
            for selector in selectors:
                loc = page.locator(selector)
                if loc.count():
                    return loc.first
            # Fallback: visible button whose text is Post/Poster/Répondre.
            for label in ("Poster", "Post", "Répondre", "Reply"):
                loc = page.get_by_role(
                    "button", name=re.compile(f"^{re.escape(label)}$", re.IGNORECASE)
                )
                if loc.count():
                    return loc.first
            page.wait_for_timeout(500)
        raise XWebError("Bouton de publication X introuvable")

    def _attach_media(self, page, paths: List[str]) -> None:
        if not paths:
            return
        max_media = int(getattr(self.config.media, "max_media_per_tweet", 4))
        if len(paths) > max_media:
            raise XWebMediaError(f"X accepte au maximum {max_media} médias")
        valid = []
        image_limit = int(self.config.media.max_size_mb) * 1024 * 1024
        video_limit = int(getattr(self.config.media, "max_video_size_mb", 50)) * 1024 * 1024
        for path in paths:
            p = Path(path)
            if not p.exists():
                raise XWebMediaError(f"Média introuvable: {p}")
            is_video = p.suffix.lower() == ".mp4"
            limit = video_limit if is_video else image_limit
            if p.stat().st_size > limit:
                raise XWebMediaError(f"Média trop volumineux ({p.stat().st_size} octets): {p}")
            allowed = {".mp4"} if is_video else {".jpg", ".jpeg", ".png", ".webp", ".gif"}
            if p.suffix.lower() not in allowed:
                raise XWebMediaError(f"Format média non supporté: {p.suffix}")
            valid.append(str(p.resolve()))
        inputs = page.locator('[data-testid="fileInput"]')
        if inputs.count() == 0:
            raise XWebMediaError("Champ d'import d'image X introuvable")
        inputs = inputs.last
        # Retry the complete upload-and-wait operation. The previous code could
        # break out of the wait loop without actually starting a new upload.
        last_exc: Optional[Exception] = None
        upload_errors = (
            "La photo n'a pas pu être téléversée",
            "The photo could not be uploaded",
            "impossible de téléverser",
            "could not upload",
            "Une erreur s'est produite",
        )
        for attempt in range(3):
            try:
                inputs.set_input_files(valid)
                deadline = time.time() + 60
                while time.time() < deadline:
                    try:
                        attachments = page.locator('[data-testid="attachments"]')
                        count = attachments.locator('img, video').count()
                        if count == 0:
                            count = attachments.locator('[role="progressbar"]').count()
                    except Exception:
                        count = 0
                    if count >= len(valid):
                        return
                    try:
                        body = page.locator("body").inner_text().lower()
                    except Exception:
                        body = ""
                    matching_error = next(
                        (message for message in upload_errors if message.lower() in body),
                        None,
                    )
                    if matching_error:
                        raise XWebMediaError(
                            f"Échec du téléversement de l'image: {matching_error}"
                        )
                    page.wait_for_timeout(500)
                raise XWebMediaError("Délai dépassé pendant le téléversement de l'image")
            except Exception as exc:
                last_exc = exc
                if attempt < 2:
                    logger.warning(
                        "Upload media tentative {} échouée, nouvelle tentative: {}",
                        attempt + 1, exc,
                    )
                    page.wait_for_timeout(2000)
                else:
                    raise XWebMediaError(
                        f"Import du média impossible après 3 tentatives: {last_exc}"
                    ) from last_exc

    @staticmethod
    def _extract_tweet_id(payload: Dict) -> Optional[str]:
        # Fast path for the normal CreateTweet response.
        try:
            result = payload["data"]["create_tweet"]["tweet_results"]["result"]
            if isinstance(result, dict):
                if result.get("rest_id") and str(result["rest_id"]).isdigit():
                    return str(result["rest_id"])
                nested = result.get("tweet")
                if isinstance(nested, dict) and nested.get("rest_id"):
                    return str(nested["rest_id"])
        except (KeyError, TypeError):
            pass

        def walk(obj):
            if isinstance(obj, dict):
                legacy = obj.get("legacy")
                if (
                    obj.get("rest_id")
                    and str(obj["rest_id"]).isdigit()
                    and isinstance(legacy, dict)
                    and (legacy.get("full_text") or legacy.get("created_at"))
                ):
                    return str(obj["rest_id"])
                nested = obj.get("tweet")
                if (
                    isinstance(nested, dict)
                    and nested.get("rest_id")
                    and isinstance(nested.get("legacy"), dict)
                ):
                    return str(nested["rest_id"])
                for value in obj.values():
                    found = walk(value)
                    if found:
                        return found
            elif isinstance(obj, list):
                for value in obj:
                    found = walk(value)
                    if found:
                        return found
            return None

        return walk(payload)

    @staticmethod
    def _extract_error(payload: Dict) -> Optional[str]:
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if errors:
            messages = []
            for error in errors:
                if isinstance(error, dict):
                    messages.append(str(error.get("message") or error.get("code") or ""))
            return " | ".join(m for m in messages if m)[:800]
        return None

    def _click_and_capture(self, page, button, timeout: int = 75000) -> Dict:
        try:
            with page.expect_response(
                lambda r: "/graphql/" in r.url and "CreateTweet" in r.url,
                timeout=timeout,
            ) as response_info:
                button.click()
            response = response_info.value
            try:
                payload = response.json()
            except Exception as exc:
                raise XWebUncertainError(
                    f"X a répondu sans JSON exploitable (status {response.status}): {exc}"
                ) from exc
            if response.status >= 400:
                raise XWebError(f"X a refusé le post (HTTP {response.status}): {payload}")
            error = self._extract_error(payload)
            if error:
                raise XWebError(f"X a refusé le post: {error}")
            tweet_id = self._extract_tweet_id(payload)
            if not tweet_id:
                raise XWebUncertainError(
                    "Réponse CreateTweet reçue sans identifiant de post; "
                    "le post a peut-être été publié."
                )
            return payload
        except PlaywrightTimeoutError as exc:
            # The UI may have succeeded without a captured CreateTweet response
            # (rare, e.g. module worker). Inspect the toast before deciding.
            toast_text = ""
            try:
                toast = page.locator('[data-testid="toast"]')
                if toast.count():
                    toast_text = toast.first.inner_text()
            except Exception:
                pass
            if any(s in toast_text.lower() for s in ("envoyé", "sent", "publié", "posted")):
                match = re.search(r"/status/(\d+)", page.content())
                if match:
                    return {"data": {"create_tweet": {"tweet_results": {"result": {"rest_id": match.group(1)}}}}}
            raise XWebUncertainError(
                f"Pas de réponse CreateTweet après 75 s. Toast: {toast_text[:200] or 'aucun'}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_auth(self) -> str:
        with self._session() as page:
            page.goto("https://x.com/home", wait_until="domcontentloaded",
                      timeout=self.timeout_ms)
            page.wait_for_timeout(5000)
            self._verify_account(page)
            return page.locator('[data-testid="SideNav_AccountSwitcher_Button"]').first.inner_text().strip()

    def post_tweet(
        self,
        text: str,
        media_paths: Optional[List[str]] = None,
        in_reply_to_tweet_id: Optional[str] = None,
    ) -> str:
        text = (text or "").strip()
        if not text:
            raise XWebError("Texte vide: publication refusée")
        length = weighted_length(text)
        if length > self.config.length.max_chars:
            raise XWebError(
                f"Post trop long ({length}/{self.config.length.max_chars} caractères pondérés)"
            )
        media_paths = [str(p) for p in (media_paths or []) if p]
        with self._session() as page:
            reply = bool(in_reply_to_tweet_id)
            try:
                if reply:
                    page.goto(
                        f"https://x.com/i/status/{in_reply_to_tweet_id}",
                        wait_until="domcontentloaded",
                        timeout=self.timeout_ms,
                    )
                    page.wait_for_timeout(4500)
                    self._verify_account(page)
                    if self._is_missing_tweet(page):
                        raise XWebError("Le post auquel répondre est introuvable ou supprimé")
                else:
                    page.goto("https://x.com/home", wait_until="domcontentloaded",
                              timeout=self.timeout_ms)
                    page.wait_for_timeout(4500)
                    self._verify_account(page)

                editor = self._find_editor(page)
                editor.wait_for(state="visible", timeout=15000)
                # Remove an autosaved draft, if present, before writing.
                try:
                    current = editor.inner_text().strip()
                    if current:
                        editor.fill("")
                except Exception:
                    pass
                editor.click()
                editor.fill(text)
                page.wait_for_timeout(500)
                actual = editor.inner_text().strip()
                if not actual:
                    raise XWebError("Le texte n'a pas pu être saisi dans l'éditeur X")

                # Upload des medias avec fallback: si ca echoue, on publie sans
                # les medias plutot que de perdre le tweet complet.
                if media_paths:
                    try:
                        self._attach_media(page, media_paths)
                    except XWebMediaError as exc:
                        logger.warning(
                            "Upload media echoue definitivement, publication sans media: {}", exc
                        )
                        media_paths = []

                button = self._find_post_button(page, reply=reply)
                # Let X finish enabling the button after media processing.
                deadline = time.time() + 35
                while time.time() < deadline and button.is_disabled():
                    page.wait_for_timeout(500)
                if button.is_disabled():
                    raise XWebError("Bouton de publication toujours désactivé")
                payload = self._click_and_capture(page, button)
                tweet_id = self._extract_tweet_id(payload)
                if not tweet_id:
                    raise XWebUncertainError("Post sans identifiant retourné")
                page.wait_for_timeout(1500)
                logger.info(
                    "{} publié via X web: {}",
                    "Réponse" if reply else "Post",
                    tweet_id,
                )
                return str(tweet_id)
            except (XWebError, XWebAuthError):
                raise
            except Exception as exc:
                self.last_error = str(exc)
                raise XWebError(f"Publication web impossible: {exc}") from exc

    def reply_to(self, comment_id: str, text: str) -> str:
        return self.post_tweet(text, in_reply_to_tweet_id=str(comment_id))

    @staticmethod
    def _is_missing_tweet(page) -> bool:
        try:
            body = page.locator("body").inner_text().lower()
        except Exception:
            return False
        markers = (
            "ce post n'existe pas",
            "ce post n’existe pas",
            "cette page n'existe pas",
            "cette page n’existe pas",
            "hmm... cette page n'existe pas",
            "this post is unavailable",
            "this page doesn't exist",
        )
        return any(marker in body for marker in markers)

    def delete_tweet(self, tweet_id: str) -> bool:
        with self._session() as page:
            page.goto(
                f"https://x.com/{self.expected_handle}/status/{tweet_id}",
                wait_until="domcontentloaded",
                timeout=self.timeout_ms,
            )
            page.wait_for_timeout(4500)
            if self._is_missing_tweet(page):
                return True
            carets = page.locator('[data-testid="caret"]:visible')
            if carets.count() == 0:
                raise XWebError("Menu du post introuvable pour la suppression")
            carets.first.click()
            page.wait_for_timeout(900)
            deleted_menu = False
            for label in ("Supprimer", "Delete"):
                item = page.get_by_role("menuitem", name=re.compile(f"^{label}$", re.IGNORECASE))
                if item.count():
                    item.first.click()
                    deleted_menu = True
                    break
            if not deleted_menu:
                raise XWebError("Option Supprimer introuvable")
            page.wait_for_timeout(1200)
            confirmed = False
            for label in ("Supprimer", "Delete"):
                button = page.get_by_role("button", name=re.compile(f"^{label}$", re.IGNORECASE))
                if button.count():
                    button.last.click()
                    confirmed = True
                    break
            if not confirmed:
                raise XWebError("Confirmation de suppression introuvable")
            page.wait_for_timeout(3000)
            body = page.locator("body").inner_text().lower()
            if "supprimé" in body or "deleted" in body or self._is_missing_tweet(page):
                logger.info("Post {} supprimé", tweet_id)
                return True
            raise XWebError("La suppression du post n'a pas été confirmée")
