"""X/Twitter browser-session helpers.

The bot never asks for the account password. It reuses an already logged-in
browser session:

* ``data/x_universecule_state.json`` is the @UniverseCule writing session
  (exported from a logged-in Brave/Chrome profile).
* ``data/x_reader_state.json`` is a read-only session used to browse public
  source profiles and comment threads. It is imported automatically from
  Firefox when it is missing.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
READER_STATE = DATA_DIR / "x_reader_state.json"
WRITER_STATE = DATA_DIR / "x_universecule_state.json"


class XSessionError(RuntimeError):
    pass


def _firefox_profile_dirs() -> List[Path]:
    candidates: List[Path] = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "Mozilla" / "Firefox" / "Profiles")
    home = Path.home()
    candidates.extend([
        home / ".mozilla" / "firefox",
        home / "snap" / "firefox" / "common" / ".mozilla" / "firefox",
        home / "Library" / "Application Support" / "Firefox" / "Profiles",
    ])
    out: List[Path] = []
    for base in candidates:
        if not base.exists():
            continue
        if (base / "cookies.sqlite").exists():
            out.append(base)
        for profile in base.iterdir():
            if profile.is_dir() and (profile / "cookies.sqlite").exists():
                out.append(profile)
    return out


def _firefox_cookie_rows(profile: Path) -> List[Dict]:
    tmp = Path(tempfile.gettempdir()) / f"ff_cookies_{os.getpid()}_{profile.name}.sqlite"
    try:
        shutil.copy2(profile / "cookies.sqlite", tmp)
        conn = sqlite3.connect(str(tmp))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT host, path, name, value, expiry, isSecure, isHttpOnly, sameSite "
            "FROM moz_cookies WHERE host LIKE '%x.com' OR host LIKE '%twitter.com'"
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.debug("Impossible de lire les cookies Firefox {}: {}", profile, exc)
        return []
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _same_site(value) -> Optional[str]:
    # Firefox: 0=no restriction, 1=lax, 2=strict, 3=none. Playwright accepts
    # Lax/Strict/None; omitting the key is safest for unknown values.
    mapping = {0: "Lax", 1: "Lax", 2: "Strict", 3: "None"}
    return mapping.get(int(value)) if value is not None else None


def _playwright_cookie(row: Dict) -> Optional[Dict]:
    name = str(row.get("name") or "")
    value = row.get("value")
    host = str(row.get("host") or "")
    if not name or value is None or not host:
        return None
    expiry = row.get("expiry")
    if expiry:
        expiry = int(expiry)
        expires = expiry // 1000 if expiry > 10_000_000_000 else expiry
    else:
        expires = -1
    cookie = {
        "name": name,
        "value": str(value),
        "domain": host,
        "path": str(row.get("path") or "/"),
        "expires": expires,
        "httpOnly": bool(row.get("isHttpOnly")),
        "secure": bool(row.get("isSecure")) or host.endswith("x.com") or host.endswith("twitter.com"),
    }
    same_site = _same_site(row.get("sameSite"))
    if same_site:
        cookie["sameSite"] = same_site
    return cookie


def import_firefox_cookies() -> List[Dict]:
    """Return Playwright cookies from the first Firefox profile with auth_token."""
    for profile in _firefox_profile_dirs():
        rows = _firefox_cookie_rows(profile)
        if not rows:
            continue
        cookies = [c for c in (_playwright_cookie(r) for r in rows) if c]
        names = {c["name"] for c in cookies}
        if "auth_token" in names and "ct0" in names:
            logger.info("Session X importée depuis le profil Firefox {}", profile.name)
            return cookies
    return []


def storage_state_from_cookies(cookies: Iterable[Dict]) -> Dict:
    return {"cookies": list(cookies), "origins": []}


def has_auth_cookie(state: Dict) -> bool:
    cookies = state.get("cookies") if isinstance(state, dict) else None
    if not cookies:
        return False
    names = {c.get("name") for c in cookies if isinstance(c, dict)}
    return "auth_token" in names and "ct0" in names


def load_state(path: Path) -> Dict:
    if not path.exists():
        raise XSessionError(f"Fichier de session introuvable: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise XSessionError(f"Session X illisible ({path}): {exc}") from exc
    if not has_auth_cookie(data):
        raise XSessionError(
            f"La session X {path} ne contient pas auth_token/ct0. "
            "Reconnectez le compte dans le navigateur puis réexportez la session."
        )
    return data


def save_state(path: Path, state: Dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def ensure_reader_state(force: bool = False) -> Path:
    if not force and READER_STATE.exists():
        try:
            load_state(READER_STATE)
            return READER_STATE
        except XSessionError:
            logger.warning("Session lecteur invalide, nouvel import Firefox...")
    cookies = import_firefox_cookies()
    if not cookies:
        raise XSessionError(
            "Aucune session X Firefox trouvée (auth_token/ct0). "
            "Connectez-vous à x.com dans Firefox, puis relancez."
        )
    save_state(READER_STATE, storage_state_from_cookies(cookies))
    logger.info("Session lecteur enregistrée dans {}", READER_STATE)
    return READER_STATE


def ensure_writer_state(force: bool = False) -> Path:
    if not force and WRITER_STATE.exists():
        try:
            load_state(WRITER_STATE)
            return WRITER_STATE
        except XSessionError:
            logger.warning("Session @UniverseCule invalide.")
    # Try Firefox as a fallback only if the user also has the writer account
    # logged in there.
    cookies = import_firefox_cookies()
    if cookies:
        candidate = storage_state_from_cookies(cookies)
        # We cannot know the account without opening a browser. Save as reader
        # fallback only; writer state must be explicitly exported.
        del candidate
    raise XSessionError(
        f"Session @UniverseCule introuvable ou invalide: {WRITER_STATE}. "
        "Connectez @UniverseCule dans Brave/Chrome et exportez la session "
        "(voir README, section Sessions)."
    )


def state_summary(path: Path) -> Dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    try:
        data = load_state(path)
    except XSessionError as exc:
        return {"path": str(path), "exists": True, "valid": False, "error": str(exc)}
    auth = next((c for c in data["cookies"] if c.get("name") == "auth_token"), {})
    return {
        "path": str(path),
        "exists": True,
        "valid": True,
        "cookies": len(data["cookies"]),
        "auth_expires": auth.get("expires", -1),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gestion des sessions X du bot")
    parser.add_argument("action", choices=["reader", "writer", "status"], nargs="?", default="status")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.action == "reader":
        print(ensure_reader_state(force=args.force))
    elif args.action == "writer":
        print(ensure_writer_state(force=args.force))
    else:
        print(json.dumps({
            "reader": state_summary(READER_STATE),
            "writer": state_summary(WRITER_STATE),
        }, indent=2, ensure_ascii=False))
