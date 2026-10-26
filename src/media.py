"""Download source images and videos into the local cache for X re-upload."""
from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlparse

from loguru import logger

from .config import get_config
from .x_session import PROJECT_ROOT


class MediaDownloadError(RuntimeError):
    pass


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
_EXT_BY_TYPE = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png",
    "image/webp": ".webp", "image/gif": ".gif", "video/mp4": ".mp4",
}


def _cache_dir(config) -> Path:
    path = Path(config.media.cache_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _guess_extension(url: str, content_type: str) -> str:
    if content_type:
        ext = _EXT_BY_TYPE.get(content_type.split(";")[0].strip().lower())
        if ext:
            return ext
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS:
        return ".jpg" if suffix == ".jpeg" else suffix
    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip()) if content_type else None
    return guessed if guessed in _IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS else ".jpg"


def _valid_image_magic(data: bytes) -> bool:
    return (
        data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"GIF87a") or data.startswith(b"GIF89a")
        or (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


def _valid_video_magic(data: bytes) -> bool:
    # MP4/MOV stores an ftyp box near the beginning of the file.
    return len(data) >= 12 and data[4:8] == b"ftyp"


def download_media(
    urls: Iterable[str], source_tweet_id: str, database: Database, config=None,
    media_types: Optional[Iterable[str]] = None, max_images: Optional[int] = None,
) -> List[str]:
    config = config or get_config()
    if not config.media.enabled:
        return []
    max_media = int(getattr(config.media, "max_media_per_tweet", 4))
    type_list = list(media_types or [])
    max_image_bytes = int(config.media.max_size_mb) * 1024 * 1024
    max_video_bytes = int(getattr(config.media, "max_video_size_mb", 50)) * 1024 * 1024
    timeout = int(config.media.timeout_seconds)
    cache_dir = _cache_dir(config)
    paths: List[str] = []
    seen = set()
    for index, url in enumerate(urls):
        if not url or url in seen or len(paths) >= max_media:
            continue
        seen.add(url)
        media_type = type_list[index] if index < len(type_list) else "photo"
        is_video = media_type == "video"
        max_bytes = max_video_bytes if is_video else max_image_bytes
        cached = database.get_cached_media(url)
        if cached and Path(cached["local_path"]).exists() and Path(cached["local_path"]).stat().st_size > 0:
            paths.append(cached["local_path"])
            continue
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        safe_id = re.sub(r"[^0-9A-Za-z_-]", "", str(source_tweet_id))[:24] or "src"
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
                "Referer": "https://x.com/",
                "Accept": "video/mp4,image/*,*/*;q=0.8",
            })
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                if is_video and content_type and not content_type.startswith("video/"):
                    continue
                if not is_video and content_type and not content_type.startswith("image/"):
                    continue
                data = response.read(max_bytes + 1)
            if len(data) > max_bytes:
                logger.warning("Média ignoré (trop volumineux): {}", url[:90])
                continue
            valid = _valid_video_magic(data) if is_video else _valid_image_magic(data)
            if not valid:
                logger.warning("Média ignoré (fichier invalide): {}", url[:90])
                continue
            ext = ".mp4" if is_video else _guess_extension(url, content_type)
            local_path = cache_dir / f"{safe_id}_{index}_{key}{ext}"
            tmp = local_path.with_suffix(local_path.suffix + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, local_path)
            database.cache_media(url, str(local_path), "video" if is_video else "photo", len(data))
            paths.append(str(local_path))
            logger.debug("Média téléchargé: {} ({} octets)", local_path.name, len(data))
        except Exception as exc:
            logger.warning("Téléchargement média impossible ({}): {}", url[:80], exc)
    return paths


class MediaDownloader:
    def __init__(self, database: Database, config=None):
        self.db = database
        self.config = config or get_config()

    def download(self, urls: Iterable[str], source_tweet_id: str, media_types=None) -> List[str]:
        return download_media(urls, source_tweet_id, self.db, self.config, media_types=media_types)
