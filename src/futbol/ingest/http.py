"""Descargas HTTP con caché en disco."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)


class SourceUnavailable(Exception):
    """La fuente no responde o devolvió un error; el pipeline debe continuar sin ella."""


def import_soccerdata(cache_dir: Path):
    """Importa soccerdata usando nuestra caché y sin que reconfigure el logging global."""
    import os

    os.environ.setdefault("SOCCERDATA_DIR", str(cache_dir))
    os.environ.setdefault("SOCCERDATA_LOGLEVEL", "WARNING")
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    import soccerdata as sd  # noqa: PLC0415

    for h in list(root.handlers):
        if h not in before:
            root.removeHandler(h)
    for h in before:
        if h not in root.handlers:
            root.addHandler(h)
    root.setLevel(level)
    for name, lg in logging.root.manager.loggerDict.items():
        if name.startswith("futbol") and isinstance(lg, logging.Logger):
            lg.disabled = False
    for name in ("soccerdata", "tls_requests", "TLSLibrary"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        lg.handlers.clear()
        lg.propagate = False
    return sd


def make_session(cfg: dict) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": cfg["http"]["user_agent"]})
    return s


def cached_get(
    url: str,
    cache_path: Path,
    *,
    session: requests.Session,
    refresh: bool = False,
    timeout: float = 20,
    retries: int = 2,
    pause: float = 0.5,
    headers: dict | None = None,
) -> bytes:
    """Devuelve el contenido de `url`, usando `cache_path` si existe (salvo refresh)."""
    cache_path = Path(cache_path)
    if cache_path.exists() and cache_path.stat().st_size > 0 and not refresh:
        log.debug("caché: %s", cache_path)
        return cache_path.read_bytes()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, timeout=timeout, headers=headers, allow_redirects=True)
            if r.status_code == 404:
                raise SourceUnavailable(f"404 en {url}")
            r.raise_for_status()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(cache_path)
            time.sleep(pause)
            return r.content
        except SourceUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.debug("fallo %s (intento %d): %s", url, attempt + 1, e)
            time.sleep(pause * (attempt + 1))
    raise SourceUnavailable(f"{url}: {last_err}")
