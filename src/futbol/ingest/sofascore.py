"""Cliente OPCIONAL para la API de SofaScore (api.sofascore.com/api/v1).

- Desactivado por defecto (`sources.sofascore: false`).
- Caché en disco (data/raw/sofascore/) y pausa entre peticiones (por defecto 2 s).
- Usa un cliente HTTP estándar. No intenta evadir protecciones anti-bot: si SofaScore
  responde 403 u otro error, se registra un warning y el pipeline continúa sin esta fuente.
- Ningún paso crítico del pipeline depende de estos datos.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

from ..config import get_path
from .http import make_session

log = logging.getLogger(__name__)


class SofaScoreBlocked(Exception):
    pass


class SofaScoreClient:
    def __init__(self, cfg: dict, refresh: bool = False):
        sc = cfg.get("sofascore", {})
        self.base = sc.get("base_url", "https://api.sofascore.com/api/v1").rstrip("/")
        self.pause = float(sc.get("pause_seconds", 2.0))
        self.timeout = cfg["http"]["timeout"]
        self.refresh = refresh
        self.cache = get_path(cfg, "raw") / "sofascore"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = make_session(cfg)
        self.session.headers.update({"Accept": "application/json"})
        self._last = 0.0

    def _cache_path(self, path: str) -> Path:
        h = hashlib.sha1(path.encode()).hexdigest()[:16]
        return self.cache / f"{path.strip('/').replace('/', '_')[:80]}_{h}.json"

    def get(self, path: str) -> dict:
        cp = self._cache_path(path)
        if cp.exists() and not self.refresh:
            return json.loads(cp.read_text(encoding="utf-8"))
        wait = self.pause - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()
        r = self.session.get(f"{self.base}/{path.lstrip('/')}", timeout=self.timeout)
        if r.status_code in (401, 403, 429):
            raise SofaScoreBlocked(f"HTTP {r.status_code} en {path}")
        r.raise_for_status()
        data = r.json()
        cp.write_text(json.dumps(data), encoding="utf-8")
        return data

    def seasons(self, tournament_id: int) -> list[dict]:
        return self.get(f"unique-tournament/{tournament_id}/seasons").get("seasons", [])

    def last_events(self, tournament_id: int, season_id: int, page: int = 0) -> list[dict]:
        return self.get(f"unique-tournament/{tournament_id}/season/{season_id}/events/last/{page}").get("events", [])


def download(cfg: dict, refresh: bool = False) -> str:
    """Descarga eventos recientes de cada liga. Devuelve 'ok', 'no disponible' o 'desactivado'."""
    if not cfg["sources"].get("sofascore", False):
        log.info("SofaScore desactivado en config (módulo opcional).")
        return "desactivado"
    client = SofaScoreClient(cfg, refresh=refresh)
    total = 0
    try:
        for lg in cfg["leagues"]:
            tid = lg.get("sofascore_tournament")
            if not tid:
                continue
            seasons = client.seasons(tid)
            if not seasons:
                continue
            sid = seasons[0]["id"]
            for page in range(int(cfg.get("sofascore", {}).get("max_pages", 3))):
                ev = client.last_events(tid, sid, page)
                total += len(ev)
                if not ev:
                    break
        log.info("SofaScore: %d eventos descargados/cacheados", total)
        return "ok"
    except Exception as e:  # noqa: BLE001  (403, red, formato...)
        log.warning("SofaScore no disponible (%s). El pipeline continúa sin esta fuente.", str(e)[:200])
        return "no disponible"
