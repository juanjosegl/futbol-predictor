"""Ingesta de datos: football-data (obligatoria), Understat, ClubElo y SofaScore (opcionales)."""
from __future__ import annotations

import logging

from ..config import get_path
from . import clubelo, football_data, football_data_org, sofascore, understat
from .team_names import TeamNameMapper

log = logging.getLogger(__name__)


def canonical_mapper(cfg: dict) -> TeamNameMapper:
    df = football_data.load_all(cfg)
    by_league = {lg: sorted(set(g["home_team"]) | set(g["away_team"])) for lg, g in df.groupby("league")}
    return TeamNameMapper(by_league, threshold=cfg["team_names"]["fuzzy_threshold"])


def download_all(cfg: dict, refresh: bool = False, refresh_current: bool = False) -> dict:
    status = {}
    football_data.download(cfg, refresh=refresh, refresh_current=refresh_current)
    status["football_data"] = "ok"
    fx = football_data.download_fixtures(cfg, refresh=refresh or refresh_current)

    def _fdorg():
        st = football_data_org.download(cfg, refresh=True)
        if st == "ok" and fx is None:  # football-data.co.uk no dio próximos partidos: se usa el calendario de la API
            football_data_org.write_upcoming(cfg, football_data_org.load(cfg, canonical_mapper(cfg)))
        return st

    for name, fn in [
        ("football_data_org", _fdorg),
        ("understat", lambda: understat.download(cfg, refresh=refresh)),
        ("clubelo", lambda: clubelo.download(cfg, lambda: canonical_mapper(cfg), refresh=refresh)),
        ("sofascore", lambda: sofascore.download(cfg, refresh=refresh)),
    ]:
        try:
            status[name] = fn()
        except Exception as e:  # noqa: BLE001  nada opcional puede romper el pipeline
            log.warning("%s falló: %s. Se continúa sin esa fuente.", name, e)
            status[name] = f"error: {e}"
    raw = get_path(cfg, "raw")
    status["understat_files"] = len(list((raw / "understat").glob("*.json"))) if (raw / "understat").exists() else 0
    status["clubelo_files"] = len(list((raw / "clubelo").glob("team_*.csv"))) if (raw / "clubelo").exists() else 0
    log.info("Estado de fuentes: %s", status)
    import datetime as dt
    import json

    status["checked_at"] = dt.datetime.now().isoformat(timespec="seconds")
    (get_path(cfg, "reports") / "sources_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False),
                                                                    encoding="utf-8")
    return status
