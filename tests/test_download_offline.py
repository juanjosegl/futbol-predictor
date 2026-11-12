"""Si football-data no responde, la descarga debe cortar al primer fallo y usar la caché."""
from __future__ import annotations

import copy

from futbol.config import season_codes
from futbol.ingest import football_data
from futbol.ingest.http import SourceUnavailable


def test_site_down_short_circuits_and_uses_cache(cfg, tmp_path, monkeypatch):
    cfg = copy.deepcopy(cfg)
    cfg["_root"] = str(tmp_path)
    raw = tmp_path / cfg["paths"]["raw"] / "football_data"
    raw.mkdir(parents=True)
    codes = [lg["code"] for lg in cfg["leagues"]]
    seasons = season_codes(cfg)
    for c in codes:
        for s in seasons:
            (raw / f"{c}_{s}.csv").write_text("Div,Date,HomeTeam,AwayTeam\n", encoding="utf-8")

    calls = []

    def fake_get(url, dest, **kw):
        calls.append(url)
        if kw.get("refresh"):
            raise SourceUnavailable(f"{url}: timeout")
        return dest.read_bytes()

    monkeypatch.setattr(football_data, "cached_get", fake_get)
    monkeypatch.setattr(football_data, "_blocked_hint", lambda s: "")
    monkeypatch.setattr(football_data, "current_season_code", lambda: seasons[-1])

    paths = football_data.download(cfg, refresh_current=True)

    refreshed = [u for u in calls if f"/{seasons[-1]}/" in u]
    assert len(refreshed) == 1, "tras el primer fallo de red no debe reintentar otras ligas"
    assert len(paths) == len(codes) * len(seasons), "todas las temporadas deben venir de la caché"
    assert cfg["_football_data_down"] is True
