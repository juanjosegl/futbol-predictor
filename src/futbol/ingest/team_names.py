"""Mapeo de nombres de equipos entre fuentes.

Los nombres canónicos son los de football-data.co.uk (fuente principal). Para cada nombre de
otra fuente se intenta, en orden: diccionario manual -> coincidencia exacta normalizada ->
fuzzy matching (rapidfuzz) restringido a los equipos de la misma liga.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

# Alias conocidos (nombre en otra fuente -> nombre football-data). Claves normalizadas al usarse.
MANUAL_ALIASES: dict[str, str] = {
    # Inglaterra
    "Manchester United": "Man United", "Manchester Utd": "Man United", "Man Utd": "Man United",
    "Manchester City": "Man City", "Newcastle United": "Newcastle", "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest", "Forest": "Nott'm Forest", "West Bromwich Albion": "West Brom",
    "Tottenham Hotspur": "Tottenham", "Spurs": "Tottenham", "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton", "Leicester City": "Leicester", "Leeds United": "Leeds",
    "Norwich City": "Norwich", "West Ham United": "West Ham", "Sheffield Utd": "Sheffield United",
    "Luton Town": "Luton", "Ipswich Town": "Ipswich", "AFC Bournemouth": "Bournemouth",
    "Cardiff City": "Cardiff", "Huddersfield Town": "Huddersfield", "Coventry City": "Coventry",
    "Sunderland AFC": "Sunderland",
    # España
    "Atletico Madrid": "Ath Madrid", "Atlético Madrid": "Ath Madrid", "Atletico": "Ath Madrid",
    "Athletic Club": "Ath Bilbao", "Athletic Bilbao": "Ath Bilbao", "Bilbao": "Ath Bilbao",
    "Real Betis": "Betis", "Celta Vigo": "Celta", "Espanyol": "Espanol", "RCD Espanyol": "Espanol",
    "Rayo Vallecano": "Vallecano", "Real Sociedad": "Sociedad", "Deportivo Alaves": "Alaves",
    "Real Valladolid": "Valladolid", "SD Huesca": "Huesca", "Cadiz": "Cadiz", "Leganes": "Leganes",
    "Real Oviedo": "Oviedo", "Sporting Gijon": "Sp Gijon", "Gijon": "Sp Gijon",
    # Italia
    "AC Milan": "Milan", "Internazionale": "Inter", "Inter Milan": "Inter", "AS Roma": "Roma",
    "Parma Calcio 1913": "Parma", "SPAL 2013": "Spal", "SPAL": "Spal", "Hellas Verona": "Verona",
    # Alemania
    "Bayern Munich": "Bayern Munich", "Bayern": "Bayern Munich", "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Dortmund", "Borussia M.Gladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach", "Gladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt", "Frankfurt": "Ein Frankfurt",
    "RasenBallsport Leipzig": "RB Leipzig", "Leipzig": "RB Leipzig", "Bayer Leverkusen": "Leverkusen",
    "FC Cologne": "FC Koln", "Koeln": "FC Koln", "1. FC Köln": "FC Koln", "Hertha Berlin": "Hertha",
    "Fortuna Duesseldorf": "Fortuna Dusseldorf", "Duesseldorf": "Fortuna Dusseldorf",
    "Werder Bremen": "Werder Bremen", "Werder": "Werder Bremen", "Schalke 04": "Schalke 04",
    "Schalke": "Schalke 04", "Mainz 05": "Mainz", "FSV Mainz 05": "Mainz",
    "Arminia Bielefeld": "Bielefeld", "Greuther Fuerth": "Greuther Furth", "Fuerth": "Greuther Furth",
    "VfL Bochum": "Bochum", "Darmstadt 98": "Darmstadt", "FC Heidenheim": "Heidenheim",
    "St. Pauli": "St Pauli", "Holstein Kiel": "Holstein Kiel", "Holstein": "Holstein Kiel",
    "VfB Stuttgart": "Stuttgart", "TSG Hoffenheim": "Hoffenheim", "SC Freiburg": "Freiburg",
    "VfL Wolfsburg": "Wolfsburg", "1. FC Union Berlin": "Union Berlin", "FC Augsburg": "Augsburg",
    "Hamburger SV": "Hamburg", "HSV": "Hamburg", "Paderborn": "Paderborn",
    # Francia
    "Paris Saint Germain": "Paris SG", "Paris Saint-Germain": "Paris SG", "PSG": "Paris SG",
    "Saint-Etienne": "St Etienne", "Saint Etienne": "St Etienne", "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon", "Stade Rennais": "Rennes", "Clermont Foot": "Clermont",
    "Paris FC": "Paris FC", "Le Havre": "Le Havre", "Brest": "Brest", "Stade Brestois": "Brest",
}

_STOPWORDS = {"fc", "cf", "afc", "ac", "as", "sc", "ssc", "us", "ss", "rc", "cd", "ud", "sd", "rcd",
              "club", "de", "calcio", "vfl", "vfb", "tsg", "fsv", "sv", "1", "olympique", "stade"}


def normalize(name: str) -> str:
    """Minúsculas, sin acentos, sin puntuación ni prefijos/sufijos típicos (FC, CF, ...)."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t not in _STOPWORDS]
    return " ".join(tokens) if tokens else s.strip()


class TeamNameMapper:
    """Empareja nombres de otras fuentes con los nombres canónicos (football-data)."""

    def __init__(self, canonical_by_league: dict[str, Iterable[str]], threshold: int = 85,
                 aliases: dict[str, str] | None = None):
        self.canonical_by_league = {lg: sorted(set(v)) for lg, v in canonical_by_league.items()}
        self.all_canonical = sorted({t for v in self.canonical_by_league.values() for t in v})
        self.threshold = threshold
        aliases = aliases if aliases is not None else MANUAL_ALIASES
        self.aliases = {normalize(k): v for k, v in aliases.items()}
        self._norm = {}
        for t in self.all_canonical:
            self._norm.setdefault(normalize(t), t)
        self.unmatched: list[dict] = []
        self.matches: list[dict] = []
        self._cache: dict[tuple[str, str | None], str | None] = {}

    def map(self, name: str, league: str | None = None, source: str = "") -> str | None:
        key = (name, league)
        if key in self._cache:
            return self._cache[key]
        result, method, score = self._map(name, league)
        self._cache[key] = result
        rec = {"source": source, "league": league, "name": name, "mapped": result,
               "method": method, "score": score}
        (self.matches if result else self.unmatched).append(rec)
        if result is None:
            log.debug("Sin emparejar: %s (%s, %s)", name, league, source)
        return result

    def _map(self, name: str, league: str | None):
        candidates = self.canonical_by_league.get(league, self.all_canonical) if league else self.all_canonical
        if name in candidates:
            return name, "exact", 100.0
        n = normalize(name)
        alias = self.aliases.get(n)
        if alias and alias in candidates:
            return alias, "manual", 100.0
        cand_norm = {normalize(c): c for c in candidates}
        if n in cand_norm:
            return cand_norm[n], "normalized", 100.0
        if not cand_norm:
            return None, "none", 0.0
        best = process.extractOne(n, list(cand_norm.keys()), scorer=fuzz.WRatio)
        if best and best[1] >= self.threshold:
            return cand_norm[best[0]], "fuzzy", float(best[1])
        return None, "unmatched", float(best[1]) if best else 0.0

    def report(self) -> pd.DataFrame:
        cols = ["source", "league", "name", "mapped", "method", "score"]
        return pd.DataFrame(self.unmatched, columns=cols).drop_duplicates()

    def save_report(self, path: Path) -> pd.DataFrame:
        rep = self.report()
        rep.to_csv(path, index=False)
        fuzzy = pd.DataFrame(self.matches)
        if not fuzzy.empty:
            fuzzy = fuzzy[fuzzy["method"] == "fuzzy"].drop_duplicates()
            fuzzy.to_csv(Path(path).with_name("fuzzy_matches.csv"), index=False)
        if len(rep):
            log.warning("%d nombres de equipo sin emparejar (ver %s)", len(rep), path)
        else:
            log.info("Todos los nombres de equipo se emparejaron.")
        return rep
