from futbol.ingest.team_names import TeamNameMapper, normalize

CANON = {
    "E0": ["Man United", "Man City", "Nott'm Forest", "Wolves", "Tottenham", "Newcastle", "Arsenal"],
    "SP1": ["Ath Madrid", "Ath Bilbao", "Betis", "Celta", "Espanol", "Sociedad", "Real Madrid", "Alaves"],
    "D1": ["Bayern Munich", "M'gladbach", "Ein Frankfurt", "RB Leipzig", "FC Koln", "Dortmund"],
}


def mapper(**kw):
    return TeamNameMapper(CANON, threshold=85, **kw)


def test_normalize_strips_accents_and_suffixes():
    assert normalize("Atlético  Madrid") == "atletico madrid"
    assert normalize("Arsenal FC") == "arsenal"
    assert normalize("1. FC Köln") == "koln"


def test_exact_and_manual_aliases():
    m = mapper()
    assert m.map("Arsenal", "E0") == "Arsenal"
    assert m.map("Manchester United", "E0") == "Man United"
    assert m.map("Manchester City", "E0") == "Man City"
    assert m.map("Nottingham Forest", "E0") == "Nott'm Forest"
    assert m.map("Atlético Madrid", "SP1") == "Ath Madrid"
    assert m.map("Athletic Club", "SP1") == "Ath Bilbao"
    assert m.map("RasenBallsport Leipzig", "D1") == "RB Leipzig"
    assert m.map("Borussia M.Gladbach", "D1") == "M'gladbach"


def test_fuzzy_matching():
    m = mapper(aliases={})
    assert m.map("Tottenham Hotspur", "E0") == "Tottenham"
    assert m.map("Newcastle Utd", "E0") == "Newcastle"
    assert m.map("Real Madrid CF", "SP1") == "Real Madrid"
    assert any(r["method"] == "fuzzy" for r in m.matches)


def test_restricted_to_league():
    m = mapper()
    # "Real Madrid" no existe en la Premier: no debe emparejarse con otro equipo de esa liga
    assert m.map("Real Madrid", "E0") is None


def test_unmatched_report():
    m = mapper()
    assert m.map("Equipo Inventado XYZ", "E0") is None
    rep = m.report()
    assert len(rep) == 1
    assert rep.iloc[0]["name"] == "Equipo Inventado XYZ"
    assert rep.iloc[0]["league"] == "E0"


def test_save_report(tmp_path):
    m = mapper()
    m.map("Desconocido FC", "SP1")
    m.map("Tottenham Hotspur", "E0")
    rep = m.save_report(tmp_path / "unmatched_teams.csv")
    assert (tmp_path / "unmatched_teams.csv").exists()
    assert list(rep["name"]) == ["Desconocido FC"]
