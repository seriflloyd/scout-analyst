import pandas as pd

from src.tools import match_tools


def test_normalize_name():
    assert match_tools.normalize_name("Luis Suárez") == "luis suarez"
    assert match_tools.normalize_name("  Cristiano   Ronaldo ") == "cristiano ronaldo"


def test_fuzzy_match_players_accepts_subset_name_rejects_wrong_person():
    candidates = pd.DataFrame({
        "player_id": [1, 2, 3],
        "name": ["Luis Suárez", "Sergio Busquets", "Karim Benzema"],
    })
    source_names = [
        "Luis Alberto Suárez Díaz",   # candidates[0]'in tam adi -> eslesmeli
        "Sergio Ramos García",        # candidates'ta yok, sadece 'Sergio' ortak -> eslesmemeli
        "Karim Benzema",              # tam ayni -> eslesmeli
    ]

    result = match_tools.fuzzy_match_players(source_names, candidates, score_threshold=90.0)

    suarez_row = result[result["source_name"] == "Luis Alberto Suárez Díaz"].iloc[0]
    assert suarez_row["player_id"] == 1
    assert suarez_row["score"] >= 90

    ramos_row = result[result["source_name"] == "Sergio Ramos García"].iloc[0]
    assert pd.isna(ramos_row["player_id"])

    benzema_row = result[result["source_name"] == "Karim Benzema"].iloc[0]
    assert benzema_row["player_id"] == 3


def test_fuzzy_match_players_rejects_tied_candidates():
    """'Sergio Ramos Garcia' hem 'Sergio Ramos' hem 'Sergio Garcia' adaylarina
    skor=100 ile esit derecede uyar - hangisinin dogru kisi oldugu skordan
    ayirt edilemeyecegi icin eslesme reddedilmeli (rastgele/sirali secim
    yapilmamali)."""
    candidates = pd.DataFrame({
        "player_id": [1, 2],
        "name": ["Sergio Ramos", "Sergio García"],
    })
    result = match_tools.fuzzy_match_players(
        ["Sergio Ramos García"], candidates, score_threshold=90.0
    )
    assert result.iloc[0]["player_id"] is None or pd.isna(result.iloc[0]["player_id"])


def test_fuzzy_match_players_alt_names_resolves_common_surname_collision():
    """'Fernando Llorente Torres' (gercek adi Fernando Llorente) tam legal
    isminde HEM 'Torres' HEM 'Llorente' soyismini tasidigindan, aday
    havuzundaki iki FARKLI gercek oyuncuya ('Fernando Torres' ve 'Fernando
    Llorente') skor=100 ile ayni anda catisir - alt_names verilmeden bu
    catisma (tie_margin) yuzunden reddedilmeli. alt_names (StatsBomb
    player_nickname, orn. 'Fernando Llorente') verildiginde ise sadece dogru
    adaya net sekilde eslesmeli, cunku kisa/dogru nickname'de rakip soyisim
    (Torres) artik yer almiyor."""
    candidates = pd.DataFrame({
        "player_id": [1, 2],
        "name": ["Fernando Torres", "Fernando Llorente"],
    })
    source_names = ["Fernando Llorente Torres"]

    without_alt = match_tools.fuzzy_match_players(source_names, candidates, score_threshold=90.0)
    assert without_alt.iloc[0]["player_id"] is None or pd.isna(without_alt.iloc[0]["player_id"])

    alt_names = {"Fernando Llorente Torres": "Fernando Llorente"}
    with_alt = match_tools.fuzzy_match_players(
        source_names, candidates, score_threshold=90.0, alt_names=alt_names
    )
    assert with_alt.iloc[0]["player_id"] == 2


def test_get_player_nicknames_skips_missing_and_blank():
    lineups_df = pd.DataFrame({
        "player_id": [1, 1, 2, 3],
        "player_nickname": ["Fernando Torres", "Fernando Torres", None, "  "],
    })
    nicknames = match_tools.get_player_nicknames(lineups_df)
    assert nicknames == {1: "Fernando Torres"}


def test_demote_ambiguous_matches_rejects_many_to_one_collisions():
    """Tek kelimelik bir Transfermarkt adi (orn. 'Pedro'), onu tam adinin
    icinde barindiran birden fazla FARKLI StatsBomb oyuncusuyla skor=100
    eslesebilir (token_set_ratio'nun bilinen zaafi) - bu durumda hicbiri
    guvenilir degildir, ikisi de reddedilmelidir."""
    match_result = pd.DataFrame({
        "source_name": ["Pedro Bigas Rigo", "João Pedro Cavaco Cancelo", "Karim Benzema"],
        "player_id": [65278, 65278, 3],
        "matched_name": ["Pedro", "Pedro", "Karim Benzema"],
        "score": [100.0, 100.0, 100.0],
    })

    result = match_tools.demote_ambiguous_matches(match_result)

    assert result[result["source_name"] == "Pedro Bigas Rigo"]["player_id"].isna().all()
    assert result[result["source_name"] == "João Pedro Cavaco Cancelo"]["player_id"].isna().all()
    assert result[result["source_name"] == "Karim Benzema"]["player_id"].iloc[0] == 3


def test_build_transfermarkt_candidates_filters_window_and_min_value():
    players = pd.DataFrame({
        "player_id": [1, 2, 3],
        "name": ["Player A", "Player B", "Player C"],
        "date_of_birth": ["1990-01-01", "1991-01-01", "1992-01-01"],
        "position": ["Attack", "Midfield", "Defender"],
        "sub_position": ["Centre-Forward", "Central Midfield", "Centre-Back"],
    })
    valuations = pd.DataFrame({
        "player_id": [1, 1, 2, 3],
        "date": ["2015-08-01", "2016-01-01", "2015-09-01", "2014-06-01"],
        "market_value_in_eur": [5_000_000, 8_000_000, 50_000, 9_000_000],
        "player_club_domestic_competition_id": ["ES1", "ES1", "ES1", "ES1"],
    })

    candidates = match_tools.build_transfermarkt_candidates(
        players, valuations,
        competition_code="ES1", season_start="2015-07-01", season_end="2016-06-30",
        min_market_value=100_000,
    )

    # player 1: pencere icinde en son (2016-01-01) deger alinmali
    assert candidates.loc[candidates["player_id"] == 1, "market_value_in_eur"].iloc[0] == 8_000_000
    # player 2: taban-deger altinda (50_000 < 100_000) -> elenmeli
    assert 2 not in candidates["player_id"].values
    # player 3: pencere disinda (2014) -> aday havuzunda olmamali
    assert 3 not in candidates["player_id"].values


def test_build_statsbomb_value_pool_end_to_end():
    npxg_df = pd.DataFrame({
        "player_name": ["Luis Alberto Suárez Díaz", "Isimsiz Bilinmeyen Oyuncu"],
        "npxg": [20.0, 2.0],
        "minutes": [1800.0, 950.0],
        "npxg_per90": [1.0, 0.19],
    })
    players = pd.DataFrame({
        "player_id": [1],
        "name": ["Luis Suárez"],
        "date_of_birth": ["1987-01-24"],
        "position": ["Attack"],
        "sub_position": ["Centre-Forward"],
    })
    valuations = pd.DataFrame({
        "player_id": [1],
        "date": ["2016-01-01"],
        "market_value_in_eur": [80_000_000],
        "player_club_domestic_competition_id": ["ES1"],
    })

    matched, unmatched = match_tools.build_statsbomb_value_pool(
        npxg_df, players, valuations,
        competition_code="ES1", season_start="2015-07-01", season_end="2016-06-30",
        min_market_value=100_000,
    )

    assert len(matched) == 1
    assert matched.iloc[0]["player_name"] == "Luis Alberto Suárez Díaz"
    assert matched.iloc[0]["market_value_in_eur"] == 80_000_000
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["player_name"] == "Isimsiz Bilinmeyen Oyuncu"


def test_build_statsbomb_value_pool_applies_minutes_threshold():
    """Kucuk-orneklem gurultusunu elemek icin, eslesmis olsa bile min_minutes
    altinda kalan oyuncular sonuc havuzundan dusmeli (build_eligible_pool()'daki
    apply_minutes_threshold() ile ayni davranis)."""
    npxg_df = pd.DataFrame({
        "player_name": ["Luis Alberto Suárez Díaz", "Karim Benzema"],
        "npxg": [20.0, 1.0],
        "minutes": [1800.0, 320.0],
        "npxg_per90": [1.0, 0.28],
    })
    players = pd.DataFrame({
        "player_id": [1, 2],
        "name": ["Luis Suárez", "Karim Benzema"],
        "date_of_birth": ["1987-01-24", "1987-12-19"],
        "position": ["Attack", "Attack"],
        "sub_position": ["Centre-Forward", "Centre-Forward"],
    })
    valuations = pd.DataFrame({
        "player_id": [1, 2],
        "date": ["2016-01-01", "2016-01-01"],
        "market_value_in_eur": [80_000_000, 40_000_000],
        "player_club_domestic_competition_id": ["ES1", "ES1"],
    })

    matched, unmatched = match_tools.build_statsbomb_value_pool(
        npxg_df, players, valuations,
        competition_code="ES1", season_start="2015-07-01", season_end="2016-06-30",
        min_market_value=100_000, min_minutes=900,
    )

    assert len(matched) == 1
    assert matched.iloc[0]["player_name"] == "Luis Alberto Suárez Díaz"
    assert "Karim Benzema" not in matched["player_name"].values
