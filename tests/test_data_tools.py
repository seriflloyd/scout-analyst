import pandas as pd
import pytest
from src.tools import data_tools


def test_per90():
    df = pd.DataFrame({"minutes": [900, 450], "goals": [10, 5], "assists": [0, 0]})
    out = data_tools.add_per90(df)
    assert out["goals_per90"].iloc[0] == 1.0
    assert out["goals_per90"].iloc[1] == 1.0


def test_threshold():
    df = pd.DataFrame({"minutes": [1000, 100]})
    assert len(data_tools.apply_minutes_threshold(df, 900)) == 1


def test_build_eligible_pool_filters_floor_value_artifact(tmp_path):
    """Transfermarkt'ta taban deger (< MIN_MARKET_VALUE) alan sahte bir kaydin
    build_eligible_pool() tarafindan elendigini, esik ustu kaydin ise
    havuzda kaldigini dogrular."""
    season_year = pd.Timestamp.now().year

    competitions = pd.DataFrame({
        "competition_id": ["GB1"],
        "competition_code": ["premier-league"],
        "name": ["premier-league"],
        "sub_type": ["first_tier"],
        "type": ["domestic_league"],
        "country_id": [189],
        "country_name": ["England"],
        "domestic_league_code": ["GB1"],
        "confederation": ["europa"],
        "total_clubs": [20],
        "url": [""],
    })

    def make_appearances(player_id, player_name):
        return [
            {
                "appearance_id": f"{player_id}_{i}",
                "game_id": 1000 * player_id + i,
                "player_id": player_id,
                "player_club_id": 1,
                "player_current_club_id": 1,
                "date": f"{season_year}-03-{i + 1:02d}",
                "player_name": player_name,
                "competition_id": "GB1",
                "yellow_cards": 0,
                "red_cards": 0,
                "goals": 0,
                "assists": 0,
                "minutes_played": 90,
            }
            for i in range(10)
        ]

    appearances = pd.DataFrame(
        make_appearances(1, "Eligible Player") + make_appearances(2, "Floor Value Player")
    )

    players = pd.DataFrame({
        "player_id": [1, 2],
        "name": ["Eligible Player", "Floor Value Player"],
        "position": ["Attack", "Attack"],
        "sub_position": ["Centre-Forward", "Centre-Forward"],
        "date_of_birth": ["1998-01-01", "1998-01-01"],
        "country_of_citizenship": ["England", "England"],
        "foot": ["right", "right"],
    })

    valuations = pd.DataFrame({
        "player_id": [1, 2],
        "date": [f"{season_year}-01-15", f"{season_year}-01-15"],
        "market_value_in_eur": [5_000_000, 50_000],
        "current_club_name": ["Test FC", "Test FC"],
        "current_club_id": [1, 1],
        "player_club_domestic_competition_id": ["GB1", "GB1"],
    })

    appearances_path = tmp_path / "appearances.csv"
    players_path = tmp_path / "players.csv"
    valuations_path = tmp_path / "valuations.csv"
    competitions_path = tmp_path / "competitions.csv"
    output_path = tmp_path / "eligible_pool.parquet"

    appearances.to_csv(appearances_path, index=False)
    players.to_csv(players_path, index=False)
    valuations.to_csv(valuations_path, index=False)
    competitions.to_csv(competitions_path, index=False)

    pool = data_tools.build_eligible_pool(
        appearances_path=str(appearances_path),
        players_path=str(players_path),
        valuations_path=str(valuations_path),
        competitions_path=str(competitions_path),
        output_path=str(output_path),
    )

    assert 1 in pool["player_id"].values
    assert 2 not in pool["player_id"].values
    assert output_path.exists()


def test_build_eligible_pool_computes_contract_years_remaining(tmp_path):
    """4 oyuncu, sozlesme bitis tarihine gore 4 farkli durum:
    1) season_end'e ~100 gun kalmis (~0.27 yil) -> esik (0.5) ALTINDA,
       is_free_agent_soon True.
    2) season_end'e ~1000 gun kalmis (~2.74 yil) -> esik ustunde, False.
    3) contract_expiration_date bilinmiyor (NaN) -> contract_years_remaining
       NaN, is_free_agent_soon False (NaN degil - bilinmeyen durum bedava
       transfer olarak isaretlenmez).
    4) contract_expiration_date season_end'den ONCE (veride tutarsizlik) ->
       negatif fark 0'a kirpiimaz, dogrudan NaN olur; is_free_agent_soon False.
    """
    season_year = pd.Timestamp.now().year
    season_end = pd.Timestamp(f"{season_year}-12-31")

    competitions = pd.DataFrame({
        "competition_id": ["GB1"],
        "competition_code": ["premier-league"],
        "name": ["premier-league"],
        "sub_type": ["first_tier"],
        "type": ["domestic_league"],
        "country_id": [189],
        "country_name": ["England"],
        "domestic_league_code": ["GB1"],
        "confederation": ["europa"],
        "total_clubs": [20],
        "url": [""],
    })

    def make_appearances(player_id, player_name):
        return [
            {
                "appearance_id": f"{player_id}_{i}",
                "game_id": 1000 * player_id + i,
                "player_id": player_id,
                "player_club_id": 1,
                "player_current_club_id": 1,
                "date": f"{season_year}-03-{i + 1:02d}",
                "player_name": player_name,
                "competition_id": "GB1",
                "yellow_cards": 0,
                "red_cards": 0,
                "goals": 0,
                "assists": 0,
                "minutes_played": 90,
            }
            for i in range(10)
        ]

    player_ids = [1, 2, 3, 4]
    appearances = pd.DataFrame(
        make_appearances(1, "Soon Free Agent")
        + make_appearances(2, "Long Contract")
        + make_appearances(3, "Unknown Contract")
        + make_appearances(4, "Stale Past Date")
    )

    contract_dates = {
        1: (season_end + pd.Timedelta(days=100)).strftime("%Y-%m-%d"),
        2: (season_end + pd.Timedelta(days=1000)).strftime("%Y-%m-%d"),
        3: None,
        4: (season_end - pd.Timedelta(days=200)).strftime("%Y-%m-%d"),
    }
    players = pd.DataFrame({
        "player_id": player_ids,
        "name": ["Soon Free Agent", "Long Contract", "Unknown Contract", "Stale Past Date"],
        "position": ["Attack"] * 4,
        "sub_position": ["Centre-Forward"] * 4,
        "date_of_birth": ["1998-01-01"] * 4,
        "country_of_citizenship": ["England"] * 4,
        "foot": ["right"] * 4,
        "contract_expiration_date": [contract_dates[pid] for pid in player_ids],
    })

    valuations = pd.DataFrame({
        "player_id": player_ids,
        "date": [f"{season_year}-01-15"] * 4,
        "market_value_in_eur": [5_000_000] * 4,
        "current_club_name": ["Test FC"] * 4,
        "current_club_id": [1] * 4,
        "player_club_domestic_competition_id": ["GB1"] * 4,
    })

    appearances_path = tmp_path / "appearances.csv"
    players_path = tmp_path / "players.csv"
    valuations_path = tmp_path / "valuations.csv"
    competitions_path = tmp_path / "competitions.csv"
    output_path = tmp_path / "eligible_pool.parquet"

    appearances.to_csv(appearances_path, index=False)
    players.to_csv(players_path, index=False)
    valuations.to_csv(valuations_path, index=False)
    competitions.to_csv(competitions_path, index=False)

    pool = data_tools.build_eligible_pool(
        appearances_path=str(appearances_path),
        players_path=str(players_path),
        valuations_path=str(valuations_path),
        competitions_path=str(competitions_path),
        output_path=str(output_path),
    )
    by_id = pool.set_index("player_id")

    assert by_id.loc[1, "contract_years_remaining"] == pytest.approx(100 / 365.25, abs=1e-6)
    assert by_id.loc[1, "is_free_agent_soon"] == True  # noqa: E712

    assert by_id.loc[2, "contract_years_remaining"] == pytest.approx(1000 / 365.25, abs=1e-6)
    assert by_id.loc[2, "is_free_agent_soon"] == False  # noqa: E712

    assert pd.isna(by_id.loc[3, "contract_years_remaining"])
    assert by_id.loc[3, "is_free_agent_soon"] == False  # noqa: E712

    assert pd.isna(by_id.loc[4, "contract_years_remaining"])
    assert by_id.loc[4, "is_free_agent_soon"] == False  # noqa: E712


def test_build_eligible_pool_contract_years_remaining_only_for_latest_season(tmp_path):
    """players.csv oyuncu basina TEK (guncel) contract_expiration_date tutar,
    sezona ozgu sozlesme gecmisi yoktur. Bu yuzden ayni oyuncunun havuzdaki
    EN ESKI sezonunda contract_years_remaining NaN/False olmali (gecerli bir
    ham tarih farki olsa bile) - sadece EN GUNCEL sezonda hesaplanmali.
    Aksi halde oyuncunun DAHA SONRA imzaladigi bir sozlesme, cok daha eski
    bir sezonu icin 'kalan sure' gibi kullanilarak look-ahead bias
    yaratirdi (bkz. build_eligible_pool() docstring'i)."""
    season_a = pd.Timestamp.now().year - 1
    season_b = pd.Timestamp.now().year
    season_b_end = pd.Timestamp(f"{season_b}-12-31")

    competitions = pd.DataFrame({
        "competition_id": ["GB1"],
        "competition_code": ["premier-league"],
        "name": ["premier-league"],
        "sub_type": ["first_tier"],
        "type": ["domestic_league"],
        "country_id": [189],
        "country_name": ["England"],
        "domestic_league_code": ["GB1"],
        "confederation": ["europa"],
        "total_clubs": [20],
        "url": [""],
    })

    def make_appearances(player_id, season_year):
        return [
            {
                "appearance_id": f"{player_id}_{season_year}_{i}",
                "game_id": 1000 * player_id + 100 * season_year + i,
                "player_id": player_id,
                "player_club_id": 1,
                "player_current_club_id": 1,
                "date": f"{season_year}-03-{i + 1:02d}",
                "player_name": "Long Career Player",
                "competition_id": "GB1",
                "yellow_cards": 0,
                "red_cards": 0,
                "goals": 0,
                "assists": 0,
                "minutes_played": 90,
            }
            for i in range(10)
        ]

    appearances = pd.DataFrame(
        make_appearances(1, season_a) + make_appearances(1, season_b)
    )

    players = pd.DataFrame({
        "player_id": [1],
        "name": ["Long Career Player"],
        "position": ["Attack"],
        "sub_position": ["Centre-Forward"],
        "date_of_birth": ["1998-01-01"],
        "country_of_citizenship": ["England"],
        "foot": ["right"],
        "contract_expiration_date": [(season_b_end + pd.Timedelta(days=500)).strftime("%Y-%m-%d")],
    })

    valuations = pd.DataFrame({
        "player_id": [1, 1],
        "date": [f"{season_a}-01-15", f"{season_b}-01-15"],
        "market_value_in_eur": [5_000_000, 5_000_000],
        "current_club_name": ["Test FC", "Test FC"],
        "current_club_id": [1, 1],
        "player_club_domestic_competition_id": ["GB1", "GB1"],
    })

    appearances_path = tmp_path / "appearances.csv"
    players_path = tmp_path / "players.csv"
    valuations_path = tmp_path / "valuations.csv"
    competitions_path = tmp_path / "competitions.csv"
    output_path = tmp_path / "eligible_pool.parquet"

    appearances.to_csv(appearances_path, index=False)
    players.to_csv(players_path, index=False)
    valuations.to_csv(valuations_path, index=False)
    competitions.to_csv(competitions_path, index=False)

    pool = data_tools.build_eligible_pool(
        appearances_path=str(appearances_path),
        players_path=str(players_path),
        valuations_path=str(valuations_path),
        competitions_path=str(competitions_path),
        output_path=str(output_path),
    )
    by_season = pool.set_index("season")

    assert by_season.loc[season_b, "contract_years_remaining"] == pytest.approx(500 / 365.25, abs=1e-6)
    assert by_season.loc[season_b, "is_free_agent_soon"] == False  # noqa: E712

    # ayni oyuncu, ayni (gecerli) ham contract_expiration_date - ama eski sezon icin NaN/False olmali
    assert pd.isna(by_season.loc[season_a, "contract_years_remaining"])
    assert by_season.loc[season_a, "is_free_agent_soon"] == False  # noqa: E712
