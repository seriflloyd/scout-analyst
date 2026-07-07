import pandas as pd
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
