import pandas as pd

from src.tools import event_tools


def test_compute_goals_per90_counts_all_goals_including_penalties():
    events_df = pd.DataFrame({
        "type": ["Shot", "Shot", "Shot", "Shot", "Pass"],
        "player_id": [1, 1, 2, 2, 1],
        "player": ["A", "A", "B", "B", "A"],
        "shot_outcome": ["Goal", "Saved", "Goal", "Goal", None],
        "shot_type": ["Open Play", "Open Play", "Penalty", "Open Play", None],
    })
    minutes_df = pd.DataFrame({
        "player_id": [1, 2],
        "player_name": ["A", "B"],
        "minutes": [900.0, 450.0],
    })

    out = event_tools.compute_goals_per90(events_df, minutes_df)

    a = out[out["player_id"] == 1].iloc[0]
    assert a["goals"] == 1
    assert a["goals_per90"] == 0.1

    b = out[out["player_id"] == 2].iloc[0]
    # penalti dahil TUM goller sayilir (npxg_per90'dan farkli olarak)
    assert b["goals"] == 2
    assert b["goals_per90"] == 0.4


def test_compute_progressive_passes_per90_classifies_known_distances():
    """(60,40) -> (90,40): baslangic kaleye uzakligi 60, bitis 30, azalma
    %50 -> progresif (>=%25 esigini gecer).
    (60,40) -> (70,40): baslangic 60, bitis 50, azalma %16.7 -> progresif DEGIL.
    Bir set-parca (Corner) pasi, geometrik olarak progresif olsa bile
    (60,40)->(100,40) haric tutulmali. Tamamlanmamis bir pas (pass_outcome
    dolu) da haric tutulmali. Hic pas atmayan bir oyuncu (orn. kaleci) 0
    progressive_passes_per90 ile listede kalmali (TUM pozisyonlar kapsanir)."""
    events_df = pd.DataFrame({
        "type": ["Pass", "Pass", "Pass", "Pass"],
        "player_id": [1, 2, 3, 4],
        "player": ["Progresif", "Progresif Degil", "Set Parca", "Tamamlanmamis"],
        "location": [[60.0, 40.0], [60.0, 40.0], [60.0, 40.0], [60.0, 40.0]],
        "pass_end_location": [[90.0, 40.0], [70.0, 40.0], [100.0, 40.0], [90.0, 40.0]],
        "pass_outcome": [None, None, None, "Incomplete"],
        "pass_type": [None, None, "Corner", None],
    })
    minutes_df = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5],
        "player_name": ["Progresif", "Progresif Degil", "Set Parca", "Tamamlanmamis", "Kaleci"],
        "minutes": [900.0, 900.0, 900.0, 900.0, 900.0],
    })

    out = event_tools.compute_progressive_passes_per90(events_df, minutes_df)
    by_id = out.set_index("player_id")

    assert by_id.loc[1, "progressive_passes"] == 1
    assert by_id.loc[2, "progressive_passes"] == 0
    assert by_id.loc[3, "progressive_passes"] == 0
    assert by_id.loc[4, "progressive_passes"] == 0
    # hic pas atmayan oyuncu da 0 ile listede kalmali, listeden dusmemeli
    assert by_id.loc[5, "progressive_passes"] == 0
    assert len(out) == 5


def test_compute_npxg_per90_separates_open_play_from_set_piece():
    """Ayni oyuncunun biri acik oyun (Regular Play), biri set-parca (From
    Corner) iki sutu var. compute_npxg_per90() (varsayilan filtre) sadece
    acik oyun sutunu, compute_set_piece_npxg_per90() sadece set-parca sutunu
    saymali - ikisi toplaminda TUM (penalti haric) sutlari kapsamali, hicbir
    sut iki kanalda birden sayilmamali ya da hic sayilmadan kaybolmamali.
    Penalti (play_pattern='Regular Play' olsa bile shot_type='Penalty') her
    iki kanaldan da haric tutulmali."""
    events_df = pd.DataFrame({
        "type": ["Shot", "Shot", "Shot", "Shot"],
        "player_id": [1, 1, 1, 1],
        "player": ["A", "A", "A", "A"],
        "shot_type": ["Open Play", "Open Play", "Open Play", "Penalty"],
        "shot_statsbomb_xg": [0.3, 0.2, 0.9, 0.76],
        "play_pattern": ["Regular Play", "From Corner", "From Counter", "Regular Play"],
    })
    minutes_df = pd.DataFrame({
        "player_id": [1],
        "player_name": ["A"],
        "minutes": [900.0],
    })

    open_play = event_tools.compute_npxg_per90(events_df, minutes_df)
    set_piece = event_tools.compute_set_piece_npxg_per90(events_df, minutes_df)

    assert open_play.loc[open_play["player_id"] == 1, "npxg"].iloc[0] == 0.3 + 0.9
    assert set_piece.loc[set_piece["player_id"] == 1, "npxg"].iloc[0] == 0.2


def test_load_multi_league_events_combines_multiple_competitions(monkeypatch):
    """Iki sahte 'lig' (competition_id, season_id cifti) icin sb.matches() ve
    get_events_cached/get_lineups_cached sahtelenir; load_multi_league_events()
    her ikisinin mac verisini tek bir events_df/lineups_df'te dogru birlestirmeli
    (hicbir mac kaybolmamali, hicbir mac tekrarlanmamali)."""
    fake_matches = {
        (1, 100): pd.DataFrame({"match_id": [11, 12], "competition_name": ["Lig A", "Lig A"]}),
        (2, 200): pd.DataFrame({"match_id": [21], "competition_name": ["Lig B"]}),
    }
    fake_events = {
        11: pd.DataFrame({"match_id": [11], "type": ["Shot"]}),
        12: pd.DataFrame({"match_id": [12], "type": ["Pass"]}),
        21: pd.DataFrame({"match_id": [21], "type": ["Shot"]}),
    }
    fake_lineups = {
        11: pd.DataFrame({"match_id": [11], "player_id": [1]}),
        12: pd.DataFrame({"match_id": [12], "player_id": [2]}),
        21: pd.DataFrame({"match_id": [21], "player_id": [3]}),
    }

    class FakeSb:
        @staticmethod
        def matches(competition_id, season_id):
            return fake_matches[(competition_id, season_id)]

    monkeypatch.setattr(event_tools, "sb", FakeSb)
    monkeypatch.setattr(event_tools, "get_events_cached", lambda match_id, cache_dir=None: fake_events[match_id])
    monkeypatch.setattr(event_tools, "get_lineups_cached", lambda match_id, cache_dir=None: fake_lineups[match_id])

    events_df, lineups_df = event_tools.load_multi_league_events([(1, 100), (2, 200)])

    assert sorted(events_df["match_id"].tolist()) == [11, 12, 21]
    assert sorted(lineups_df["match_id"].tolist()) == [11, 12, 21]
    assert len(events_df) == 3
    assert len(lineups_df) == 3

    events_league = events_df.set_index("match_id")["league"]
    assert events_league[11] == "Lig A"
    assert events_league[12] == "Lig A"
    assert events_league[21] == "Lig B"
    lineups_league = lineups_df.set_index("match_id")["league"]
    assert lineups_league[11] == "Lig A"
    assert lineups_league[21] == "Lig B"


def test_get_player_leagues_picks_most_played_league():
    """Bir oyuncu (player_id=1) Lig A'da 2 mac, Lig B'de 1 mac oynamissa
    (sezon ici transfer/loan senaryosu), en sik oynadigi Lig A donmeli. Tek
    ligde oynayan bir oyuncu (player_id=2) o ligle esleşmeli."""
    lineups_df = pd.DataFrame({
        "player_id": [1, 1, 1, 2],
        "match_id": [101, 102, 201, 301],
        "league": ["Lig A", "Lig A", "Lig B", "Lig B"],
    })

    leagues = event_tools.get_player_leagues(lineups_df)

    assert leagues == {1: "Lig A", 2: "Lig B"}


def test_get_primary_position_group_uses_minutes_weighted_majority():
    """Bir oyuncu mac icinde kisa sureli 'Left Wing' (Attack) sonra uzun
    sureli 'Right Back' (Defender) oynamissa, dakika agirlikli birincil
    pozisyonu Defender olmali (en sik gorulen etiket degil, en cok dakika)."""
    lineups_df = pd.DataFrame({
        "match_id": [1, 1],
        "player_id": [10, 10],
        "position": ["Left Wing", "Right Back"],
        "from": ["0:00", "20:00"],
        "to": ["20:00", None],
        "from_period": [1, 1],
        "to_period": [1, None],
    })
    events_df = pd.DataFrame({
        "match_id": [1],
        "type": ["Half End"],
        "period": [1],
        "minute": [45],
        "second": [0],
    })

    result = event_tools.get_primary_position_group(lineups_df, events_df)

    assert result.loc[result["player_id"] == 10, "position_group"].iloc[0] == "Defender"
