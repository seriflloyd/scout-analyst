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
