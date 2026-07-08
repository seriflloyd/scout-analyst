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
