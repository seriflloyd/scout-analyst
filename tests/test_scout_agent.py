"""Scout ajanının API çağrısı GEREKTİRMEYEN kısımlarının testleri: arac
dispatcher'ı (_run_tool) ve JSON aday listesi parse mantığı (_parse_candidates_json).
Gercek Anthropic API çağrısı (run_scout) icin bkz. scripts/manual_scout_integration.py
(pytest bunu ÇALIŞTIRMAZ)."""
import itertools

import numpy as np
import pandas as pd

from src.agents import scout


def _synthetic_pool(n=30, seed=1):
    rng = np.random.default_rng(seed)
    npxg_per90 = rng.uniform(0, 1.0, n)
    age = rng.uniform(18, 34, n)
    minutes = rng.integers(900, 3000, n)
    position = list(itertools.islice(itertools.cycle(["Attack", "Midfield", "Defender"]), n))
    league = list(itertools.islice(itertools.cycle(["GB1", "ES1", "IT1", "FR1"]), n))
    birth_year = (2016 - age.astype(int))
    date_of_birth = [f"{y}-06-15" for y in birth_year]

    true_log_value = (
        14.0 + 1.2 * npxg_per90 - 0.004 * age ** 2 + 0.3 * np.log(minutes)
        + rng.normal(0, 0.05, n)
    )
    market_value = np.exp(true_log_value)

    return pd.DataFrame({
        "player_id": range(n),
        "player_name": [f"Oyuncu {i}" for i in range(n)],
        "season_end": pd.Timestamp("2016-06-30"),
        "date_of_birth": date_of_birth,
        "position": position,
        "league": league,
        "minutes": minutes,
        "npxg_per90": npxg_per90,
        "market_value_in_eur": market_value,
    })


def test_run_tool_load_matched_pool_reads_parquet_into_state(monkeypatch):
    fake_df = _synthetic_pool(n=5)
    monkeypatch.setattr(scout.pd, "read_parquet", lambda path: fake_df)

    state = {}
    out = scout._run_tool("load_matched_pool", {}, state)

    assert state["df"] is fake_df
    assert out["satir_sayisi"] == 5
    assert "npxg_per90" in out["sutunlar"]


def test_run_tool_requires_load_matched_pool_first():
    state = {}  # 'df' hic yuklenmedi
    out = scout._run_tool("percentile_by_group", {"metric": "npxg_per90"}, state)

    assert "hata" in out


def test_run_tool_percentile_by_group_adds_pct_column():
    state = {"df": _synthetic_pool(n=20)}

    out = scout._run_tool("percentile_by_group", {"metric": "npxg_per90"}, state)

    assert "npxg_per90_pct" in state["df"].columns
    assert len(out["en_yuksek_5"]) == 5
    # en yuksek percentile 100'e (grup icindeki en iyi) yakin olmali
    assert max(r["npxg_per90_pct"] for r in out["en_yuksek_5"]) == 100.0


def test_run_tool_run_value_residuals_returns_n_and_top10():
    state = {"df": _synthetic_pool(n=30)}

    out = scout._run_tool("run_value_residuals", {}, state)

    assert out["N"] == 30
    assert 0.0 <= out["r_squared"] <= 1.0
    assert len(out["en_negatif_10"]) == 10
    assert "value_residual" in state["df"].columns
    # deger artigina gore artan sirali olmali (en negatif ilk satirda)
    assert state["df"]["value_residual"].is_monotonic_increasing


def test_run_tool_get_similar_players_returns_top_n():
    df = _synthetic_pool(n=20)
    df = scout.scout_tools.percentile_by_group(df, "npxg_per90", "position")
    state = {"df": df}

    out = scout._run_tool(
        "get_similar_players",
        {"player_id": 0, "feature_cols": ["npxg_per90_pct"], "top_n": 3},
        state,
    )

    assert len(out["benzer_oyuncular"]) == 3


def test_run_tool_unknown_tool_returns_error():
    state = {"df": _synthetic_pool(n=5)}
    out = scout._run_tool("bilinmeyen_arac", {}, state)
    assert "hata" in out


def test_parse_candidates_json_extracts_list_from_surrounding_text():
    text = 'Iste adaylar:\n[{"player_name": "A", "value_residual": -1.2}]\nTesekkurler.'
    result = scout._parse_candidates_json(text)
    assert result == [{"player_name": "A", "value_residual": -1.2}]


def test_parse_candidates_json_returns_empty_list_on_malformed_input():
    assert scout._parse_candidates_json("bu bir liste degil") == []
    assert scout._parse_candidates_json("[{malformed json") == []
