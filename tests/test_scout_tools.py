import itertools

import numpy as np
import pandas as pd

from src.tools import scout_tools


def test_value_residuals_v2_sign():
    """Bilinen bir sentetik iliskiyle uretilen veride, kasitli olarak
    trendin cok altinda fiyatlanan bir oyuncunun en negatif value_residual'a,
    kasitli olarak cok ustunde fiyatlanan bir oyuncunun ise en pozitif
    value_residual'a sahip oldugunu dogrular."""
    rng = np.random.default_rng(42)
    n = 40

    goals_per90 = rng.uniform(0, 1.2, n)
    age = rng.uniform(18, 34, n)
    minutes = rng.integers(900, 3000, n)
    position = list(itertools.islice(itertools.cycle(["Attack", "Midfield"]), n))
    league = list(itertools.islice(itertools.cycle(["GB1", "ES1"]), n))
    birth_year = (2024 - age.astype(int))
    date_of_birth = [f"{y}-06-15" for y in birth_year]

    true_log_value = (
        14.0
        + 1.2 * goals_per90
        - 0.004 * age ** 2
        + 0.3 * np.log(minutes)
        + rng.normal(0, 0.05, n)
    )
    market_value = np.exp(true_log_value)

    df = pd.DataFrame({
        "player_id": range(n),
        "player_name": [f"Player {i}" for i in range(n)],
        "season": 2024,
        "season_end": pd.Timestamp("2024-12-31"),
        "date_of_birth": date_of_birth,
        "position": position,
        "league": league,
        "minutes": minutes,
        "goals_per90": goals_per90,
        "market_value_in_eur": market_value,
        "contract_years_remaining": rng.uniform(0, 5, n),
    })

    undervalued_idx = 0
    df.loc[undervalued_idx, "market_value_in_eur"] = market_value[undervalued_idx] / 50

    overvalued_idx = 1
    df.loc[overvalued_idx, "market_value_in_eur"] = market_value[overvalued_idx] * 50

    result, model = scout_tools.value_residuals(df)

    assert model.rsquared > 0
    assert result.iloc[0]["player_id"] == undervalued_idx
    assert result.iloc[0]["value_residual"] < 0
    assert result.iloc[-1]["player_id"] == overvalued_idx
    assert result.iloc[-1]["value_residual"] > 0


def test_value_residuals_low_signal_flag():
    """perf_col (goals_per90) tam 0 olan satirlarda dusuk_sinyal_guvenilirligi
    True, digerlerinde False olmali."""
    rng = np.random.default_rng(7)
    n = 30

    goals_per90 = rng.uniform(0.05, 1.2, n)
    zero_idx = [0, 5, 10]
    goals_per90[zero_idx] = 0.0

    age = rng.uniform(18, 34, n)
    minutes = rng.integers(900, 3000, n)
    position = list(itertools.islice(itertools.cycle(["Attack", "Midfield"]), n))
    league = list(itertools.islice(itertools.cycle(["GB1", "ES1"]), n))
    birth_year = (2024 - age.astype(int))
    date_of_birth = [f"{y}-06-15" for y in birth_year]

    true_log_value = (
        14.0
        + 1.2 * goals_per90
        - 0.004 * age ** 2
        + 0.3 * np.log(minutes)
        + rng.normal(0, 0.05, n)
    )
    market_value = np.exp(true_log_value)

    df = pd.DataFrame({
        "player_id": range(n),
        "season": 2024,
        "season_end": pd.Timestamp("2024-12-31"),
        "date_of_birth": date_of_birth,
        "position": position,
        "league": league,
        "minutes": minutes,
        "goals_per90": goals_per90,
        "market_value_in_eur": market_value,
        "contract_years_remaining": rng.uniform(0, 5, n),
    })

    result, _ = scout_tools.value_residuals(df)

    flagged = result[result["player_id"].isin(zero_idx)]
    not_flagged = result[~result["player_id"].isin(zero_idx)]

    assert flagged["dusuk_sinyal_guvenilirligi"].all()
    assert not not_flagged["dusuk_sinyal_guvenilirligi"].any()


def test_value_residuals_without_contract_column_falls_back_to_v2():
    """contract_years_remaining sutunu df'te HIC yoksa (orn. match_tools.py
    fuzzy-match ciktisi gibi sezon-indeksli olmayan tarihsel veri) value_residuals()
    hata firlatmamali, kovaryati sessizce atlayip eski (contract'siz) 7-parametreli
    modele donmeli - cunku bu durumda contract_years_remaining hicbir sekilde
    kavramsal olarak hesaplanamaz (bkz. reports/contract_years_remaining_zamanlama_sorunu.md)."""
    rng = np.random.default_rng(3)
    n = 30

    goals_per90 = rng.uniform(0, 1.2, n)
    age = rng.uniform(18, 34, n)
    minutes = rng.integers(900, 3000, n)
    position = list(itertools.islice(itertools.cycle(["Attack", "Midfield"]), n))
    league = list(itertools.islice(itertools.cycle(["GB1", "ES1"]), n))
    birth_year = (2024 - age.astype(int))
    date_of_birth = [f"{y}-06-15" for y in birth_year]

    true_log_value = (
        14.0 + 1.2 * goals_per90 - 0.004 * age ** 2 + 0.3 * np.log(minutes)
        + rng.normal(0, 0.05, n)
    )
    market_value = np.exp(true_log_value)

    df = pd.DataFrame({
        "player_id": range(n),
        "season": 2024,
        "season_end": pd.Timestamp("2024-12-31"),
        "date_of_birth": date_of_birth,
        "position": position,
        "league": league,
        "minutes": minutes,
        "goals_per90": goals_per90,
        "market_value_in_eur": market_value,
    })
    assert "contract_years_remaining" not in df.columns

    result, model = scout_tools.value_residuals(df)

    assert len(result) == n
    assert "contract_years_remaining" not in model.params.index
