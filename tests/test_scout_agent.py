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


def test_run_tool_run_value_residuals_stores_backfill_source():
    """_backfill_from_value_residuals()'un okuyacagi state anahtarlari
    (value_residuals_df, value_residuals_perf_col) run_value_residuals
    calistiginda dogru sekilde set edilmeli."""
    state = {"df": _synthetic_pool(n=30)}

    scout._run_tool("run_value_residuals", {}, state)

    assert "value_residuals_df" in state
    assert "value_residual" in state["value_residuals_df"].columns
    assert state["value_residuals_perf_col"] == "npxg_per90"


def test_backfill_overrides_wrong_llm_number_with_real_tool_output():
    """Kritik senaryo: LLM'in JSON'unda dogru isimle birlikte YANLIS bir sayi
    (orn. percentile'i ham deger sanip yazmasi) gelirse, geri-doldurma bu
    sayiyi YOK SAYMALI ve gercek DataFrame'deki dogru degeri kullanmalidir -
    LLM'in yazdigi hatali sayi sonuca asla sizmamali."""
    df = _synthetic_pool(n=15)
    vr_df, _ = scout.scout_tools.value_residuals(df, perf_col="npxg_per90", value_col="market_value_in_eur")

    target = vr_df.iloc[0]
    real_npxg = float(target["npxg_per90"])
    real_residual = float(target["value_residual"])
    real_value = float(target["market_value_in_eur"])

    # LLM'in (yanlis) JSON'u: dogru isim, ama uydurma/yanlis sayilar (orn.
    # bir onceki gercek calistirmada gozlemlenen percentile-karistirma hatasi)
    llm_candidates = [{
        "player_name": target["player_name"],
        "position": target["position"],
        "gerekce": "iyi performans, dusuk deger",
        "npxg_per90": 0.31,               # YANLIS - gercek degerle eslesmiyor
        "value_residual": -999.0,          # YANLIS
        "market_value_in_eur": 1,          # YANLIS
    }]

    result = scout._backfill_from_value_residuals(llm_candidates, vr_df, "npxg_per90")

    assert len(result) == 1
    entry = result[0]
    assert entry["npxg_per90"] == real_npxg
    assert entry["value_residual"] == real_residual
    assert entry["market_value_in_eur"] == real_value
    # LLM'in yazdigi hatali sayilar sonuca hic yansimamali
    assert entry["npxg_per90"] != 0.31
    assert entry["value_residual"] != -999.0


def test_backfill_drops_candidate_with_no_reliable_name_match():
    """LLM'in yazdigi isim, arac ciktisindaki HICBIR oyuncuya
    NAME_MATCH_SCORE_THRESHOLD uzerinde eslesmezse, o aday (sessizce degil)
    dusurulmelidir - dogrulanamayan bir aday sonuca sizmamali."""
    df = _synthetic_pool(n=10)
    vr_df, _ = scout.scout_tools.value_residuals(df, perf_col="npxg_per90", value_col="market_value_in_eur")

    llm_candidates = [{"player_name": "Tamamen Alakasiz Bir Isim Zzzqx", "position": "Attack", "gerekce": "?"}]

    result = scout._backfill_from_value_residuals(llm_candidates, vr_df, "npxg_per90")

    assert result == []


def test_backfill_returns_empty_when_value_residuals_never_ran():
    """run_value_residuals hic cagrilmadiysa (value_residuals_df=None), hicbir
    adayin sayisal alani dogrulanamaz - tum adaylar dusurulmeli."""
    llm_candidates = [{"player_name": "Herhangi Biri", "position": "Attack", "gerekce": "?"}]

    result = scout._backfill_from_value_residuals(llm_candidates, None, "npxg_per90")

    assert result == []


def test_backfill_tolerates_slightly_misspelled_names():
    """LLM ismi hafifce yanlis/eksik yazmis olsa bile (orn. aksan/bosluk farki),
    rapidfuzz toleransli eslestirme dogru oyuncuyu bulmali."""
    df = _synthetic_pool(n=10)
    vr_df, _ = scout.scout_tools.value_residuals(df, perf_col="npxg_per90", value_col="market_value_in_eur")
    target = vr_df.iloc[0]

    misspelled = target["player_name"].replace("Oyuncu", "oyuncu ")  # kucuk harf + fazla bosluk
    llm_candidates = [{"player_name": misspelled, "position": target["position"], "gerekce": "?"}]

    result = scout._backfill_from_value_residuals(llm_candidates, vr_df, "npxg_per90")

    assert len(result) == 1
    assert result[0]["player_name"] == target["player_name"]
