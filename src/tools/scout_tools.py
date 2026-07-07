"""Scout hesap katmani: percentile, shrinkage, deger artigi, benzerlik."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics.pairwise import cosine_similarity


def percentile_by_group(df: pd.DataFrame, metric: str, group: str = "position") -> pd.DataFrame:
    """Metrigi grup (pozisyon) icinde 0-100 percentile'a cevirir."""
    out = df.copy()
    out[f"{metric}_pct"] = out.groupby(group)[metric].rank(pct=True) * 100
    return out


def shrink_to_group_mean(df: pd.DataFrame, metric: str, k: int = 900,
                         group: str = "position") -> pd.DataFrame:
    """Empirik Bayes cekmesi: az dakikali oyuncuyu grup ortalamasina yaklastirir."""
    out = df.copy()
    grp_mean = out.groupby(group)[metric].transform("mean")
    w = out["minutes"] / (out["minutes"] + k)
    out[f"{metric}_shrunk"] = w * out[metric] + (1 - w) * grp_mean
    return out


def value_residuals(df: pd.DataFrame, perf_col: str,
                    value_col: str = "market_value"):
    """log(piyasa degeri) ~ performans regresyonu.
    NEGATIF artik = piyasa degeri, performansin ongordugunden DUSUK = degerinin altinda oyuncu."""
    d = df.dropna(subset=[perf_col, value_col]).copy()
    d = d[d[value_col] > 0]
    X = sm.add_constant(d[[perf_col]])
    y = np.log(d[value_col])
    model = sm.OLS(y, X).fit()
    d["value_residual"] = model.resid
    return d.sort_values("value_residual"), model


def similar_players(df: pd.DataFrame, player_id: int,
                    feature_cols: list, top_n: int = 5) -> pd.DataFrame:
    """Percentile profillerine kosinus benzerligiyle en yakin oyunculari bulur."""
    d = df.dropna(subset=feature_cols).reset_index(drop=True)
    M = d[feature_cols].to_numpy()
    idx = d.index[d["player_id"] == player_id][0]
    sims = cosine_similarity(M[idx:idx + 1], M)[0]
    d = d.assign(similarity=sims).sort_values("similarity", ascending=False)
    return d[d["player_id"] != player_id].head(top_n)
