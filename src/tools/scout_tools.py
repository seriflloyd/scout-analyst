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


def value_residuals(df: pd.DataFrame, perf_col: str = "goals_per90",
                    value_col: str = "market_value_in_eur"):
    """v2: log(piyasa degeri) ~ perf_col + age + age^2 + pozisyon kukla
    degiskenleri + lig kukla degiskenleri + log(dakika) (statsmodels OLS).

    age, date_of_birth ile o oyuncu-sezonun bitis tarihi (season_end sutunu
    varsa o kullanilir, yoksa season'dan 31 Aralik olarak turetilir) arasindaki
    farktan yil cinsinden hesaplanir (look-ahead bias'siz, sezon ici performansla
    ayni doneme ait yas).

    NEGATIF value_residual = oyuncunun piyasa degeri, modelin performans ve
    yas/lig/pozisyona gore ongordugunden DUSUK -> "degerinin altinda"
    (undervalued) oyuncu adayi. POZITIF artik ise performansina gore piyasada
    ASIRI DEGERLI oyuncu demektir.

    dusuk_sinyal_guvenilirligi: performans metrigi (perf_col) 0 olan
    oyuncularda deger artigi buyuk olcude yas/dakika/lig tarafindan
    aciklanir, gercek performans sinyali degil - bu sutun Elestirmen
    ajaninin bu farki ayirt etmesini saglar.

    Donus: (value_residual'a gore artan sirali DataFrame, sm.OLS fit sonucu).
    Ikinci eleman tam bir statsmodels RegressionResultsWrapper oldugundan
    model.rsquared, model.params ve model.summary() dogrudan erisilebilir.

    perf_col secimi: goals_per90 piyasa-referans modeli icin, npxg_per90 (veya
    benzer sansa-gore-duzeltilmis metrikler) asil scouting sinyali icin tercih
    edilir - dusuk R-squared burada zayiflik degil, aranan sinyalin isaretidir.

    contract_years_remaining kovaryati (df'te sutun VARSA): contract_years_remaining
    dusukse piyasa degeri performans-disi bir nedenle (yaklasan bonservissiz
    transfer riski) dusuk olabilir - bu ayrimi yapmadan value_residual'i tek
    basina "degerinin altinda" diye yorumlamak yanlis olur. contract_years_remaining
    eksik olan satirlar (build_eligible_pool()'da sozlesme bitis tarihi
    bilinmeyen/tutarsiz oyuncular) dropna ile regresyondan otomatik dusecek;
    bu yuzden dusen satir sayisi stdout'a raporlanir.

    contract_years_remaining df'te sutun olarak HIC yoksa (orn. build_eligible_pool()
    disinda, match_tools.py fuzzy-match ciktisi gibi tarihsel/sezon-indeksli
    olmayan veri kaynaklarinda - bkz. reports/contract_years_remaining_zamanlama_sorunu.md)
    kovaryat sessizce ATLANIR, model eski (contract'siz) 7-parametreli haline
    doner - cunku bu durumda contract_years_remaining kavramsal olarak
    hesaplanamaz (players.csv sezona degil oyuncuya tek bir guncel sozlesme
    tarihi bagliyor), zorla eklemek look-ahead bias yaratir.
    """
    required = [perf_col, "date_of_birth", "position", "league", "minutes", value_col]
    has_contract = "contract_years_remaining" in df.columns
    base_valid_count = len(df.dropna(subset=required))

    required_full = required + ["contract_years_remaining"] if has_contract else required
    d = df.dropna(subset=required_full).copy()
    if has_contract:
        dropped_for_contract = base_valid_count - len(d)
        print(f"value_residuals: contract_years_remaining eksikligi nedeniyle "
              f"{dropped_for_contract} satir dustu ({base_valid_count} -> {len(d)})")

    d = d[(d[value_col] > 0) & (d["minutes"] > 0)]

    if "season_end" in d.columns:
        season_end = pd.to_datetime(d["season_end"])
    else:
        season_end = pd.to_datetime(d["season"].astype(int).astype(str) + "-12-31")

    d["age"] = (season_end - pd.to_datetime(d["date_of_birth"])).dt.days / 365.25
    d["age_sq"] = d["age"] ** 2
    d["log_minutes"] = np.log(d["minutes"])

    perf_cols = [perf_col, "age", "age_sq", "log_minutes"]
    if has_contract:
        perf_cols.append("contract_years_remaining")

    dummies = pd.get_dummies(d[["position", "league"]], drop_first=True, dtype=float)
    X = pd.concat([d[perf_cols], dummies], axis=1)
    X = sm.add_constant(X)
    y = np.log(d[value_col])

    model = sm.OLS(y, X).fit()
    d["value_residual"] = model.resid
    d["dusuk_sinyal_guvenilirligi"] = d[perf_col] == 0
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
