"""StatsBomb <-> Transfermarkt oyuncu eslestirme araclari. LLM bu dosyadaki hicbir hesabi yapmaz."""
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process


def normalize_name(name: str) -> str:
    """Aksanlari kaldirir, kucuk harfe cevirir, fazla bosluklari temizler."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_name.lower().split())


def build_transfermarkt_candidates(players: pd.DataFrame, valuations: pd.DataFrame,
                                   competition_code: str, season_start: str, season_end: str,
                                   min_market_value: float) -> pd.DataFrame:
    """Verilen lig kodu ve sezon penceresinde en az bir piyasa degeri kaydi olan
    oyunculardan aday havuzu olusturur (players.csv'nin guncel/genel
    current_club_domestic_competition_id alani yerine, valuations'daki
    player_club_domestic_competition_id + tarih penceresi kullanilir - boylece
    sadece o sezon gercekten o ligde olan oyuncular alinir).

    Her oyuncu icin pencere ici en son (sezon sonuna en yakin, sonrasi degil -
    look-ahead bias'siz) piyasa degeri secilir; MIN_MARKET_VALUE altindaki
    taban-deger artefaktlari elenir.
    """
    v = valuations.copy()
    v["date"] = pd.to_datetime(v["date"])
    v = v[
        (v["player_club_domestic_competition_id"] == competition_code)
        & (v["date"] >= pd.Timestamp(season_start))
        & (v["date"] <= pd.Timestamp(season_end))
    ]
    v = v.sort_values("date")
    latest = v.groupby("player_id", as_index=False).last()
    latest = latest[latest["market_value_in_eur"] >= min_market_value]

    candidates = latest.merge(
        players[["player_id", "name", "date_of_birth", "position", "sub_position"]],
        on="player_id", how="left",
    )
    return candidates[["player_id", "name", "date_of_birth", "position", "sub_position", "market_value_in_eur"]]


def fuzzy_match_players(source_names: list, candidates: pd.DataFrame, name_col: str = "name",
                        score_threshold: float = 90.0, tie_margin: float = 1.0) -> pd.DataFrame:
    """Her source_names elemani (orn. StatsBomb'un tam/dogum adi) icin
    candidates[name_col] (orn. Transfermarkt'in bilinen adi) icinde rapidfuzz
    token_set_ratio ile en iyi eslesmeyi bulur - bu skor, kisa bilinen adin
    (orn. 'Luis Suarez') uzun tam adin (orn. 'Luis Alberto Suarez Diaz') alt
    kumesi olup olmadigina bakar, bu yuzden isim uzunluk farkina karsi
    dayaniklidir. Skor score_threshold altindaysa eslesme reddedilir.

    Ayrica en iyi iki aday birbirine tie_margin puan icinde ise (orn. 'Sergio
    Ramos Garcia' kaynagi, aday havuzunda hem 'Sergio Ramos' hem 'Sergio
    Garcia' varsa ikisine de skor=100 ile 'eslesir') eslesme belirsiz sayilir
    ve reddedilir - token_set_ratio'nun ust-skor yakinsamasi hangisinin dogru
    kisi oldugunu ayirt edemez, process.extractOne'in rastgele/sirali sectigi
    taraf sessizce yanlis olabilir.

    Donen tablo: source_name, player_id, matched_name, score. Reddedilen
    (esik alti veya belirsiz) satirlarda player_id None olur; matched_name/score
    yine de en iyi adayi ve skorunu tasir - tanisal amacli (raporlamada 'neye
    yakin ama esik alti/belirsiz kaldi' gorunsun diye)."""
    choices = candidates[name_col].tolist()
    rows = []
    for src in source_names:
        top_matches = process.extract(src, choices, scorer=fuzz.token_set_ratio, limit=2)
        if not top_matches:
            rows.append({"source_name": src, "player_id": None, "matched_name": None, "score": 0.0})
            continue
        matched_name, score, idx = top_matches[0]
        candidate_player_id = candidates.iloc[idx]["player_id"]
        is_tied = len(top_matches) > 1 and (score - top_matches[1][1]) < tie_margin
        accepted = score >= score_threshold and not is_tied
        rows.append({
            "source_name": src,
            "player_id": candidate_player_id if accepted else None,
            "matched_name": matched_name,
            "score": score,
        })
    return pd.DataFrame(rows)


def demote_ambiguous_matches(match_result: pd.DataFrame) -> pd.DataFrame:
    """Ayni Transfermarkt player_id'sine birden fazla farkli source_name'in
    eslestigi durumlari 'belirsiz' sayip player_id'yi None'a ceker.

    Bu, token_set_ratio'nun bilinen bir zaafini duzeltir: Transfermarkt'ta
    tek kelimelik/kisa bir isim (orn. 'Pedro', 'Roberto', 'Danilo') varsa,
    o tek kelimeyi iceren HER uzun StatsBomb tam adi (farkli kisiler olsalar
    bile) skor=100 ile eslesir. Iki farkli gercek oyuncu ayni tek adaya
    eslestiginde hangisinin (varsa) dogru oldugu skor uzerinden guvenilir
    sekilde ayirt edilemez, bu yuzden ikisi de reddedilip 'eslesemedi'
    olarak raporlanir - yanlis ama emin gorunen bir eslemeyi sessizce kabul
    etmektense boyle davranmak tercih edilir."""
    d = match_result.copy()
    counts = d.loc[d["player_id"].notna(), "player_id"].value_counts()
    ambiguous_ids = counts[counts > 1].index
    d.loc[d["player_id"].isin(ambiguous_ids), "player_id"] = None
    return d


def build_statsbomb_value_pool(npxg_df: pd.DataFrame, players: pd.DataFrame, valuations: pd.DataFrame,
                               competition_code: str, season_start: str, season_end: str,
                               min_market_value: float, score_threshold: float = 90.0):
    """StatsBomb npxG/90 sonuclarini (npxg_df: player_name, npxg, minutes, npxg_per90)
    Transfermarkt oyuncu + piyasa degeri verisiyle isim tabanli fuzzy match ile
    birlestirir.

    Donus: (matched_df, unmatched_df).
    matched_df: value_residuals() icin gerekli sutunlari icerir (npxg_per90,
    minutes, position, date_of_birth, league, season_end, market_value_in_eur).
    unmatched_df: eslesemeyen StatsBomb oyuncularini (en iyi adayi ve skoruyla,
    tanisal amacli) listeler - eslesmeme nedeni uc turlu olabilir: (1) adayin
    skoru esigin altinda kaldi, (2) o oyuncu aday havuzunda (o sezon o ligde
    kayitli piyasa degeri) hic yok, (3) demote_ambiguous_matches() tarafindan
    belirsiz bulundu (ayni Transfermarkt kaydina birden fazla farkli StatsBomb
    oyuncusu eslesti - hangisinin dogru oldugu skor uzerinden ayirt edilemedi).
    """
    candidates = build_transfermarkt_candidates(
        players, valuations, competition_code, season_start, season_end, min_market_value
    )

    # npxg_df'in kendi 'player_id' sutunu StatsBomb ID'si; Transfermarkt player_id'siyle
    # karismamasi icin ayri isimlendirilir.
    source = npxg_df.rename(columns={"player_id": "statsbomb_player_id"})

    match_result = fuzzy_match_players(
        source["player_name"].tolist(), candidates, score_threshold=score_threshold
    )
    match_result = demote_ambiguous_matches(match_result)
    merged = source.merge(match_result, left_on="player_name", right_on="source_name", how="left")

    matched = merged[merged["player_id"].notna()].copy()
    unmatched = merged[merged["player_id"].isna()][["player_name", "matched_name", "score"]].reset_index(drop=True)

    matched = matched.merge(
        candidates[["player_id", "date_of_birth", "position", "market_value_in_eur"]],
        on="player_id", how="left",
    )
    matched["league"] = competition_code
    matched["season_end"] = pd.Timestamp(season_end)
    matched = matched.reset_index(drop=True)

    return matched, unmatched
