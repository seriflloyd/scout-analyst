"""Veri yükleme ve hazırlama araçları. LLM bu dosyadaki hiçbir hesabı yapmaz."""
from pathlib import Path

import pandas as pd

from src import config

# Top-5 lig: (competitions.csv icindeki name, country_name) ciftleri.
# "bundesliga" adi hem Almanya (L1) hem Avusturya (A1) icin kullanildigindan
# ulke adiyla eslestirmek gerekir; kod tahmin edilmez, competitions.csv'den bulunur.
_TOP5_LEAGUES = pd.DataFrame({
    "name": ["premier-league", "laliga", "bundesliga", "serie-a", "ligue-1"],
    "country_name": ["England", "Spain", "Germany", "Italy", "France"],
})


def load_appearances(path: str = "data/raw/appearances.csv") -> pd.DataFrame:
    """Maç bazlı oyuncu kayıtlarını yükler."""
    return pd.read_csv(path)


def load_valuations(path: str = "data/raw/player_valuations.csv") -> pd.DataFrame:
    """Oyuncu piyasa değeri geçmişini yükler."""
    return pd.read_csv(path)


def aggregate_player_season(appearances: pd.DataFrame) -> pd.DataFrame:
    """Oyuncu-sezon düzeyine toplar: toplam dakika, gol, asist, maç sayısı."""
    df = appearances.copy()
    df["season"] = pd.to_datetime(df["date"]).dt.year
    agg = (
        df.groupby(["player_id", "player_name", "season"], as_index=False)
          .agg(minutes=("minutes_played", "sum"),
               goals=("goals", "sum"),
               assists=("assists", "sum"),
               matches=("game_id", "nunique"))
    )
    return agg


def add_per90(df: pd.DataFrame, cols: tuple = ("goals", "assists")) -> pd.DataFrame:
    """Seçilen metrikleri 90 dakikaya normalize eder."""
    out = df.copy()
    for c in cols:
        out[f"{c}_per90"] = out[c] / out["minutes"] * 90
    return out


def apply_minutes_threshold(df: pd.DataFrame, min_minutes: int = 900) -> pd.DataFrame:
    """Küçük örneklem gürültüsünü elemek için dakika eşiği uygular."""
    return df[df["minutes"] >= min_minutes].reset_index(drop=True)


def get_top5_competition_ids(competitions: pd.DataFrame) -> list:
    """competitions.csv icinden top-5 lig (Premier League, LaLiga, Bundesliga,
    Serie A, Ligue 1) competition_id kodlarini bulur. Ayni isimli farkli
    ulke ligleri (orn. Almanya/Avusturya bundesligasi) ulke adiyla ayrilir.
    """
    top5 = competitions.merge(_TOP5_LEAGUES, on=["name", "country_name"])
    top5 = top5[top5["type"] == "domestic_league"]
    return top5["competition_id"].tolist()


def build_eligible_pool(
    appearances_path: str = "data/raw/appearances.csv",
    players_path: str = "data/raw/players.csv",
    valuations_path: str = "data/raw/player_valuations.csv",
    competitions_path: str = "data/raw/competitions.csv",
    n_seasons: int = 3,
    min_minutes: int = config.MIN_MINUTES,
    output_path: str = "data/processed/eligible_pool.parquet",
) -> pd.DataFrame:
    """Scouting icin uygun oyuncu-sezon havuzunu olusturur.

    Adimlar: top-5 lige ait mac katilimlarini alip oyuncu-sezon bazinda
    toplar; her oyuncu-sezona en sik oynadigi competition_id'yi ("league")
    ekler; her oyuncu-sezona, o sezonun bitisine (31 Aralik) kadar bilinen
    en guncel player_valuations kaydini (merge_asof, direction="backward")
    esler - bu sekilde sezon sonrasindaki degerler kullanilmayarak
    look-ahead bias engellenir; taban-deger artefaktlarini (< MIN_MARKET_VALUE)
    ve son n_seasons disindaki sezonlari eler; dakika esigini uygular;
    goals_per90 / assists_per90 sutunlarini ekler; sonucu parquet olarak
    kaydeder.

    Ayrica players.csv'deki contract_expiration_date'ten iki sutun turetir:
    - contract_years_remaining: (contract_expiration_date - season_end) / 365.25.
      Sozlesme bitis tarihi bilinmiyorsa VEYA season_end'den ONCEYSE (veride
      tutarsizlik/guncel olmayan kayit - gercek durum bilinmedigi icin negatif
      bir deger degil, dogrudan NaN birakilir; 0'a da kirpilmaz, cunku bu da
      fabrike bir deger olurdu) NaN olur.
    - is_free_agent_soon: contract_years_remaining <= 0.5 ise True (yarim
      yildan az kalmis, bonservissiz transfer adayi); contract_years_remaining
      NaN ise False (bilinmeyen durum "yakinda bedava" olarak ISARETLENMEZ).

    ONEMLI KISIT: players.csv oyuncu basina TEK (guncel/son bilinen) bir
    contract_expiration_date tutar - sezona ozgu sozlesme gecmisi YOKTUR.
    Bu yuzden contract_years_remaining, SADECE havuzdaki EN GUNCEL sezon
    icin hesaplanir (season == max(kept seasons)); n_seasons penceresindeki
    daha eski sezonlarda NaN/False birakilir. Aksi halde (eski sezonlar icin
    de hesaplansaydi) look-ahead bias olurdu: oyuncunun DAHA SONRA imzaladigi
    bir sozlesmenin bitis tarihi, o oyuncunun cok daha eski bir sezonundaki
    "kalan sure"si gibi kullanilmis olurdu - ampirik olarak dogrulandi (bkz.
    reports/contract_years_remaining_zamanlama_sorunu.md): 2015/16 La Liga
    verisinde bu sekilde hesaplanan degerler 6.9-14 yil araliginda cikiyordu
    (gercekci bir futbol sozlesmesi suresi degil), cunku season_end'den
    onceki degil GELECEKTEKI (2023-2030 arasi) bir contract_expiration_date
    kullaniliyordu.
    """
    appearances = pd.read_csv(appearances_path)
    players = pd.read_csv(players_path)
    valuations = pd.read_csv(valuations_path)
    competitions = pd.read_csv(competitions_path)

    top5_ids = get_top5_competition_ids(competitions)
    appearances = appearances[appearances["competition_id"].isin(top5_ids)].copy()

    player_season = aggregate_player_season(appearances)

    appearances["season"] = pd.to_datetime(appearances["date"]).dt.year
    league_mode = (
        appearances.groupby(["player_id", "season"])["competition_id"]
        .agg(lambda s: s.mode().iloc[0])
        .rename("league")
        .reset_index()
    )
    player_season = player_season.merge(league_mode, on=["player_id", "season"], how="left")

    last_seasons = sorted(player_season["season"].unique())[-n_seasons:]
    player_season = player_season[player_season["season"].isin(last_seasons)].copy()
    player_season["season_end"] = pd.to_datetime(
        player_season["season"].astype(int).astype(str) + "-12-31"
    )

    valuations = valuations.copy()
    valuations["date"] = pd.to_datetime(valuations["date"])
    valuations = valuations[["player_id", "date", "market_value_in_eur"]].sort_values("date")

    player_season = player_season.sort_values("season_end")
    pool = pd.merge_asof(
        player_season,
        valuations,
        left_on="season_end",
        right_on="date",
        by="player_id",
        direction="backward",
    )

    pool = pool.dropna(subset=["market_value_in_eur"])
    pool = pool[pool["market_value_in_eur"] >= config.MIN_MARKET_VALUE]

    pool = apply_minutes_threshold(pool, min_minutes)

    player_meta = players[[
        "player_id", "name", "position", "sub_position",
        "date_of_birth", "country_of_citizenship", "foot",
    ]].copy()
    player_meta["contract_expiration_date"] = (
        players["contract_expiration_date"] if "contract_expiration_date" in players.columns
        else pd.NaT
    )
    pool = pool.merge(player_meta, on="player_id", how="left")
    pool = pool.drop(columns=["date"]).reset_index(drop=True)

    pool = add_per90(pool)

    pool["contract_expiration_date"] = pd.to_datetime(pool["contract_expiration_date"], errors="coerce")
    years_remaining = (pool["contract_expiration_date"] - pool["season_end"]).dt.days / 365.25
    is_latest_season = pool["season"] == max(last_seasons)
    pool["contract_years_remaining"] = years_remaining.where(years_remaining >= 0).where(is_latest_season)
    pool["is_free_agent_soon"] = (pool["contract_years_remaining"] <= 0.5).fillna(False)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(output_file, index=False)

    print(f"eligible_pool: {len(pool)} satir -> {output_file}")

    return pool


def summarize_columns(df: pd.DataFrame) -> dict:
    """LLM'e gönderilecek kompakt veri özeti üretir (asla ham veri gönderme)."""
    return {
        "satir_sayisi": int(len(df)),
        "sutunlar": {
            c: {
                "tip": str(df[c].dtype),
                "eksik_orani": round(float(df[c].isna().mean()), 3),
                "ornek": str(df[c].dropna().iloc[0]) if df[c].notna().any() else None,
            }
            for c in df.columns
        },
    }
