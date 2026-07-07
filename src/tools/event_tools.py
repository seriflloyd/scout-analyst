"""StatsBomb event verisi araclari. LLM bu dosyadaki hicbir hesabi yapmaz."""
import time
from pathlib import Path

import pandas as pd
import requests
from statsbombpy import sb

_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 5


def _with_retry(fetch_fn):
    """raw.githubusercontent.com 429 (Too Many Requests) gibi gecici hatalarda
    exponential backoff ile yeniden dener."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fetch_fn()
        except requests.exceptions.HTTPError as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(f"  indirme hatasi ({e}), {wait} sn sonra tekrar denenecek "
                  f"({attempt + 1}/{_MAX_RETRIES})", flush=True)
            time.sleep(wait)


def get_events_cached(match_id: int, cache_dir: str = "data/raw/statsbomb/events") -> pd.DataFrame:
    """Bir macin event verisini yerel parquet cache'inden okur; yoksa
    sb.events() ile indirir ve cache'ler (429 gibi gecici hatalarda backoff'la
    yeniden dener)."""
    cache_path = Path(cache_dir) / f"{match_id}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    events = _with_retry(lambda: sb.events(match_id=match_id))
    events.to_parquet(cache_path, index=False)
    return events


def get_lineups_cached(match_id: int, cache_dir: str = "data/raw/statsbomb/lineups") -> pd.DataFrame:
    """Bir macin lineup verisini yerel parquet cache'inden okur; yoksa
    sb.lineups() ile indirir (429 gibi gecici hatalarda backoff'la yeniden
    dener), her oyuncunun pozisyon/sure segmentlerini (positions listesi)
    duzlestirip cache'ler.

    Donen tablo: match_id, team, player_id, player_name, position, from, to,
    from_period, to_period. 'to' bos (None) ise oyuncu o segmentin ait oldugu
    periyodun sonuna (veya mac sonuna) kadar sahadadir."""
    cache_path = Path(cache_dir) / f"{match_id}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    lineups = _with_retry(lambda: sb.lineups(match_id=match_id))
    rows = []
    for team, team_df in lineups.items():
        for _, player in team_df.iterrows():
            for pos in player["positions"]:
                rows.append({
                    "match_id": match_id,
                    "team": team,
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "position": pos["position"],
                    "from": pos["from"],
                    "to": pos["to"],
                    "from_period": pos["from_period"],
                    "to_period": pos["to_period"],
                })
    flat = pd.DataFrame(rows)
    flat.to_parquet(cache_path, index=False)
    return flat


def _parse_clock_to_seconds(clock: str) -> int:
    """Mac saatini ('MM:SS' veya 'HH:MM:SS') toplam saniyeye cevirir."""
    parts = [int(p) for p in clock.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts
    return hours * 3600 + minutes * 60 + seconds


def compute_player_minutes(lineups_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Birden fazla macin lineup pozisyon segmentlerinden (get_lineups_cached
    cikisi, 'match_id' sutunuyla concat edilmis) oyuncu basina toplam oynanan
    dakikayi hesaplar.

    Yontem: statsbombpy'de aggregated player-match istatistikleri (player_match_stats)
    acik veri icin kimlik dogrulama gerektirdiginden kullanilamaz; bunun yerine
    StatsBomb'un resmi olarak onerdigi yontem izlenir - lineups()'un 'positions'
    alanindaki her segmentin bitis zamani (to) doluysa dogrudan kullanilir; bos ise
    (oyuncu o periyodun sonuna kadar sahadadir) periyodun gercek bitis zamani
    events_df'teki 'Half End' event'inin minute:second degerinden (uzatma dakikalari
    dahil) alinir. Boylece sabit 90/45 dakika varsayilmadan, kirmizi kart ve
    uzatma sureleri dogru yansitilir.
    """
    period_end = (
        events_df[events_df["type"] == "Half End"]
        .groupby(["match_id", "period"])
        .apply(lambda g: int((g["minute"] * 60 + g["second"]).max()))
        .to_dict()
    )

    d = lineups_df.copy()
    d["from_sec"] = d["from"].map(_parse_clock_to_seconds)

    def _to_seconds(row):
        if pd.notna(row["to"]):
            return _parse_clock_to_seconds(row["to"])
        return period_end.get((row["match_id"], row["from_period"]), row["from_sec"])

    d["to_sec"] = d.apply(_to_seconds, axis=1)
    d["segment_minutes"] = (d["to_sec"] - d["from_sec"]) / 60

    return (
        d.groupby(["player_id", "player_name"], as_index=False)["segment_minutes"]
        .sum()
        .rename(columns={"segment_minutes": "minutes"})
    )


def compute_npxg_per90(events_df: pd.DataFrame, minutes_df: pd.DataFrame) -> pd.DataFrame:
    """Oyuncu basina npxG/90 (penalti haric beklenen gol, 90 dakikaya normalize).

    events_df: birden fazla macin ham event verisi (get_events_cached ile
    indirilip concat edilmis); 'type', 'player_id', 'player', 'shot_type',
    'shot_statsbomb_xg' sutunlarini icermelidir.
    minutes_df: compute_player_minutes() ciktisi - player_id basina toplam
    oynanan dakika (ayni mac kumesi uzerinden).

    Penaltilar (shot_type == "Penalty") npxG toplamina dahil edilmez, cunku
    penalti xG'si oyuncunun acik oyun performansini degil sabit bir durumu
    yansitir.
    """
    shots = events_df[(events_df["type"] == "Shot") & (events_df["shot_type"] != "Penalty")]
    npxg = (
        shots.groupby(["player_id", "player"], as_index=False)["shot_statsbomb_xg"]
        .sum()
        .rename(columns={"shot_statsbomb_xg": "npxg", "player": "player_name"})
    )
    out = npxg.merge(minutes_df[["player_id", "minutes"]], on="player_id", how="left")
    out["npxg_per90"] = out["npxg"] / out["minutes"] * 90
    return out.sort_values("npxg_per90", ascending=False).reset_index(drop=True)
