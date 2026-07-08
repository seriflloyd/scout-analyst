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


def _add_segment_minutes(lineups_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """lineups_df'teki her pozisyon segmentine gercek suresini (segment_minutes)
    ekler. Yontem: statsbombpy'de aggregated player-match istatistikleri
    (player_match_stats) acik veri icin kimlik dogrulama gerektirdiginden
    kullanilamaz; bunun yerine StatsBomb'un resmi olarak onerdigi yontem
    izlenir - 'positions' alanindaki her segmentin bitis zamani (to) doluysa
    dogrudan kullanilir; bos ise (oyuncu o periyodun sonuna kadar sahadadir)
    periyodun gercek bitis zamani events_df'teki 'Half End' event'inin
    minute:second degerinden (uzatma dakikalari dahil) alinir. Boylece sabit
    90/45 dakika varsayilmadan, kirmizi kart ve uzatma sureleri dogru
    yansitilir."""
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
    return d


def compute_player_minutes(lineups_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Birden fazla macin lineup pozisyon segmentlerinden (get_lineups_cached
    cikisi, 'match_id' sutunuyla concat edilmis) oyuncu basina toplam oynanan
    dakikayi hesaplar (bkz. _add_segment_minutes icin yontem aciklamasi)."""
    d = _add_segment_minutes(lineups_df, events_df)
    return (
        d.groupby(["player_id", "player_name"], as_index=False)["segment_minutes"]
        .sum()
        .rename(columns={"segment_minutes": "minutes"})
    )


_POSITION_GROUPS = {
    "Goalkeeper": "Goalkeeper",
    "Center Back": "Defender", "Left Back": "Defender", "Left Center Back": "Defender",
    "Left Wing Back": "Defender", "Right Back": "Defender", "Right Center Back": "Defender",
    "Right Wing Back": "Defender",
    "Center Attacking Midfield": "Midfield", "Center Defensive Midfield": "Midfield",
    "Center Midfield": "Midfield", "Left Center Midfield": "Midfield",
    "Left Defensive Midfield": "Midfield", "Left Midfield": "Midfield",
    "Right Center Midfield": "Midfield", "Right Defensive Midfield": "Midfield",
    "Right Midfield": "Midfield",
    "Center Forward": "Attack", "Left Center Forward": "Attack", "Left Wing": "Attack",
    "Right Center Forward": "Attack", "Right Wing": "Attack",
}


def get_primary_position_group(lineups_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Her oyuncu icin, StatsBomb'un ayrintili pozisyon etiketlerini (orn.
    'Right Center Back', 'Left Wing') dort genis gruba (Goalkeeper, Defender,
    Midfield, Attack) indirger ve en cok dakika oynadigi grubu 'birincil
    pozisyon' olarak atar (bir oyuncu sezon icinde farkli pozisyonlarda/
    mevkilerde oynamis olabilir - basit en-sik-gorulen etiket yerine dakika
    agirlikli secim yapilir, boylece kisa sureli tactical shift'ler baskin
    cikmaz).

    Donen tablo: player_id, position_group."""
    d = _add_segment_minutes(lineups_df, events_df)
    d["position_group"] = d["position"].map(_POSITION_GROUPS)
    d = d.dropna(subset=["position_group"])
    by_group = d.groupby(["player_id", "position_group"], as_index=False)["segment_minutes"].sum()
    idx = by_group.groupby("player_id")["segment_minutes"].idxmax()
    return by_group.loc[idx, ["player_id", "position_group"]].reset_index(drop=True)


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


def compute_goals_per90(events_df: pd.DataFrame, minutes_df: pd.DataFrame) -> pd.DataFrame:
    """Oyuncu basina gercek gol/90 (penaltiler dahil - tum atilan goller,
    90 dakikaya normalize). npxg_per90'in aksine penaltiler haric tutulmaz,
    cunku bu ham gerceklesen gol sayisidir (goals_per90 sutunuyla ayni
    tanim: data_tools.aggregate_player_season'daki 'goals' de tum golleri
    sayar).

    events_df: birden fazla macin ham event verisi (get_events_cached ile
    indirilip concat edilmis); 'type', 'player_id', 'player', 'shot_outcome'
    sutunlarini icermelidir.
    minutes_df: compute_player_minutes() ciktisi - player_id basina toplam
    oynanan dakika (ayni mac kumesi uzerinden).
    """
    goal_shots = events_df[(events_df["type"] == "Shot") & (events_df["shot_outcome"] == "Goal")]
    goals = (
        goal_shots.groupby(["player_id", "player"], as_index=False)
        .size()
        .rename(columns={"size": "goals", "player": "player_name"})
    )
    out = goals.merge(minutes_df[["player_id", "minutes"]], on="player_id", how="left")
    out["goals_per90"] = out["goals"] / out["minutes"] * 90
    return out.sort_values("goals_per90", ascending=False).reset_index(drop=True)


# StatsBomb'un 120x80 saha koordinat sisteminde rakip kale merkezi. Her takimin
# kendi event'leri kendi hucum yonune gore normalize edildiginden (iki takim da
# kendi sutlarini x~120 civarinda yapar), rakip kale her pas icin sabit kabul
# edilebilir.
_GOAL_CENTER = (120.0, 40.0)

# StatsBomb'un resmi tanimi "set parca haric" der (bkz. compute_progressive_passes_per90
# docstring'i); dead-ball yeniden baslatmalari haric tutulur.
_SET_PIECE_PASS_TYPES = {"Free Kick", "Corner", "Throw-in", "Kick Off", "Goal Kick"}


def _distance_to_goal(loc) -> float:
    x, y = loc[0], loc[1]
    return ((_GOAL_CENTER[0] - x) ** 2 + (_GOAL_CENTER[1] - y) ** 2) ** 0.5


def compute_progressive_passes_per90(events_df: pd.DataFrame, minutes_df: pd.DataFrame) -> pd.DataFrame:
    """Oyuncu basina progresif pas/90 (TUM pozisyonlar - kaleciler dahil).

    Progresif pas tanimi: StatsBomb'un kendi resmi tanimi kullanilir (bkz.
    blogarchive.statsbomb.com, "The Art of Progression: An Analysis of
    Passing vs. Ball Carrying") - basarili (set-parca haric) bir pas, topu
    rakip kale merkezine olan KALAN mesafenin en az %25'i kadar
    yaklastiriyorsa progresiftir. Bu tanim su gerekcelerle secildi:
    (1) StatsBomb'un kendi resmi/yayinlanmis tanimidir - kullandigimiz veri
    saglayicisiyla dogrudan tutarli; (2) tek, sabit bir yuzde esigi kullanir -
    FBref/Opta'nin "son 6 pasin en uzak noktasindan >=10 yarda" gibi
    possession-sequence baglami gerektiren tanimindan farkli olarak, tek bir
    event satirindan (baslangic+bitis konumu) dogrudan hesaplanabilir, ek
    state/sequence takibi gerektirmez; (3) bolgeye gore degisen (orn.
    30/15/10 yarda) heuristiklerden farkli olarak tek ve tutarli bir kural
    olup uygulamada belirsizlik yaratmaz.

    Hesap: her pasin location -> pass_end_location arasindaki rakip-kale-
    merkezine-uzaklik azalisi, baslangic uzakligina bolunur; oran >= 0.25 ise
    progresif sayilir. Baslangic uzakligi 0 olan (kale cizgisinde baslayan)
    paslar orantisiz bolme nedeniyle progresif SAYILMAZ.

    events_df: 'type','player_id','player','location','pass_end_location',
    'pass_outcome','pass_type' sutunlarini icermelidir.
    minutes_df: compute_player_minutes() ciktisi - player_id, player_name,
    minutes (TUM oynayan oyuncular, sadece pas atanlar degil).

    TUM pozisyonlari kapsar: cikti minutes_df'teki HERKESI icerir (kaleciler
    dahil), progresif pasi olmayanlar 0 progressive_passes_per90 alir - sadece
    en az bir progresif pas atmis oyunculara indirgenmez (npxg/goals_per90'in
    aksine, orada sadece sut/gol atanlar goruluyordu).
    """
    passes = events_df[
        (events_df["type"] == "Pass")
        & (events_df["pass_outcome"].isna())
        & (~events_df["pass_type"].isin(_SET_PIECE_PASS_TYPES))
    ].dropna(subset=["location", "pass_end_location"]).copy()

    start_dist = passes["location"].map(_distance_to_goal)
    end_dist = passes["pass_end_location"].map(_distance_to_goal)
    reduction = start_dist - end_dist
    passes["is_progressive"] = (start_dist > 0) & (reduction >= 0.25 * start_dist)

    prog_counts = (
        passes[passes["is_progressive"]]
        .groupby(["player_id", "player"], as_index=False)
        .size()
        .rename(columns={"size": "progressive_passes", "player": "player_name"})
    )

    out = minutes_df[["player_id", "player_name", "minutes"]].merge(
        prog_counts[["player_id", "progressive_passes"]], on="player_id", how="left"
    )
    out["progressive_passes"] = out["progressive_passes"].fillna(0)
    out["progressive_passes_per90"] = out["progressive_passes"] / out["minutes"] * 90
    return out.sort_values("progressive_passes_per90", ascending=False).reset_index(drop=True)
