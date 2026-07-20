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

    Donen tablo: match_id, team, player_id, player_name, player_nickname,
    country, position, from, to, from_period, to_period. 'to' bos (None) ise
    oyuncu o segmentin ait oldugu periyodun sonuna (veya mac sonuna) kadar
    sahadadir. player_nickname/country, StatsBomb'un oyuncuya ait bilinen kisa
    ad (orn. 'Fernando Torres') ve uyrugu (match_tools.py'deki isim eslestirme
    icin - full legal isim yerine kisa ad kullanmak ortak-soyisim
    catismalarini azaltir, bkz. match_tools.fuzzy_match_players)."""
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
                    "player_nickname": player["player_nickname"],
                    "country": player["country"],
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


def load_multi_league_events(competitions: list, events_cache_dir: str = "data/raw/statsbomb/events",
                             lineups_cache_dir: str = "data/raw/statsbomb/lineups") -> tuple:
    """Birden fazla lig/sezonun event+lineup verisini tek bir DataFrame ciftinde
    birlestirir - get_events_cached/get_lineups_cached'i (mevcut cache/retry
    davranisiyla, degistirilmeden) her mac icin cagiran ust-duzey bir birlestirici.

    competitions: [(competition_id, season_id), ...] cifti listesi (orn.
    [(2, 27), (12, 27)] -> Premier League 2015/16 + Serie A 2015/16). Her cift
    icin sb.matches() ile mac listesi cekilir; StatsBomb match_id'leri tum
    liglerde global olarak benzersiz oldugundan cakisma/tekrar riski yoktur.

    Her macin tum event ve lineup satirlarina, sb.matches() ciktisindaki
    'competition_name' degeri (orn. 'La Liga', 'Premier League') 'league'
    sutunu olarak eklenir - boylece birlesik veride hangi satirin hangi ligden
    geldigi ayirt edilebilir (bkz. get_player_leagues()).

    Donus: (events_df, lineups_df) - verilen tum lig/sezonlarin mac event ve
    lineup satirlarinin concat edilmis hali (ikisinde de 'league' sutunu var).
    compute_player_minutes, compute_npxg_per90, compute_goals_per90 gibi
    asagi-akis fonksiyonlari tek-lig cagrisindaki gibi degismeden calisir (bu
    fonksiyonlar zaten events_df/lineups_df'in hangi ligden geldigine
    bakmaz - fazladan 'league' sutunu onlari etkilemez)."""
    all_events = []
    all_lineups = []
    for competition_id, season_id in competitions:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
        for _, match_row in matches.iterrows():
            match_id = match_row["match_id"]
            league = match_row["competition_name"]
            all_events.append(get_events_cached(match_id, cache_dir=events_cache_dir).assign(league=league))
            all_lineups.append(get_lineups_cached(match_id, cache_dir=lineups_cache_dir).assign(league=league))

    events_df = pd.concat(all_events, ignore_index=True)
    lineups_df = pd.concat(all_lineups, ignore_index=True)
    return events_df, lineups_df


def get_player_leagues(lineups_df: pd.DataFrame) -> dict:
    """lineups_df (load_multi_league_events ciktisi, 'league' sutunu icermeli)
    icinden oyuncu basina EN COK MAC oynadigi ligi cikarir - bir oyuncu ayni
    sezon icinde (loan/transfer) birden fazla ligde oynamis olsa bile en sik
    gorulen lig alinir (get_primary_position_group'un dakika-agirlikli
    secimine benzer bir 'coklu-lig' kenar durumu koruma yontemi, ama burada
    dakika yerine mac sayisi kullanilir - segment_minutes hesaplamak icin
    events_df gerekir, bu fonksiyon sadece lineups_df alir).

    Donus: {player_id: lig_adi} sozlugu (lig_adi = 'league' sutunundaki deger,
    orn. StatsBomb competition_name) - match_tools.build_multi_league_value_pool()'a
    player_leagues olarak aktarilir."""
    counts = (
        lineups_df.groupby(["player_id", "league"])["match_id"]
        .nunique()
        .reset_index()
    )
    idx = counts.groupby("player_id")["match_id"].idxmax()
    return counts.loc[idx].set_index("player_id")["league"].to_dict()


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


# StatsBomb'un 'play_pattern' alani, sutun ait oldugu POSSESSION'in hangi
# oyun durumundan (restart) baslatildigini belirtir - sutun kendisinin nasil
# vuruldugunu (kafa/ayak) DEGIL. Ampirik dogrulama (Suarez/Messi/Ronaldo,
# 380 mac cache'i, shot_body_part dagilimi): 'From Corner'+'From Free Kick'
# etiketli sutlarin sadece %13.5'i kafa (Head), %86.5'i ayak (Left/Right
# Foot) - yani bu play_pattern'ler cogunlukla "corner/frikik sonrasi acilan
# possession icinde baska turlu bir sut" anlamina gelir, "duran topla kafa
# vurusu" degil (bkz. compute_set_piece_npxg_per90 docstring'i). Yine de acik
# oyun (Regular Play, From Counter, From Keeper) ile set-parca kaynakli
# possession'lari ayirmak anlamlidir - bitiricilik orn. bir kontratak
# aninda mi, yoksa rakibin defansif dizilimi henuz toparlanmamisken bir
# dead-ball sonrasi mi olustu farkli taktiksel baglamlari yansitir.
# 'From Goal Kick', 'From Kick Off' ve 'Other' kasitli olarak disarida
# birakilir (cok nadir gorulur, ne acik oyun ne set-parca possession
# profiline net oturur).
_OPEN_PLAY_SHOT_PATTERNS = frozenset({"Regular Play", "From Counter", "From Keeper"})
_SET_PIECE_SHOT_PATTERNS = frozenset({"From Corner", "From Free Kick", "From Throw In"})


def _npxg_per90_for_patterns(events_df: pd.DataFrame, minutes_df: pd.DataFrame,
                              play_patterns) -> pd.DataFrame:
    """compute_npxg_per90 ve compute_set_piece_npxg_per90 arasinda paylasilan
    ortak hesap: verilen play_pattern kumesine giren (penalti haric) sutlardan
    npxG/90 hesaplar."""
    shots = events_df[
        (events_df["type"] == "Shot")
        & (events_df["shot_type"] != "Penalty")
        & (events_df["play_pattern"].isin(play_patterns))
    ]
    npxg = (
        shots.groupby(["player_id", "player"], as_index=False)["shot_statsbomb_xg"]
        .sum()
        .rename(columns={"shot_statsbomb_xg": "npxg", "player": "player_name"})
    )
    out = npxg.merge(minutes_df[["player_id", "minutes"]], on="player_id", how="left")
    out["npxg_per90"] = out["npxg"] / out["minutes"] * 90
    return out.sort_values("npxg_per90", ascending=False).reset_index(drop=True)


def compute_npxg_per90(events_df: pd.DataFrame, minutes_df: pd.DataFrame,
                        play_pattern_filter=_OPEN_PLAY_SHOT_PATTERNS) -> pd.DataFrame:
    """Oyuncu basina npxG/90 (penalti haric beklenen gol, 90 dakikaya normalize).

    events_df: birden fazla macin ham event verisi (get_events_cached ile
    indirilip concat edilmis); 'type', 'player_id', 'player', 'shot_type',
    'shot_statsbomb_xg', 'play_pattern' sutunlarini icermelidir.
    minutes_df: compute_player_minutes() ciktisi - player_id basina toplam
    oynanan dakika (ayni mac kumesi uzerinden).

    Penaltilar (shot_type == "Penalty") npxG toplamina dahil edilmez, cunku
    penalti xG'si oyuncunun acik oyun performansini degil sabit bir durumu
    yansitir.

    play_pattern_filter: sadece bu kumedeki 'play_pattern' degerlerine sahip
    sutlar sayilir. Varsayilan _OPEN_PLAY_SHOT_PATTERNS ({'Regular Play',
    'From Counter', 'From Keeper'}) - set-parca sutlari (corner/frikik/taca)
    haric tutulur, cunku bunlar acik oyun bitiricilik sinyalini bulanik-
    lastirir (bkz. compute_set_piece_npxg_per90 - set-parca kanali ayri
    raporlanir). Karisik (eski davranis, filtre uygulanmamis) bir sonuc
    isteniyorsa play_pattern_filter=events_df["play_pattern"].unique()
    verilebilir.
    """
    return _npxg_per90_for_patterns(events_df, minutes_df, play_pattern_filter)


def compute_set_piece_npxg_per90(events_df: pd.DataFrame, minutes_df: pd.DataFrame) -> pd.DataFrame:
    """Oyuncu basina SADECE set-parca npxG/90 (penalti haric, corner/frikik/
    taca kaynakli sutlar; 90 dakikaya normalize).

    compute_npxg_per90()'in acik-oyun kanaliyla birlikte ikinci bir kanal
    olarak dusunulmelidir: bir oyuncunun toplam npxg_per90'i buyuk olcude
    set-parca kaynakliysa, bu fonksiyon o bagimliligi ayri gosterir.

    DIKKAT - bu kanal "kafa vurusu/duran top yetenegi" DEGIL, "sutun ait
    oldugu possession'in bir corner/frikik/tacadan baslamis olmasi"
    anlamina gelir (bkz. _SET_PIECE_SHOT_PATTERNS yorumu). Ampirik olarak
    (Suarez/Messi/Ronaldo, 380 mac) bu kanaldaki sutlarin %86.5'i ayakla
    atilmis, sadece %13.5'i kafa - ozellikle 'From Free Kick' sutlarinin
    cogu muhtemelen dogrudan serbest vurus (elit frikik atan oyuncularda)
    veya possession devam ederken atilan acik-oyun-tarzi bir sut, klasik
    "kafa vurusu"ndan cok farkli bir beceri profili. Yorumlarken "bu
    oyuncunun npxG'si set-parca possession'lara ne kadar bagimli" sorusuna
    cevap verir, "bu oyuncu ne kadar iyi kafa vurur" sorusuna degil.

    events_df / minutes_df: compute_npxg_per90() ile ayni sekil ve sutunlar.
    """
    return _npxg_per90_for_patterns(events_df, minutes_df, _SET_PIECE_SHOT_PATTERNS)


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
