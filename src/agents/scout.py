"""Scout ajanı: eşleşen çok-lig havuzunda değerinin altındaki oyuncuları
araçlarla bulur, yapılandırılmış (JSON) aday listesi döner."""
import json
import re

import anthropic
import pandas as pd
from rapidfuzz import fuzz, process

from src import config
from src.tools import scout_tools

DEFAULT_POOL_PATH = "data/processed/multi_league_1516_matched.parquet"

# Backfill sirasinda LLM'in yazdigi isimle arac ciktisindaki isim arasinda
# kabul edilecek en dusuk rapidfuzz skoru (asagida NAME_MATCH_SCORE_THRESHOLD
# olarak kullanilir).
NAME_MATCH_SCORE_THRESHOLD = 85.0

# _backfill_from_value_residuals()'da LLM'in JSON'undan degil, run_value_residuals
# aracinin son DataFrame'inden geri doldurulan sutunlar - bkz. run_scout docstring'i.
# 'npxg_per90' HER ZAMAN geri doldurulur - run_value_residuals artik LLM'e perf_col
# secme sansi vermez (TOOLS semasinda yok, config.DEFAULT_PERF_COL'a sabitlenmistir),
# bu yuzden entry[perf_col] atamasi zaten hep 'npxg_per90'in kendisidir; yine de
# ayri sabit bir kolon olarak tutulur ki config.DEFAULT_PERF_COL ileride degisse
# bile ham npxg_per90 referansi kaybolmasin.
_BACKFILL_COLUMNS = ["npxg_per90", "value_residual", "market_value_in_eur", "league", "dusuk_sinyal_guvenilirligi"]

SYSTEM = (
    "Sen bir futbol scoutusun. Elindeki araclarla data/processed/"
    "multi_league_1516_matched.parquet'teki oyunculardan degerinin altinda "
    "olanlari bul. Hesap yapma; sadece arac ciktilari uzerinden konus. Karar "
    "sirasi: once (istersen) percentile_by_group ile performansi grup-ici "
    "baglama otur (bu SADECE teshis/rapor amaclidir - 'bu oyuncu pozisyonunda "
    "kacinci yuzdelikte' gibi bir baglam sunar, Yazar/Elestirmen'e gosterilecek "
    "ek bilgidir; value_residuals'in GIRDISI DEGILDIR ve olamaz - value_residuals "
    "her zaman ham performans metrigiyle, sabit ve degistirilemez sekilde "
    "calisir), sonra value_residuals ile deger artigini hesapla, en negatif "
    "5-10 adayi sec, ilgi cekici olanlar icin similar_players ile benzer "
    "oyuncu bul. Turkce, kisa ve net yaz.\n\n"
    "ONEMLI: Nihai JSON yanitinda HICBIR sayisal deger (npxg_per90, "
    "value_residual, market_value_in_eur, percentile vb.) YAZMA - bu degerleri "
    "hafizandan/gorduklerinden aktarirsan yanlis hatirlayabilirsin (orn. "
    "percentile'i ham degerle karistirmak gibi). Sayisal alanlar sistem "
    "tarafindan arac ciktisindan otomatik geri doldurulacak; senin gorevin "
    "sadece HANGI oyuncularin secildigini (isim+pozisyon+kisa gerekce) net "
    "sekilde belirtmek."
)

TOOLS = [
    {
        "name": "load_matched_pool",
        "description": "data/processed/multi_league_1516_matched.parquet dosyasini yukler, "
                        "oyuncu havuzunu aktif hale getirir. Analiz baslamadan once ilk arac "
                        "cagrisi bu olmalidir. Satir sayisi ve sutun listesini dondurur.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "percentile_by_group",
        "description": "Aktif havuzdaki bir performans metrigini (orn. 'npxg_per90') grup "
                        "(varsayilan 'position') icinde 0-100 percentile'a cevirir; '<metric>_pct' "
                        "sutunu ekler ve aktif havuzu gunceller.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "Percentile'a cevrilecek sutun, orn. 'npxg_per90'"},
                "group": {"type": "string", "description": "Gruplama sutunu (varsayilan 'position')"},
            },
            "required": ["metric"],
        },
    },
    {
        "name": "run_value_residuals",
        "description": "Aktif havuz uzerinde deger artigi (value_residual) regresyonunu, "
                        "Faz B-C'de dogrulanan SABIT modelle (ham npxg_per90 performans sutunu - "
                        "config.DEFAULT_PERF_COL, secilemez) calistirir - piyasa degerinin "
                        "performans/yas/pozisyon/lige gore beklenenden ne kadar dusuk/yuksek "
                        "oldugunu olcer, aktif havuzu (value_residual'a gore artan sirali) gunceller. "
                        "N, R-squared ve en negatif (degerinin en altinda) 10 satiri dondurur. "
                        "Parametre almaz.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_similar_players",
        "description": "Verilen oyuncuya (Transfermarkt player_id) secilen ozellik sutunlarindaki "
                        "(feature_cols) kosinus benzerligiyle en cok benzeyen top_n oyunculari bulur.",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {"type": "integer", "description": "Transfermarkt player_id"},
                "feature_cols": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Benzerlik icin kullanilacak sutunlar, orn. ['npxg_per90_pct']",
                },
                "top_n": {"type": "integer", "description": "Kac benzer oyuncu donsun (varsayilan 5)"},
            },
            "required": ["player_id", "feature_cols"],
        },
    },
]


def _run_tool(name: str, args: dict, state: dict) -> dict:
    if name == "load_matched_pool":
        df = pd.read_parquet(state.get("pool_path", DEFAULT_POOL_PATH))
        state["df"] = df
        return {"satir_sayisi": int(len(df)), "sutunlar": df.columns.tolist()}

    if "df" not in state:
        return {"hata": "once load_matched_pool cagirilmali"}

    if name == "percentile_by_group":
        metric = args["metric"]
        group = args.get("group", "position")
        try:
            df = scout_tools.percentile_by_group(state["df"], metric, group)
        except Exception as e:
            return {"hata": str(e)}
        state["df"] = df
        pct_col = f"{metric}_pct"
        cols = [c for c in ["player_name", group, pct_col] if c in df.columns]
        return {
            "satir_sayisi": int(len(df)),
            "en_yuksek_5": df.sort_values(pct_col, ascending=False).head(5)[cols].to_dict(orient="records"),
        }

    if name == "run_value_residuals":
        # perf_col LLM'e birakilmaz (TOOLS semasinda yok) - args'ta yanlislikla/kotu
        # niyetle bir 'perf_col' gelse bile YOKSAYILIR, her zaman Faz B-C'de
        # dogrulanan config.DEFAULT_PERF_COL kullanilir (bkz. config.py notu).
        perf_col = config.DEFAULT_PERF_COL
        try:
            df, model = scout_tools.value_residuals(state["df"], perf_col=perf_col)
        except Exception as e:
            return {"hata": str(e)}
        state["df"] = df
        # LLM'in JSON'una GUVENILMEYECEK sayisal alanlarin geri-doldurulacagi
        # kaynak - run_scout()'un _backfill_from_value_residuals() adimi bunu
        # kullanir (bkz. dosya basindaki not).
        state["value_residuals_df"] = df
        state["value_residuals_perf_col"] = perf_col
        cols = [c for c in ["player_name", "position", "league", "value_residual", perf_col,
                             "market_value_in_eur", "player_id"] if c in df.columns]
        return {
            "N": int(model.nobs),
            "r_squared": round(float(model.rsquared), 4),
            "en_negatif_10": df.head(10)[cols].to_dict(orient="records"),
        }

    if name == "get_similar_players":
        try:
            result = scout_tools.similar_players(
                state["df"], player_id=args["player_id"],
                feature_cols=args["feature_cols"], top_n=args.get("top_n", 5),
            )
        except Exception as e:
            return {"hata": str(e)}
        cols = [c for c in ["player_name", "position", "league", "similarity"] if c in result.columns]
        return {"benzer_oyuncular": result[cols].to_dict(orient="records")}

    return {"hata": f"bilinmeyen arac: {name}"}


def _final_text(resp) -> str:
    texts = [b.text for b in resp.content if b.type == "text"]
    return texts[-1] if texts else ""


def _parse_candidates_json(text: str) -> list:
    """LLM'in son (arac cagirmayan) yanitindan JSON aday listesini cikarir -
    modelin talimata ragmen ekstra aciklama metni eklemesi ihtimaline karsi
    metin icindeki ilk '[...]' bloguna regex ile bakar. Parse basarisiz olursa
    (bos/bicimsiz yanit) bos liste doner - graf'in candidates alani hep bir
    liste olarak kalir, None/hata firlatmaz."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []


def _backfill_from_value_residuals(candidates: list, value_residuals_df, perf_col: str) -> list:
    """LLM'in JSON'undaki player_name/position/gerekce dismindaki HICBIR
    sayisal/kategorik alana GUVENILMEZ - bir onceki calistirmada LLM'in
    percentile'i ham deger sanip yazdigi gozlemlendi (bkz. Faz D duzeltme
    commit'i). Bunun yerine her adayin _BACKFILL_COLUMNS + perf_col
    degerleri, run_value_residuals aracinin EN SON DONDURDUGU DataFrame'den
    (value_residuals_df) player_name uzerinden rapidfuzz ile (LLM ismi hafif
    yanlis/eksik yazmis olabilir - orn. aksan/soyisim kisaltmasi) toleransli
    eslestirilerek GERI DOLDURULUR.

    value_residuals_df hic yoksa (run_value_residuals hic cagrilmadiysa) VEYA
    bir aday NAME_MATCH_SCORE_THRESHOLD uzerinde guvenilir bir isme
    eslesmezse, o aday SESSIZCE degil LOGLANARAK dusurulur - boylece
    dogrulanamayan bir sayi asla LLM'in yazdigi haliyle sizmaz.

    Donus: her biri player_name (arac ciktisindaki DOGRU/TAM isim), position,
    gerekce ve _BACKFILL_COLUMNS + perf_col alanlarini iceren aday listesi."""
    if value_residuals_df is None or value_residuals_df.empty:
        print(f"run_scout: value_residuals_df yok - {len(candidates)} aday sayisal dogrulama "
              f"olmadan kabul edilemez, tumu dusuruldu")
        return []

    names = value_residuals_df["player_name"].tolist()
    backfilled = []
    for c in candidates:
        query = (c or {}).get("player_name")
        if not query:
            print(f"run_scout: aday player_name icermiyor, dusuruldu: {c}")
            continue

        match = process.extractOne(query, names, scorer=fuzz.token_set_ratio)
        if match is None or match[1] < NAME_MATCH_SCORE_THRESHOLD:
            best = f"'{match[0]}' (skor={match[1]:.1f})" if match else "aday yok"
            print(f"run_scout: '{query}' icin guvenilir isim eslesmesi bulunamadi "
                  f"(esik={NAME_MATCH_SCORE_THRESHOLD}, en iyi: {best}) - aday dusuruldu")
            continue

        matched_name, score, idx = match
        row = value_residuals_df.iloc[idx]
        entry = {
            "player_name": row["player_name"],
            "position": c.get("position", row.get("position")),
            "gerekce": c.get("gerekce", ""),
        }
        entry[perf_col] = row[perf_col] if perf_col in row.index else None
        for col in _BACKFILL_COLUMNS:
            entry[col] = row[col] if col in row.index else None
        backfilled.append(entry)

    return backfilled


def _finish(resp, state: dict) -> list:
    """LLM'in son (arac cagirmayan) yanitini JSON aday listesine parse eder,
    ardindan HER sayisal/kategorik alani (LLM'in JSON'unda ASLA bulunmamasi
    gereken degerler - bkz. SYSTEM ve run_scout prompt'u) tool ciktisindan
    geri doldurur. LLM'in JSON'u istenmeden sayi icerse bile bu sayilar
    _backfill_from_value_residuals icinde ATILIP tool ciktisiyle degistirilir
    (perf_col/entry[col] atamalari LLM'in olasi degerlerinin UZERINE yazar)."""
    raw_candidates = _parse_candidates_json(_final_text(resp))
    perf_col = state.get("value_residuals_perf_col", config.DEFAULT_PERF_COL)
    return _backfill_from_value_residuals(raw_candidates, state.get("value_residuals_df"), perf_col)


def run_scout(state: dict, question: str) -> list:
    """Explorer'daki run_explorer ile ayni tool-use dongusunu (MAX_TOOL_CALLS
    sinirli) kullanir, ama son (arac cagirmayan) LLM yanitini serbest metin
    olarak degil, yapilandirilmis bir JSON aday listesi olarak parse edip
    doner - ilk kullanici mesajina bu bicim talimati eklenir, boylece LLM'in
    dogal 'artik arac cagirmiyorum' yanit turu zaten JSON olur.

    KRITIK: LLM'in JSON'u SADECE player_name, position ve gerekce (kisa metin
    aciklama) icermelidir - hicbir sayisal/kategorik alan (npxg_per90,
    value_residual, market_value_in_eur, league, percentile vb.) istenmez,
    cunku LLM bunlari yanlis aktarabilir (ampirik olarak gozlemlendi: bir
    onceki calistirmada LLM percentile degerini ham deger sanip yazmisti).
    Bu alanlar _finish() icinde run_value_residuals aracinin EN SON
    DONDURDUGU DataFrame'den player_name eslestirmesiyle GERI DOLDURULUR
    (bkz. _backfill_from_value_residuals).

    state: bu cagriya ozel bos/gecici bir sozluk (load_matched_pool burada
    aktif DataFrame'i 'df' anahtarina, run_value_residuals ise backfill
    kaynagini 'value_residuals_df'e yazar) - agent'lar arasi paylasilmaz.

    MAX_TOOL_CALLS turunde hala arac cagirmaya devam ediyorsa (nadiren), araclari
    kapatip (tools olmadan) bir son tur ile zorla JSON yanit istenir - boylece
    fonksiyon her zaman bir liste (bos da olabilir) doner, asla None/exception
    firlatmaz."""
    client = anthropic.Anthropic()
    prompt = (
        f"{question}\n\n"
        "Arac cagirmayi bitirdiginde SADECE (baska hicbir aciklama/metin eklemeden) "
        "secilen adaylari bir JSON LISTESI olarak yaz. HER adayda SADECE su alanlar "
        "olsun: player_name (arac ciktisindaki TAM/DOGRU isim), position, gerekce "
        "(neden secildigine dair kisa bir metin). BASKA HICBIR ALAN (sayisal veya "
        "degil) YAZMA - ozellikle npxg_per90, value_residual, market_value_in_eur, "
        "league, percentile gibi degerleri YAZMA; bunlar sistem tarafindan otomatik "
        "doldurulacak."
    )
    messages = [{"role": "user", "content": prompt}]

    for _ in range(config.MAX_TOOL_CALLS):
        resp = client.messages.create(
            model=config.MODEL_SCOUT,
            max_tokens=config.MAX_TOKENS,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            return _finish(resp, state)

        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                out = _run_tool(block.name, block.input, state)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out, ensure_ascii=False, default=str),
                })
        messages.append({"role": "user", "content": results})

    # MAX_TOOL_CALLS'a ulasildi, hala arac cagiriyor - araclari kapatip zorla nihai JSON iste
    resp = client.messages.create(
        model=config.MODEL_SCOUT,
        max_tokens=config.MAX_TOKENS,
        system=SYSTEM,
        messages=messages,
    )
    return _finish(resp, state)
