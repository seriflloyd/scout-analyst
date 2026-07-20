"""Scout ajanı: eşleşen çok-lig havuzunda değerinin altındaki oyuncuları
araçlarla bulur, yapılandırılmış (JSON) aday listesi döner."""
import json
import re

import anthropic
import pandas as pd

from src import config
from src.tools import scout_tools

DEFAULT_POOL_PATH = "data/processed/multi_league_1516_matched.parquet"

SYSTEM = (
    "Sen bir futbol scoutusun. Elindeki araclarla data/processed/"
    "multi_league_1516_matched.parquet'teki oyunculardan degerinin altinda "
    "olanlari bul. Hesap yapma; sadece arac ciktilari uzerinden konus. Karar "
    "sirasi: once percentile_by_group ile performansi bagla, sonra "
    "value_residuals ile deger artigini hesapla, en negatif 5-10 adayi sec, "
    "ilgi cekici olanlar icin similar_players ile benzer oyuncu bul. Turkce, "
    "kisa ve net yaz."
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
        "description": "Aktif havuz uzerinde deger artigi (value_residual) regresyonunu calistirir - "
                        "piyasa degerinin performans/yas/pozisyon/lige gore beklenenden ne kadar "
                        "dusuk/yuksek oldugunu olcer, aktif havuzu (value_residual'a gore artan sirali) "
                        "gunceller. N, R-squared ve en negatif (degerinin en altinda) 10 satiri dondurur.",
        "input_schema": {
            "type": "object",
            "properties": {
                "perf_col": {"type": "string", "description": "Performans sutunu (varsayilan 'npxg_per90')"},
            },
            "required": [],
        },
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
        perf_col = args.get("perf_col", "npxg_per90")
        try:
            df, model = scout_tools.value_residuals(state["df"], perf_col=perf_col)
        except Exception as e:
            return {"hata": str(e)}
        state["df"] = df
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


def run_scout(state: dict, question: str) -> list:
    """Explorer'daki run_explorer ile ayni tool-use dongusunu (MAX_TOOL_CALLS
    sinirli) kullanir, ama son (arac cagirmayan) LLM yanitini serbest metin
    olarak degil, yapilandirilmis bir JSON aday listesi olarak parse edip
    doner - ilk kullanici mesajina bu bicim talimati eklenir, boylece LLM'in
    dogal 'artik arac cagirmiyorum' yanit turu zaten JSON olur.

    state: bu cagriya ozel bos/gecici bir sozluk (load_matched_pool burada
    aktif DataFrame'i 'df' anahtarina yazar) - agent'lar arasi paylasilmaz.

    MAX_TOOL_CALLS turunde hala arac cagirmaya devam ediyorsa (nadiren), araclari
    kapatip (tools olmadan) bir son tur ile zorla JSON yanit istenir - boylece
    fonksiyon her zaman bir liste (bos da olabilir) doner, asla None/exception
    firlatmaz."""
    client = anthropic.Anthropic()
    prompt = (
        f"{question}\n\n"
        "Arac cagirmayi bitirdiginde SADECE (baska hicbir aciklama/metin eklemeden) "
        "secilen adaylari, her biri en az player_name, position, value_residual, "
        "npxg_per90, market_value_in_eur alanlarini iceren bir JSON LISTESI olarak yaz."
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
            return _parse_candidates_json(_final_text(resp))

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
    return _parse_candidates_json(_final_text(resp))
