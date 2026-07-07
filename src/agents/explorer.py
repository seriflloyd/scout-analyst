"""Kaşif ajanı: veri özetini araçlarla toplar, LLM yorumlar."""
import json
import anthropic
from src import config
from src.tools import data_tools

SYSTEM = (
    "Sen bir futbol verisi kaşifisin. Elindeki araçlarla veri setini tanı, "
    "kalite sorunlarını (eksik veri, uç değer) raporla. Hesap yapma; "
    "yalnızca araç çıktıları üzerinden konuş. Türkçe, kısa ve net yaz."
)

TOOLS = [
    {
        "name": "summarize_columns",
        "description": "Aktif veri çerçevesindeki sütunların tip/eksik/örnek özetini döndürür.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "aggregate_and_per90",
        "description": "Veriyi oyuncu-sezon düzeyine toplar, gol/asist per-90 ekler, dakika eşiği uygular. Kaç oyuncu kaldığını ve ilk 5 satırı döndürür.",
        "input_schema": {
            "type": "object",
            "properties": {"min_minutes": {"type": "integer", "description": "Dakika eşiği"}},
            "required": [],
        },
    },
]


def _run_tool(name: str, args: dict, state: dict) -> dict:
    if name == "summarize_columns":
        return data_tools.summarize_columns(state["raw"])
    if name == "aggregate_and_per90":
        mm = args.get("min_minutes", config.MIN_MINUTES)
        agg = data_tools.aggregate_player_season(state["raw"])
        agg = data_tools.add_per90(agg)
        agg = data_tools.apply_minutes_threshold(agg, mm)
        state["prepared"] = agg
        return {"kalan_oyuncu_sezon": int(len(agg)),
                "ilk5": agg.head().to_dict(orient="records")}
    return {"hata": f"bilinmeyen arac: {name}"}


def run_explorer(state: dict, question: str) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

    for _ in range(config.MAX_TOOL_CALLS):
        resp = client.messages.create(
            model=config.MODEL_EXPLORER,
            max_tokens=config.MAX_TOKENS,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            return next(b.text for b in resp.content if b.type == "text")

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

    return "Arac cagrisi limitine ulasildi."
