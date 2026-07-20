"""Faz D LangGraph iskeleti: Kaşif -> Scout -> Eleştirmen -> Yazar akışının
state tanımı ve yönlendirme mantığı.

Kaşif (explorer) ve Scout gerçek LLM çağrısı yapar (run_explorer, run_scout);
Eleştirmen/Yazar node'ları henüz STUB'tır (deterministik, LLM'siz) - graf
kablolaması ve revizyon döngüsü testlerde run_explorer/run_scout monkeypatch'iyle
gerçek API çağrısı olmadan doğrulanır (bkz. tests/test_graph.py).
"""
from typing import TypedDict

from langgraph.graph import StateGraph, END

from src import config
from src.tools import data_tools
from src.agents.explorer import run_explorer
from src.agents.scout import run_scout


class ScoutState(TypedDict):
    """Ajan döngüsü boyunca taşınan paylaşılan durum."""
    question: str
    data_note: str          # Kaşif çıktısı (veri kalitesi/özet notu)
    candidates: list         # Scout çıktısı (aday oyuncu dict'leri)
    critic_verdict: str      # Eleştirmen kararı: "ONAYLA" | "REDDET" | ""
    critic_notes: str        # Eleştirmen gerekçesi
    revision_count: int      # Kaç kez revizyona dönüldü
    report: str              # Yazar çıktısı (nihai rapor)


def explorer_node(state: ScoutState) -> dict:
    """Kaşif: gerçek run_explorer'ı (src/agents/explorer.py) çağırır - bu zaten
    çalışan, araçlarla veri özeti toplayıp LLM ile yorumlayan bir fonksiyondur
    (mock gerekmez). Testlerde graph.run_explorer / graph.data_tools.load_appearances
    monkeypatch'lenerek gerçek API çağrısı ve ağır CSV okuması atlanır."""
    raw = data_tools.load_appearances()
    note = run_explorer({"raw": raw}, state["question"])
    return {"data_note": note}


def scout_node(state: ScoutState) -> dict:
    """Scout: gerçek run_scout'u (src/agents/scout.py) çağırır - tool-use ile
    multi_league_1516_matched.parquet üzerinde deger artığı analizi yapıp
    yapılandırılmış (JSON) aday listesi döner.

    Revizyon sayacı burada artırılır: route_after_critic revizyon için "scout"a
    dönmeye karar verdiğinde, bu düğüm yeniden çalışır. LangGraph koşullu
    kenarları (conditional edge) state'i kalıcı olarak GÜNCELLEYEMEZ (ampirik
    olarak doğrulandı - koşullu kenar içindeki in-place mutasyon bir sonraki
    süper-adıma taşınmaz, sonsuz döngüye yol açar), bu yüzden görevde
    route_after_critic'e atfedilen "scout'a dönmeden önce revision_count'u 1
    artır" işlemi, kalıcı olması için scout'un yeniden-giriş güncellemesi olarak
    gerçeklenir. İlk çalıştırmada (henüz aday yokken) sayaç artmaz; sonraki her
    çalıştırma (candidates zaten doluyken = bir revizyon dönüşü) sayacı 1 artırır
    - bu, görevdeki 'router eski revision_count'u kontrol eder, sonra artırır,
    kritik bir sonraki turda artmış değeri görür' kontrol akışının birebir
    çalışan karşılığıdır."""
    is_revision = bool(state.get("candidates"))
    revision_count = state.get("revision_count", 0) + (1 if is_revision else 0)
    candidates = run_scout(state, state["question"])
    return {"candidates": candidates, "revision_count": revision_count}


def critic_node(state: ScoutState) -> dict:
    """Eleştirmen (STUB): revision_count'a göre deterministik karar. revision_count
    < MAX_REVISIONS iken REDDET döndürür (revizyon döngüsünü test için tetikler),
    revision_count >= MAX_REVISIONS iken ONAYLA döndürür - böylece döngünün
    gerçekten sonlandığı doğrulanabilir."""
    revision_count = state.get("revision_count", 0)
    if revision_count < config.MAX_REVISIONS:
        return {
            "critic_verdict": "REDDET",
            "critic_notes": f"Revizyon {revision_count}: kanıt yetersiz, daha fazla aday/gerekçe iste.",
        }
    return {
        "critic_verdict": "ONAYLA",
        "critic_notes": "Yeterli kanıt sağlandı, aday listesi onaylandı.",
    }


def writer_node(state: ScoutState) -> dict:
    """Yazar (STUB): candidates + critic_notes'tan basit bir rapor stringi üretir.

    .get() ile defansif okunur: gerçek Scout'un (run_scout) JSON çıktısı en az
    player_name/position/value_residual/npxg_per90/market_value_in_eur
    alanlarını garanti eder ('league' garanti değildir), bu yüzden doğrudan
    dict indexleme yerine eksik alanlarda KeyError fırlatmayan .get() kullanılır."""
    candidates = state.get("candidates", [])
    if candidates:
        satirlar = "\n".join(
            f"- {c.get('player_name', '?')} ({c.get('league', c.get('position', '?'))}): "
            f"value_residual={c.get('value_residual', '?')}"
            for c in candidates
        )
    else:
        satirlar = "- (aday bulunamadı)"
    report = (
        f"SCOUTING RAPORU\n"
        f"Soru: {state.get('question', '')}\n"
        f"Aday sayısı: {len(candidates)}\n"
        f"{satirlar}\n"
        f"Eleştirmen notu: {state.get('critic_notes', '')}"
    )
    return {"report": report}


def route_after_critic(state: ScoutState) -> str:
    """Eleştirmenden sonra yönlendirme: karar ONAYLA ise VEYA revision_count
    config.MAX_REVISIONS'a ulaştıysa "writer"a (döngüyü sonlandır), aksi halde
    "scout"a (revizyon) döner. Revizyon sayacının artırılması scout_node'da
    gerçeklenir (bkz. scout_node docstring'i - koşullu kenar state'i kalıcı
    güncelleyemez)."""
    if state.get("critic_verdict") == "ONAYLA" or state.get("revision_count", 0) >= config.MAX_REVISIONS:
        return "writer"
    return "scout"


def build_graph():
    """StateGraph'ı kurar ve derlenmiş uygulamayı döndürür."""
    g = StateGraph(ScoutState)
    g.add_node("explorer", explorer_node)
    g.add_node("scout", scout_node)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)

    g.set_entry_point("explorer")
    g.add_edge("explorer", "scout")
    g.add_edge("scout", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"writer": "writer", "scout": "scout"})
    g.add_edge("writer", END)

    return g.compile()


app = build_graph()
