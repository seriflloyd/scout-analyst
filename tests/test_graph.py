"""Faz D LangGraph iskeleti testleri: STUB node'larla (gerçek API çağrısı yok,
hızlı) state akışı ve revizyon döngüsünün sonlanması doğrulanır."""
import pandas as pd
import pytest

from src import config, graph


@pytest.fixture
def stubbed_explorer(monkeypatch):
    """Kaşif node'unun gerçek LLM çağrısını ve ağır appearances.csv okumasını
    monkeypatch ile atlar - graf iskeleti testleri hızlı ve ağdan bağımsız
    kalır (Scout/Eleştirmen/Yazar zaten stub)."""
    monkeypatch.setattr(graph, "run_explorer", lambda state, question: "sahte kesif notu")
    monkeypatch.setattr(graph.data_tools, "load_appearances", lambda *a, **k: pd.DataFrame())


def test_normal_flow_threads_state_explorer_to_writer(stubbed_explorer):
    """explorer->scout->critic->...->writer boyunca state'in her düğümden doğru
    aktığını doğrular: Kaşif notu, Scout adayları, Eleştirmen kararı ve Yazar
    raporu nihai state'te dolu olmalı."""
    final = graph.app.invoke({"question": "Değerinin altında oyuncular kim?"})

    assert final["data_note"] == "sahte kesif notu"          # Kaşif çıktısı aktı
    assert len(final["candidates"]) == 2                       # Scout çıktısı aktı
    assert final["candidates"][0]["player_name"] == "Test Oyuncu A"
    assert final["critic_verdict"] in ("ONAYLA", "REDDET")     # Eleştirmen çalıştı
    assert final["report"]                                     # Yazar çıktısı aktı

    # Düğüm ziyaret sırası: explorer ve writer tam bir kez, arada scout/critic döngüsü
    visited = [node for chunk in graph.app.stream({"question": "soru"}) for node in chunk]
    assert visited[0] == "explorer"
    assert visited[-1] == "writer"
    assert "scout" in visited and "critic" in visited


def test_revision_loop_terminates_at_max_revisions(stubbed_explorer):
    """Eleştirmen stub'ı revision_count < MAX_REVISIONS iken REDDET döndürür;
    döngü sonsuza gitmemeli - revision_count MAX_REVISIONS'a ulaşınca Eleştirmen
    zorla ONAYLA'ya geçip Yazar'a gitmeli."""
    final = graph.app.invoke({"question": "soru"})

    assert final["revision_count"] == config.MAX_REVISIONS     # sayaç tam sınıra ulaştı
    assert final["critic_verdict"] == "ONAYLA"                 # sınırda zorla onaylandı
    assert final["report"]                                     # döngü writer'da sonlandı

    # Eleştirmen tam MAX_REVISIONS+1 kez çalışmalı (rc=0,1,...,MAX): sonsuz döngü yok
    visited = [node for chunk in graph.app.stream({"question": "soru"}) for node in chunk]
    assert visited.count("critic") == config.MAX_REVISIONS + 1
    assert visited.count("scout") == config.MAX_REVISIONS + 1
    assert visited.count("writer") == 1


def test_final_state_report_is_populated(stubbed_explorer):
    """Nihai state'te report alanı boş olmamalı ve rapor içeriği aday bilgisini
    taşımalı."""
    final = graph.app.invoke({"question": "soru"})

    assert final["report"].strip() != ""
    assert "SCOUTING RAPORU" in final["report"]
    assert "Test Oyuncu A" in final["report"]
