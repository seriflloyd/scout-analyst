# Otonom Scouting Analisti

## Proje amacı
LLM ajanlarıyla (Kaşif → Scout → Eleştirmen → Yazar) futbolda
değerinin altında kalan oyuncuları tespit eden çok-ajanlı sistem.

## Değişmez kurallar
- LLM ASLA hesap yapmaz. Tüm sayılar src/tools/ altındaki saf Python
  fonksiyonlarından gelir; LLM sadece araç seçer ve yorumlar.
- Model adları ve limitler yalnızca src/config.py'de tanımlanır.
- Ajan döngülerinde araç çağrısı limiti: config.MAX_TOOL_CALLS.
- Her yeni tools/ fonksiyonu için pytest testi yazılır.

## Kod stili
- Python 3.12, type hints zorunlu, docstring'ler Türkçe.
- Dosya yapısı: src/tools (hesap), src/agents (LLM), src/graph.py (akış).

## Komutlar
- Test: pytest
- Çalıştırma: python -m src.main --help
