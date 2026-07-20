"""Scout ajaninin (src/agents/scout.py) GERCEK Anthropic API cagrisiyla,
gercek data/processed/multi_league_1516_matched.parquet uzerinde manuel
entegrasyon calistirmasi.

pytest TARAFINDAN OTOMATIK CALISTIRILMAZ (dosya adi test_*.py / *_test.py
kalibina uymaz, tests/ altinda degil) - gercek API cagrisi ve ucret gerektirir.
Manuel calistirma: python -m scripts.manual_scout_integration
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.agents.scout import run_scout

load_dotenv()

QUESTION = "Bu 4 ligde performansina gore degerinin altinda kalan orta saha bul"

if __name__ == "__main__":
    print(f"Soru: {QUESTION}\n")
    candidates = run_scout({}, QUESTION)
    print(f"\n{len(candidates)} aday donduruldu:\n")
    for c in candidates:
        print(c)
