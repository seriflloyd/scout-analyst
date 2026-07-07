import argparse
import sys
from dotenv import load_dotenv
from src.tools import data_tools
from src.agents.explorer import run_explorer

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


def main() -> None:
    p = argparse.ArgumentParser(description="Otonom Scouting Analisti")
    p.add_argument("--question", default="Bu veri setini tani ve scouting'e hazirla.")
    args = p.parse_args()

    state = {"raw": data_tools.load_appearances()}
    print(run_explorer(state, args.question))


if __name__ == "__main__":
    main()
