"""Orquestrador do pipeline completo: fetch -> features -> train -> evaluate.

Uso:
    python run_pipeline.py            # corre tudo
    python run_pipeline.py --skip-fetch   # reaproveita data/raw já existente
"""
from __future__ import annotations

import argparse

from src import evaluate, features, fetch_data, train


def main() -> None:
    parser = argparse.ArgumentParser(description="US Recession Predictor — pipeline")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="não recolher da FRED; usar data/raw existente")
    args = parser.parse_args()

    if not args.skip_fetch:
        print("=" * 60, "\n[1/4] Recolha de dados (FRED)\n", "=" * 60, sep="")
        fetch_data.main()

    print("\n" + "=" * 60 + "\n[2/4] Engenharia de features + target\n" + "=" * 60)
    features.main()

    print("\n" + "=" * 60 + "\n[3/4] Treino (walk-forward)\n" + "=" * 60)
    train.main()

    print("\n" + "=" * 60 + "\n[4/4] Avaliação + figuras\n" + "=" * 60)
    evaluate.main()

    print("\n[OK] Pipeline completo. Ve reports/figures/.")


if __name__ == "__main__":
    main()
