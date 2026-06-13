"""Previsão ao vivo num só comando.

Imprime a probabilidade estimada de recessão nos próximos 12 meses, com base
nos dados mais recentes disponíveis. Reutiliza predict_live() de train.py.

Uso:
    python -m src.predict                 # usa o melhor modelo (XGBoost)
    python -m src.predict --model logistic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .fetch_data import load_config
from .features import TARGET_COL
from .train import predict_live

ROOT = Path(__file__).resolve().parents[1]


def risk_label(prob: float) -> str:
    """Traduz a probabilidade num rotulo legivel."""
    if prob < 0.20:
        return "BAIXO (territorio de expansao)"
    if prob < 0.50:
        return "MODERADO (sinais a vigiar)"
    return "ELEVADO (alerta de recessao)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Previsão ao vivo de recessão")
    parser.add_argument("--model", default="xgboost",
                        choices=["xgboost", "random_forest", "logistic"],
                        help="modelo a usar (default: xgboost)")
    args = parser.parse_args()

    config = load_config()
    df = pd.read_parquet(ROOT / config["data"]["processed_path"])

    live = predict_live(df, config)
    if live.empty:
        print("Sem meses por prever — corre o pipeline para obter dados recentes.")
        return

    horizon = config["target"]["lag_months"]
    series = live[args.model]
    last_date = series.index[-1]
    last_prob = series.iloc[-1]
    target_date = last_date + pd.DateOffset(months=horizon)

    print("=" * 56)
    print(f"  PREVISAO DE RECESSAO - modelo: {args.model}")
    print("=" * 56)
    print(f"\n  Dados mais recentes: {last_date:%Y-%m}")
    print(f"  Probabilidade de recessao ate {target_date:%Y-%m}: {last_prob:.1%}")
    print(f"  Nivel de risco: {risk_label(last_prob)}")

    print(f"\n  Trajetoria dos ultimos meses:")
    for date, prob in series.tail(6).items():
        bar = "#" * int(prob * 40)
        print(f"    {date:%Y-%m}  {prob:5.1%}  {bar}")
    print()


if __name__ == "__main__":
    main()
