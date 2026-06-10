"""Recolha de dados macro da FRED.

Lê as séries definidas em config.yaml, alinha-as numa grelha mensal comum
e grava o resultado em data/raw/series.parquet.

Uso:
    python -m src.fetch_data
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from fredapi import Fred

import os

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Carrega o config.yaml como dicionário."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key() -> str:
    """Lê a FRED API key do ambiente (.env). Falha com mensagem clara se ausente.

    Mantemos o segredo fora do repositório: o código lê de uma variável de
    ambiente, nunca de um literal no código.
    """
    load_dotenv(ROOT / ".env")
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY não encontrada.\n"
            "  1. Obtém uma chave gratuita em "
            "https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "  2. Copia .env.example para .env e preenche FRED_API_KEY=<a tua chave>"
        )
    return key


def fetch_series(config: dict | None = None) -> pd.DataFrame:
    """Recolhe todas as séries da FRED e devolve um DataFrame mensal alinhado.

    - Séries diárias (spreads) são reamostradas para o último valor do mês.
    - Todas as séries são alinhadas no mesmo índice mensal (frequency do config).
    - Inclui a série-alvo bruta (USREC) para mais tarde construir o target.
    """
    config = config or load_config()
    fred = Fred(api_key=get_api_key())

    start = config["data"]["start_date"]
    freq = config["data"]["frequency"]

    all_ids = {**config["series"], "recession": config["target_series"]}

    frames = {}
    for name, fred_id in all_ids.items():
        print(f"  - a recolher {name:<14} ({fred_id}) ...")
        s = fred.get_series(fred_id, observation_start=start)
        s.index = pd.to_datetime(s.index)
        # Reamostra para mensal. 'last' usa a observação mais recente do mês
        # (apropriado para séries de nível como taxas e spreads).
        s = s.resample(freq).last()
        frames[name] = s

    df = pd.DataFrame(frames)
    # USREC é mensal e {0,1}; após o resample pode aparecer NaN no mês corrente.
    df["recession"] = df["recession"].ffill()
    return df


def main() -> None:
    config = load_config()
    print("A recolher séries da FRED...")
    df = fetch_series(config)

    out = ROOT / config["data"]["raw_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nGravado {df.shape[0]} linhas x {df.shape[1]} colunas em {out}")
    print(f"Intervalo: {df.index.min():%Y-%m} a {df.index.max():%Y-%m}")


if __name__ == "__main__":
    main()
