"""Engenharia de features e construção da variável-alvo.

Lê data/raw/series.parquet, transforma os indicadores em features e cria o
target binário com lag de 12 meses. Grava data/processed/dataset.parquet.

Uso:
    python -m src.features
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .fetch_data import load_config

ROOT = Path(__file__).resolve().parents[1]

# Nome da coluna que vamos prever.
TARGET_COL = "target"

# Colunas que NÃO são features de input (o alvo e a recessão atual crua).
NON_FEATURE_COLS = ("recession", TARGET_COL)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Lista as colunas de features (tudo menos o alvo e a recessão crua).

    Fonte única de verdade — usada no treino, na previsão e na avaliação.
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Transforma as séries brutas em features de modelação.

    - Séries em config['features']['yoy_pct_change'] (CPI, M2, produção
      industrial) viram variação percentual homóloga (YoY, 12 meses): os
      níveis brutos não são estacionários, a variação é que carrega sinal.
    - As restantes (spreads, taxas, desemprego, sentimento) ficam em nível.
    """
    out = df.copy()
    yoy_cols = config["features"]["yoy_pct_change"]
    for col in yoy_cols:
        # variação % face a 12 meses atrás
        out[f"{col}_yoy"] = out[col].pct_change(periods=12) * 100
        out = out.drop(columns=col)
    return out


def build_target(df: pd.DataFrame, lag_months: int = 12) -> pd.Series:
    """Constrói o alvo binário: "haverá recessão dentro de `lag_months`?".

    A coluna `recession` (USREC) marca se um dado mês ESTÁ em recessão. Mas
    queremos PREVER o futuro: o label da linha de hoje deve ser 1 se houver
    recessão daqui a `lag_months`. Conseguimos isto deslocando a série de
    recessão para trás no tempo (shift negativo): o valor de daqui a N meses
    "vem" para a linha de hoje.

    Consequência importante: as últimas `lag_months` linhas ficam com NaN
    (ainda não sabemos o futuro) — esse será o nosso conjunto de predição "ao vivo".

    Args:
        df: DataFrame com uma coluna "recession" em {0, 1}, índice temporal mensal ordenado.
        lag_months: horizonte de previsão em meses (default 12).

    Returns:
        pd.Series com o target binário, mesmo índice de `df`, com NaN nas
        últimas `lag_months` linhas.
    """
    # shift(-lag) puxa o valor de daqui a `lag` meses para a linha de hoje.
    # Assim o label de hoje responde a "haverá recessão dentro de `lag` meses?".
    # As últimas `lag` linhas ficam NaN (futuro desconhecido) -> conjunto "ao vivo".
    return df["recession"].shift(-lag_months)


def main() -> None:
    config = load_config()
    raw_path = ROOT / config["data"]["raw_path"]
    df = pd.read_parquet(raw_path)

    feats = build_features(df, config)
    feats[TARGET_COL] = build_target(df, config["target"]["lag_months"])

    # Remove o aquecimento inicial (NaN das transformações YoY). Mantém as
    # linhas finais sem target — são separadas explicitamente a jusante.
    feature_cols = feature_columns(feats)
    feats = feats.dropna(subset=feature_cols)

    out = ROOT / config["data"]["processed_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out)

    labelled = feats[TARGET_COL].notna().sum()
    print(f"Gravado dataset: {feats.shape[0]} linhas, {len(feature_cols)} features.")
    print(f"  -{labelled} linhas com label (treino/teste)")
    print(f"  -{feats.shape[0] - labelled} linhas sem label (predição ao vivo)")
    print(f"  -features: {feature_cols}")


if __name__ == "__main__":
    main()
