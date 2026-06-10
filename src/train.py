"""Treino de modelos com walk-forward validation.

Carrega o dataset processado, treina Logistic Regression, Random Forest e
XGBoost com validação walk-forward (expanding window) e grava as previsões
out-of-sample em data/processed/predictions.parquet.

Uso:
    python -m src.train
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .fetch_data import load_config
from .features import TARGET_COL

ROOT = Path(__file__).resolve().parents[1]
PRED_PATH = "data/processed/predictions.parquet"


def build_models(config: dict) -> dict[str, Pipeline]:
    """Define os três modelos. Logistic precisa de scaling; árvores não.

    A Logistic Regression é o baseline interpretável (é o que a Fed de NY usa);
    Random Forest e XGBoost captam não-linearidades e interações.
    """
    rs = config["models"]["random_state"]
    m = config["models"]
    return {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=m["logistic"]["max_iter"])),
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=m["random_forest"]["n_estimators"],
                max_depth=m["random_forest"]["max_depth"],
                random_state=rs,
            )),
        ]),
        "xgboost": Pipeline([
            ("clf", XGBClassifier(
                n_estimators=m["xgboost"]["n_estimators"],
                max_depth=m["xgboost"]["max_depth"],
                learning_rate=m["xgboost"]["learning_rate"],
                eval_metric="logloss",
                random_state=rs,
            )),
        ]),
    }


def walk_forward_split(
    n_samples: int,
    min_train: int,
    step: int = 1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Gera índices (treino, teste) para validação walk-forward (expanding window).

    Esta é a peça que separa um projeto sério de séries temporais de um amador.
    Com cross-validation aleatório, o modelo treinaria com 2010 e testaria em
    2005 — a "ver o futuro". Aqui, em cada passo, treina-se SÓ com o passado e
    prevê-se o(s) ponto(s) imediatamente a seguir; depois a janela cresce.

    Esquema (expanding window):
        passo 0:  treino [0 .. min_train),           teste [min_train .. min_train+step)
        passo 1:  treino [0 .. min_train+step),       teste [.. +step)
        passo 2:  treino [0 .. min_train+2*step),     teste [.. +step)
        ...até esgotar as amostras.

    Args:
        n_samples: número total de linhas com label.
        min_train: nº mínimo de meses de treino antes da primeira previsão.
        step: de quantos em quantos meses a janela avança (e tamanho do bloco de teste).

    Yields:
        Tuplos (train_idx, test_idx) de arrays numpy de índices inteiros.

    ─────────────────────────────────────────────────────────────────────
    A TUA VEZ (Learning Mode):
    Implementa o gerador. Invariante CRÍTICA a garantir: max(train_idx) <
    min(test_idx) SEMPRE (nunca treinar com dados >= ao teste). O teste em
    tests/test_train.py verifica exatamente isto.
    Pista: um `while` sobre uma posição `t` que começa em `min_train`; o treino
    é `np.arange(0, t)`, o teste é `np.arange(t, t+step)`; incrementa `t += step`
    até `t >= n_samples`. Atenção a não deixar o último bloco de teste ultrapassar
    n_samples.
    Decisão de design: "expanding" (treino cresce sempre) vs "rolling" (janela de
    tamanho fixo que desliza). Implementa expanding; pensa em que situação rolling
    seria melhor (mudanças estruturais na economia?).
    ─────────────────────────────────────────────────────────────────────
    """
    t = min_train
    while t < n_samples:
        train_idx = np.arange(0, t)            # tudo o que é estritamente passado
        test_idx = np.arange(t, min(t + step, n_samples))  # bloco seguinte
        yield train_idx, test_idx
        t += step


def run_walk_forward(df_labelled: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Corre a validação walk-forward para cada modelo.

    Devolve um DataFrame indexado por data com a probabilidade prevista
    out-of-sample de cada modelo (mais a coluna do valor real `target`).
    """
    feature_cols = [c for c in df_labelled.columns if c not in ("recession", TARGET_COL)]
    X = df_labelled[feature_cols].to_numpy()
    y = df_labelled[TARGET_COL].to_numpy().astype(int)
    idx = df_labelled.index

    min_train = config["models"]["walk_forward"]["min_train_months"]
    step = config["models"]["walk_forward"]["step_months"]
    models = build_models(config)

    # Acumula previsões: {modelo: {data: prob}}
    preds = {name: pd.Series(index=idx, dtype=float) for name in models}

    splits = list(walk_forward_split(len(y), min_train, step))
    print(f"Walk-forward: {len(splits)} passos, treino mínimo {min_train} meses.")

    for name, model in models.items():
        for train_idx, test_idx in splits:
            # Salta blocos de teste degenerados ou sem ambas as classes no treino.
            if len(np.unique(y[train_idx])) < 2:
                continue
            model.fit(X[train_idx], y[train_idx])
            proba = model.predict_proba(X[test_idx])[:, 1]
            preds[name].iloc[test_idx] = proba
        print(f"  -{name} concluído.")

    result = pd.DataFrame(preds)
    result[TARGET_COL] = y
    result["recession"] = df_labelled["recession"].values
    return result


def predict_live(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Previsão "ao vivo": treina cada modelo em TODOS os meses com label e
    prevê os últimos meses sem label (o futuro ainda desconhecido).

    Devolve um DataFrame com a mesma forma do walk-forward (uma coluna por
    modelo), indexado pelas datas sem label. target/recession ficam NaN.
    """
    feature_cols = [c for c in df.columns if c not in ("recession", TARGET_COL)]
    train = df[df[TARGET_COL].notna()]
    live = df[df[TARGET_COL].isna()]
    if live.empty:
        return pd.DataFrame()

    Xtr = train[feature_cols].to_numpy()
    ytr = train[TARGET_COL].to_numpy().astype(int)
    out = {}
    for name, model in build_models(config).items():
        model.fit(Xtr, ytr)
        out[name] = model.predict_proba(live[feature_cols].to_numpy())[:, 1]
    result = pd.DataFrame(out, index=live.index)
    result[TARGET_COL] = np.nan
    result["recession"] = np.nan
    return result


def main() -> None:
    config = load_config()
    df = pd.read_parquet(ROOT / config["data"]["processed_path"])
    df_labelled = df[df[TARGET_COL].notna()].copy()

    result = run_walk_forward(df_labelled, config)
    result = result.dropna(how="all", subset=list(build_models(config).keys()))

    # Anexa a previsão ao vivo (futuro) para estender o gráfico até hoje.
    live = predict_live(df, config)
    if not live.empty:
        result = pd.concat([result, live])
        print(f"Previsão ao vivo: {len(live)} meses ({live.index.min():%Y-%m} a {live.index.max():%Y-%m}).")

    out = ROOT / PRED_PATH
    result.to_parquet(out)
    print(f"\nPrevisões gravadas em {out} ({len(result)} linhas).")


if __name__ == "__main__":
    main()
