"""Avaliação e visualização dos resultados.

Lê as previsões walk-forward e produz:
  - AUC-ROC por modelo (impresso + gráfico de curvas ROC)
  - Probabilidade de recessão prevista ao longo do tempo, sobreposta às
    recessões reais (barras cinzentas, estilo FRED)
  - Feature importance do melhor modelo de árvore

Uso:
    python -m src.evaluate
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .fetch_data import load_config
from .features import TARGET_COL, feature_columns
from .train import MODEL_NAMES, PRED_PATH, build_models, run_walk_forward

ROOT = Path(__file__).resolve().parents[1]


def compute_auc(preds: pd.DataFrame) -> dict[str, float]:
    """AUC-ROC out-of-sample por modelo. Métrica principal: robusta a classes
    desbalanceadas (recessões são raras, ~15% dos meses)."""
    scores = {}
    for name in MODEL_NAMES:
        # Só linhas com label conhecido E previsão deste modelo.
        mask = preds[name].notna() & preds[TARGET_COL].notna()
        y = preds[TARGET_COL][mask].astype(int)
        scores[name] = roc_auc_score(y, preds[name][mask])
    return scores


def _shade_recessions(ax, recession: pd.Series) -> None:
    """Desenha barras cinzentas nos períodos de recessão real (estilo FRED)."""
    rec = recession.fillna(0).astype(int)
    in_rec = False
    start = None
    for date, val in rec.items():
        if val == 1 and not in_rec:
            in_rec, start = True, date
        elif val == 0 and in_rec:
            ax.axvspan(start, date, color="grey", alpha=0.3, lw=0)
            in_rec = False
    if in_rec:
        ax.axvspan(start, rec.index[-1], color="grey", alpha=0.3, lw=0)


def plot_recession_probability(preds: pd.DataFrame, scores: dict, out_dir: Path) -> Path:
    """Gráfico-âncora do projeto: prob. prevista vs. recessões reais."""
    fig, ax = plt.subplots(figsize=(13, 5))
    _shade_recessions(ax, preds["recession"])
    for name in MODEL_NAMES:
        ax.plot(preds.index, preds[name], label=f"{name} (AUC={scores[name]:.3f})", lw=1.3)

    # Fronteira histórico | previsão ao vivo: onde o target deixa de ser conhecido.
    live = preds[preds[TARGET_COL].isna()]
    if not live.empty:
        boundary = live.index.min()
        ax.axvline(boundary, color="black", ls=":", lw=1)
        ax.text(boundary, 0.95, " previsão ao vivo", fontsize=8, va="top")
    ax.set_title("Probabilidade prevista de recessão (12 meses à frente) vs. recessões reais NBER",
                 fontsize=12, pad=30)
    ax.set_ylabel("P(recessão dentro de 12 meses)")
    ax.set_ylim(0, 1)
    # Legenda horizontal por cima do gráfico (entre o título e o plot), para não
    # tapar as linhas dos anos ~2000 nem o título.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
              fontsize=9, frameon=False)
    ax.margins(x=0.01)

    # Eixo X: uma marca por ano (em vez do automático de 4 em 4), rodadas.
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    # Carimbo de geração: ajuda a situar a frescura do gráfico.
    last_data = preds.index.max()
    fig.text(0.99, 0.01,
             f"Gerado em {date.today():%Y-%m-%d}  |  dados ate {last_data:%Y-%m}",
             ha="right", va="bottom", fontsize=8, color="grey")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = out_dir / "recession_probability.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(preds: pd.DataFrame, scores: dict, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in MODEL_NAMES:
        mask = preds[name].notna() & preds[TARGET_COL].notna()
        y = preds[TARGET_COL][mask].astype(int)
        fpr, tpr, _ = roc_curve(y, preds[name][mask])
        ax.plot(fpr, tpr, label=f"{name} (AUC={scores[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="aleatório")
    ax.set_xlabel("Falsos positivos")
    ax.set_ylabel("Verdadeiros positivos")
    ax.set_title("Curvas ROC (walk-forward out-of-sample)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = out_dir / "roc_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feature_importance(config: dict, out_dir: Path) -> Path | None:
    """Treina o XGBoost em todos os dados com label e mostra a importância das
    features — responde a "qual indicador mais previu recessões?"."""
    df = pd.read_parquet(ROOT / config["data"]["processed_path"])
    df = df[df[TARGET_COL].notna()]
    feature_cols = feature_columns(df)

    model = build_models(config)["xgboost"]
    model.fit(df[feature_cols].to_numpy(), df[TARGET_COL].astype(int).to_numpy())
    importances = model.named_steps["clf"].feature_importances_

    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(np.array(feature_cols)[order], importances[order], color="steelblue")
    ax.set_title("Importância das features (XGBoost) — o que prevê recessões?")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    config = load_config()
    pred_path = ROOT / PRED_PATH
    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
    else:
        # Conveniência: corre o walk-forward se ainda não houver previsões.
        df = pd.read_parquet(ROOT / config["data"]["processed_path"])
        preds = run_walk_forward(df[df[TARGET_COL].notna()].copy(), config)

    scores = compute_auc(preds)
    print("\nAUC-ROC (walk-forward, out-of-sample):")
    for name, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name:<15} {s:.4f}")

    out_dir = ROOT / config["reports"]["figures_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = plot_recession_probability(preds, scores, out_dir)
    p2 = plot_roc_curves(preds, scores, out_dir)
    p3 = plot_feature_importance(config, out_dir)
    print(f"\nFiguras gravadas:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
