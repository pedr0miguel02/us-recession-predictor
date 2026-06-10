"""Testes das duas peças críticas: o target (sem fuga) e o split walk-forward.

Corre com:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.features import build_target
from src.train import walk_forward_split


def test_build_target_lag():
    """O label de hoje deve refletir a recessão daqui a `lag` meses."""
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    # Recessão nos meses 12..15 (índices 12-15).
    rec = pd.Series(0, index=idx)
    rec.iloc[12:16] = 1
    df = pd.DataFrame({"recession": rec})

    target = build_target(df, lag_months=12)

    # O mês 0 (2000-01) deve "ver" a recessão do mês 12 -> label 1.
    assert target.iloc[0] == 1
    # As últimas 12 linhas não têm futuro conhecido -> NaN.
    assert target.iloc[-12:].isna().all()


def test_walk_forward_no_leakage():
    """Invariante crítica: nunca treinar com dados >= ao bloco de teste."""
    splits = list(walk_forward_split(n_samples=50, min_train=20, step=5))
    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min(), "fuga de dados: treino sobrepõe teste"
        assert test_idx.max() < 50, "bloco de teste ultrapassa n_samples"


def test_walk_forward_expanding():
    """A janela de treino deve crescer (expanding)."""
    splits = list(walk_forward_split(n_samples=50, min_train=20, step=5))
    train_sizes = [len(tr) for tr, _ in splits]
    assert train_sizes == sorted(train_sizes), "treino não está a crescer monotonicamente"
