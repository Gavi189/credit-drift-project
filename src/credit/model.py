"""Treino, persistência e avaliação do modelo de Credit Scoring."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from credit import config


def train_baseline(reference: pd.DataFrame) -> tuple[GradientBoostingClassifier, dict[str, float]]:
    """Treina o classificador base no dataset de Referência.

    Args:
        reference: Dataset limpo de referência com a coluna alvo.

    Returns:
        Tupla (modelo treinado, métricas no holdout de referência).
    """
    x_train, x_test, y_train, y_test = train_test_split(
        reference[config.FEATURES],
        reference[config.TARGET],
        test_size=0.25,
        random_state=config.SEED,
        stratify=reference[config.TARGET],
    )
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=config.SEED)
    model.fit(x_train, y_train)
    return model, evaluate(model, x_test, y_test)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS de separação entre bons e maus pagadores (métrica clássica de crédito).

    Args:
        y_true: Rótulos reais.
        y_score: Probabilidades previstas.

    Returns:
        Máxima distância entre as curvas acumuladas de TPR e FPR.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(model: GradientBoostingClassifier, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Calcula métricas de desempenho e calibração.

    Args:
        model: Modelo treinado.
        x: Features.
        y: Rótulos reais.

    Returns:
        Dicionário com AUC, KS, acurácia, precisão, recall, F1, Brier e
        taxas média prevista vs. observada de inadimplência.
    """
    proba = model.predict_proba(x)[:, 1]
    pred = (proba >= config.DECISION_THRESHOLD).astype(int)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "ks": ks_statistic(y.to_numpy(), proba),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
        "mean_predicted_default": float(proba.mean()),
        "observed_default_rate": float(y.mean()),
    }


def save_model(model: GradientBoostingClassifier, path=config.MODEL_PATH) -> None:
    """Persiste o modelo em disco.

    Args:
        model: Modelo treinado.
        path: Destino do arquivo joblib.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path=config.MODEL_PATH) -> GradientBoostingClassifier:
    """Carrega o modelo persistido.

    Args:
        path: Arquivo joblib do modelo.

    Returns:
        Modelo treinado.
    """
    return joblib.load(path)
