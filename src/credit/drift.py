"""Detecção de Data Drift e Concept Drift (PSI, Kolmogorov-Smirnov, Evidently)."""

from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from scipy import stats

from credit import config


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Calcula o Population Stability Index (PSI) entre duas distribuições.

    Os cortes (bins) são quantis da referência; valores fora da faixa da
    referência caem nos bins extremos.

    Args:
        reference: Amostra de referência.
        current: Amostra atual (produção).
        bins: Número de faixas.

    Returns:
        PSI (< 0,10 estável; 0,10–0,25 atenção; > 0,25 mudança crítica).
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_level(value: float) -> str:
    """Classifica o PSI em estável / atenção / crítico.

    Args:
        value: Valor do PSI.

    Returns:
        Um entre ``"estavel"``, ``"atencao"`` e ``"critico"``.
    """
    if value >= config.PSI_CRITICAL:
        return "critico"
    return "atencao" if value >= config.PSI_WARNING else "estavel"


def drift_table(reference: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    """Compara cada feature entre referência e produção com PSI e KS.

    Args:
        reference: Dataset de referência.
        production: Dataset de produção.

    Returns:
        Tabela por feature, ordenada por PSI decrescente, com a decisão ``drift``.
    """
    rows = []
    for col in config.NUMERIC_DRIFT_FEATURES:
        ks_stat, p_value = stats.ks_2samp(reference[col], production[col])
        value = psi(reference[col], production[col])
        rows.append(
            {
                "feature": col,
                "media_referencia": float(reference[col].mean()),
                "media_producao": float(production[col].mean()),
                "variacao_pct": float((production[col].mean() / reference[col].mean() - 1) * 100),
                "psi": value,
                "psi_nivel": psi_level(value),
                "ks_stat": float(ks_stat),
                "ks_p_value": float(p_value),
                "drift": bool(value >= config.PSI_WARNING and p_value < config.KS_P_VALUE),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def concept_drift_summary(ref_metrics: dict, prod_metrics: dict) -> dict[str, float]:
    """Resume a degradação de desempenho (sinal de Concept Drift).

    Args:
        ref_metrics: Métricas do modelo na referência (holdout).
        prod_metrics: Métricas do modelo em produção (rótulos já observados).

    Returns:
        Quedas de AUC/KS/recall e erro de calibração entre previsto e observado.
    """
    return {
        "auc_drop": ref_metrics["auc"] - prod_metrics["auc"],
        "ks_drop": ref_metrics["ks"] - prod_metrics["ks"],
        "recall_drop": ref_metrics["recall"] - prod_metrics["recall"],
        "calibration_gap": prod_metrics["observed_default_rate"] - prod_metrics["mean_predicted_default"],
        "target_rate_shift": prod_metrics["observed_default_rate"] - ref_metrics["observed_default_rate"],
    }


def run_evidently_report(
    reference: pd.DataFrame, production: pd.DataFrame, method: str, html_path: Path
) -> dict[str, dict]:
    """Executa o DataDriftPreset do Evidently AI e salva o relatório HTML.

    Args:
        reference: Referência (já com a coluna ``prediction``).
        production: Produção (já com a coluna ``prediction``).
        method: Teste estatístico do Evidently (``"psi"`` ou ``"ks"``).
        html_path: Destino do relatório HTML.

    Returns:
        Dicionário por coluna com valor do teste e resumo de colunas em drift.
    """
    cols = [*config.FEATURES, "prediction"]
    definition = DataDefinition(numerical_columns=cols)
    ref_ds = Dataset.from_pandas(reference[cols], data_definition=definition)
    prod_ds = Dataset.from_pandas(production[cols], data_definition=definition)
    snapshot = Report([DataDriftPreset(method=method)]).run(prod_ds, ref_ds)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))

    out: dict[str, dict] = {}
    for metric in snapshot.dict()["metrics"]:
        name = metric["metric_name"]
        if name.startswith("DriftedColumnsCount"):
            out["_resumo"] = metric["value"]
        elif name.startswith("ValueDrift"):
            column = name.split("column=")[1].split(",")[0]
            out[column] = {"value": float(metric["value"])}
    return out
