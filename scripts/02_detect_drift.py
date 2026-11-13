"""Etapa 2 — Simulação e detecção de Drift.

1. Carrega o dataset de Produção (já com Data Drift + Concept Drift simulados).
2. Passa pelo contrato de dados (estrutura válida ≠ distribuição estável).
3. Gera predições com o modelo baseline.
4. Compara Referência × Produção: PSI e Kolmogorov-Smirnov por feature.
5. Gera relatórios HTML do Evidently AI (PSI e KS).
6. Mede degradação de desempenho (Concept Drift) e gera alertas.

Uso:
    python scripts/02_detect_drift.py
"""

import logging

import pandas as pd

from credit import config, contract, drift, model, observability


def main() -> None:
    """Executa a Etapa 2 de ponta a ponta."""
    reference = pd.read_csv(config.DATA_REFERENCE)
    production = pd.read_csv(config.DATA_PRODUCTION)
    clf = model.load_model()

    with observability.track_stage("data_contract_production") as ctx:
        contract.enforce_contract(production)
        ctx["rows"] = len(production)
        ctx["contract_passed"] = True

    with observability.track_stage("predict") as ctx:
        reference["prediction"] = clf.predict_proba(reference[config.FEATURES])[:, 1]
        production["prediction"] = clf.predict_proba(production[config.FEATURES])[:, 1]
        ctx["rows"] = len(production)

    with observability.track_stage("drift_statistics") as ctx:
        table = drift.drift_table(reference, production)
        table.to_csv(config.REPORTS_DIR / "drift_table.csv", index=False)
        ctx["features_with_drift"] = int(table["drift"].sum())
        ctx["rows"] = len(production)

    with observability.track_stage("evidently_reports") as ctx:
        ev_psi = drift.run_evidently_report(
            reference, production, "psi", config.REPORTS_DIR / "evidently_drift_psi.html"
        )
        ev_ks = drift.run_evidently_report(
            reference, production, "ks", config.REPORTS_DIR / "evidently_drift_ks.html"
        )
        ctx["evidently_drifted_psi"] = ev_psi["_resumo"]["count"]
        ctx["evidently_drifted_ks"] = ev_ks["_resumo"]["count"]

    with observability.track_stage("performance_monitoring"):
        ref_metrics = observability.load_metrics()["model_reference"]
        prod_metrics = model.evaluate(clf, production[config.FEATURES], production[config.TARGET])
        concept = drift.concept_drift_summary(ref_metrics, prod_metrics)

    contract_rate = observability.load_metrics()["contract"]["bad_batch_failure_rate"]
    alerts = observability.build_alerts(table.to_dict(orient="records"), concept, contract_rate)
    for alert in alerts:
        level = logging.ERROR if alert["severity"] == "critical" else logging.WARNING
        observability.log_event(
            level,
            f"ALERTA {alert['type']}: {alert['message']}",
            stage="alert",
            status=alert["severity"],
            alert_type=alert["type"],
        )

    observability.update_metrics("model_production", prod_metrics)
    observability.update_metrics("concept_drift", concept)
    observability.update_metrics(
        "drift",
        {
            "n_features": len(table),
            "n_drifted": int(table["drift"].sum()),
            "drift_share": float(table["drift"].mean()),
            "evidently_drifted_psi": ev_psi["_resumo"]["count"],
            "evidently_drifted_ks": ev_ks["_resumo"]["count"],
            "features": table.to_dict(orient="records"),
        },
    )
    observability.update_metrics(
        "alerts",
        {"total": len(alerts), "critical": sum(a["severity"] == "critical" for a in alerts), "items": alerts},
    )
    observability.log_to_mlflow(
        "etapa2_drift",
        params={
            "psi_warning": config.PSI_WARNING,
            "psi_critical": config.PSI_CRITICAL,
            "n_production": len(production),
        },
        metrics={
            **{f"prod_{k}": v for k, v in prod_metrics.items()},
            **{f"concept_{k}": v for k, v in concept.items()},
            "n_drifted_features": int(table["drift"].sum()),
            "n_alerts": len(alerts),
        },
        artifacts=[
            config.REPORTS_DIR / "evidently_drift_psi.html",
            config.REPORTS_DIR / "evidently_drift_ks.html",
            config.REPORTS_DIR / "drift_table.csv",
        ],
        tags={"stage": "drift"},
    )

    pd.set_option("display.width", 200)
    print("\n=== Drift por feature (PSI + KS) ===")
    cols = [
        "feature",
        "media_referencia",
        "media_producao",
        "variacao_pct",
        "psi",
        "psi_nivel",
        "ks_p_value",
        "drift",
    ]
    print(table[cols].round(4).to_string(index=False))
    print("\n=== Desempenho: referência x produção ===")
    for key in ("auc", "ks", "recall", "brier", "mean_predicted_default", "observed_default_rate"):
        print(f"  {key:<24} ref={ref_metrics[key]:.4f}  prod={prod_metrics[key]:.4f}")
    print(f"\n=== {len(alerts)} alertas gerados ===")
    for alert in alerts:
        print(f"  [{alert['severity'].upper():<8}] {alert['type']:<14} {alert['message']}")


if __name__ == "__main__":
    main()
