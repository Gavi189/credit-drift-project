"""Etapa 1 — Contrato de dados + modelo baseline.

1. Gera e salva os datasets (Referência, Produção, lote corrompido).
2. Valida a Referência contra o contrato Pandera (deve passar).
3. Treina o classificador base com a Referência.
4. Tenta ingerir o lote corrompido: o contrato BLOQUEIA e envia linhas à quarentena.

Uso:
    python scripts/01_validate_and_train.py
"""

import logging

from credit import config, contract, data_gen, model, observability


def main() -> None:
    """Executa a Etapa 1 de ponta a ponta."""
    for path in (config.DATA_REFERENCE, config.DATA_PRODUCTION, config.DATA_BAD_BATCH):
        path.parent.mkdir(parents=True, exist_ok=True)

    with observability.track_stage("generate_datasets") as ctx:
        reference = data_gen.generate_reference()
        production = data_gen.generate_production()
        bad_batch = data_gen.generate_bad_batch()
        reference.to_csv(config.DATA_REFERENCE, index=False)
        production.to_csv(config.DATA_PRODUCTION, index=False)
        bad_batch.to_csv(config.DATA_BAD_BATCH, index=False)
        ctx["rows"] = len(reference) + len(production) + len(bad_batch)

    with observability.track_stage("data_contract_reference") as ctx:
        contract.enforce_contract(reference)
        ctx["rows"] = len(reference)
        ctx["contract_passed"] = True

    with observability.track_stage("train_baseline") as ctx:
        clf, ref_metrics = model.train_baseline(reference)
        model.save_model(clf)
        ctx["rows"] = len(reference)
        ctx["auc"] = round(ref_metrics["auc"], 4)

    # --- Demonstração: lote ruim é bloqueado pelo contrato ---
    blocked = False
    report = contract.validate_batch(bad_batch)
    try:
        with observability.track_stage("data_contract_bad_batch") as ctx:
            ctx["rows"] = len(bad_batch)
            contract.enforce_contract(bad_batch.copy())
    except contract.DataContractError as exc:
        blocked = True
        observability.log_event(
            logging.WARNING,
            f"INGESTÃO BLOQUEADA: {exc}",
            stage="ingestion",
            status="blocked",
            failed_rows=report.n_failed_rows,
        )

    print("\n=== Violações encontradas no lote corrompido ===")
    for failure in report.failures:
        print(f"  - {failure['column']:<26} {failure['check']:<28} n={failure['n']}")

    observability.update_metrics(
        "contract",
        {
            "reference_passed": True,
            "bad_batch_blocked": blocked,
            "bad_batch_rows": report.n_rows,
            "bad_batch_failed_rows": report.n_failed_rows,
            "bad_batch_failure_rate": report.failure_rate,
            "violations": report.failures,
            "rules": contract.RULES_DOC,
        },
    )
    observability.update_metrics("model_reference", ref_metrics)
    observability.log_to_mlflow(
        "etapa1_baseline",
        params={
            "model": "GradientBoostingClassifier",
            "n_reference": len(reference),
            "contract": contract.CONTRACT.name,
        },
        metrics={
            **{f"ref_{k}": v for k, v in ref_metrics.items()},
            "bad_batch_failure_rate": report.failure_rate,
        },
        artifacts=[config.METRICS_PATH],
        tags={"stage": "validation"},
    )
    print("\n=== Modelo baseline (holdout da Referência) ===")
    for key in ("auc", "ks", "accuracy", "recall", "f1", "brier"):
        print(f"  {key:<10} {ref_metrics[key]:.4f}")
    print(f"\nLote bloqueado pelo contrato: {blocked}")


if __name__ == "__main__":
    main()
