"""Testes do contrato de dados, drift, PII e observabilidade."""

import numpy as np
import pandas as pd
import pytest

from credit import config, data_gen, pii
from credit.contract import DataContractError, enforce_contract, validate_batch
from credit.drift import drift_table, psi, psi_level
from credit.observability import build_alerts


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    """Dataset de referencia valido."""
    return data_gen.generate_reference()


@pytest.fixture(scope="module")
def production() -> pd.DataFrame:
    """Dataset de producao com drift."""
    return data_gen.generate_production()


def test_reference_passes_contract(reference):
    """Dados validos devem passar no contrato."""
    assert validate_batch(reference).passed


def test_bad_batch_is_blocked(tmp_path):
    """Lote ruim deve ser bloqueado e enviado para quarentena."""
    bad = data_gen.generate_bad_batch()
    with pytest.raises(DataContractError):
        enforce_contract(bad, tmp_path / "quarantine.csv")


def test_bad_batch_has_at_least_three_rule_types():
    """O contrato deve ter 3+ regras duras violadas pelo lote ruim."""
    report = validate_batch(data_gen.generate_bad_batch())
    assert len({v["check"] for v in report.failures}) >= 3


def test_pii_column_is_rejected(reference):
    """Coluna de PII (cpf) nao pode passar pelo contrato (strict=True)."""
    df = reference.head(50).copy()
    df["cpf"] = "123.456.789-00"
    assert not validate_batch(df).passed


def test_psi_identical_distributions_is_near_zero():
    """PSI de amostras iguais deve ser ~0."""
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x.copy()) < 0.01


def test_psi_shifted_distribution_is_critical():
    """Deslocamento grande deve gerar PSI critico."""
    rng = np.random.default_rng(0)
    value = psi(rng.normal(0, 1, 5000), rng.normal(2, 1, 5000))
    assert value > config.PSI_CRITICAL
    assert psi_level(value) == "critico"


def test_drift_table_flags_only_altered_features(reference, production):
    """Features alteradas devem ter drift; controles nao."""
    rows = {r["feature"]: r for r in drift_table(reference, production).to_dict("records")}
    assert rows["taxa_desemprego_regional"]["drift"]
    assert rows["renda_mensal"]["drift"]
    assert not rows["idade"]["drift"]


def test_pseudonymize_is_deterministic_and_not_plain():
    """Pseudonimo e estavel e nao revela o CPF."""
    a = pii.pseudonymize("123.456.789-00", salt="s")
    assert a == pii.pseudonymize("123.456.789-00", salt="s")
    assert "123" not in a


def test_drop_pii_removes_columns():
    """drop_pii remove colunas sensiveis."""
    df = pd.DataFrame({"cpf": ["1"], "idade": [30]})
    assert list(pii.drop_pii(df).columns) == ["idade"]


def test_build_alerts_flags_critical_psi():
    """Alerta critico deve surgir para PSI alto."""
    rows = [{"feature": "x", "psi": 0.9, "psi_nivel": "critico"}]
    concept = {"auc_drop": 0.0, "calibration_gap": 0.0}
    assert any(a["severity"] == "critical" for a in build_alerts(rows, concept, 0.0))
