"""Contrato de dados (Data Contract) com Pandera.

O contrato é a primeira linha de defesa: lotes que violam as regras são
bloqueados antes de chegar ao modelo, e as linhas rejeitadas vão para
quarentena com o motivo registrado.
"""

from dataclasses import dataclass, field

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from credit import config

CONTRACT = pa.DataFrameSchema(
    columns={
        config.ID_COLUMN: pa.Column(str, unique=True, nullable=False),  # R1: sem duplicatas
        "idade": pa.Column(int, pa.Check.in_range(18, 100), nullable=False),  # R2: maior de idade
        "renda_mensal": pa.Column(  # R3: renda obrigatória e positiva
            float, [pa.Check.gt(0), pa.Check.le(200_000)], nullable=False
        ),
        "taxa_desemprego_regional": pa.Column(float, pa.Check.in_range(0, 30), nullable=False),
        "valor_emprestimo": pa.Column(float, pa.Check.in_range(100, 500_000), nullable=False),
        "prazo_meses": pa.Column(int, pa.Check.isin([6, 12, 18, 24, 36, 48]), nullable=False),
        "tempo_emprego_anos": pa.Column(float, pa.Check.in_range(0, 60), nullable=False),
        "num_dividas_ativas": pa.Column(int, pa.Check.in_range(0, 50), nullable=False),
        "comprometimento_renda": pa.Column(float, pa.Check.in_range(0, 2), nullable=False),
        config.TARGET: pa.Column(int, pa.Check.isin([0, 1]), nullable=False, required=False),
    },
    strict=True,  # R4: coluna inesperada (ex.: CPF, nome) bloqueia o lote — minimização LGPD
    coerce=False,
    name="credit_scoring_contract_v1",
)

RULES_DOC = {
    "R1": "cliente_id único (sem duplicatas)",
    "R2": "idade entre 18 e 100 anos",
    "R3": "renda_mensal não nula, > 0 e <= 200.000",
    "R4": "nenhuma coluna fora do contrato (bloqueia PII acidental)",
    "R5": "taxa_desemprego_regional entre 0 e 30",
    "R6": "prazo_meses ∈ {6, 12, 18, 24, 36, 48}",
    "R7": "inadimplente ∈ {0, 1}",
}


class DataContractError(Exception):
    """Lote de dados rejeitado pelo contrato."""


@dataclass
class ValidationReport:
    """Resultado da validação de um lote contra o contrato."""

    passed: bool
    n_rows: int
    n_failed_rows: int = 0
    failures: list[dict] = field(default_factory=list)
    failed_index: list[int] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Fração de linhas reprovadas no lote."""
        return self.n_failed_rows / self.n_rows if self.n_rows else 0.0


def validate_batch(df: pd.DataFrame) -> ValidationReport:
    """Valida um lote contra o contrato, coletando TODAS as violações.

    Args:
        df: Lote de dados a validar.

    Returns:
        ValidationReport com as violações agrupadas por coluna e regra.
    """
    try:
        CONTRACT.validate(df, lazy=True)
        return ValidationReport(passed=True, n_rows=len(df))
    except SchemaErrors as exc:
        cases = exc.failure_cases
        grouped = (
            cases.groupby(["column", "check"], dropna=False)
            .agg(n=("failure_case", "size"), exemplo=("failure_case", "first"))
            .reset_index()
        )
        failed_idx = sorted({int(i) for i in cases["index"].dropna().tolist()})
        return ValidationReport(
            passed=False,
            n_rows=len(df),
            n_failed_rows=len(failed_idx),
            failures=grouped.astype(str).to_dict(orient="records"),
            failed_index=failed_idx,
        )


def enforce_contract(df: pd.DataFrame, quarantine_path=config.DATA_QUARANTINE_LOG) -> pd.DataFrame:
    """Bloqueia a ingestão se o lote violar o contrato.

    Linhas reprovadas são gravadas em quarentena e uma ``DataContractError``
    é levantada — o lote NÃO segue para o modelo.

    Args:
        df: Lote a ingerir.
        quarantine_path: CSV onde as linhas rejeitadas são guardadas.

    Returns:
        O mesmo DataFrame, se (e somente se) o lote estiver íntegro.

    Raises:
        DataContractError: Se qualquer regra do contrato for violada.
    """
    report = validate_batch(df)
    if report.passed:
        return df
    if report.failed_index:
        valid_idx = [i for i in report.failed_index if i in df.index]
        df.loc[valid_idx].to_csv(quarantine_path, index=False)
    raise DataContractError(
        f"Lote rejeitado: {report.n_failed_rows}/{report.n_rows} linhas violam o contrato "
        f"({len(report.failures)} tipos de violação)."
    )
