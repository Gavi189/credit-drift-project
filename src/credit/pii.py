"""Tratamento de dados pessoais (LGPD): pseudonimização e minimização."""

import hashlib
import os

import pandas as pd

# Colunas que NUNCA podem chegar ao modelo ou aos relatórios.
PII_COLUMNS = ("cpf", "nome", "email", "telefone", "endereco")


def pseudonymize(value: str, salt: str | None = None) -> str:
    """Gera um identificador pseudonimizado (SHA-256 com salt) a partir de um valor.

    Args:
        value: Identificador original (ex.: CPF).
        salt: Segredo usado no hash; por padrão lido de ``PII_SALT``.

    Returns:
        Hash hexadecimal truncado em 16 caracteres.
    """
    salt = salt if salt is not None else os.getenv("PII_SALT", "dev-only-salt")
    digest = hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    return digest[:16]


def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas de PII direta (princípio da minimização, LGPD art. 6º, III).

    Args:
        df: DataFrame possivelmente contendo PII.

    Returns:
        Cópia do DataFrame sem as colunas listadas em ``PII_COLUMNS``.
    """
    return df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
