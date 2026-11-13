"""Geração dos datasets de Referência, Produção (com drift) e lote corrompido.

Os dados são sintéticos, inspirados no German Credit Data. O dataset de
Referência representa o mundo em que o modelo foi treinado; o de Produção
simula a passagem do tempo sob inflação e aumento do desemprego.
"""

import numpy as np
import pandas as pd

from credit import config
from credit.pii import pseudonymize


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Função sigmoide.

    Args:
        x: Array de entrada.

    Returns:
        Valores no intervalo (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))


def _base_features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Gera as features de um mundo "normal" (distribuição de referência).

    Args:
        n: Número de clientes.
        rng: Gerador pseudoaleatório.

    Returns:
        DataFrame com as features numéricas e ``cliente_id`` pseudonimizado.
    """
    renda = np.clip(rng.lognormal(mean=8.2, sigma=0.45, size=n), 1500, 40000)
    valor = np.clip(rng.lognormal(mean=8.6, sigma=0.6, size=n), 500, 60000)
    prazo = rng.choice([6, 12, 18, 24, 36, 48], size=n, p=[0.1, 0.25, 0.2, 0.25, 0.15, 0.05])
    return pd.DataFrame(
        {
            "cliente_id": [pseudonymize(f"cpf-{i}-{rng.integers(1e9)}") for i in range(n)],
            "idade": np.clip(rng.normal(38, 11, n), 19, 80).round().astype(int),
            "renda_mensal": renda.round(2),
            "taxa_desemprego_regional": np.clip(rng.normal(6.5, 1.2, n), 2, 15).round(2),
            "valor_emprestimo": valor.round(2),
            "prazo_meses": prazo,
            "tempo_emprego_anos": np.clip(rng.gamma(2.5, 2.2, n), 0, 40).round(1),
            "num_dividas_ativas": np.clip(rng.poisson(1.4, n), 0, 10),
            "comprometimento_renda": np.clip((valor / prazo) / renda, 0.01, 1.5).round(4),
        }
    )


def _default_probability(df: pd.DataFrame, desemprego_weight: float, renda_weight: float) -> np.ndarray:
    """Calcula a probabilidade real de inadimplência (o "mundo verdadeiro").

    Args:
        df: Features dos clientes.
        desemprego_weight: Peso do desemprego regional no risco.
        renda_weight: Peso (negativo) da renda no risco.

    Returns:
        Array com a probabilidade de inadimplência por cliente.
    """
    logit = (
        -2.2
        + 2.8 * df["comprometimento_renda"]
        + 0.35 * df["num_dividas_ativas"]
        + desemprego_weight * (df["taxa_desemprego_regional"] - 6.5)
        + renda_weight * (np.log(df["renda_mensal"]) - 8.2)
        - 0.04 * (df["idade"] - 38)
        - 0.05 * df["tempo_emprego_anos"]
    )
    return _sigmoid(logit.to_numpy())


def generate_reference(n: int = config.N_REFERENCE, seed: int = config.SEED) -> pd.DataFrame:
    """Gera o dataset de Referência (usado no treino do modelo).

    Args:
        n: Número de linhas.
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame limpo, com a coluna alvo ``inadimplente``.
    """
    rng = np.random.default_rng(seed)
    df = _base_features(n, rng)
    prob = _default_probability(df, desemprego_weight=0.25, renda_weight=-0.9)
    df[config.TARGET] = (rng.uniform(size=n) < prob).astype(int)
    return df


def generate_production(n: int = config.N_PRODUCTION, seed: int = config.SEED + 1) -> pd.DataFrame:
    """Gera o dataset de Produção, com Data Drift e Concept Drift.

    Data Drift (mudança em P(X)):
        * inflação reduz a renda real dos clientes (~-25%);
        * desemprego regional sobe (~+2,5 p.p.);
        * valor dos empréstimos sobe (~+30%), elevando o comprometimento.

    Concept Drift (mudança em P(y|X)):
        * o desemprego passa a pesar muito mais no risco e a renda
          protege menos — a relação aprendida pelo modelo envelhece.

    Args:
        n: Número de linhas.
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame de produção com a coluna alvo (rótulos observados depois).
    """
    rng = np.random.default_rng(seed)
    df = _base_features(n, rng)
    df["renda_mensal"] = (df["renda_mensal"] * rng.normal(0.75, 0.05, n)).clip(1200).round(2)
    df["taxa_desemprego_regional"] = (df["taxa_desemprego_regional"] + 2.5).clip(upper=18).round(2)
    df["valor_emprestimo"] = (df["valor_emprestimo"] * 1.3).round(2)
    df["comprometimento_renda"] = np.clip(
        (df["valor_emprestimo"] / df["prazo_meses"]) / df["renda_mensal"], 0.01, 1.5
    ).round(4)
    prob = _default_probability(df, desemprego_weight=0.22, renda_weight=-0.7)
    df[config.TARGET] = (rng.uniform(size=n) < prob).astype(int)
    return df


def generate_bad_batch(seed: int = config.SEED + 2) -> pd.DataFrame:
    """Gera um lote com erros propositais para demonstrar o contrato falhando.

    Erros injetados: menores de idade, renda nula, renda negativa, duplicatas,
    desemprego impossível, prazo inválido e alvo fora de {0, 1}.

    Args:
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame de 200 linhas com anomalias controladas.
    """
    rng = np.random.default_rng(seed)
    df = generate_reference(n=200, seed=seed)
    df.loc[0:9, "idade"] = rng.integers(10, 18, 10)  # menores de idade
    df.loc[10:19, "renda_mensal"] = np.nan  # renda nula
    df.loc[20:24, "renda_mensal"] = -500.0  # renda negativa
    df.loc[30:34, "taxa_desemprego_regional"] = 250.0  # fora do range
    df.loc[40:42, "prazo_meses"] = 0  # prazo inválido
    df.loc[50:51, config.TARGET] = 7  # alvo inválido
    return pd.concat([df, df.iloc[60:70]], ignore_index=True)  # duplicatas de cliente_id
