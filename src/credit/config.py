"""Constantes e caminhos centrais do projeto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_REFERENCE = ROOT / "data" / "reference" / "reference.csv"
DATA_PRODUCTION = ROOT / "data" / "production" / "production.csv"
DATA_BAD_BATCH = ROOT / "data" / "quarantine" / "bad_batch.csv"
DATA_QUARANTINE_LOG = ROOT / "data" / "quarantine" / "rejected_rows.csv"

MODEL_PATH = ROOT / "models" / "credit_model.joblib"
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = ROOT / "logs"
PIPELINE_LOG = LOGS_DIR / "pipeline.jsonl"
METRICS_PATH = REPORTS_DIR / "pipeline_metrics.json"

SEED = 42
DECISION_THRESHOLD = 0.25  # ponto de corte de política de crédito (classe desbalanceada)
N_REFERENCE = 6000
N_PRODUCTION = 3000

TARGET = "inadimplente"
ID_COLUMN = "cliente_id"
FEATURES = [
    "idade",
    "renda_mensal",
    "taxa_desemprego_regional",
    "valor_emprestimo",
    "prazo_meses",
    "tempo_emprego_anos",
    "num_dividas_ativas",
    "comprometimento_renda",
]
NUMERIC_DRIFT_FEATURES = FEATURES  # todas numéricas

# Limiares de drift (convenções de mercado para PSI)
PSI_WARNING = 0.10
PSI_CRITICAL = 0.25
KS_P_VALUE = 0.05
