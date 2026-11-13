"""Observabilidade do pipeline: logs estruturados, métricas de saúde e alertas."""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime

from credit import config

_LOGGER_NAME = "credit.pipeline"


class JsonFormatter(logging.Formatter):
    """Formata cada log como uma linha JSON (fácil de ingerir em Datadog/ELK/Loki)."""

    def format(self, record: logging.LogRecord) -> str:
        """Converte o registro em JSON.

        Args:
            record: Registro de log.

        Returns:
            Linha JSON com timestamp, nível, estágio, status e campos extras.
        """
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_logger() -> logging.Logger:
    """Cria (uma única vez) o logger do pipeline com saída em arquivo e console.

    Returns:
        Logger configurado com ``JsonFormatter``.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(config.PIPELINE_LOG, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.propagate = False
    return logger


def log_event(level: int, message: str, **fields) -> None:
    """Registra um evento estruturado.

    Args:
        level: Nível do ``logging`` (INFO, WARNING, ERROR).
        message: Mensagem legível.
        **fields: Campos estruturados (stage, status, rows, duration_s...).
    """
    get_logger().log(level, message, extra={"fields": fields})


@contextmanager
def track_stage(stage: str, **fields):
    """Mede duração e registra sucesso/falha de uma etapa do pipeline.

    Args:
        stage: Nome da etapa (ex.: ``"data_contract"``).
        **fields: Campos extras anexados ao log.

    Yields:
        Dicionário mutável onde a etapa pode registrar contagens (``rows`` etc.).
    """
    context: dict = dict(fields)
    start = time.perf_counter()
    log_event(logging.INFO, f"{stage} iniciado", stage=stage, status="started", **context)
    try:
        yield context
    except Exception as exc:
        context["duration_s"] = round(time.perf_counter() - start, 3)
        log_event(
            logging.ERROR,
            f"{stage} FALHOU: {exc}",
            stage=stage,
            status="failed",
            error=type(exc).__name__,
            **context,
        )
        raise
    context["duration_s"] = round(time.perf_counter() - start, 3)
    log_event(logging.INFO, f"{stage} concluído", stage=stage, status="success", **context)


def read_log_events() -> list[dict]:
    """Lê todos os eventos já gravados em ``logs/pipeline.jsonl``.

    Returns:
        Lista de eventos (dicionários), do mais antigo ao mais recente.
    """
    if not config.PIPELINE_LOG.exists():
        return []
    lines = config.PIPELINE_LOG.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_metrics() -> dict:
    """Carrega o arquivo de métricas de saúde (ou um dicionário vazio).

    Returns:
        Métricas acumuladas pelas etapas do pipeline.
    """
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    return {}


def update_metrics(section: str, values: dict) -> None:
    """Atualiza uma seção do arquivo de métricas de saúde.

    Args:
        section: Nome da seção (``contract``, ``model``, ``drift``, ``alerts``...).
        values: Valores a gravar na seção.
    """
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    metrics[section] = values
    metrics["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def build_alerts(drift_rows: list[dict], concept: dict, contract_failure_rate: float) -> list[dict]:
    """Gera alertas a partir dos resultados de drift, desempenho e qualidade.

    Args:
        drift_rows: Linhas de ``drift_table`` como dicionários.
        concept: Resultado de ``concept_drift_summary``.
        contract_failure_rate: Fração de linhas reprovadas no último lote ruim.

    Returns:
        Lista de alertas com ``severity``, ``type`` e ``message``.
    """
    alerts: list[dict] = []
    for row in drift_rows:
        if row["psi_nivel"] == "critico":
            alerts.append(
                {
                    "severity": "critical",
                    "type": "data_drift",
                    "message": f"PSI crítico em {row['feature']}: {row['psi']:.2f}",
                }
            )
        elif row["psi_nivel"] == "atencao":
            alerts.append(
                {
                    "severity": "warning",
                    "type": "data_drift",
                    "message": f"PSI em atenção em {row['feature']}: {row['psi']:.2f}",
                }
            )
    if concept["auc_drop"] > 0.02:
        alerts.append(
            {
                "severity": "critical",
                "type": "concept_drift",
                "message": f"AUC caiu {concept['auc_drop']:.3f} em produção",
            }
        )
    if abs(concept["calibration_gap"]) > 0.05:
        alerts.append(
            {
                "severity": "critical",
                "type": "calibration",
                "message": f"Modelo subestima risco em {concept['calibration_gap'] * 100:.1f} p.p.",
            }
        )
    if contract_failure_rate > 0:
        alerts.append(
            {
                "severity": "warning",
                "type": "data_quality",
                "message": f"Lote bloqueado: {contract_failure_rate * 100:.1f}% linhas inválidas",
            }
        )
    return alerts


def log_to_mlflow(
    run_name: str, params: dict, metrics: dict, artifacts: list, tags: dict | None = None
) -> None:
    """Registra uma execução no MLflow (falha de tracking nunca derruba o pipeline).

    Args:
        run_name: Nome da run.
        params: Parâmetros (config, limiares, tamanhos de dataset).
        metrics: Métricas numéricas.
        artifacts: Caminhos de arquivos anexados (relatórios HTML, tabelas).
        tags: Tags livres (ex.: ``stage``).
    """
    import os

    import mlflow

    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}"))
        mlflow.set_experiment("credit-scoring-observability")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.set_tags(tags or {})
            for path in artifacts:
                mlflow.log_artifact(str(path))
        log_event(logging.INFO, f"MLflow: run '{run_name}' registrada", stage="mlflow", status="success")
    except Exception as exc:  # noqa: BLE001 - tracking é auxiliar
        log_event(logging.WARNING, f"MLflow indisponível: {exc}", stage="mlflow", status="skipped")
