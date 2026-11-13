"""Etapa 3 - Gera o dashboard HTML de observabilidade e metricas Prometheus."""

import html
import json
import logging
from collections import defaultdict

from credit import config
from credit.observability import load_metrics, log_event, read_log_events, update_metrics


def _bar(value: float, vmax: float, color: str) -> str:
    pct = min(100.0, 100.0 * value / vmax) if vmax else 0.0
    return f'<div class="bar"><span style="width:{pct:.1f}%;background:{color}"></span></div>'


def _stage_summary(events: list[dict]) -> list[dict]:
    """Ultimo status, duracao e linhas por etapa do pipeline."""
    stages: dict[str, dict] = defaultdict(dict)
    for e in events:
        if e.get("stage") and e.get("status") in ("success", "failed"):
            stages[e["stage"]] = e
    return list(stages.values())


def build_prometheus(metrics: dict, stages: list[dict]) -> str:
    """Formato texto Prometheus (integravel ao stack da Fase 3)."""
    lines = []
    for s in stages:
        name = s["stage"]
        ok = 1 if s["status"] == "success" else 0
        lines.append(f'pipeline_stage_success{{stage="{name}"}} {ok}')
        lines.append(f'pipeline_stage_duration_seconds{{stage="{name}"}} {s.get("duration_s", 0)}')
        if "rows" in s:
            lines.append(f'pipeline_rows_processed{{stage="{name}"}} {s["rows"]}')
    alerts = metrics.get("alerts", {})
    lines.append(f"pipeline_alerts_total {alerts.get('total', 0)}")
    lines.append(f"pipeline_alerts_critical {alerts.get('critical', 0)}")
    for f in metrics.get("drift", {}).get("features", []):
        lines.append(f'feature_psi{{feature="{f["feature"]}"}} {f["psi"]:.4f}')
        lines.append(f'feature_ks_statistic{{feature="{f["feature"]}"}} {f["ks_stat"]:.4f}')
    lines.append(f"model_auc_drop {metrics.get('concept_drift', {}).get('auc_drop', 0):.4f}")
    return "\n".join(lines) + "\n"


def build_html(metrics: dict, stages: list[dict], events: list[dict]) -> str:
    """Monta o HTML autocontido (CSS inline, sem CDN)."""
    drift = metrics.get("drift", {})
    alerts = metrics.get("alerts", {})
    contract = metrics.get("contract", {})
    ref, prod = metrics.get("model_reference", {}), metrics.get("model_production", {})
    total_rows = sum(s.get("rows", 0) for s in stages)
    total_time = sum(s.get("duration_s", 0) for s in stages)
    n_ok = sum(1 for s in stages if s["status"] == "success")
    health = "SAUDÁVEL" if n_ok == len(stages) else "DEGRADADO"
    if alerts.get("critical", 0):
        health += " · COM ALERTAS CRÍTICOS"

    stage_rows = "".join(
        f"<tr><td>{html.escape(s['stage'])}</td>"
        f"<td class='{'ok' if s['status'] == 'success' else 'bad'}'>{s['status']}</td>"
        f"<td>{s.get('duration_s', 0):.3f}s</td><td>{s.get('rows', '-')}</td></tr>"
        for s in stages
    )
    colors = {"estavel": "#2a9d8f", "atencao": "#e9c46a", "critico": "#d62828"}
    psi_max = max([f["psi"] for f in drift.get("features", [])] + [0.5])
    drift_rows = "".join(
        f"<tr><td>{f['feature']}</td><td>{f['psi']:.3f}</td>"
        f"<td>{_bar(f['psi'], psi_max, colors.get(f['psi_nivel'], '#888'))}</td>"
        f"<td>{html.escape(f['psi_nivel'])}</td><td>{f['ks_stat']:.3f}</td>"
        f"<td>{f['variacao_pct']:+.1f}%</td></tr>"
        for f in drift.get("features", [])
    )
    perf_rows = "".join(
        f"<tr><td>{k}</td><td>{ref.get(k, 0):.4f}</td><td>{prod.get(k, 0):.4f}</td>"
        f"<td>{prod.get(k, 0) - ref.get(k, 0):+.4f}</td></tr>"
        for k in (
            "auc",
            "ks",
            "recall",
            "precision",
            "brier",
            "mean_predicted_default",
            "observed_default_rate",
        )
    )
    alert_rows = "".join(
        f"<tr><td class='{'bad' if a['severity'] == 'critical' else 'warn'}'>{a['severity']}</td>"
        f"<td>{a['type']}</td><td>{html.escape(a['message'])}</td></tr>"
        for a in alerts.get("items", [])
    )
    viol_rows = "".join(
        f"<tr><td>{html.escape(str(v['column']))}</td><td>{html.escape(str(v['check']))}</td><td>{v['n']}</td></tr>"
        for v in contract.get("violations", [])
    )
    recent = "".join(
        f"<li><code>{e['ts'][11:19]}</code> [{e['level']}] {html.escape(e['message'])}</li>"
        for e in events[-12:]
    )
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Observabilidade - Credit Scoring</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f4f6f8;color:#1d2733}}
header{{background:#14213d;color:#fff;padding:20px 32px}} main{{padding:24px 32px;max-width:1200px;margin:auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#fff;border-radius:8px;padding:14px;box-shadow:0 1px 3px #0002}}
.card b{{display:block;font-size:26px}} section{{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:18px;box-shadow:0 1px 3px #0002}}
table{{width:100%;border-collapse:collapse;font-size:14px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid #e5e8eb}}
.ok{{color:#2a9d8f;font-weight:600}} .bad{{color:#d62828;font-weight:600}} .warn{{color:#b8860b;font-weight:600}}
.bar{{background:#e5e8eb;height:12px;border-radius:6px;width:180px}} .bar span{{display:block;height:12px;border-radius:6px}}
li{{font-size:13px;margin:3px 0}}</style></head><body>
<header><h1 style="margin:0">Observabilidade do Pipeline · Credit Scoring</h1>
<div>Status geral: <b>{health}</b> · atualizado em {html.escape(str(metrics.get("updated_at", "")))}</div></header><main>
<div class="cards">
<div class="card">Etapas OK<b>{n_ok}/{len(stages)}</b></div>
<div class="card">Linhas processadas<b>{total_rows:,}</b></div>
<div class="card">Latência total<b>{total_time:.2f}s</b></div>
<div class="card">Alertas (críticos)<b>{alerts.get("total", 0)} ({alerts.get("critical", 0)})</b></div>
<div class="card">Features com drift<b>{drift.get("n_drifted", 0)}/{drift.get("n_features", 0)}</b></div></div>
<section><h2>Saúde operacional por etapa</h2><table><tr><th>Etapa</th><th>Status</th><th>Latência</th><th>Linhas</th></tr>{stage_rows}</table></section>
<section><h2>Drift de dados (PSI e KS)</h2><table><tr><th>Feature</th><th>PSI</th><th></th><th>Nível</th><th>KS</th><th>Δ média</th></tr>{drift_rows}</table></section>
<section><h2>Desempenho: referência vs produção</h2><table><tr><th>Métrica</th><th>Referência</th><th>Produção</th><th>Δ</th></tr>{perf_rows}</table></section>
<section><h2>Alertas</h2><table><tr><th>Severidade</th><th>Tipo</th><th>Mensagem</th></tr>{alert_rows}</table></section>
<section><h2>Contrato de dados: lote ruim bloqueado ({contract.get("bad_batch_failed_rows", 0)}/{contract.get("bad_batch_rows", 0)} linhas)</h2>
<table><tr><th>Coluna</th><th>Regra violada</th><th>Qtd</th></tr>{viol_rows}</table></section>
<section><h2>Últimos logs (logs/pipeline.jsonl)</h2><ul>{recent}</ul></section>
</main></body></html>"""


def main() -> None:
    """Gera dashboard.html e metrics.prom."""
    metrics = load_metrics()
    events = read_log_events()
    stages = _stage_summary(events)
    (config.REPORTS_DIR / "dashboard.html").write_text(build_html(metrics, stages, events), encoding="utf-8")
    (config.REPORTS_DIR / "metrics.prom").write_text(build_prometheus(metrics, stages), encoding="utf-8")
    update_metrics(
        "stages",
        {s["stage"]: {k: s.get(k) for k in ("status", "duration_s", "rows")} for s in stages},
    )
    log_event(logging.INFO, "dashboard gerado", stage="build_dashboard", status="success")
    print(f"Dashboard: {config.REPORTS_DIR / 'dashboard.html'}")
    print(json.dumps({"etapas": len(stages), "alertas": metrics["alerts"]["total"]}))


if __name__ == "__main__":
    main()
