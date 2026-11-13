# Observabilidade e Governança de Modelo de Crédito (Tech Challenge FIAP – Fase 04)

Pipeline de **credit scoring** para uma fintech com: contrato de dados (Pandera), detecção de drift
(PSI, KS e Evidently AI), observabilidade (logs JSON, MLflow, dashboard, métricas Prometheus) e
governança (LGPD, análise causal, mitigação de viés).

## Como executar (Windows – PowerShell/CMD)

```
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts\run_all.py
pytest -q
ruff check .
```

Depois abra `reports\dashboard.html` e `reports\evidently_drift_psi.html` no navegador.
MLflow (opcional): `mlflow ui --backend-store-uri sqlite:///mlflow.db` e acesse http://127.0.0.1:5000.

## Estrutura

| Caminho | Função |
|---|---|
| `src/credit/contract.py` | Data contract Pandera (7 regras duras), quarentena |
| `src/credit/data_gen.py` | Geração de referência, produção (com drift) e lote ruim |
| `src/credit/model.py` | Baseline GradientBoosting + métricas |
| `src/credit/drift.py` | PSI, KS, drift conceitual, relatório Evidently |
| `src/credit/observability.py` | Logs JSON, métricas de saúde, alertas, MLflow |
| `src/credit/pii.py` | Pseudonimização e remoção de PII (LGPD) |
| `scripts/01_validate_and_train.py` | **Etapa 1** |
| `scripts/02_detect_drift.py` | **Etapa 2** |
| `scripts/03_build_dashboard.py` | **Etapa 3** |
| `scripts/run_all.py` | Executa 1→3 |
| `tests/` | 10 testes automatizados (pytest) · CI em `.github/workflows/ci.yml` |

## Etapa 1 – Contrato de dados e baseline

Regras duras (violá-las **bloqueia** a ingestão e envia as linhas para `data/quarantine/`):

| Regra | Descrição |
|---|---|
| R1 | `cliente_id` único |
| R2 | idade entre 18 e 100 |
| R3 | `renda_mensal` não nula, > 0 e ≤ 200.000 |
| R4 | nenhuma coluna fora do contrato (`strict=True`: bloqueia PII como CPF/nome) |
| R5 | `taxa_desemprego_regional` entre 0 e 30 |
| R6 | `prazo_meses` ∈ {6,12,18,24,36,48} |
| R7 | alvo `inadimplente` ∈ {0,1} |

**Prova:** o lote ruim (210 linhas) teve **55 linhas reprovadas (26,2%)** em 7 tipos de violação e foi
bloqueado. A referência (6.000 linhas) passou.

Baseline (GradientBoosting, limiar 0,25): AUC **0,714**, KS **0,331**, recall 0,455, Brier 0,138.

## Etapa 2 – Drift em produção

Produção (3.000 linhas) foi gerada com **data drift** em 4 variáveis (renda −25%, desemprego +2,5 p.p.,
valor do empréstimo +30%, comprometimento de renda recalculado) e **mudança leve de conceito**
(a relação variável→inadimplência muda). Quatro variáveis de controle permanecem estáveis.

| Feature | PSI | Nível | KS |
|---|---|---|---|
| taxa_desemprego_regional | 3,55 | crítico | 0,70 |
| renda_mensal | 0,38 | crítico | – |
| comprometimento_renda | 0,32 | crítico | – |
| valor_emprestimo | 0,18 | atenção | – |
| idade, tempo_emprego, dívidas, prazo | < 0,10 | estável | – |

Relatórios Evidently: `reports/evidently_drift_psi.html` e `reports/evidently_drift_ks.html`.
Tabela completa: `reports/drift_table.csv`.

| Métrica | Referência | Produção |
|---|---|---|
| AUC | 0,714 | 0,686 (−0,028) |
| KS | 0,331 | 0,283 (−0,047) |
| Brier | 0,138 | 0,221 |
| Inadimplência observada | 18,6% | 37,4% |
| Recall | 0,455 | 0,734 |

**Leitura honesta:** o drift de dados é forte e inequívoco. O drift de conceito é **modesto**
(AUC −0,028; calibração média preservada). O recall sobe porque os scores sobem com a renda menor e o
desemprego maior, não porque o modelo melhorou. Brier e acurácia pioram → recomenda-se retreino.

## Etapa 3 – Observabilidade

* **Logs centralizados:** `logs/pipeline.jsonl` (um JSON por evento: etapa, status, linhas, latência).
* **MLflow:** experimento `credit-scoring-observability` (runs `etapa1_baseline`, `etapa2_drift`).
* **Dashboard:** `reports/dashboard.html` (autocontido, sem CDN).
* **Métricas Prometheus:** `reports/metrics.prom` (integra com Prometheus/Grafana da Fase 3).

Métricas de saúde operacional documentadas:

| Métrica | Onde | Uso |
|---|---|---|
| Status por etapa (`pipeline_stage_success`) | logs + dashboard | pipeline funcionando? |
| Latência por etapa (`duration_s`) | logs + dashboard | gargalos / SLA |
| Volume de dados (`rows`) | logs + dashboard | quedas/picos de ingestão |
| Alertas total/críticos | `pipeline_metrics.json` | acionamento de equipe |
| PSI/KS por feature | `drift_table.csv`, `.prom` | drift de dados |
| `auc_drop`, `calibration_gap` | `pipeline_metrics.json` | drift de conceito |

Regras de alerta: PSI > 0,25 crítico; PSI 0,10–0,25 atenção; queda de AUC > 0,02; |gap de calibração|
> 0,05; lote bloqueado pelo contrato. Execução atual: **6 alertas (4 críticos)**.

## Etapa 4 – Governança

### 1. Plano de proteção de dados (LGPD – Lei 13.709/2018)

| Tema | Decisão |
|---|---|
| **PII tratada** | CPF, nome, e-mail, telefone, endereço. **Nunca entram** no dataset do modelo: o contrato (`strict=True`) rejeita essas colunas e `pii.drop_pii` as remove. |
| **Identificador** | `cliente_id` é pseudonimizado (SHA-256 com sal em variável de ambiente `PII_SALT`, fora do repositório). |
| **Minimização** (art. 6º, III) | Só 8 variáveis financeiras necessárias à análise de risco. Sem raça, religião, saúde, orientação ou dados sensíveis (art. 5º, II). |
| **Base legal da decisão de crédito** | Art. 7º, **V** (execução de contrato/procedimentos preliminares a pedido do titular) e art. 7º, **X** (proteção do crédito). Não se usa consentimento como base principal. |
| **Decisão automatizada** (art. 20) | O titular pode **solicitar revisão** por pessoa natural e explicação dos critérios. O score é insumo; casos limítrofes (score próximo ao limiar) vão para analista. |
| **Transparência** (art. 6º, VI) | Variáveis e lógica do modelo documentadas; fatores principais informáveis ao cliente. |
| **Retenção** | Dados de treino/monitoramento pseudonimizados: 5 anos (prazo regulatório de relações de crédito), depois eliminação ou anonimização (art. 15–16). Logs operacionais: 12 meses. Quarentena: 30 dias, depois exclusão. |
| **Direitos do titular** (art. 18) | Acesso, correção, eliminação, portabilidade e revisão atendidos via canal do Encarregado (DPO). |
| **Segurança** (art. 46) | Sal/segredos fora do Git (`.env`), acesso por perfil, logs sem PII, quarentena com acesso restrito. |
| **Governança** | Registro das operações (ROPA), Relatório de Impacto (RIPD, art. 38) para a decisão automatizada, nomeação do Encarregado (art. 41). |

### 2. Análise causal

Diagrama causal assumido (macroeconomia → comportamento → inadimplência):

```
Recessão/inflação ─▶ Desemprego regional ↑ ─▶ Renda mensal ↓ ─▶ Comprometimento de renda ↑ ─▶ Inadimplência ↑
                                  └──────────────────────────────────────────────────────────────▲
Política de crédito (valor, prazo) ───────────────▶ Comprometimento de renda
```

* **Data drift (P(X) muda):** choque macro deslocou desemprego, renda e comprometimento. É a causa
  *raiz* do PSI crítico; as variáveis de controle (idade, tempo de emprego) não se movem.
* **Concept drift (P(y|X) muda):** a relação variável→inadimplência enfraquece (AUC −0,028), pois o
  desemprego regional afeta clientes de modo diferente do que o treino aprendeu.
* **Correlação ≠ causalidade:** desemprego e inadimplência sobem juntos, mas o efeito passa pela renda
  (mediação). Variáveis omitidas (ex.: inflação do crédito) podem confundir; por isso o modelo é preditivo,
  **não** deve ser lido como "reduzir desemprego causa X% menos calotes" sem desenho causal.
* **Intervenções:** (1) retreinar com janela recente; (2) recalibrar probabilidades; (3) rever limiar de
  corte; (4) endurecer política de concessão (limitar comprometimento de renda); (5) monitoramento
  contínuo com gatilho de PSI > 0,25.

### 3. Mitigação de viés

* **Sem atributos protegidos** nas features (nada de sexo, raça, religião, endereço detalhado).
* **Variáveis proxy:** `taxa_desemprego_regional` pode agir como proxy de região/nível socioeconômico;
  monitorar impacto por faixa regional e renda.
* **Idade:** usada como risco financeiro legítimo (≥ 18, validada pelo contrato) e auditada por faixa
  etária para evitar discriminação indevida (ex.: idosos).
* **Auditoria periódica:** comparar taxa de aprovação, recall e falsos positivos por faixa de renda e idade
  (paridade de oportunidades); investigar disparidades > 20% (regra dos 4/5).
* **Humano no circuito** para casos limítrofes e contestações (art. 20 LGPD).
* **Dados representativos:** o drift de produção (população mais pobre/desempregada) mostra que o
  treino pode sub-representar cenários de crise; incluir esses cenários no retreino.

### 4. Limitações

Dados **sintéticos** (nenhuma pessoa real). O drift de conceito é intencionalmente moderado; num cenário
real a perda de AUC só seria confirmada quando os rótulos (inadimplência) amadurecessem.

## Histórico de commits

Segue **Conventional Commits** (`feat`, `test`, `docs`, `ci`, `chore`). Veja `git log --oneline`.
