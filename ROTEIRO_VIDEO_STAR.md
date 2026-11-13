# Roteiro do vídeo (máx. 5 min) – método STAR

**[0:00–0:30] Abertura** – "Sou Gabriel, e vou mostrar como monitoramos e governamos um modelo de
crédito para uma fintech, do contrato de dados à LGPD." Mostre o repositório no GitHub.

## S – Situação (0:30–1:00)
"Modelos de crédito perdem qualidade quando a economia muda. A fintech precisa detectar isso rápido
e provar conformidade com a LGPD." Mostre o diagrama do README.

## T – Tarefa (1:00–1:30)
"Meu objetivo: (1) barrar dados ruins, (2) detectar drift de dados e conceito, (3) dar visibilidade
operacional e (4) documentar governança."

## A – Ação (1:30–4:00)
1. **Contrato (1:30–2:15)** – Rode `python scripts\run_all.py`. Mostre: referência passou; lote ruim
   teve **55 de 210 linhas reprovadas (26,2%)** e foi bloqueado em 7 regras (idade 15, renda negativa,
   desemprego 250, alvo 7, ID duplicado).
2. **Drift (2:15–3:15)** – Abra `reports\evidently_drift_psi.html`. "Desemprego PSI **3,55**, renda
   **0,38**, comprometimento **0,32**, valor **0,18**; idade e demais ficaram estáveis." Mostre AUC
   **0,714 → 0,686** e Brier **0,138 → 0,221**. Seja honesto: "o drift de conceito é modesto; o recall
   subiu só porque os scores subiram."
3. **Observabilidade (3:15–3:45)** – Abra `reports\dashboard.html`: etapas, latência, linhas, **6 alertas
   (4 críticos)**. Cite logs JSON, MLflow e `metrics.prom`.
4. **Governança (3:45–4:00)** – README: PII bloqueada pelo contrato, pseudonimização, base legal art. 7º
   V e X, revisão humana art. 20, retenção, análise causal e viés.

## R – Resultado (4:00–4:50)
"Dados inválidos são barrados antes do modelo; drift detectado em 4 de 8 variáveis com PSI, KS e
Evidently; alertas automáticos; 10 testes passando e CI no GitHub. Recomendação: **retreinar e recalibrar**."

## Fechamento (4:50–5:00)
"Obrigado! Código, relatórios e README estão no repositório."

**Dicas:** grave tela + voz, deixe o terminal em fonte grande, ensaie para caber em 5 min.
