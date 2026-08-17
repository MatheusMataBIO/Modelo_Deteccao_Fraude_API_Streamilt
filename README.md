# Detecção de Fraude em Cartão de Crédito — Pipeline Two-Stage

Sistema de detecção de fraude em transações de cartão de crédito, com arquitetura de
duas etapas (não-supervisionada + supervisionada), threshold de decisão otimizado por
custo de negócio, explicabilidade via SHAP, monitoramento de drift e deploy via API.

**Repositório:** [Modelo_Deteccao_Fraude_API_Streamlit](https://github.com/MatheusMataBIO/Modelo_Deteccao_Fraude_API_Streamilt)

**Demo ao vivo:**
- 🖥️ **Dashboard:** [modelodeteccaofraudeapistreamilt-uxkunfsvdutnquheulouwr.streamlit.app](https://modelodeteccaofraudeapistreamilt-uxkunfsvdutnquheulouwr.streamlit.app/)
- ⚙️ **API:** [modelo-deteccao-fraude-api-streamilt.onrender.com/docs](https://modelo-deteccao-fraude-api-streamilt.onrender.com/docs)

> A API roda no plano gratuito do Render e estiver inativa há um tempo, a
> primeira chamada pode levar até ~1 minuto para responder (o serviço "acorda").
> Isso afeta tanto o acesso direto à API quanto a aba de simulação individual
> do dashboard.

---

## Índice

- [O problema](#o-problema)
- [Por que uma arquitetura two-stage](#por-que-uma-arquitetura-two-stage)
- [Dataset](#dataset)
- [Arquitetura da solução](#arquitetura-da-solução)
- [Escolha das métricas](#escolha-das-métricas)
- [Metodologia e decisões técnicas](#metodologia-e-decisões-técnicas)
- [Resultados](#resultados)
- [Explicabilidade (SHAP)](#explicabilidade-shap)
- [Monitoramento de drift](#monitoramento-de-drift)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Como executar](#como-executar)
- [Deploy](#deploy)
- [Limitações conhecidas](#limitações-conhecidas)
- [Stack técnica](#stack-técnica)

---

## O problema

O mercado brasileiro de meios de pagamento movimentou R$ 4,7 trilhões em transações
com cartão em 2023, com perdas bilionárias anuais por fraude. O desafio técnico por
trás desse número é maior do que parece: em carteiras reais, fraudes representam
menos de 0,2% das transações.

Esse desbalanceamento extremo cria uma armadilha estatística. Um modelo treinado
nesse cenário aprende rapidamente que **aprovar todas as transações** já garante
99,8% de acurácia sem detectar uma única fraude. Métricas convencionais (Accuracy,
ROC-AUC) mascaram esse comportamento, fazendo um sistema inútil parecer excelente no
papel.

Outro erro comum em projetos desse tipo é usar **split aleatório** entre treino e
teste. Isso permite que o modelo "veja o futuro" durante o treinamento, inflando
artificialmente a performance reportada em produção, onde o modelo só tem acesso
ao passado, esse ganho desaparece.

Este projeto foi desenhado para evitar essas duas armadilhas desde a primeira
decisão de arquitetura.

---

## Por que uma arquitetura two-stage

A abordagem mais comum em tutoriais é treinar um único classificador supervisionado
sobre os dados desbalanceados — tem uma limitação estrutural: exemplos de fraude
rotulados são escassos (492 casos em 284 mil transações) e enviesados, já que só
capturam padrões de fraude **já identificados no passado**. Fraudadores mudam de
tática; um modelo treinado só nesses casos conhecidos tende a generalizar mal para
fraudes com padrão novo.

A solução adotada aqui divide o problema em duas etapas complementares:

**Stage 1 — Isolation Forest (não-supervisionado).**
Treinado **apenas com transações legítimas**, aprende o perfil do comportamento
"normal" e pontua o quão anômala é cada transação, sem depender de rótulos de
fraude. As transações mais suspeitas (top 28% por score de anomalia) avançam para a
próxima etapa; o restante é aprovado automaticamente. Isso reduz o volume analisado
pela etapa seguinte e concentra a proporção de fraude nesse subconjunto — de 0,18%
para cerca de 0,6%, uma melhora de mais de 3x.

**Stage 2 — LightGBM (supervisionado).**
Atuando apenas sobre o subconjunto filtrado, opera num cenário de desbalanceamento
muito menos severo, o que melhora sua capacidade de aprender fronteiras de decisão
relevantes. Os hiperparâmetros foram otimizados via **Optuna**, incluindo tratamento
explícito de desbalanceamento (`scale_pos_weight`).

O resultado é um funil: toda transação passa pelo Stage 1 (barato, rápido, sem
depender de rótulos); apenas a fração mais suspeita é avaliada pelo Stage 2 (mais
caro computacionalmente, mas com informação supervisionada).

---

## Dataset

**Credit Card Fraud Detection** — Kaggle (`mlg-ulb/creditcardfraud`)

- 284.807 transações realizadas por portadores de cartão europeus, em setembro de 2013
- 492 fraudes (0,172% do total)
- Features `V1`–`V28`: componentes principais (PCA) de variáveis originais
  confidenciais — sem interpretação direta de negócio
- `Time`: segundos desde a primeira transação do dataset
- `Amount`: valor da transação (moeda não especificada oficialmente pela fonte)
- `Class`: variável-alvo (0 = legítima, 1 = fraude)

Após remoção de 1.081 transações duplicadas (19 delas fraudes), o dataset final
utilizado contém 283.726 transações e 473 fraudes.

---

## Arquitetura da solução

```
Transação nova
      │
      ▼
┌─────────────────────┐
│  Feature Engineering │  RobustScaler (Amount) + encoding cíclico (Hour)
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  Stage 1             │  Isolation Forest — score de anomalia
│  (não-supervisionado) │  threshold fixo (top 28% do treino)
└─────────┬────────────┘
          │
   score < threshold          score ≥ threshold
          │                          │
          ▼                          ▼
   ┌─────────────┐          ┌──────────────────────┐
   │  Aprovada    │          │  Stage 2              │
   │  automática  │          │  (LightGBM, otimizado  │
   └─────────────┘          │  via Optuna)           │
                             └──────────┬─────────────┘
                                        ▼
                             ┌──────────────────────┐
                             │  Threshold de negócio  │
                             │  (0,18 — minimização    │
                             │  de custo total)        │
                             └──────────┬─────────────┘
                                        ▼
                              Bloqueada / Aprovada
```

**Decisões técnicas centrais:**

- **Split temporal** (não aleatório): treino com as transações mais antigas (80%),
  teste com as mais recentes (20%) replica a condição real de produção, em que o
  modelo nunca vê o futuro durante o treino.
- **RobustScaler em `Amount`**: robusto a outliers extremos de valor (mediana/IQR em
  vez de média/desvio padrão).
- **Encoding cíclico (seno/cosseno) de `Hour`**: preserva a proximidade real entre
  23h e 0h, que seriam tratadas como extremos opostos em uma representação linear.
- **Threshold de decisão via custo de negócio**, não fixado em 0,5 (ver seção de
  resultados).

---

## Escolha das métricas

| Categoria | Métricas | Papel |
|---|---|---|
| **Principais** | PR-AUC, Recall | Guiam todas as decisões de modelagem e otimização |
| **Secundárias** | Precision, F1-Score, FPR, matriz de confusão | Reportadas para comparação entre modelos |
| **Contraste pedagógico** | ROC-AUC, Accuracy | Reportadas apenas para evidenciar a armadilha do desbalanceamento — nunca usadas para decisão |
| **Negócio** | Custo total por threshold, valor recuperado, valor perdido | Guiam a escolha do ponto de operação final |
| **Monitoramento** | CSI (features), PSI (score do modelo) | Avaliam estabilidade ao longo do tempo |

Accuracy e ROC-AUC são mantidas no projeto exclusivamente para fins didáticos. Uma
demonstração feita com os dados reais deste projeto (notebook 6) mostra que a
diferença de Accuracy entre "aprovar todas as transações sem nenhum modelo" e o
pipeline completo é de apenas 0,06 pontos percentuais — apesar de um cenário não
detectar nenhuma fraude e o outro detectar quase 80% delas. Essa é a prova prática de
por que essas métricas nunca guiaram nenhuma decisão de arquitetura ou threshold
neste projeto.

---

## Metodologia e decisões técnicas

- **Comparação justa entre modelos**: antes de aplicar otimização de
  hiperparâmetros, todos os candidatos (LightGBM, XGBoost, Random Forest, Regressão
  Logística) foram avaliados com configuração padrão, para não favorecer
  artificialmente o modelo que recebeu mais esforço de tuning.
- **Otimização via Optuna** aplicada tanto ao LightGBM quanto ao Random Forest
  (o principal concorrente na comparação baseline), com validação cruzada
  estratificada — o Random Forest recebeu um espaço de busca mais restrito por
  limitação computacional, ressalva documentada explicitamente no projeto.
- **Matriz de custo *example-dependent cost-sensitive*** (Bahnsen et al., 2014):
  o custo de uma fraude não detectada é o valor (`Amount`) da própria transação; o
  custo de um bloqueio indevido ou de um acerto é um valor administrativo fixo,
  refletindo o custo operacional de revisar um caso sinalizado.
- **Explicabilidade via SHAP**, tanto em nível global (quais features mais pesam nas
  decisões do modelo) quanto individual (por que uma transação específica foi
  classificada de determinada forma).
- **Correção de rastreabilidade de dados**: durante o desenvolvimento, foi
  identificado que o salvamento de arquivos intermediários em Parquet
  (`index=False`) quebrava o vínculo entre transações processadas e o dataset
  original, comprometendo o cálculo de custo de negócio. A correção — introdução de
  um identificador explícito (`transaction_id`) propagado por todo o pipeline — está
  documentada no notebook 1 e nos ajustes subsequentes, como registro de rigor e
  transparência metodológica.

---

## Resultados

### Stage 1 — Isolation Forest

| Métrica | Valor |
|---|---|
| Corte de filtragem | Top 28% mais anômalas |
| Recall preservado (teste) | 95,95% |
| Redução de volume para o Stage 2 | ~72% |

### Stage 2 — LightGBM (otimizado via Optuna)

| Métrica | Valor (threshold 0,5) |
|---|---|
| PR-AUC | 0,8340 |
| Recall | 78,87% |
| Precision | 81,16% |

### Pipeline completo (ponta a ponta, threshold de negócio = 0,18)

| Métrica | Valor |
|---|---|
| PR-AUC | 0,8003 |
| **Recall** | **79,73%** |
| Precision | 70,24% |
| F1-Score | 0,7468 |
| Matriz de confusão (teste) | TP=59, FP=25, FN=15, TN=56.645 |

O recall final (79,73%) é a composição de perdas sequenciais entre as duas etapas
(95,95% × 83,1%) e reportar o recall isolado de qualquer etapa superestimaria a
performance real do sistema em produção.

### Impacto de negócio (threshold otimizado por custo)

| Cenário | Custo total | Redução vs. sem modelo |
|---|---|---|
| Sem modelo algum | € 7.478,08 | — |
| Threshold padrão (0,5) | ~€ 4.017 | ~46% |
| **Threshold otimizado (0,18)** | **€ 2.988,20** | **~60%** |

No ponto de operação escolhido, o pipeline recupera **71,3%** do valor total de
fraude presente no conjunto de teste (€ 5.329,88 de € 7.478,08), com apenas 25
bloqueios indevidos em mais de 15 mil transações analisadas pelo Stage 2.

> Os valores de custo administrativo usados na matriz de custo são estimativas
> ilustrativas de mercado — o dataset não especifica moeda oficialmente nem fornece
> parâmetros reais de uma operação. Os resultados demonstram a metodologia correta
> de otimização de threshold por custo de negócio, não uma validação financeira
> definitiva.

---

## Explicabilidade (SHAP)

A análise SHAP sobre o modelo final revelou que **V4** e **V8** concentram a maior
importância nas decisões do Stage 2 que é um resultado que diverge do ranking de
separabilidade univariada obtido na EDA (liderado por V17, V14, V12), evidenciando
que o LightGBM captura interações não-lineares entre variáveis que uma análise
univariada simples não seria capaz de detectar. V12 se manteve relevante em ambas as
análises.

Casos individuais de erro (um falso positivo e um falso negativo) foram analisados
em detalhe no notebook de avaliação, ilustrando tanto um erro "defensável" (transação
legítima com padrão estatístico atípico) quanto uma limitação real do modelo diante
de um perfil de fraude pouco representado no treino.

---

## Monitoramento de drift

Como o dataset cobre apenas ~48 horas de transações, o monitoramento de drift foi
implementado como uma **simulação metodológica**, dividindo o conjunto de teste em
janelas temporais sucessivas e comparando suas distribuições:

- **CSI** (Characteristic Stability Index): estabilidade das features de entrada
  mais relevantes ao modelo.
- **PSI** (Population Stability Index): estabilidade da distribuição do score de
  saída do modelo — também chamado de "drift de score".

Todos os valores observados permaneceram muito abaixo dos limiares usuais de alerta
(CSI e PSI < 0,03, frente a um limiar de atenção de 0,10), resultado esperado dado o
curto intervalo de tempo coberto pelo dataset. O cálculo foi validado com a
biblioteca **Evidently AI**.

> A aba de Monitoramento do dashboard exibe um **snapshot estático** desse
> resultado — não recalcula CSI/PSI em tempo real, já que o dataset não tem um
> fluxo contínuo de transações novas para comparar. Ver limitações abaixo.

---

## Estrutura do repositório

```
.
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── main.py            # Endpoints FastAPI (/health, /predict, /explain)
│   ├── pipeline.py         # Lógica do pipeline two-stage
│   └── schemas.py          # Validação de entrada/saída (Pydantic)
├── models/
│   ├── cost_benefit_metadata.json
│   ├── isolation_forest_stage1.pkl
│   ├── lightgbm_stage2.pkl
│   ├── robust_scaler_amount.pkl
│   └── stage1_metadata.json
├── dashboard/
│   ├── app.py
│   ├── requirements.txt
│   ├── README.md
│   └── data/
│       └── test_scores_with_amount.parquet
└── notebooks/
    ├── modelo_fraude_EDA.ipynb
    ├── modelo_fraude_feature_engineering.ipynb
    ├── modelo_fraude_detecção_anomalias_stage1.ipynb
    ├── modelo_fraude_classificador_stage2.ipynb
    ├── modelo_fraude_analise_custo-beneficio_threshold_de_decisao.ipynb
    ├── modelo_fraude_avaliação_explicabilidade.ipynb
    ├── modelo_fraude_monitoramento.ipynb
    └── modelo_fraude_preparacao_deploy.ipynb
```

### Notebooks

| Notebook | Conteúdo |
|---|---|
| `modelo_fraude_EDA` | Análise exploratória, validação de padrões, decisões técnicas iniciais |
| `modelo_fraude_feature_engineering` | Split temporal, encoding cíclico, RobustScaler |
| `modelo_fraude_detecção_anomalias_stage1` | Treinamento e seleção do Isolation Forest, curva de sensibilidade |
| `modelo_fraude_classificador_stage2` | Comparação de modelos, otimização via Optuna |
| `modelo_fraude_analise_custo-beneficio_threshold_de_decisao` | Matriz de custo, varredura de threshold |
| `modelo_fraude_avaliação_explicabilidade` | Avaliação ponta a ponta do pipeline, SHAP |
| `modelo_fraude_monitoramento` | Simulação de drift (CSI/PSI), Evidently AI |
| `modelo_fraude_preparacao_deploy` | Geração de artefatos fixos para uso em produção |

---

## Como executar

### Notebooks

Desenvolvidos e testados no **Google Colab** (camada gratuita), com artefatos
persistidos no Google Drive. Requer upload do dataset (`creditcard.csv`, via Kaggle
API) e execução sequencial dos notebooks, na ordem listada acima.

### API localmente

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

```bash
curl http://localhost:7860/health
```

### Via Docker

```bash
docker build -t fraud-api .
docker run -p 7860:7860 fraud-api
```

### Dashboard localmente

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy

- **API**: FastAPI + Docker, hospedada no [Render](https://render.com) (camada
  gratuita para Web Services com Docker).
- **Dashboard**: Streamlit, hospedado no Streamlit Community Cloud.

> Nota: o escopo original previa deploy no Hugging Face Spaces (SDK Docker). Essa
> opção deixou de ser gratuita durante o desenvolvimento do projeto (Docker e Gradio
> passaram a exigir plano pago), o que motivou a migração para Render + Streamlit
> Community Cloud, mantendo a arquitetura de API separada do dashboard.

---

## Limitações conhecidas

- A separabilidade das features neste dataset é atipicamente alta para um problema
  real de detecção de fraude — os resultados de performance não devem ser
  extrapolados diretamente para expectativas de produção real.
- Os valores de custo de negócio são estimativas ilustrativas, não validadas
  financeiramente com dados reais de operação.
- A comparação entre LightGBM e Random Forest não teve orçamento de otimização
  perfeitamente simétrico, por limitação computacional do ambiente gratuito.
- A ausência de identificador de cliente/cartão no dataset impede engenharia de
  features baseada em histórico (frequência de uso, ticket médio, etc.).
- O período coberto pelo dataset (~48 horas) é curto demais para uma análise de
  drift genuína — o notebook de monitoramento demonstra a metodologia correta, não
  evidência definitiva de estabilidade em produção real.
- A aba de Monitoramento do dashboard exibe um snapshot estático do resultado de
  CSI/PSI, não um recálculo dinâmico — não há fluxo de transações novas no dataset
  para alimentar um monitoramento contínuo real.
- A API roda em plano gratuito (Render), com "sleep" após períodos de inatividade —
  a primeira requisição após um tempo sem uso pode levar até ~1 minuto.

---

## Stack técnica

**Modelagem:** Python, scikit-learn (Isolation Forest, RobustScaler), LightGBM,
XGBoost, Optuna, SHAP

**Dados e visualização:** pandas, NumPy, Matplotlib, Seaborn, Plotly

**Monitoramento:** Evidently AI

**Deploy:** FastAPI, Docker, Render, Streamlit

**Ambiente de desenvolvimento:** Google Colab, Google Drive
