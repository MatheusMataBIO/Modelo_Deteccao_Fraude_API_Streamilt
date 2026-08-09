# Dashboard — Detecção de Fraude (Streamlit)

Dashboard interativo com três abas:

1. **Visão de Negócio** — threshold de decisão dinâmico, aplicado em tempo real
   sobre o conjunto de teste (recall, precision, custo total, curva de custo).
2. **Simulação Individual** — formulário que consome a API (`/predict`, `/explain`)
   para uma transação específica, incluindo explicação SHAP.
3. **Monitoramento** — snapshot do resultado de CSI/PSI do notebook 07.

## Configuração necessária antes de rodar

**1. Adicionar o arquivo de dados**

Copie `test_scores_with_amount.parquet` (gerado no notebook
`modelo_fraude_analise_custo-beneficio_threshold_de_decisao`) para:

```
dashboard/data/test_scores_with_amount.parquet
```

**2. Configurar a URL da API**

Crie o arquivo `.streamlit/secrets.toml` (não versionado no Git) com:

```toml
API_URL = "https://modelo-deteccao-fraude-api-streamilt.onrender.com"
```

No Streamlit Community Cloud, essa mesma configuração é feita pela interface web,
em **App settings → Secrets**, com o mesmo conteúdo acima.

## Rodar localmente

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Deploy no Streamlit Community Cloud

1. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com GitHub
2. **New app** → selecione o repositório e a branch `main`
3. **Main file path**: `dashboard/app.py`
4. Em **Advanced settings → Secrets**, cole o conteúdo do `secrets.toml` acima
5. **Deploy**

> Nota: como a API roda no plano gratuito do Render, a primeira chamada após um
> período de inatividade pode levar até ~1 minuto para responder (o serviço
> "acorda"). Isso é esperado e está sinalizado na aba de simulação individual.
