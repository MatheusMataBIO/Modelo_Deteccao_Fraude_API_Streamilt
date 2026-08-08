"""
API de detecção de fraude em cartão de crédito — pipeline two-stage.

Endpoints:
  GET  /health    -> health-check
  POST /predict   -> aplica o pipeline completo a uma transação
  POST /explain   -> predição + explicação SHAP (contribuição por feature)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import TransactionInput, PredictionOutput
from app.pipeline import pipeline, FEATURE_ORDER

app = FastAPI(
    title="Fraud Detection API — Two-Stage Pipeline",
    description=(
        "API do projeto de detecção de fraude em cartão de crédito. "
        "Aplica um pipeline de duas etapas: Isolation Forest (Stage 1, "
        "filtro de anomalia) seguido de LightGBM (Stage 2, classificação), "
        "com decisão final via threshold de custo de negócio."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Confirma que a API está no ar e os artefatos foram carregados."""
    return {
        "status": "ok",
        "anomaly_threshold": pipeline.anomaly_threshold,
        "business_threshold": pipeline.business_threshold,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    """Aplica o pipeline completo (Stage 1 -> Stage 2 -> threshold) a uma transação."""
    try:
        resultado = pipeline.predict(transaction.model_dump())
        return resultado
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/explain")
def explain(transaction: TransactionInput):
    """
    Predição + explicação SHAP da contribuição de cada feature.

    Só gera explicação SHAP se a transação avançou ao Stage 2 (o Stage 1
    é um modelo não-supervisionado sem explicação SHAP direta neste projeto).
    """
    try:
        resultado = pipeline.predict(transaction.model_dump())

        if not resultado["passou_stage1"]:
            return {
                **resultado,
                "shap_values": None,
                "note": "Transação aprovada já no Stage 1 — não avaliada pelo LightGBM, sem explicação SHAP.",
            }

        X = pipeline._build_features(transaction.model_dump())
        shap_values = pipeline.shap_explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        contribuicoes = {
            feat: float(val) for feat, val in zip(FEATURE_ORDER, shap_values[0])
        }
        # Ordena por impacto absoluto, mais relevante primeiro
        contribuicoes_ordenadas = dict(
            sorted(contribuicoes.items(), key=lambda item: abs(item[1]), reverse=True)
        )

        return {**resultado, "shap_values": contribuicoes_ordenadas}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
