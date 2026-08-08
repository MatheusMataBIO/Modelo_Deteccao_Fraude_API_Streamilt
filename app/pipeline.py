"""
Pipeline de inferência do sistema de detecção de fraude two-stage.

Reproduz, para UMA transação por vez, exatamente o que os notebooks
02-05 fizeram em batch:
  1. Feature engineering (encoding cíclico de Hour, RobustScaler em Amount)
  2. Stage 1 (Isolation Forest) — filtro de anomalia com threshold fixo
  3. Stage 2 (LightGBM) — classificação, apenas se passou pelo Stage 1
  4. Threshold de negócio — decisão final (bloquear / aprovar)
"""

import json
import numpy as np
import joblib
import shap
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"

# Ordem exata das 28 componentes PCA, conforme usada no treino
V_FEATURES = [f"V{i}" for i in range(1, 29)]

# Ordem exata das features de entrada do modelo (deve bater com o
# notebook 02: V1-V28, Hour_sin, Hour_cos, Amount_scaled)
FEATURE_ORDER = V_FEATURES + ["Hour_sin", "Hour_cos", "Amount_scaled"]


class FraudDetectionPipeline:
    """Carrega os artefatos uma única vez e expõe um método de predição."""

    def __init__(self):
        self.scaler = joblib.load(MODELS_DIR / "robust_scaler_amount.pkl")
        self.iso_forest = joblib.load(MODELS_DIR / "isolation_forest_stage1.pkl")
        self.lgbm = joblib.load(MODELS_DIR / "lightgbm_stage2.pkl")

        with open(MODELS_DIR / "stage1_metadata.json") as f:
            stage1_meta = json.load(f)
        with open(MODELS_DIR / "cost_benefit_metadata.json") as f:
            cost_meta = json.load(f)

        self.anomaly_threshold = float(stage1_meta["anomaly_score_threshold_producao"])
        self.business_threshold = float(cost_meta["threshold_otimo"])

        # TreeExplainer é construído uma única vez (no boot da API),
        # em vez de recriado a cada chamada de /explain — evita custo
        # repetido de inicialização a cada requisição
        self.shap_explainer = shap.TreeExplainer(self.lgbm)

    def _build_features(self, transaction: dict) -> np.ndarray:
        """Aplica feature engineering (idêntico ao notebook 02) a uma transação bruta."""
        hour = (transaction["Time"] / 3600) % 24
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        amount_scaled = self.scaler.transform([[transaction["Amount"]]])[0, 0]

        valores_v = [transaction[v] for v in V_FEATURES]
        vetor = valores_v + [hour_sin, hour_cos, amount_scaled]
        return np.array(vetor).reshape(1, -1)

    def predict(self, transaction: dict) -> dict:
        """
        Executa o pipeline completo para uma transação.
        Retorna score(s), decisão final e qual estágio decidiu.
        """
        X = self._build_features(transaction)

        # --- Stage 1: Isolation Forest ---
        anomaly_score = float(-self.iso_forest.score_samples(X)[0])
        passou_stage1 = anomaly_score >= self.anomaly_threshold

        if not passou_stage1:
            # Transação considerada normal pelo Stage 1 — aprovada
            # automaticamente, nunca chega ao Stage 2 (mesmo
            # comportamento reproduzido na avaliação do notebook 06)
            return {
                "anomaly_score": anomaly_score,
                "passou_stage1": False,
                "fraud_score": 0.0,
                "decision": "approved",
                "decided_by": "stage1",
            }

        # --- Stage 2: LightGBM ---
        fraud_score = float(self.lgbm.predict_proba(X)[0, 1])
        decision = "blocked" if fraud_score >= self.business_threshold else "approved"

        return {
            "anomaly_score": anomaly_score,
            "passou_stage1": True,
            "fraud_score": fraud_score,
            "decision": decision,
            "decided_by": "stage2",
        }


# Instância única, carregada uma vez quando o processo da API sobe
pipeline = FraudDetectionPipeline()
