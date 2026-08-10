"""
Dashboard — Detecção de Fraude em Cartão de Crédito (Pipeline Two-Stage)

Três abas:
  1. Visão de Negócio     -> threshold dinâmico aplicado ao conjunto de teste
  2. Simulação Individual -> formulário que chama a API (/predict, /explain)
  3. Monitoramento        -> resultado estático de CSI/PSI (notebook 07)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------
# Configuração geral
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Detecção de Fraude — Pipeline Two-Stage",
    page_icon="🕵️",
    layout="wide",
)

API_URL = st.secrets.get("API_URL", "https://modelo-deteccao-fraude-api-streamilt.onrender.com")
CUSTO_ADMINISTRATIVO = 10.0
THRESHOLD_OTIMO_PADRAO = 0.18

DATA_PATH = Path(__file__).parent / "data" / "test_scores_with_amount.parquet"


@st.cache_data
def carregar_dados_teste():
    """Carrega os scores do conjunto de teste (produto do notebook 05)."""
    return pd.read_parquet(DATA_PATH)


def calcular_metricas(df: pd.DataFrame, threshold: float) -> dict:
    """Recalcula custo, recall, precision e matriz de confusão para um threshold."""
    y_true = df["Class"].values
    y_score = df["y_score_lgbm"].values
    amount = df["Amount"].values

    y_pred = (y_score >= threshold).astype(int)

    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)

    tp, fp, fn, tn = tp_mask.sum(), fp_mask.sum(), fn_mask.sum(), tn_mask.sum()

    custo_fn = amount[fn_mask].sum()
    custo_fp = fp * CUSTO_ADMINISTRATIVO
    custo_tp = tp * CUSTO_ADMINISTRATIVO
    custo_total = custo_fn + custo_fp + custo_tp

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    valor_total_fraude = amount[y_true == 1].sum()
    valor_recuperado = amount[tp_mask].sum()

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "recall": recall, "precision": precision,
        "custo_total": custo_total,
        "valor_total_fraude": valor_total_fraude,
        "valor_recuperado": valor_recuperado,
        "pct_recuperado": (valor_recuperado / valor_total_fraude * 100) if valor_total_fraude > 0 else 0,
    }


# ==================================================================
# CABEÇALHO
# ==================================================================
st.title("🕵️ Detecção de Fraude em Cartão de Crédito")
st.caption(
    "Pipeline two-stage — Isolation Forest (Stage 1) + LightGBM (Stage 2), "
    "com threshold de decisão otimizado por custo de negócio."
)

aba_negocio, aba_simulacao, aba_monitoramento = st.tabs(
    ["📊 Visão de Negócio", "🔎 Simulação Individual", "📈 Monitoramento"]
)

# ==================================================================
# ABA 1 — VISÃO DE NEGÓCIO (threshold dinâmico)
# ==================================================================
with aba_negocio:
    st.subheader("Impacto do threshold de decisão no resultado de negócio")
    st.write(
        "Ajuste o threshold abaixo e veja, em tempo real, como o recall, o custo "
        "total e o valor de fraude recuperado mudam. O valor **0,18** é o "
        "threshold ótimo, calculado por minimização de custo total (notebook 05)."
    )

    try:
        df_teste = carregar_dados_teste()

        col_slider, col_info = st.columns([3, 1])
        with col_slider:
            threshold = st.slider(
                "Threshold de decisão (fraud_score mínimo para bloquear)",
                min_value=0.01, max_value=0.99, value=THRESHOLD_OTIMO_PADRAO, step=0.01,
            )
        with col_info:
            if abs(threshold - THRESHOLD_OTIMO_PADRAO) < 0.005:
                st.success("Threshold ótimo ✅")
            else:
                st.info("Threshold customizado")

        metricas = calcular_metricas(df_teste, threshold)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall", f"{metricas['recall']*100:.1f}%")
        c2.metric("Precision", f"{metricas['precision']*100:.1f}%")
        c3.metric("Custo Total", f"€ {metricas['custo_total']:,.2f}")
        c4.metric("Valor Recuperado", f"{metricas['pct_recuperado']:.1f}%")

        st.markdown("##### Matriz de confusão")
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"Verdadeiros Positivos (fraude detectada): **{metricas['tp']}**")
            st.write(f"Falsos Negativos (fraude não detectada): **{metricas['fn']}**")
        with col_b:
            st.write(f"Falsos Positivos (bloqueio indevido): **{metricas['fp']}**")
            st.write(f"Verdadeiros Negativos (aprovação correta): **{metricas['tn']}**")

        # ----------------------------------------------------------------
        # Curva de custo total por threshold, com marcador no ponto atual
        # ----------------------------------------------------------------
        st.markdown("##### Curva de custo total por threshold")
        thresholds_curva = np.arange(0.01, 1.0, 0.01)
        custos_curva = [calcular_metricas(df_teste, t)["custo_total"] for t in thresholds_curva]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds_curva, y=custos_curva, mode="lines",
            name="Custo total", line=dict(color="steelblue", width=2)
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="tomato",
                       annotation_text=f"Threshold atual = {threshold:.2f}")
        fig.update_layout(
            xaxis_title="Threshold", yaxis_title="Custo total (€)",
            height=400, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width="stretch")

        st.caption(
            "Custo de Falso Negativo = valor (Amount) da transação fraudulenta. "
            "Custo de Falso Positivo e Verdadeiro Positivo = €10 (custo administrativo "
            "de revisão), conforme matriz *example-dependent cost-sensitive* "
            "(Bahnsen et al., 2014). Valores ilustrativos — ver limitações no README."
        )

    except FileNotFoundError:
        st.error(
            "Arquivo de dados não encontrado. Copie `test_scores_with_amount.parquet` "
            "(gerado no notebook 05) para a pasta `dashboard/data/` deste repositório."
        )

# ==================================================================
# ABA 2 — SIMULAÇÃO INDIVIDUAL (consome a API)
# ==================================================================
with aba_simulacao:
    st.subheader("Simular uma transação")
    st.write(
        "Preencha os dados de uma transação (ou use o exemplo pré-carregado, "
        "que é uma fraude real do dataset) e veja a decisão do pipeline completo."
    )

    exemplo = {
        "Time": 406.0, "Amount": 0.0,
        "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979,
        "V5": -0.5222, "V6": -1.4265, "V7": -2.5373, "V8": 1.3916,
        "V9": -2.7700, "V10": -2.7722, "V11": 3.2020, "V12": -2.8999,
        "V13": -0.5952, "V14": -4.2892, "V15": 0.3897, "V16": -1.1407,
        "V17": -2.8300, "V18": -0.0168, "V19": 0.4169, "V20": 0.1269,
        "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202,
        "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1433,
    }

    usar_exemplo = st.checkbox("Usar transação de exemplo (fraude conhecida)", value=True)

    with st.form("form_transacao"):
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time (segundos)", value=exemplo["Time"] if usar_exemplo else 0.0)
        with col2:
            amount_val = st.number_input("Amount", value=exemplo["Amount"] if usar_exemplo else 0.0, min_value=0.0)

        st.markdown("###### Componentes V1–V28")
        valores_v = {}
        cols = st.columns(4)
        for i in range(1, 29):
            chave = f"V{i}"
            with cols[(i - 1) % 4]:
                valores_v[chave] = st.number_input(
                    chave, value=exemplo[chave] if usar_exemplo else 0.0,
                    format="%.4f", key=f"input_{chave}",
                )

        col_btn1, col_btn2 = st.columns(2)
        submitted_predict = col_btn1.form_submit_button("🔮 Prever", width="stretch")
        submitted_explain = col_btn2.form_submit_button("🧠 Prever + Explicar (SHAP)", width="stretch")

    if submitted_predict or submitted_explain:
        payload = {"Time": time_val, "Amount": amount_val, **valores_v}
        endpoint = "/explain" if submitted_explain else "/predict"

        with st.spinner("Consultando a API... (pode levar até 1 min se o serviço estava inativo)"):
            try:
                resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=90)
                resp.raise_for_status()
                resultado = resp.json()

                decisao = resultado["decision"]
                cor = "🔴" if decisao == "blocked" else "🟢"
                st.markdown(f"### {cor} Decisão: **{'BLOQUEADA' if decisao == 'blocked' else 'APROVADA'}**")

                c1, c2, c3 = st.columns(3)
                c1.metric("Score de anomalia (Stage 1)", f"{resultado['anomaly_score']:.4f}")
                c2.metric("Passou ao Stage 2?", "Sim" if resultado["passou_stage1"] else "Não")
                c3.metric("Score de fraude (Stage 2)", f"{resultado['fraud_score']:.4f}")

                if resultado.get("shap_values"):
                    st.markdown("###### Contribuição por feature (SHAP)")
                    df_shap = pd.DataFrame(
                        resultado["shap_values"].items(), columns=["Feature", "Contribuição SHAP"]
                    ).sort_values("Contribuição SHAP", key=abs, ascending=False).head(15)

                    fig_shap = go.Figure(go.Bar(
                        x=df_shap["Contribuição SHAP"], y=df_shap["Feature"],
                        orientation="h",
                        marker_color=["tomato" if v > 0 else "steelblue" for v in df_shap["Contribuição SHAP"]],
                    ))
                    fig_shap.update_layout(
                        height=450, margin=dict(t=20, b=20),
                        xaxis_title="Contribuição para o score de fraude",
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_shap, width="stretch")

            except requests.exceptions.RequestException as exc:
                st.error(f"Erro ao consultar a API: {exc}")

# ==================================================================
# ABA 3 — MONITORAMENTO (resultado estático do notebook 07)
# ==================================================================
with aba_monitoramento:
    st.subheader("Estabilidade de Features e Score do Modelo (CSI / PSI)")
    st.write(
        "Resultado do monitoramento de drift simulado no notebook 07, usando "
        "janelas temporais do conjunto de teste. Valores abaixo de 0,10 indicam "
        "estabilidade; entre 0,10–0,25, alerta moderado; acima de 0,25, drift "
        "significativo."
    )

    # Snapshot estático dos resultados do notebook 07 (Evidently AI)
    dados_csi_psi = pd.DataFrame({
        "Feature": ["Amount_scaled", "V12", "V14", "V17", "V22", "V4", "V8", "score_lgbm (PSI)"],
        "Janela 1": [0.0088, 0.0059, 0.0069, 0.0054, 0.0072, 0.0063, 0.0019, 0.0081],
        "Janela 2": [0.0098, 0.0228, 0.0047, 0.0117, 0.0147, 0.0051, 0.0055, 0.0107],
        "Janela 3": [0.0085, 0.0297, 0.0044, 0.0087, 0.0205, 0.0209, 0.0096, 0.0073],
    })

    st.dataframe(
        dados_csi_psi.style.background_gradient(
            cmap="RdYlGn_r", subset=["Janela 1", "Janela 2", "Janela 3"], vmin=0, vmax=0.10
        ).format({"Janela 1": "{:.4f}", "Janela 2": "{:.4f}", "Janela 3": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )

    st.success(
        "✅ Nenhuma feature ou o score do modelo ultrapassou o limiar de alerta "
        "(0,10) nas janelas analisadas — resultado esperado dado o curto período "
        "coberto pelo dataset (~48h). Ver limitações metodológicas no README."
    )

    st.caption(
        "Este é um snapshot estático do notebook 07 — em um cenário de produção "
        "real, esse cálculo rodaria continuamente sobre dados novos."
    )
