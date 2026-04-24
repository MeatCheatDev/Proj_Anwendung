from typing import Any

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import pandas as pd
import shap  # type: ignore[import-untyped]
import streamlit as st
from pathlib import Path

from src.counterfactual import run_app_counterfactual
from src.services.risk_service import evaluate_uq, prepare_patient_data
from src.shap_analysis import run_app_shap
from src.uq_variance import calculate_uq

FEATURE_COLUMNS: list[str] = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang']

# ==========================================
# 1. PHASE 0: BACKEND & DATEN LADEN
# ==========================================

@st.cache_resource
def lade_backend() -> tuple[Any, Any, pd.DataFrame, pd.Series]:  # type: ignore[type-arg]
    """Lädt Modell, SHAP-Explainer und Trainingsdaten."""
    base_dir: Path = Path(__file__).resolve().parent
    model_path: Path = base_dir / 'models' / 'heart_model.joblib'
    artifact: dict[str, Any] = joblib.load(model_path)

    return (
        artifact['model'],
        shap.TreeExplainer(artifact['model']),
        artifact['x_train'],
        artifact['y_train'],
    )

st.set_page_config(page_title="Herzkrankheit Risiko", layout="wide")

model, shap_explainer, x_train, y_train = lade_backend()

# ==========================================
# 2. UI (Die Slider)
# ==========================================
st.title("🫀 Echte KI-Analyse")
st.divider()

col1, col2 = st.columns(2)
with col1:
    age: int = st.slider("Alter", 18, 100, 50)
    sex_input: str | None = st.selectbox("Geschlecht", ["Männlich", "Weiblich"])
    cp_input: str | None = st.selectbox("Brustschmerzen", [
        "Typische Angina (Schwer)", "Atypische Angina", "Nicht-anginöser Schmerz", "Keine Beschwerden"
    ])
    trestbps: int | float | None = st.number_input("Blutdruck", 90, 200, 120)

with col2:
    chol: int | float | None = st.number_input("Cholesterin", 100, 400, 200)
    thalach: int = st.slider("Max. Herzfrequenz", 60, 220, 150)
    fbs_input: str | None = st.radio("Blutzucker > 120?", ["Nein", "Ja"])
    exang_input: str | None = st.radio("Belastungsangina?", ["Nein", "Ja"])

# ==========================================
# 3. AUSWERTUNG
# ==========================================
if st.button("Risiko auswerten 🚀", type="primary", use_container_width=True):
    # Daten über Service aufbereiten
    user_df: pd.DataFrame = prepare_patient_data(
        age,
        str(sex_input),
        str(cp_input),
        int(trestbps),
        int(chol),
        str(fbs_input),
        thalach,
        str(exang_input),
    )

    st.success("Werte erfasst! Die KI rechnet...")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("📊 Diagnose")

        # Vorhersage
        echtes_risiko: float = model.predict_proba(user_df)[0][0]
        st.metric("Berechnetes Risiko", f"{echtes_risiko * 100:.1f} %")

        # UQ Bewertung über Service holen
        _, echte_uq = calculate_uq(model, user_df)
        uq_bewertung = evaluate_uq(echte_uq[0])

        st.markdown("---")
        st.subheader("🤖 KI-Verlässlichkeit")

        getattr(st, uq_bewertung.status_color)(f"**{uq_bewertung.titel}**")
        st.caption(uq_bewertung.beschreibung)

        st.write(
            f"<span style='font-size: 0.8em; color: gray;'>"
            f"Technische Modell-Varianz: {uq_bewertung.varianz:.4f} / max. 0.25</span>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("💡 Was musst du ändern? (DiCE)")

        with st.spinner("Berechne Lebensstil-Änderungen..."):
            cf_df: pd.DataFrame = run_app_counterfactual(model, x_train, y_train, user_df)
            st.dataframe(cf_df)

    with res_col2:
        st.subheader("🔍 Warum ist das so? (SHAP)")

        fig = run_app_shap(shap_explainer, user_df)
        st.pyplot(fig)
        plt.clf()