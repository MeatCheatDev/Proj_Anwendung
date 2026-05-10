import re
from typing import Any

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import pandas as pd
import shap  # type: ignore[import-untyped]
import streamlit as st
from pathlib import Path

from src.counterfactual import format_counterfactual_for_display, run_app_counterfactual
from src.services.risk_service import evaluate_uq, prepare_patient_data
from src.shap_analysis import run_app_shap
from src.uq_variance import calculate_uq
from src.mapping import CP_MAP, SEX_MAP

FEATURE_COLUMNS: list[str] = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang']

# BACKEND & DATEN LADEN
@st.cache_resource
def lade_backend() -> tuple[Any, Any, pd.DataFrame, pd.Series]:  # type: ignore[type-arg]

    # Lädt Modell, SHAP-Explainer und Trainingsdaten.

    base_dir: Path = Path(__file__).resolve().parent
    model_path: Path = base_dir / 'models' / 'heart_model.joblib'
    artifact: dict[str, Any] = joblib.load(model_path)

    return (
        artifact['model'],
        shap.TreeExplainer(artifact['model']),
        artifact['x_train'],
        artifact['y_train'],
    )

model, shap_explainer, x_train, y_train = lade_backend()



# SEITENKONFIGURATION
st.set_page_config(
    page_title="Herzkrankheit Risiko-Analyse",
    page_icon="🫀",
    layout="wide"

)

# NAVIGATIONSBAR

st.markdown("""
<style>
    /* Navbar Container - STICKY mit höherer Priorität */
    .navbar-container {
        background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%) !important;
        padding: 18px !important;
        border-radius: 12px !important;
        margin-bottom: 32px !important;
        margin-left: auto !important;      /* ← neu */
        margin-right: auto !important;     /* ← neu */
        box-shadow: 0 4px 12px rgba(25, 118, 210, 0.2) !important;
        position: sticky !important;
        width: 65% !important;
    }

    /* Navbar Inhalt */
    .navbar-content {
        display: flex;
        justify-content: space-around;
        gap: 16px;
        flex-wrap: wrap;
    }

    /* Navbar Links */
    .nav-link {
        color: white !important;
        font-weight: 600;
        cursor: pointer;
        text-decoration: none !important;
        font-size: 30px;
        padding: 12px 20px;
        border-radius: 8px;
        transition: all 0.3s ease;
        user-select: none;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .nav-link:hover {
        background-color: rgba(255, 255, 255, 0.25);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .nav-link:active {
        transform: translateY(0);
    }

    /* Primary Button Styling - Blau wie die Navbar */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%) !important;
        color: white !important;
        border: none !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%) !important;
    }
</style>

<div class="navbar-container">
    <div class="navbar-content">
        <a href="#informationen" class="nav-link">ℹ️ Informationen</a>
        <a href="#risikorechner" class="nav-link">🩺 Risikorechner</a>
        <a href="#praevention" class="nav-link">💪 Prävention</a>
    </div>
</div>
""", unsafe_allow_html=True)

#1: EINFÜHRUNG
st.markdown('<h2 id="informationen"></h2>', unsafe_allow_html=True)
st.header("ℹ️ Was bringt diese Seite?")

intro_col1, intro_col2 = st.columns([2, 1])

with intro_col1:
    st.markdown("""
    Diese Anwendung nutzt künstliche Intelligenz und Datenwissenschaft, um Ihr persönliches Risiko 
    für Herzkrankheiten zu analysieren. Basierend auf wissenschaftlichen Studien und medizinischen 
    Parametern berechnet unser **Random Forest Machine Learning Modell** eine individuelle 
    Risikoeinschätzung – speziell für Sie.

    **Das bekommen Sie in dieser App:**
    - 📈 Eine prozentuale Risikoeinschätzung basierend auf Ihren Vitaldaten
    - 🎯 Aussagen zur Sicherheit und Zuverlässigkeit der Prognose (Konfidenz-Score)
    - 🔍 Detaillierte Einsicht, welche Faktoren Ihr Risiko am meisten beeinflussen (SHAP-Analyse)
    - 💡 Konkrete Empfehlungen zur Risikoreduktion

    Nutzen Sie diese Informationen, um mit Ihrem Arzt oder Ihrer Ärztin ein gezielteres Gespräch 
    zu führen und ggf. präventive Maßnahmen einzuleiten.
    """)

with intro_col2:
    pass

st.warning("""
**⚠️ Wichtiger Disclaimer:** Diese App wurde zu Demonstrationszwecken erstellt. Bei Fragen zu Ihrer Herzgesundheit konsultieren Sie bitte einen Arzt. Handeln Sie nicht ohne medizinische Fachberatung.
""")

st.divider()



# 2: RECHNER - Eingabe-Formular
st.markdown('<h2 id="risikorechner"></h2>', unsafe_allow_html=True)
st.title("🫀 Risikorechner – Ihre Vitaldaten")
st.divider()

col1, col2 = st.columns(2)

with col1:
    age: int | float | None = st.slider("Alter", 18, 100, 50)
    sex_input: str | None = st.selectbox("Geschlecht", list(SEX_MAP.values()))
    options = list(CP_MAP.values())

    cp_input = st.selectbox(
        "Brustschmerzen",
        options,
        index=len(options) - 1  # Auswahl auf das letzte Element (Keine Brustschmerzen) - Setzt default Wert.
    )
    trestbps: int | float | None = st.slider("Systolischer Ruhe-Blutdruck (Sys | mmHg)", 90, 200, 120)

with col2:
    chol: int | float | None = st.number_input("Cholesterin (mg/dl)", 100, 400, 200)
    thalach: int = st.slider("Max. Herzfrequenz", 60, 220, 150)
    fbs_input: str | None = st.radio("Nüchternblutzucker über 120 mg/dl?", ["Nein", "Ja"])
    exang_input: str | None = st.radio("Belastungsangina?", ["Nein", "Ja"])

# 3. AUSWERTUNG
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
        st.subheader("KI-Verlässlichkeit")

        getattr(st, uq_bewertung.status_color)(f"**{uq_bewertung.titel}**")
        st.caption(uq_bewertung.beschreibung)

        st.write(
            f"<span style='font-size: 0.8em; color: gray;'>"
            f"Technische Modell-Varianz: {uq_bewertung.varianz:.4f} / max. 0.25</span>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("💡 Was musst du ändern?")
        st.write(
            f"<span style='font-size: 0.8em; color: gray;'>"
            f"DiCE</span>",
            unsafe_allow_html=True,
        )

        if echtes_risiko < 0.25:
            st.success("✅ Dein Risikoscore ist bereits sehr niedrig – alles tippi toppi! Keine Änderungen notwendig.")
        else:
            with st.spinner("Berechne Lebensstil-Änderungen..."):
                cf_df: pd.DataFrame = run_app_counterfactual(model, x_train, y_train, user_df)
                st.dataframe(
                    format_counterfactual_for_display(cf_df),
                    hide_index=True,
                    use_container_width=True,
                )

            # Erklärung für SHAP
            with st.expander("ℹ️ Wie lese ich diese Tabelle?"):
                st.markdown("""
                Um die Diagnose "Gesund" zu bekommen, müssen die Werte aus der Tabelle eingegeben werden.
                """)


    with res_col2:

        #Bug-fix, damit die Schrift nicht Schwarz wird. Leider ein Streamlit Bug:
        plt.style.use("dark_background")

        st.subheader("🔍 Warum ist das so?")
        st.write(
            "<p style='font-size: 0.8em; color: gray; margin-top: 1rem;'>SHAP</p>",
            unsafe_allow_html=True,
        )

        st.markdown("### Die wichtigsten Einflussfaktoren:")
        fig, erklaerungen = run_app_shap(shap_explainer, user_df)
        for satz in erklaerungen:
            satz_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', satz)
            st.markdown(
                f"<p style='font-size: 1.15rem; margin: 0.2rem 0;'>· {satz_html}</p>",
                unsafe_allow_html=True,
            )
        st.write(
            "</br></br>",
            unsafe_allow_html=True,
        )


        # SHAP Skizze
        plt.style.use("dark_background")
        fig.patch.set_facecolor('#0e1117')
        fig.set_size_inches(5, 3.5)  # Kleiner als vorher
        st.pyplot(fig, use_container_width=False)
        plt.clf()

        # Erklärung für SHAP
        with st.expander("ℹ️ Wie lese ich diesen Chart?"):
            st.markdown("""
            - **f(x) = 0.XX** ist der Risiko-Score des Modells für diesen Patienten (0 = gesund, 1 = krank)
            - **E[f(X)] = 0.XXX** ist der Durchschnittsscore über alle Trainingspatienten
            - 🔴 **Rote Balken** treiben den Score nach oben → erhöhen das Risiko
            - 🔵 **Blaue Balken** ziehen den Score nach unten → senken das Risiko
            - Die Länge zeigt, wie stark der Einfluss ist
            """)

# PRÄVENTION - Wie man Herzprobleme vorbeugen kann

st.markdown('<h2 id="praevention"></h2>', unsafe_allow_html=True)
st.header("💪 Herzgesundheit fördern – 4 Säulen der Prävention")

st.markdown("""
Neben der Risikoanalyse ist die Prävention essentiell. Diese vier Faktoren reduzieren Ihr 
Risiko für Herzkrankheiten nachweislich und signifikant:
""")


prev_col1, prev_col2, prev_col3, prev_col4 = st.columns(4, gap="medium")

with prev_col1:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #1976D2;
        box-shadow: 0 2px 8px rgba(25, 118, 210, 0.15);
        min-height: 350px;
        display: flex;
        flex-direction: column;
        font-family: 'Segoe UI', sans-serif;
    ">
        <h3 style="color: #1565C0; margin: 0 0 12px 0; font-size: 18px;">🏃 Bewegung</h3>
        <p style="color: #333; margin: 12px 0; font-size: 14px; line-height: 1.6; flex-grow: 1;">
            <strong>150 Minuten</strong> moderate Ausdaueraktivität pro Woche 
            reduzieren das Herzinfarkt-Risiko um bis zu <strong>35%</strong>.
        </p>
        <small style="color: #666; margin-top: 12px;">
            <em>Quelle: American Heart Association</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

with prev_col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #F3E5F5 0%, #E1BEE7 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #7B1FA2;
        box-shadow: 0 2px 8px rgba(123, 31, 162, 0.15);
        min-height: 350px;
        display: flex;
        flex-direction: column;
        font-family: 'Segoe UI', sans-serif;
    ">
        <h3 style="color: #6A1B9A; margin: 0 0 12px 0; font-size: 18px;">🍎 Ernährung</h3>
        <p style="color: #333; margin: 12px 0; font-size: 14px; line-height: 1.6; flex-grow: 1;">
            Eine <strong>mediterrane Diät</strong> mit viel Fisch, Gemüse und 
            ungesättigten Fetten senkt das Herzrisiko nachweislich.
        </p>
        <small style="color: #666; margin-top: 12px;">
            <em>Quelle: PREDIMED-Studie (2013)</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

with prev_col3:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #FCE4EC 0%, #F8BBD0 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #C2185B;
        box-shadow: 0 2px 8px rgba(194, 24, 91, 0.15);
        min-height: 350px;
        display: flex;
        flex-direction: column;
        font-family: 'Segoe UI', sans-serif;
    ">
        <h3 style="color: #AD1457; margin: 0 0 12px 0; font-size: 18px;">😴 Schlaf</h3>
        <p style="color: #333; margin: 12px 0; font-size: 14px; line-height: 1.6; flex-grow: 1;">
            <strong>7–9 Stunden</strong> Schlaf pro Nacht sind essentiell. 
            Chronischer Schlafmangel erhöht das Infarktrisiko um <strong>48%</strong>.
        </p>
        <small style="color: #666; margin-top: 12px;">
            <em>Quelle: Journal of the American College of Cardiology</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

with prev_col4:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #388E3C;
        box-shadow: 0 2px 8px rgba(56, 142, 60, 0.15);
        min-height: 350px;
        display: flex;
        flex-direction: column;
        font-family: 'Segoe UI', sans-serif;
    ">
        <h3 style="color: #2E7D32; margin: 0 0 12px 0; font-size: 18px;">🧘 Stressmanagement</h3>
        <p style="color: #333; margin: 12px 0; font-size: 14px; line-height: 1.6; flex-grow: 1;">
            Chronischer Stress verdoppelt das Herzinfarktrisiko. 
            Meditation und <strong>Atemübungen</strong> helfen nachweislich.
        </p>
        <small style="color: #666; margin-top: 12px;">
            <em>Quelle: European Heart Journal (2017)</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

# FOOTER

st.divider()

st.markdown("""
<div style="text-align: center; padding: 20px; color: #999; font-size: 12px; font-style: italic;">
    Made by Fabian Mayer, Nico Then, Raffael Wellmann
</div>
""", unsafe_allow_html=True)