"""Streamlit-Oberfläche der Herzrisiko-Analyse (Präsentationsschicht).

Apple-Health-inspiriertes Dashboard mit eigenem Light-/Dark-Schalter. Die
fachliche Logik (Vorhersage, UQ, SHAP, DiCE) liegt vollständig in ``src/``.
"""

import math
import re
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import pandas as pd
import shap  # type: ignore[import-untyped]
import streamlit as st

from src.counterfactual import format_counterfactual_for_display, run_app_counterfactual
from src.services.risk_service import evaluate_uq, prepare_patient_data
from src.shap_analysis import run_app_shap
from src.uq_variance import calculate_uq
from src.mapping import CP_MAP, SEX_MAP

FEATURE_COLUMNS: list[str] = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang']

RISK_RECOMMENDATION_THRESHOLD = 0.25
RISK_HIGH_THRESHOLD = 0.50
ACCENT = "#FF375F"


@st.cache_resource
def lade_backend() -> tuple[Any, Any, pd.DataFrame, pd.Series]:  # type: ignore[type-arg]
    """Lädt Modell, SHAP-Explainer und Trainingsdaten (einmalig pro Session)."""
    base_dir: Path = Path(__file__).resolve().parent
    artifact: dict[str, Any] = joblib.load(base_dir / 'models' / 'heart_model.joblib')
    return (
        artifact['model'],
        shap.TreeExplainer(artifact['model']),
        artifact['x_train'],
        artifact['y_train'],
    )


model, shap_explainer, x_train, y_train = lade_backend()

st.set_page_config(page_title="Herz-Risiko", page_icon="🫀", layout="wide")


# ----------------------------------------------------------------------------
# Theme-Schalter (einzige Quelle der Wahrheit)
# ----------------------------------------------------------------------------
head_l, head_r = st.columns([3, 1])
with head_r:
    dark = st.toggle("Dark Mode", value=True)

if dark:
    C = dict(bg="#000000", card="#1C1C1E", card2="#2C2C2E", text="#F5F5F7", muted="#98989F",
             border="#38383A", input="#2C2C2E", track="#2C2C2E", shadow="none")
else:
    C = dict(bg="#F2F2F7", card="#FFFFFF", card2="#F2F2F7", text="#1C1C1E", muted="#8E8E93",
             border="#E3E3E8", input="#FFFFFF", track="#ECECF1", shadow="0 6px 22px rgba(0,0,0,0.07)")


def icon(body: str, color: str, size: int = 22, sw: float = 2.0) -> str:
    """Inline-SVG-Icon (Lucide-Stil, Stroke)."""
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round">{body}</svg>')


IC_HEART = '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.29 1.51 4.04 3 5.5l7 7Z"/>'
IC_ACTIVITY = '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'
IC_LEAF = '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6"/>'
IC_MOON = '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>'
IC_WIND = '<path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>'
IC_SHIELD = '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>'
IC_CHECK = '<path d="M20 6 9 17l-5-5"/>'


# ----------------------------------------------------------------------------
# Styling – CSS-Variablen treiben eigene UND native Elemente
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
    :root {{
        --bg:{C['bg']}; --card:{C['card']}; --card2:{C['card2']}; --text:{C['text']};
        --muted:{C['muted']}; --border:{C['border']}; --input:{C['input']};
        --track:{C['track']}; --shadow:{C['shadow']}; --accent:{ACCENT};
    }}
    .stApp {{ background: var(--bg) !important; }}
    [data-testid="stHeader"] {{ background: transparent !important; }}
    .block-container {{ padding-top: 2.4rem; max-width: 1060px;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", system-ui, sans-serif; }}

    /* Text */
    .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
    [data-testid="stWidgetLabel"] *, .stMarkdown, .stMarkdown * {{ color: var(--text); }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; }}
    h1,h2,h3,h4 {{ color: var(--text) !important; letter-spacing: -0.02em; }}

    /* Karten (Form + bordered container) */
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {{
        background: var(--card) !important; border: 1px solid var(--border) !important;
        border-radius: 24px !important; box-shadow: var(--shadow) !important; padding: 10px 8px; }}

    /* Selectbox */
    [data-baseweb="select"] > div {{ background: var(--input) !important; border-color: var(--border) !important; }}
    [data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] input {{ color: var(--text) !important; }}
    [data-baseweb="select"] svg {{ fill: var(--muted) !important; }}
    ul[role="listbox"], div[data-baseweb="menu"], [data-baseweb="popover"] ul,
    [data-baseweb="popover"] > div {{ background: var(--card) !important; }}
    li[role="option"], [role="listbox"] [role="option"] {{ background: var(--card) !important; }}
    li[role="option"] *, [role="option"] *, [role="listbox"] * {{ color: var(--text) !important; }}
    li[role="option"]:hover, [role="option"][aria-selected="true"] {{ background: var(--card2) !important; }}

    /* Text-/Number-Input */
    [data-baseweb="input"], [data-baseweb="base-input"] {{ background: var(--input) !important; border-color: var(--border) !important; }}
    [data-testid="stNumberInput"] input, .stTextInput input, input {{ background: var(--input) !important; color: var(--text) !important; }}
    [data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {{ background: var(--card2) !important; color: var(--text) !important; }}

    /* Radio */
    [data-testid="stRadio"] label, [data-testid="stRadio"] label * {{ color: var(--text) !important; }}

    /* Slider */
    [data-testid="stSlider"] [role="slider"] {{ background: var(--accent) !important; box-shadow: none !important; }}
    [data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ color: var(--muted) !important; }}

    /* Toggle */
    [data-testid="stToggle"] label, [data-testid="stToggle"] label * {{ color: var(--text) !important; }}

    /* Buttons */
    .stFormSubmitButton button, .stButton button[kind="primary"] {{
        background: var(--accent) !important; border: none !important; border-radius: 15px !important;
        font-weight: 600 !important; padding: 13px 0 !important; }}
    .stFormSubmitButton button *, .stButton button[kind="primary"] * {{ color: #FFFFFF !important; }}
    .stButton button[kind="secondary"] {{
        background: transparent !important; border: 1px solid var(--border) !important; border-radius: 15px !important;
        font-weight: 600 !important; padding: 11px 0 !important; }}
    .stButton button[kind="secondary"] * {{ color: var(--text) !important; }}
    .stButton button[kind="secondary"]:hover {{ background: var(--card2) !important; }}
    .stFormSubmitButton button:hover {{ filter: brightness(0.93); }}

    /* Expander + Dataframe */
    [data-testid="stExpander"] {{ background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 16px !important; }}
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: var(--text) !important; }}
    [data-testid="stDataFrame"] {{ background: var(--card) !important; }}

    /* Eigene Klassen */
    .hero {{ display:flex; align-items:center; gap:14px; }}
    .hero-ic {{ width:52px; height:52px; border-radius:16px; background:{ACCENT}1A;
        display:flex; align-items:center; justify-content:center; }}
    .hero-title {{ font-size:30px; font-weight:800; letter-spacing:-0.03em; margin:0; color:var(--text); }}
    .hero-sub {{ color:var(--muted); font-size:14.5px; margin:2px 0 0 0; }}
    .sec {{ font-size:22px; font-weight:800; letter-spacing:-0.02em; margin:6px 0 2px 0; color:var(--text); }}
    .sec-sub {{ color:var(--muted); font-size:14px; margin:0 0 14px 0; }}

    .tile {{ background:var(--card); border:1px solid var(--border); border-radius:22px;
        padding:20px 22px; box-shadow:var(--shadow); height:100%; }}
    .t-label {{ font-size:12px; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:0.05em; }}
    .t-value {{ font-size:28px; font-weight:800; color:var(--text); margin-top:4px; line-height:1.1; }}
    .t-sub {{ font-size:13px; color:var(--muted); margin-top:3px; }}

    .pill {{ display:inline-flex; align-items:center; gap:7px; padding:7px 15px; border-radius:999px; font-size:14px; font-weight:700; }}
    .dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}

    .srow {{ margin:13px 0; }}
    .stop {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }}
    .sname {{ font-size:14.5px; color:var(--text); font-weight:600; }}
    .smeta {{ color:var(--muted); font-size:12.5px; font-weight:400; }}
    .sval {{ font-size:13.5px; font-weight:800; }}
    .strack {{ position:relative; height:14px; }}
    .scenter {{ position:absolute; left:50%; top:0; width:2px; height:100%; background:var(--border); }}
    .sbar {{ position:absolute; top:3px; height:8px; border-radius:999px; }}
    .slegend {{ display:flex; gap:16px; color:var(--muted); font-size:12px; margin-top:8px; }}

    .chart-card {{ background:#FFFFFF; border-radius:14px; padding:12px; }}
    .prev {{ background:var(--card); border:1px solid var(--border); border-radius:22px; padding:20px;
        box-shadow:var(--shadow); height:100%; transition:transform .15s ease; }}
    .prev:hover {{ transform:translateY(-3px); }}
    .p-ic {{ width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center; margin-bottom:13px; }}
    .prev h4 {{ margin:0 0 6px 0; font-size:16px; color:var(--text); }}
    .prev p {{ color:var(--muted); font-size:13.5px; line-height:1.55; margin:0; }}
    .prev small {{ color:var(--muted); font-style:italic; font-size:12px; display:block; margin-top:10px; }}
    .empty {{ text-align:center; color:var(--muted); padding:46px 16px; font-size:15px; }}

    .card {{ background:var(--card); border:1px solid var(--border); border-radius:22px;
        padding:20px 22px; box-shadow:var(--shadow); }}
    .cardhead {{ font-size:18px; font-weight:800; margin:0 0 2px 0; color:var(--text); }}
    .cardsub {{ color:var(--muted); font-size:13px; margin:0 0 12px 0; }}
    .cf {{ width:100%; border-collapse:collapse; font-size:13.5px; }}
    .cf th {{ text-align:left; background:var(--track); color:var(--muted); font-weight:700; padding:9px 12px; }}
    .cf th:first-child {{ border-top-left-radius:12px; }}
    .cf th:last-child {{ border-top-right-radius:12px; }}
    .cf td {{ padding:9px 12px; border-top:1px solid var(--border); color:var(--text); }}
    .cf {{ border:1px solid var(--border); border-radius:12px; overflow:hidden; }}
    .succ {{ display:flex; align-items:center; gap:10px; background:#34C7591F; border:1px solid #34C75955;
        color:var(--text); border-radius:14px; padding:15px 16px; font-size:14px; line-height:1.4; }}

    /* Hilfe-Fragezeichen (Feather HelpCircle, Outline): nur Strichfarbe setzen */
    [data-testid="stTooltipIcon"] {{ color: var(--muted) !important; }}
    [data-testid="stTooltipIcon"] svg {{ stroke: currentColor !important; fill: none !important; }}

    /* Layout-Grids (für Media-Queries klassenbasiert) */
    .drow {{ display:grid; gap:16px; align-items:stretch; }}
    .drow.main {{ grid-template-columns:1fr 1.1fr; margin-bottom:16px; }}
    .drow.two {{ grid-template-columns:1fr 1fr; }}
    .tiles2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .gridp {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}

    /* Mobile */
    @media (max-width: 760px) {{
        .block-container {{ padding-left:0.7rem; padding-right:0.7rem; padding-top:1.6rem; }}
        .drow.main, .drow.two, .tiles2 {{ grid-template-columns:1fr !important; }}
        .gridp {{ grid-template-columns:1fr 1fr !important; }}
        .hero-title {{ font-size:25px; }}
        .sec {{ font-size:20px; }}
    }}
    @media (max-width: 460px) {{
        .gridp {{ grid-template-columns:1fr !important; }}
    }}
</style>
""", unsafe_allow_html=True)


def risk_palette(risiko: float) -> tuple[str, str]:
    if risiko < RISK_RECOMMENDATION_THRESHOLD:
        return "Niedriges Risiko", "#34C759"
    if risiko < RISK_HIGH_THRESHOLD:
        return "Erhöhtes Risiko", "#FF9F0A"
    return "Hohes Risiko", "#FF3B30"


def ring_svg(frac: float, color: str) -> str:
    frac = min(max(frac, 0.0), 1.0)
    r = 86
    c = 2 * math.pi * r
    off = c * (1 - frac)
    return f"""
    <svg width="208" height="208" viewBox="0 0 220 220">
      <circle cx="110" cy="110" r="{r}" fill="none" stroke="{C['track']}" stroke-width="20"/>
      <circle cx="110" cy="110" r="{r}" fill="none" stroke="{color}" stroke-width="20"
        stroke-linecap="round" stroke-dasharray="{c:.1f}" stroke-dashoffset="{c:.1f}"
        transform="rotate(-90 110 110)">
        <animate attributeName="stroke-dashoffset" from="{c:.1f}" to="{off:.1f}"
          dur="0.9s" fill="freeze" calcMode="spline" keyTimes="0;1" keySplines="0.34 1 0.27 1"/>
      </circle>
      <text x="110" y="103" text-anchor="middle" font-size="48" font-weight="800"
        fill="{C['text']}" font-family="-apple-system,Segoe UI,sans-serif">{frac*100:.0f}%</text>
      <text x="110" y="133" text-anchor="middle" font-size="15" fill="{color}"
        font-weight="700" font-family="-apple-system,Segoe UI,sans-serif">Risiko</text>
    </svg>
    """


# ----------------------------------------------------------------------------
# Kopfzeile
# ----------------------------------------------------------------------------
with head_l:
    st.markdown(
        f'<div class="hero"><div class="hero-ic">{icon(IC_HEART, ACCENT, 28)}</div>'
        f'<div><p class="hero-title">Herz-Risiko</p>'
        f'<p class="hero-sub">Persönliche Risiko-Analyse mit erklärbarer KI</p></div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Eingabe
# ----------------------------------------------------------------------------
st.markdown('<p class="sec">Ihre Werte</p>'
            '<p class="sec-sub">Eingaben anpassen und auswerten. Die Berechnung startet erst auf Klick.</p>',
            unsafe_allow_html=True)

with st.form("risiko_formular"):
    c1, c2 = st.columns(2)
    with c1:
        age: int | float | None = st.slider("Alter", 18, 100, 50, key="k_age", help="Lebensalter in Jahren.")
        sex_input: str | None = st.selectbox("Geschlecht", list(SEX_MAP.values()), key="k_sex")
        cp_options = list(CP_MAP.values())
        cp_input = st.selectbox("Brustschmerzen", cp_options, index=len(cp_options) - 1,
                                key="k_cp", help="Art der Brustschmerzen.")
        trestbps: int | float | None = st.slider("Ruhe-Blutdruck (systolisch, mmHg)", 90, 200, 120, key="k_trestbps")
    with c2:
        chol: int | float | None = st.number_input("Cholesterin (mg/dl)", 100, 400, 200, key="k_chol")
        thalach: int = st.slider("Max. Herzfrequenz", 60, 220, 150, key="k_thalach",
                                 help="Höchste erreichte Herzfrequenz unter Belastung.")
        fbs_input: str | None = st.radio("Nüchternblutzucker über 120 mg/dl?", ["Nein", "Ja"], horizontal=True, key="k_fbs")
        exang_input: str | None = st.radio("Belastungsangina?", ["Nein", "Ja"], horizontal=True,
                                            key="k_exang", help="Brustschmerzen unter Belastung?")
    submitted = st.form_submit_button("Risiko auswerten", type="primary", use_container_width=True)


if submitted:
    user_df: pd.DataFrame = prepare_patient_data(
        age, str(sex_input), str(cp_input), int(trestbps), int(chol),
        str(fbs_input), thalach, str(exang_input),
    )
    echtes_risiko: float = model.predict_proba(user_df)[0][0]
    _, echte_uq = calculate_uq(model, user_df)

    plt.style.use("default")
    fig, erklaerungen = run_app_shap(shap_explainer, user_df)
    plt.close(fig)  # Matplotlib-Grafik nicht mehr benötigt – eigene Balken statt Waterfall.

    cf_display: pd.DataFrame | None = None
    if echtes_risiko >= RISK_RECOMMENDATION_THRESHOLD:
        cf_display = format_counterfactual_for_display(
            run_app_counterfactual(model, x_train, y_train, user_df)
        )

    st.session_state["ergebnis"] = {
        "risiko": echtes_risiko, "uq": float(echte_uq[0]),
        "shap_saetze": erklaerungen, "cf_display": cf_display,
    }


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="sec">Übersicht</p>', unsafe_allow_html=True)

if "ergebnis" not in st.session_state:
    st.markdown('<div class="card"><div class="empty">Noch keine Auswertung.<br>'
                'Tragen Sie oben Ihre Werte ein und klicken Sie auf <b>„Risiko auswerten"</b>.</div></div>',
                unsafe_allow_html=True)
else:
    erg = st.session_state["ergebnis"]
    risiko = erg["risiko"]
    titel, farbe = risk_palette(risiko)
    uq_bewertung = evaluate_uq(erg["uq"])
    uq_titel = re.sub(r'^[\W_]+', '', uq_bewertung.titel).strip()  # führendes Emoji entfernen
    konfidenz = {"success": "Hoch", "warning": "Mittel", "error": "Niedrig"}.get(uq_bewertung.status_color, "n/v")

    # SHAP-Faktoren als Diverging-Balken aufbereiten
    parsed = []
    for satz in erg["shap_saetze"]:
        m = re.match(r"\*\*(.+?)\*\*\s*\((.+?)\)\s*(.+?)\s*um\s*\*\*([\d.,]+)%\*\*", satz)
        if m:
            name, wert, richtung, betrag = m.groups()
            parsed.append((name, wert, "erhöht" in richtung, float(betrag.replace(",", "."))))
    maxb = max((p[3] for p in parsed), default=1.0) or 1.0
    facs_html = ""
    for name, wert, up, betrag in parsed:
        col = "#FF3B30" if up else "#34C759"
        sign = "+" if up else "−"
        half = betrag / maxb * 46
        pos = (f'left:50%;width:{half:.0f}%;' if up else f'right:50%;width:{half:.0f}%;')
        facs_html += (f'<div class="srow"><div class="stop">'
                      f'<span class="sname">{name} <span class="smeta">({wert})</span></span>'
                      f'<span class="sval" style="color:{col};">{sign}{betrag:.1f}%</span></div>'
                      f'<div class="strack"><div class="scenter"></div>'
                      f'<div class="sbar" style="{pos}background:{col};"></div></div></div>')
    legend = ('<div class="slegend">'
              '<span><span class="dot" style="background:#FF3B30;"></span> erhöht das Risiko</span>'
              '<span><span class="dot" style="background:#34C759;"></span> senkt das Risiko</span></div>')

    # Empfehlung (DiCE)
    if erg["cf_display"] is None:
        rec_inner = (f'<div class="succ">{icon(IC_CHECK, "#34C759", 20)}'
                     f'<span>Ihr Risiko ist bereits niedrig, keine Änderungen notwendig.</span></div>')
    else:
        cf = erg["cf_display"]
        row = cf.iloc[0]
        rows = "".join(f'<tr><td>{c}</td><td style="text-align:right;font-weight:600;">{row[c]}</td></tr>'
                       for c in cf.columns)
        rec_inner = (f'<table class="cf"><tr><th>Merkmal</th><th style="text-align:right;">Zielwert</th></tr>'
                     f'{rows}</table>'
                     f'<p style="color:var(--muted);font-size:12px;margin-top:10px;">'
                     f'Zielwerte, um die Klassifikation auf „Gesund“ zu kippen.</p>')

    dashboard = f"""
    <div class="drow main">
      <div class="card" style="display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;">
        {ring_svg(risiko, farbe)}
        <div style="margin-top:8px;"><span class="pill" style="background:{farbe}22;color:{farbe};">
        <span class="dot" style="background:{farbe};"></span>{titel}</span></div>
      </div>
      <div style="display:flex;flex-direction:column;gap:14px;">
        <div class="tiles2">
          <div class="tile"><div class="t-label">Einstufung</div>
            <div class="t-value" style="color:{farbe};">{titel.split()[0]}</div>
            <div class="t-sub">basierend auf {risiko*100:.1f} % Risiko</div></div>
          <div class="tile"><div class="t-label">KI-Konfidenz</div>
            <div class="t-value">{konfidenz}</div>
            <div class="t-sub">Varianz {uq_bewertung.varianz:.3f} / 0.25</div></div>
        </div>
        <div class="tile" style="flex:1;"><div style="display:flex;align-items:center;gap:8px;font-weight:700;
          margin-bottom:5px;color:var(--text);">{icon(IC_SHIELD, farbe, 18)} {uq_titel}</div>
          <div style="color:var(--muted);font-size:13.5px;line-height:1.5;">{uq_bewertung.beschreibung}</div></div>
      </div>
    </div>
    <div class="drow two">
      <div class="card"><p class="cardhead">Einflussfaktoren</p>
        <p class="cardsub">Wie die Eingaben den Risiko-Score verschieben (SHAP)</p>
        {facs_html}{legend}</div>
      <div class="card"><p class="cardhead">Was Sie ändern können</p>
        <p class="cardsub">Konkreter Zielwert-Vorschlag (DiCE)</p>
        {rec_inner}</div>
    </div>
    """
    # Zu einer einzigen Zeile zusammenfügen (mit Leerzeichen, damit Tag-Attribute
    # gültig bleiben), sonst interpretiert Streamlits Markdown Einrückungen als Code-Block.
    dashboard = " ".join(line.strip() for line in dashboard.splitlines() if line.strip())
    st.markdown(dashboard, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("Zurücksetzen", type="secondary", use_container_width=True):
        # Ergebnis löschen und Eingaben auf Standardwerte zurücksetzen.
        for k in ("ergebnis", "k_age", "k_sex", "k_cp", "k_trestbps",
                  "k_chol", "k_thalach", "k_fbs", "k_exang"):
            st.session_state.pop(k, None)
        st.rerun()


# ----------------------------------------------------------------------------
# Prävention
# ----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="sec">Prävention</p>'
            '<p class="sec-sub">Vier Faktoren, die Ihr Risiko nachweislich senken.</p>',
            unsafe_allow_html=True)

prev_data = [
    ("#FF375F", IC_ACTIVITY, "Bewegung", "<b>150 Min.</b> Ausdauer pro Woche senken das Risiko um bis zu <b>35 %</b>.", "American Heart Association"),
    ("#34C759", IC_LEAF, "Ernährung", "Eine <b>mediterrane Diät</b> mit Fisch und Gemüse senkt das Herzrisiko.", "PREDIMED-Studie (2013)"),
    ("#5E9BFF", IC_MOON, "Schlaf", "<b>7 bis 9 Stunden</b> pro Nacht. Mangel erhöht das Risiko um <b>48 %</b>.", "J. Am. Coll. of Cardiology"),
    ("#FF9F0A", IC_WIND, "Stress", "Chronischer Stress verdoppelt das Risiko, <b>Atemübungen</b> helfen.", "European Heart Journal (2017)"),
]
prev_cards = "".join(
    f'<div class="prev"><div class="p-ic" style="background:{accent}1F;">{icon(ic, accent, 24)}</div>'
    f'<h4>{title}</h4><p>{text}</p><small>Quelle: {src}</small></div>'
    for accent, ic, title, text, src in prev_data
)
st.markdown(f'<div class="gridp">{prev_cards}</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.markdown('<p style="text-align:center;color:var(--muted);font-size:12px;">'
            'Wellness-Informationsangebot · ersetzt keine ärztliche Diagnose · '
            'Made by Fabian Mayer, Nico Then, Raffael Wellmann</p>', unsafe_allow_html=True)
