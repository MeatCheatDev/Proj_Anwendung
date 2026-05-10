from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap  # type: ignore[import-untyped]
import matplotlib.figure

from src.mapping import CP_MAP, SEX_MAP, FBS_MAP, EXANG_MAP, COLUMN_LABELS

_VALUE_LABELS: dict[str, dict] = {
    "sex":   SEX_MAP,
    "cp":    CP_MAP,
    "fbs":   FBS_MAP,
    "exang": EXANG_MAP,
}

def run_app_shap(
    explainer: Any,
    user_df: pd.DataFrame,
) -> tuple[matplotlib.figure.Figure, list[str]]:

    # Erklärungstexte sind eine Liste von Sätzen für die Top-Features.

    shap_values_user: Any = explainer(user_df)
    sv = shap_values_user[0, :, 0]

    # Feature-Namen und Werte für Anzeige umbenennen
    readable_names = [COLUMN_LABELS.get(f, f) for f in sv.feature_names]
    readable_data = []
    for fname, val in zip(sv.feature_names, sv.data):
        if fname in _VALUE_LABELS:
            readable_data.append(_VALUE_LABELS[fname].get(int(float(val)), val))
        else:
            readable_data.append(val)

    sv = shap.Explanation(
        values=sv.values,
        base_values=sv.base_values,
        data=np.array(readable_names, dtype=object),
        feature_names=np.array(readable_data, dtype=object),
    )


    fig: matplotlib.figure.Figure = plt.figure(figsize=(6, 4))
    shap.plots.waterfall(sv, show=False)

    # Erklärungstexte generieren
    erklaerungen = _generate_explanation(sv)

    return fig, erklaerungen


def _generate_explanation(sv: Any) -> list[str]:
    pairs = sorted(
        zip(sv.data, sv.values, sv.feature_names),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    texte = []
    for name, wert, datenwert in pairs[:4]:
        richtung = "erhöht das Risiko" if wert > 0 else "senkt das Risiko"
        texte.append(f"**{name}** ({datenwert}) {richtung} um **{abs(wert)*100:.1f}%**")
    return texte