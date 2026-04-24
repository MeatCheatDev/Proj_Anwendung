from typing import Any

import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap  # type: ignore[import-untyped]


def run_shap(model: Any, x_test: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """
    Führt SHAP-Analyse für ein gegebenes Modell und Testdaten durch.

    Args:
        model: Trainiertes Machine Learning Modell (z.B. RandomForestClassifier)
        x_test: Test-Features als DataFrame

    Returns:
        tuple: (explainer, shap_values) - SHAP Explainer und berechnete SHAP-Werte
    """
    explainer: Any = shap.TreeExplainer(model)
    shap_values: np.ndarray = explainer.shap_values(x_test)

    if len(np.array(shap_values).shape) == 3:
        shap.summary_plot(shap_values[:, :, 1], x_test)
    else:
        shap.summary_plot(shap_values, x_test)

    plt.show()

    return explainer, shap_values


def run_app_shap(explainer: Any, user_df: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    Spezielle Funktion für die Streamlit-App.
    Generiert den lokalen Waterfall-Plot und gibt das Figure-Objekt zurück.
    """
    shap_values_user: Any = explainer(user_df)

    fig: matplotlib.figure.Figure = plt.figure(figsize=(6, 4))
    shap.plots.waterfall(shap_values_user[0, :, 0], show=False)

    return fig