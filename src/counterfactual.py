"""Counterfactual-Erklärungen via DiCE.

Erzeugt für eine Patienteneingabe einen konkreten Handlungsvorschlag, also eine
minimale Merkmalskombination, welche die Klassifikation auf "gesund" kippt, und
bereitet diesen für die Anzeige auf.
"""

from typing import Any

import dice_ml  # type: ignore[import-untyped]
import pandas as pd

from src.mapping import CP_MAP, SEX_MAP, FBS_MAP, EXANG_MAP, COLUMN_LABELS

# Wichtig: Im UCI-CSV ist das Ziel invertiert kodiert. target == 1 bedeutet
# "gesund" (kein Befund), target == 0 bedeutet "krank". Mapping und Zielklasse
# richten sich nach dieser Kodierung.
_TARGET_MAP: dict[int, str] = {1: "Gesund", 0: "Krank"}

# Zielklasse des Handlungsvorschlags: 1 == "gesund" (siehe Kodierung oben).
DESIRED_CLASS_HEALTHY = 1


def format_counterfactual_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Formatiert den DiCE-DataFrame für die Anzeige (lesbare Labels statt 0/1)."""
    display_df = df.copy()

    col_map = {
        "sex": SEX_MAP,
        "cp": CP_MAP,
        "fbs": FBS_MAP,
        "exang": EXANG_MAP,
        "target": _TARGET_MAP,
    }

    for col, mapping in col_map.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda value: mapping[int(float(value))])

    return display_df.rename(columns=COLUMN_LABELS)


def run_app_counterfactual(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,  # type: ignore[type-arg]
    user_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generiert ein Counterfactual für die Nutzereingabe und gibt es als DataFrame zurück."""
    combined_df: pd.DataFrame = pd.concat([x_train, y_train], axis=1)

    data: Any = dice_ml.Data(
        dataframe=combined_df,
        continuous_features=list(x_train.columns),
        outcome_name=y_train.name,
    )

    dice_model: Any = dice_ml.Model(model=model, backend="sklearn")
    explainer: Any = dice_ml.Dice(data, dice_model, method="random")

    counterfactuals: Any = explainer.generate_counterfactuals(
        user_df, total_CFs=1, desired_class=DESIRED_CLASS_HEALTHY
    )

    return counterfactuals.cf_examples_list[0].final_cfs_df
