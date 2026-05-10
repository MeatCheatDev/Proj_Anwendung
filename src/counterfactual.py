from typing import Any

import dice_ml  # type: ignore[import-untyped]
import pandas as pd

from src.mapping import CP_MAP, SEX_MAP, FBS_MAP, EXANG_MAP, COLUMN_LABELS

_TARGET_MAP: dict[int, str] = {1: "Krank", 0: "Gesund"}


def format_counterfactual_for_display(df: pd.DataFrame) -> pd.DataFrame:
    #Formatiert den DiCE-DataFrame für die Anzeige im UI (lesbare Labels, keine 0/1).

    display_df = df.copy()

    col_map = {
        "sex":    SEX_MAP,
        "cp":     CP_MAP,
        "fbs":    FBS_MAP,
        "exang":  EXANG_MAP,
        "target": _TARGET_MAP,
    }

    for col, mapping in col_map.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: mapping[int(float(x))]
            )

    display_df = display_df.rename(columns=COLUMN_LABELS)
    return display_df


    """
    Spezielle Funktion für die Streamlit-App.
    Generiert 1 Counterfactual für den eingegebenen Nutzer und gibt es als DataFrame zurück.
    """

def run_app_counterfactual(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,  # type: ignore[type-arg]
    user_df: pd.DataFrame,
) -> pd.DataFrame:

    combined_df: pd.DataFrame = pd.concat([x_train, y_train], axis=1)

    data: Any = dice_ml.Data(
        dataframe=combined_df,
        continuous_features=list(x_train.columns),
        outcome_name=y_train.name,
    )

    m: Any = dice_ml.Model(model=model, backend="sklearn")
    exp: Any = dice_ml.Dice(data, m, method="random")

    cf: Any = exp.generate_counterfactuals(user_df, total_CFs=1, desired_class=0)

    result_df: pd.DataFrame = cf.cf_examples_list[0].final_cfs_df
    return result_df