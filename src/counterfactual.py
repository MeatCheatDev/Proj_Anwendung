from typing import Any

import dice_ml  # type: ignore[import-untyped]
import pandas as pd


def run_app_counterfactual(
    model: Any,
    x_train: pd.DataFrame,
    y_train: pd.Series,  # type: ignore[type-arg]
    user_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Spezielle Funktion für die Streamlit-App.
    Generiert 1 Counterfactual für den eingegebenen Nutzer und gibt es als DataFrame zurück.
    """
    combined_df: pd.DataFrame = pd.concat([x_train, y_train], axis=1)

    data: Any = dice_ml.Data(
        dataframe=combined_df,
        continuous_features=list(x_train.columns),
        outcome_name=y_train.name,
    )

    m: Any = dice_ml.Model(model=model, backend="sklearn")
    exp: Any = dice_ml.Dice(data, m, method="random")

    cf: Any = exp.generate_counterfactuals(user_df, total_CFs=1, desired_class="opposite")

    result_df: pd.DataFrame = cf.cf_examples_list[0].final_cfs_df
    return result_df