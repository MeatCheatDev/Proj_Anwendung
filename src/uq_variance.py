from typing import Any

import numpy as np
import pandas as pd


def calculate_uq(model: Any, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Berechnet die prädiktive Unsicherheit eines RandomForest-Modells.

    Vorgehen:

    - Für jeden Baum wird die Wahrscheinlichkeit der positiven Klasse bestimmt.
    - Falls ein Baum nur eine Klasse kennt, wird diese auf 0/1 gemappt.
    - Aus allen Baum-Prognosen werden Mittelwert und Varianz berechnet.
    """
    x_test_array: np.ndarray = x_test.values

    # Definition der Zielklasse für "positiv"
    forest_classes: list[Any] = list(model.classes_)
    positive_class: Any = 1 if 1 in forest_classes else forest_classes[-1]

    tree_preds_list: list[np.ndarray] = []

    # Wahrscheinlichkeit der positiven Klasse für jeden Baum auslesen
    for tree in model.estimators_:
        proba: np.ndarray = tree.predict_proba(x_test_array)

        if proba.shape[1] == 2:
            class_index: int = list(tree.classes_).index(positive_class)
            tree_pred: np.ndarray = proba[:, class_index]
        else:
            # Sonderfall: Baum kennt nur eine einzige Klasse (Gibt dann nur 1 oder 0 zurück)
            only_class: Any = tree.classes_[0]
            tree_pred = (
                np.ones(len(x_test_array)) if only_class == positive_class
                else np.zeros(len(x_test_array))
            )

        tree_preds_list.append(tree_pred)

    all_tree_preds: np.ndarray = np.array(tree_preds_list)

    # Varianz über alle Bäume liefert die prädiktive Unsicherheit
    mean_preds: np.ndarray = np.mean(all_tree_preds, axis=0)
    uq_variance: np.ndarray = np.var(all_tree_preds, axis=0)

    return mean_preds, uq_variance