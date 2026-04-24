import pandas as pd
from dataclasses import dataclass

from src.mapping import CP_MAP


@dataclass
class UQResult:
    varianz: float
    status_color: str
    titel: str
    beschreibung: str


def prepare_patient_data(
    age: int,
    sex_str: str,
    cp_str: str,
    trestbps: int,
    chol: int,
    fbs_str: str,
    thalach: int,
    exang_str: str,
) -> pd.DataFrame:
    cp_reversed: dict[str, int] = {v: k for k, v in CP_MAP.items()}

    return pd.DataFrame({
        'age': [age],
        'sex': [1 if sex_str == "Männlich" else 0],
        'cp':       [cp_reversed[cp_str]],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [1 if fbs_str == "Ja" else 0],
        'thalach': [thalach],
        'exang': [1 if exang_str == "Ja" else 0],
    })


def evaluate_uq(varianz: float) -> UQResult:
    """Kapselt die Geschäftslogik für die Ampel-Bewertung."""
    if varianz < 0.16:
        return UQResult(varianz, "success", "🟢 SEHR HOCH: Die KI ist sich sehr sicher.",
                        "Die Bäume im Modell sind sich weitgehend einig...")
    elif varianz < 0.22:
        return UQResult(varianz, "warning", "🟡 MITTEL: Die KI ist sich etwas unsicher.",
                        "Das Modell hat eine klare Tendenz...")
    else:
        return UQResult(varianz, "error", "🔴 SEHR NIEDRIG (WARNUNG): Die KI rät nur!",
                        "Das Modell ist komplett gespalten...")