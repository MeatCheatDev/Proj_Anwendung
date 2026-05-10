#Globales Mapping File, dass nicht in jedem File neu definiert werden muss.

CP_MAP: dict[int, str] = {
    0: "Typische Angina (Schwer)",
    1: "Atypische Angina",
    2: "Nicht-anginöser Schmerz",
    3: "Keine Beschwerden",
}

SEX_MAP: dict[int, str] = {1: "Männlich", 0: "Weiblich"}

FBS_MAP: dict[int, str] = {1: "Ja", 0: "Nein"}

EXANG_MAP: dict[int, str] = {1: "Ja", 0: "Nein"}

COLUMN_LABELS: dict[str, str] = {
    "age":      "Alter",
    "sex":      "Geschlecht",
    "cp":       "Brustschmerzen",
    "trestbps": "Ruheblutdruck (mmHg)",
    "chol":     "Cholesterin (mg/dl)",
    "fbs":      "Nüchternzucker erhöht",
    "thalach":  "Max. Herzfrequenz",
    "exang":    "Belastungsangina",
    "target":   "Diagnose",
}