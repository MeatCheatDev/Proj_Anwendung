import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path

# Features, die wir verwenden wollen. Wir wollen nicht alle aus dem UCI verwenden, nur sinnvolle.
FEATURE_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang']

# Pfade definieren
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'heart-disease-UCI.csv'
MODEL_PATH = BASE_DIR / 'models' / 'heart_model.joblib'

def train_and_save_model():
    # Daten aus CVS laden. Wir nutzen die UCI Heart Disease CSV.
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLUMNS + ['target'])

    features = df[FEATURE_COLUMNS]
    y_binary = (df['target'] > 0).astype(int)

    # Split der Daten: 20% Testset, 80% Trainingsset  --> Verhindert Overfitting
    x_train, x_test, y_train, y_test = train_test_split(
        features, y_binary, test_size=0.2, random_state=42
    )

    # Training
    print("Training startet...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(x_train, y_train)

    # Erstellt Ordner, falls noch nicht vorhanden
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Modell abspeichern. Joblib ist Speicherstandard für ML Modells wie unser Randomforest
    joblib.dump({'model': clf, 'x_train': x_train, 'y_train': y_train}, MODEL_PATH)
    print(f"Erfolg! Modell wurde gespeichert unter: {MODEL_PATH}")


# HINWEIS: Aus Sicherheitsgründen (Vermeidung von potenzieller Schadsoftware in Serialisierungsformaten
# wie Joblib/Pickle) wird die Modelldatei nicht mitgeliefert. Bitte führen Sie dieses Skript einmal
# lokal aus, um das Modell frisch zu trainieren und die .joblib-Datei sicher zu erzeugen.
if __name__ == "__main__":
    train_and_save_model()
