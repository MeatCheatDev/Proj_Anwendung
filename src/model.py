import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path

# 8 Features für die App-Nutzung
FEATURE_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach', 'exang']

# Pfade definieren
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'data' / 'heart-disease-UCI.csv'
MODEL_PATH = BASE_DIR / 'models' / 'heart_model.joblib'

def train_and_save_model():
    """
    Trainiert das Modell und speichert es dauerhaft auf der Festplatte.
    """
    # 1. Daten laden
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURE_COLUMNS + ['target'])

    features = df[FEATURE_COLUMNS]
    y_binary = (df['target'] > 0).astype(int)

    # 2. Split
    x_train, x_test, y_train, y_test = train_test_split(
        features, y_binary, test_size=0.2, random_state=42
    )

    # 3. Training
    print("Training startet...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(x_train, y_train)

    # 4. Ordner erstellen, falls er nicht existiert
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 5. DAS MODELL SPEICHERN
    joblib.dump({'model': clf, 'x_train': x_train, 'y_train': y_train}, MODEL_PATH)
    print(f"Erfolg! Modell wurde gespeichert unter: {MODEL_PATH}")

    # Performance-Check
    y_pred = clf.predict(x_test)
    print("\nModel Performance auf Testdaten:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_save_model()