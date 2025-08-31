# train_model.py
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

DATA_FILE = "demo_cicids2017.csv"
MODEL_PATH = "models/intrusion_model.pkl"

def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Run `python demo_dataset.py` first.")

    df = pd.read_csv(DATA_FILE)
    print("Loaded dataset:", df.shape)

    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n✅ Training completed. Evaluation:")
    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved as {MODEL_PATH}")

if __name__ == "__main__":
    main()
