import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Chemin vers les fichiers de données
DATA_DIR = "data"
GESTURES = ["LEFT", "RIGHT", "FIRE"]  

# Chargement des données
data = []
labels = []

for gesture in GESTURES:
    path = os.path.join(DATA_DIR, f"{gesture}.csv")
    df = pd.read_csv(path)
    data.append(df)
    labels.extend([gesture] * len(df))

# Fusion
X = pd.concat(data, axis=0)
y = labels

# Découpage en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
accuracy = model.score(X_test, y_test)
print(f"✅ Modèle entraîné avec une précision de : {accuracy:.2%}")

# Sauvegarde
joblib.dump(model, "saved_model.pkl")
print("📦 Modèle sauvegardé dans : saved_model.pkl")
