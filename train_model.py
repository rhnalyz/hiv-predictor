import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def train_and_save():
    # Load dataset
    df = pd.read_csv("pdm_aids.csv")

    # Pisahkan fitur dan target
    X = df.drop("infected", axis=1)
    y = df["infected"]

    # Seleksi 15 fitur terbaik
    selector = SelectKBest(score_func=f_classif, k=15)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]

    # Subset data dengan fitur terpilih
    X_filtered = X[selected_features]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Latih model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Simpan model dan scaler
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Model dan scaler berhasil disimpan ulang.")

if __name__ == "__main__":
    train_and_save()