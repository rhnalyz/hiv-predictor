
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Perbaikan path model
model = joblib.load("model/model_rf.pkl")

# Cek apakah scaler ada
try:
    scaler = joblib.load("model/scaler.pkl")
except:
    scaler = None

selected_features = ['time', 'trt', 'wtkg', 'drugs', 'karnof', 'oprior', 'z30',
                     'preanti', 'race', 'str2', 'strat', 'treat', 'offtrt', 'cd40', 'cd420']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            data = []
            for feature in selected_features:
                value = request.form.get(feature)
                if value is None or value.strip() == "":
                    raise ValueError(f"Fitur '{feature}' kosong atau tidak valid.")
                data.append(float(value))

            if scaler:
                data = scaler.transform([data])
            else:
                data = [data]

            pred = model.predict(data)[0]
            result = "Risiko Tinggi" if pred == 1 else "Risiko Rendah"
            return render_template("result.html", result=result)
        except Exception as e:
            return f"Terjadi kesalahan: {e}"
    return render_template("form.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
