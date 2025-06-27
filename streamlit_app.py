
import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("model.pkl")
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

st.title("Prediksi Model - Streamlit App")

st.markdown("Masukkan nilai untuk setiap fitur berikut:")

selected_features = [
    'time', 'trt', 'wtkg', 'drugs', 'karnof', 'oprior',
    'z30', 'preanti', 'race', 'str2', 'strat', 'treat',
    'offtrt', 'cd40', 'cd420'
]

input_data = {}
for feature in selected_features:
    input_data[feature] = st.number_input(f"{feature}", step=0.1)

if st.button("Prediksi"):
    try:
        data = [float(input_data[feature]) for feature in selected_features]
        if scaler:
            data = scaler.transform([data])
        else:
            data = [data]
        prediction = model.predict(data)
        st.success(f"Hasil Prediksi: {prediction[0]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
