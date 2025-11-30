import streamlit as st
import pandas as pd
import joblib

st.title("🚢 Titanic Survival Predictor")

# Load model & features
model = joblib.load("titanic_model.pkl")
features = joblib.load("titanic_features.pkl")

st.subheader("Enter Passenger Details")

pclass = st.selectbox("Passenger Class (1 = First, 3 = Third)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)

if st.button("Predict Survival"):
    # Convert inputs into DataFrame
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": 0 if sex == "male" else 1,
        "Age": age,
        "Fare": fare
    }])

    # Predict
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"🎉 Survived! (Probability: {prob:.2f})")
    else:
        st.error(f"💀 Did NOT survive (Probability: {prob:.2f})")
