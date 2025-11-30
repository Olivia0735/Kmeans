import streamlit as st
import pickle
import numpy as np

st.title("Wine Quality Clustering (K-Means, k=3)")
st.write("Enter wine chemical properties to predict which group it belongs to.")

# Load trained model
with open("kmeans_wine_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Input form
with st.form("wine_form"):
    fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.7)
    citric_acid = st.number_input("Citric Acid", value=0.0)
    residual_sugar = st.number_input("Residual Sugar", value=1.9)
    chlorides = st.number_input("Chlorides", value=0.076)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
    density = st.number_input("Density", value=0.9978)
    ph = st.number_input("pH", value=3.51)
    sulphates = st.number_input("Sulphates", value=0.56)
    alcohol = st.number_input("Alcohol", value=9.4)

    submit = st.form_submit_button("Predict Group")

# Prediction
if submit:
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, ph, sulphates, alcohol
    ]])

    cluster = kmeans.predict(input_data)[0]

    st.success(f"### 🍷 This wine belongs to **Group {cluster}**")
