import streamlit as st
import pickle
import numpy as np

st.title("🍷 Wine Clustering Prediction (K-Means k=3)")
st.write("Enter wine chemical properties to see its predicted group and description.")

# Load trained KMeans model
with open("kmeans_wine_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Group descriptions (you can adjust these)
group_info = {
    0: {
        "name": "Low-Quality / High-Acidity Wine",
        "desc": """
        - Higher volatile acidity  
        - Lower alcohol percentage  
        - Lower sulphates  
        - Tastes sharper or more sour  
        """
    },
    1: {
        "name": "Medium Balanced Wine",
        "desc": """
        - Balanced acidity  
        - Medium alcohol levels  
        - Moderate sugar  
        - Typical everyday table wine  
        """
    },
    2: {
        "name": "High-Quality / Smooth Wine",
        "desc": """
        - Higher alcohol percentage  
        - Lower volatile acidity  
        - Higher sulphates  
        - Fuller, smoother flavor  
        """
    }
}

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

# Predict result
if submit:
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, ph, sulphates, alcohol
    ]])

    cluster = kmeans.predict(input_data)[0]

    st.success(f"### 🏷 Predicted Group: **Group {cluster}**")
    st.write(f"### 🍇 Type: **{group_info[cluster]['name']}**")
    st.write("#### 📌 Group Description:")
    st.markdown(group_info[cluster]["desc"])
