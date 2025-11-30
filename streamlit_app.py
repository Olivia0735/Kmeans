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
