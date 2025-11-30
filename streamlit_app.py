import streamlit as st
import pandas as pd
import pickle

st.title("Wine Quality K-Means Clustering (k=3)")
st.write("Upload your wine dataset and get cluster predictions.")

# Load model
@st.cache_resource
def load_model():
    with open("kmeans_wine_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file, sep=';')

    st.subheader("Uploaded Dataset")
    st.dataframe(df)

    # Drop target column
    X = df.drop(columns=["quality"])

    # Predict clusters
    clusters = model.predict(X)
    df["Cluster"] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df)

    # Download clustered output
    csv = df.to_csv(index=False)
    st.download_button(
        "Download Results",
        data=csv,
        file_name="clustered_output.csv",
        mime="text/csv"
    )

    # Cluster centers
    st.subheader("Cluster Centers")
    centers = pd.DataFrame(model.cluster_centers_, columns=X.columns)
    st.dataframe(centers)

