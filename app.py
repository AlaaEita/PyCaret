import sys
import os
import streamlit as st
import pandas as pd

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Debugging statements
print("Current working directory:", os.getcwd())
print("sys.path:", sys.path)

from scripts.importer import AutoMLImporter

# Import the AutoML class using the AutoMLImporter class
AutoML = AutoMLImporter.import_auto_ml()

if AutoML is None:
    st.error("Failed to import AutoML. Please check the scripts/auto_ml.py file.")
else:
    st.title("AutoML with PyCaret")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data preview:", data.head())

        target = st.selectbox("Select the target variable", data.columns)

        if st.button("Run AutoML"):
            automl = AutoML(data, target)
            st.write("Performing EDA...")
            automl.perform_eda()
            st.write("Training models...")
            automl.train_models()
