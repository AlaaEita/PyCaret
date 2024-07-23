# AutoML with PyCaret

## Overview
This project automates the machine learning workflow using PyCaret and provides a user-friendly web app built with Streamlit.

## Installation
1. Clone the repository.
2. Create a virtual environment and install the dependencies:
    ```bash
    pip install pycaret streamlit pandas numpy
    ```

## Usage
### General Package
1. Import the `AutoML` class from `scripts.auto_ml`.
2. Load your dataset and create an `AutoML` instance:
    ```python
    import pandas as pd
    from scripts.auto_ml import AutoML

    data = pd.read_csv("path_to_your_data.csv")
    automl = AutoML(data, target="your_target_variable")
    automl.perform_eda()
    automl.train_models()
    ```

### Streamlit Web App
1. Run the Streamlit app:
    ```bash
    streamlit run app/app.py
    ```
2. Upload your dataset, select the target variable, and click "Run AutoML".

## Testing
1. Create a test script in the `tests` directory.
2. Run the test script to ensure everything works correctly:
    ```bash
    python tests/test_automl.py
    ```

## Conclusion
This project demonstrates the use of PyCaret for automated machine learning and Streamlit for building an interactive web app. It simplifies the process of loading data, performing EDA, and training machine learning models.
