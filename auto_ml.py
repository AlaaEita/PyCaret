import pandas as pd
from pycaret.classification import setup as cl_setup, compare_models as cl_compare_models
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models

class AutoML:
    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target
        self.is_classification = self._check_classification()

    def _check_classification(self):
        unique_values = self.data[self.target].nunique()
        return unique_values <= 20  # Arbitrary threshold for classification

    def perform_eda(self):
        print("Performing EDA...")
        print(self.data.describe())
        print("\nData types:\n", self.data.dtypes)
        print("\nMissing values:\n", self.data.isnull().sum())

    def train_models(self):
        if self.is_classification:
            self._train_classification_models()
        else:
            self._train_regression_models()

    def _train_classification_models(self):
        print("Training classification models...")
        setup = cl_setup(self.data, target=self.target, silent=True, verbose=False)
        best_model = cl_compare_models()
        print("Best classification model:", best_model)

    def _train_regression_models(self):
        print("Training regression models...")
        setup = reg_setup(self.data, target=self.target, silent=True, verbose=False)
        best_model = reg_compare_models()
        print("Best regression model:", best_model)








