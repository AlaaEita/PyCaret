import sys
import os

class AutoMLImporter:
    @staticmethod
    def import_auto_ml():
        try:
            # Add the parent directory to sys.path
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from scripts.auto_ml import AutoML
            return AutoML
        except ImportError as e:
            print("Error importing AutoML:", e)
            return None

