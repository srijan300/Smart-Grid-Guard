import os
import pandas as pd

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "elf_dataset.xlsx")

def load_data(filepath=DEFAULT_DATA_PATH):
    """
    Load the Electricity Load Forecasting dataset.
    """
    if not os.path.exists(filepath):
        # Fallback to local filename if running from same folder
        filepath = "elf_dataset.xlsx"
    return pd.read_excel(filepath)
