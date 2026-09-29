import os
import pandas as pd
from model_utils import train_lstm_model

if __name__ == "__main__":
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "elf_dataset.xlsx")
    if not os.path.exists(data_path):
        data_path = "elf_dataset.xlsx"

    print(f"Loading dataset from: {data_path}")
    df = pd.read_excel(data_path)
    target_col = "DEMAND"

    # Train + Save
    save_dir = os.path.dirname(__file__) or "."
    model, scaler_X, scaler_y, history, results = train_lstm_model(df, target_col, save_dir=save_dir)
    print("✅ Model trained and saved successfully!")
    print("📊 Evaluation Results:", results)
