import pandas as pd

file_path = "./assets/metrics/by_series/Tourism_M3_M4_M5_Labour_Traffic_Wiki2_error_by_series.csv"

df = pd.read_csv(file_path)

if "dataset" in df.columns and "freq" in df.columns:
    df.rename(columns={"dataset": "Dataset", "freq": "Frequency"}, inplace=True)
    df.to_csv(file_path, index=False)

    print(
        f"Updated the 'freq' column based on the 'unique_id' for the Tourism dataset."
    )
else:
    print("The required columns 'dataset' and 'freq' are not present in the CSV file.")
