import pandas as pd

file_path = "./assets/metrics/all/Tourism_M3_M4_M5_Labour_Traffic_Wiki2_all_results.csv"

df = pd.read_csv(file_path)

if "dataset" in df.columns and "freq" in df.columns:
    unique_datasets = df["dataset"].unique()
    print("Unique values in the 'dataset' column:")
    for dataset in unique_datasets:
        print(dataset)

    unique_freqs = df["freq"].unique()
    print("\nUnique values in the 'freq' column:")
    for freq in unique_freqs:
        print(freq)

    nan_freq_datasets = df[df["freq"].isna()]["dataset"].unique()

    # Print the datasets with NaN in the 'freq' column
    print("Datasets with NaN in the 'freq' column:")
    for dataset in nan_freq_datasets:
        print(dataset)
else:
    print("The required columns 'dataset' and 'freq' are not present in the CSV file.")
