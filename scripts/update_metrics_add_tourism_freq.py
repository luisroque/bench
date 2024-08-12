import pandas as pd

file_path = "./assets/metrics/all/Tourism_M3_M4_M5_Labour_Traffic_Wiki2_all_results.csv"

df = pd.read_csv(file_path)

if "dataset" in df.columns and "unique_id" in df.columns:
    for index, row in df.iterrows():
        if row["dataset"] == "Tourism":
            unique_id_parts = row["unique_id"].split("_")
            if len(unique_id_parts) > 1:
                first_letter = unique_id_parts[1][0].lower()

                if first_letter == "m":
                    df.at[index, "freq"] = "Monthly"
                elif first_letter == "q":
                    df.at[index, "freq"] = "Quarterly"

    df.to_csv(file_path, index=False)

    print(
        f"Updated the 'freq' column based on the 'unique_id' for the Tourism dataset."
    )
else:
    print(
        "The required columns 'dataset' and 'unique_id' are not present in the CSV file."
    )
