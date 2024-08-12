import os
import pandas as pd

directory = "./assets/results/by_group"

for filename in os.listdir(directory):
    if filename.endswith("neural.csv"):
        filepath = os.path.join(directory, filename)

        df = pd.read_csv(filepath)

        if "unique_id" in df.columns:
            df["unique_id"] = (
                df["unique_id"]
                .str.replace(".", "", regex=False)
                .str.replace("/", "-", regex=False)
                .str.replace("\\", "-", regex=False)
                .str.replace("_", "-", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("[", "", regex=False)
                .str.replace("]", "", regex=False)
                .str.replace("'", "", regex=False)
            )

            df.to_csv(filepath, index=False)

            print(f"Updated unique_id in: {filename}")
