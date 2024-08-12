import re
import os

import pandas as pd

from codebase.load_data.config import DATASETS_FREQ

DIRECTORY = "./assets/results/by_series"

for data_name, freq in DATASETS_FREQ.items():
    for group in freq:
        print(data_name, group)

        OUTPUT_DIR = f"./assets/results/by_group/{data_name}_{group}_classical.csv"

        if os.path.exists(OUTPUT_DIR):
            print(f"skipping {data_name}_{group}")
            continue

        files = os.listdir(DIRECTORY)

        expr = f"^cv_{data_name}_{group}"
        files_group = [x for x in files if re.search(expr, x)]

        group_results = []
        for file in files_group:
            ts_name = file.split("_")[3]
            print(ts_name)
            filepath = f"{DIRECTORY}/{file}"

            ts_result = pd.read_csv(filepath)
            ts_result["unique_id"] = ts_name

            group_results.append(ts_result)

        if group_results:
            group_df = pd.concat(group_results)

            group_df.to_csv(OUTPUT_DIR, index=False)
        else:
            print(f"No results to concatenate for {data_name} {group}")
