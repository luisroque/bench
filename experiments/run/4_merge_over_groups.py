import pandas as pd
from neuralforecast.losses.numpy import smape

from codebase.load_data.config import DATASETS, DATASETS_FREQ

cv_neural = pd.read_csv(
    "./assets/results/by_group/{}_{}_neural.csv".format("Tourism", "Quarterly")
)

for data_name, groups in DATASETS_FREQ.items():
    for group in groups:

        data_cls = DATASETS[data_name]
        INPUT_CLS = "./assets/results/by_group/{}_{}_classical.csv"
        INPUT_NEURAL = "./assets/results/by_group/{}_{}_neural.csv"
        OUTPUT_DIR = "./assets/results/by_group/{}_{}_all.csv"

        cv_cls = pd.read_csv(INPUT_CLS.format(data_name, group))
        cv_neural = pd.read_csv(INPUT_NEURAL.format(data_name, group))

        # Remove 'Auto' from all algorithm names and drop the quantiles from DeepAR
        new_columns = []
        for col in cv_neural.columns:
            if col.startswith("AutoDeepAR") and (
                "median" in col or "lo-" in col or "hi-" in col
            ):
                continue
            new_columns.append(col)

        cv_neural = cv_neural[new_columns]
        cv_neural.columns = cv_neural.columns.str.replace("Auto", "")

        cv = cv_cls.merge(
            cv_neural.drop(columns=["y"]), how="left", on=["unique_id", "ds", "cutoff"]
        )

        cv = cv.reset_index(drop=True)

        output_file = OUTPUT_DIR.format(data_name, group)

        cv.to_csv(output_file, index=False)

        print(cv.isna().mean())
        print(smape(cv["y"], cv["NHITS"]))
        print(smape(cv["y"], cv["SeasonalNaive"]))
        print(smape(cv["y"], cv["AutoTheta"]))
