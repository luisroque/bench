import pandas as pd
from neuralforecast.losses.numpy import smape

from codebase.load_data.config import DATASETS

datasets = {
    "Tourism": ["Monthly", "Quarterly"],
    "M3": ["Monthly", "Quarterly", "Yearly"],
    # "M4": ["Monthly", "Quarterly", "Yearly"],
    # "M5": ["Daily"],
}

for data_name, groups in datasets.items():
    for group in groups:

        data_cls = DATASETS[data_name]
        INPUT_CLS = "./assets/results/by_group/{}_{}_classical.csv"
        INPUT_NEURAL = "./assets/results/by_group/{}_{}_neural.csv"
        OUTPUT_DIR = "./assets/results/by_group/{}_{}_all.csv"

        cv_cls = pd.read_csv(INPUT_CLS.format(data_name, group))
        cv_neural = pd.read_csv(INPUT_NEURAL.format(data_name, group))

        # TODO: remove this line and re-run neural predictions with unique_id
        cv_neural["unique_id"] = cv_cls["unique_id"]

        cv = cv_cls.merge(
            cv_neural.drop(columns=["y"]), how="left", on=["unique_id", "ds", "cutoff"]
        )

        cv = cv.reset_index(drop=True)

        output_file = OUTPUT_DIR.format(data_name, group)

        cv.to_csv(output_file, index=False)

        print(cv.isna().mean())
        print(smape(cv["y"], cv["AutoNBEATS"]))
        print(smape(cv["y"], cv["SeasonalNaive"]))
        print(smape(cv["y"], cv["AutoTheta"]))
