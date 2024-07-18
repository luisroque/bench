import pandas as pd
from neuralforecast.losses.numpy import smape

from codebase.load_data.config import DATASETS

DS = "M4"
group = "Monthly"

data_cls = DATASETS[DS]
INPUT_CLS = "./assets/results/by_group/{}_{}_classical.csv"
INPUT_NEURAL = "./assets/results/by_group/{}_{}_neural.csv"
OUTPUT_DIR = "./assets/results/by_group/{}_{}_all.csv"

cv_cls = pd.read_csv(INPUT_CLS.format(DS, group))
cv_neural = pd.read_csv(INPUT_NEURAL.format(DS, group))

cv = cv_cls.merge(
    cv_neural.drop(columns=["y"]), how="left", on=["unique_id", "ds", "cutoff"]
)

cv = cv.reset_index(drop=True)

output_file = OUTPUT_DIR.format(DS, group)

cv.to_csv(output_file, index=False)

print(cv.isna().mean())
print(smape(cv["y"], cv["NHITS"]))
print(smape(cv["y"], cv["SeasonalNaive"]))
print(smape(cv["y"], cv["AutoTheta"]))
