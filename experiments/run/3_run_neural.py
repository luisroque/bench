import os
from neuralforecast import NeuralForecast
from neuralforecast.auto import (
    AutoRNN,
    AutoTCN,
    AutoDeepAR,
    AutoNBEATS,
    AutoTiDE,
    AutoInformer,
)
from codebase.load_data.config import DATASETS

# Define the datasets and their groups
datasets = {
    "Tourism": ["Monthly", "Quarterly"],
    "M3": ["Monthly", "Quarterly", "Yearly"],
    "M4": ["Monthly", "Quarterly", "Yearly"],
    "M5": ["Daily"],
}

for data_name, groups in datasets.items():
    for group in groups:
        data_cls = DATASETS[data_name]
        print(data_name, group)

        OUTPUT_DIR = f"./assets/results/by_group/{data_name}_{group}_neural.csv"

        if os.path.exists(OUTPUT_DIR):
            print(f"Output file for {data_name} - {group} already exists. Skipping...")
            continue

        try:
            ds = data_cls.load_data(group)
        except FileNotFoundError as e:
            print(f"Error loading data for {data_name} - {group}: {e}")
            continue

        h = data_cls.horizons_map[group]
        n_lags = data_cls.context_length[group]
        if data_name == "M4":
            freq = data_cls.frequency_map.get(group)
        else:
            freq = data_cls.frequency_pd[group]
        season_len = data_cls.frequency_map[group]
        n_series = ds.nunique()["unique_id"]

        models = [
            AutoRNN(h=h, cpus=15, gpus=0),
            AutoTCN(h=h, cpus=15, gpus=0),
            AutoDeepAR(h=h, cpus=15, gpus=0),
            AutoNBEATS(h=h, cpus=15, gpus=0),
            AutoTiDE(h=h, cpus=15, gpus=0),
            AutoInformer(h=h, cpus=15, gpus=0),
        ]

        nf = NeuralForecast(models=models, freq=freq)

        cv_nf = nf.cross_validation(df=ds, test_size=h, n_windows=None)

        cv_nf.to_csv(OUTPUT_DIR, index=False)
