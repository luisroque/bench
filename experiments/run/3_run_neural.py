import os
from neuralforecast.models import NHITS, NBEATS, DeepAR
from neuralforecast import NeuralForecast

from codebase.load_data.config import DATASETS

CONFIG = {
    "max_steps": 1500,
    "val_check_steps": 50,
    "enable_checkpointing": True,
    "start_padding_enabled": True,
    "accelerator": "cpu",
}

# Define the datasets and their groups
# TODO: Add more datasets, for example, tourism
datasets = {
    # "M3": ["Monthly", "Quarterly", "Yearly"],
    # "M4": ["Monthly", "Quarterly", "Yearly"],
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
        freq = data_cls.frequency_pd[group]
        season_len = data_cls.frequency_map[group]

        models = [
            NHITS(h=h, input_size=n_lags, **CONFIG),
            NBEATS(h=h, input_size=n_lags, **CONFIG),
            DeepAR(h=h, input_size=n_lags, **CONFIG),
        ]

        nf = NeuralForecast(models=models, freq=freq)

        cv_nf = nf.cross_validation(df=ds, test_size=h, n_windows=None)

        cv_nf.to_csv(OUTPUT_DIR, index=False)
