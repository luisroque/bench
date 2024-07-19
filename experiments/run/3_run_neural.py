import os
from neuralforecast import NeuralForecast
from neuralforecast.auto import (
    AutoRNN,
    AutoTCN,
    AutoDeepAR,
    AutoMLP,
    AutoNBEATS,
    AutoNHITS,
    AutoTiDE,
    AutoTFT,
    AutoVanillaTransformer,
    AutoInformer,
    AutoPatchTST,
    AutoTSMixer,
)
from codebase.load_data.config import DATASETS

# Set the environment variable to use CPU fallback for unsupported MPS operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Define the datasets and their groups
# TODO: Add more datasets, for example, tourism
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
        freq = data_cls.frequency_pd[group]
        season_len = data_cls.frequency_map[group]
        n_series = ds.nunique()["unique_id"]

        models = [
            AutoRNN(h=h),
            AutoTCN(h=h),
            AutoDeepAR(h=h),
            AutoMLP(h=h),
            AutoNBEATS(h=h),
            AutoNHITS(h=h),
            AutoTiDE(h=h),
            AutoTFT(h=h),
            AutoVanillaTransformer(h=h),
            AutoInformer(h=h),
            AutoPatchTST(h=h),
            AutoTSMixer(h=h, n_series=n_series),
        ]

        nf = NeuralForecast(models=models, freq=freq)

        cv_nf = nf.cross_validation(df=ds, test_size=h, n_windows=None)

        cv_nf.to_csv(OUTPUT_DIR, index=False)
