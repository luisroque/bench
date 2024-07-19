import os

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive,
    AutoETS,
    AutoARIMA,
    RandomWalkWithDrift,
    AutoTheta,
    SimpleExponentialSmoothingOptimized,
    CrostonOptimized,
)

from codebase.load_data.config import DATASETS

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

        ds = data_cls.load_data(group)
        h = data_cls.horizons_map[group]
        n_lags = data_cls.context_length[group]
        if data_name == "M4":
            freq = data_cls.frequency_map.get(group)
        else:
            freq = data_cls.frequency_pd[group]

        freq_int = data_cls.frequency_map.get(group)

        ds_grouped = ds.groupby("unique_id")
        for tsname, df in ds_grouped:
            print(data_name, group, tsname)
            filepath = f"./assets/results/by_series/cv_{data_name}_{group}_{tsname}_classical.csv"

            # Ensure the directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            if os.path.exists(filepath):
                print(f"skipping {tsname}")
                continue
            else:
                pd.DataFrame().to_csv(filepath, index=False)

            season_len = data_cls.frequency_map[group]
            if 2 * season_len >= df.shape[0]:
                season_len = 1

            cls_models = [
                RandomWalkWithDrift(),
                SeasonalNaive(season_length=season_len),
                AutoETS(season_length=season_len),
                AutoARIMA(max_P=1, max_p=1, max_D=1, max_d=1, max_q=1, max_Q=1),
                AutoTheta(season_length=season_len),
                SimpleExponentialSmoothingOptimized(),
                CrostonOptimized(),
            ]

            sf = StatsForecast(
                models=cls_models,
                freq=freq,
                n_jobs=1,
            )

            cv_result = sf.cross_validation(df=df, h=h, test_size=h, n_windows=None)

            # anomalies
            an_sf = StatsForecast(
                models=[SeasonalNaive(season_length=season_len)],
                freq=freq,
                n_jobs=1,
            )

            cv_an = an_sf.cross_validation(
                df=df,
                h=h,
                test_size=h,
                n_windows=None,
                level=[95, 99],
            )

            is_outside_99 = (cv_an["y"] >= cv_an["SeasonalNaive-hi-99"]) | (
                cv_an["y"] <= cv_an["SeasonalNaive-lo-99"]
            )
            is_outside_99 = is_outside_99.astype(int)
            cv_result["is_anomaly_99"] = is_outside_99.astype(int)

            is_outside_95 = (cv_an["y"] >= cv_an["SeasonalNaive-hi-95"]) | (
                cv_an["y"] <= cv_an["SeasonalNaive-lo-95"]
            )
            is_outside_95 = is_outside_95.astype(int)
            cv_result["is_anomaly_95"] = is_outside_95.astype(int)

            cv_result.to_csv(filepath, index=False)
