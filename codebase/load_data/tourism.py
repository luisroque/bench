import os
import pandas as pd
import numpy as np
import requests
import zipfile
from io import BytesIO
from codebase.load_data.base import LoadDataset


class TourismDataset(LoadDataset):
    DATASET_PATH = "./datasets/tourism/"
    DATASET_NAME = "Tourism"

    DATA_URL = "https://robjhyndman.com/data/27-3-Athanasopoulos1.zip"

    frequency_pd_tourism = {"Yearly": "Y", "Quarterly": "QS", "Monthly": "MS"}

    @classmethod
    def download_and_extract(cls):
        # Create dataset path if it does not exist
        if not os.path.exists(cls.DATASET_PATH):
            os.makedirs(cls.DATASET_PATH)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        # Download the file with user-agent header
        response = requests.get(cls.DATA_URL, headers=headers)
        if response.status_code == 200:
            with zipfile.ZipFile(BytesIO(response.content)) as z:
                z.extractall(cls.DATASET_PATH)
        else:
            raise Exception(
                f"Failed to download data: status code {response.status_code}"
            )

    @classmethod
    def load_data(cls, group):
        cls.download_and_extract()
        assert group in cls.data_group

        ds = {}
        train = pd.read_csv(
            os.path.join(cls.DATASET_PATH, f"{group.lower()}_in.csv"),
            header=0,
            delimiter=",",
        )
        test = pd.read_csv(
            os.path.join(cls.DATASET_PATH, f"{group.lower()}_oos.csv"),
            header=0,
            delimiter=",",
        )

        if group == "Yearly":
            train_meta = train[:2]
            meta_length = train_meta.iloc[0].astype(int)
            test = test[2:].reset_index(drop=True).T
            train = train[2:].reset_index(drop=True).T
        else:
            train_meta = train[:3]
            meta_length = train_meta.iloc[0].astype(int)
            test = test[3:].reset_index(drop=True).T
            train = train[3:].reset_index(drop=True).T

        train_set = [ts[:ts_length] for ts, ts_length in zip(train.values, meta_length)]
        test_set = [ts[:ts_length] for ts, ts_length in zip(test.values, meta_length)]

        for i, idx in enumerate(train.index):
            ds[idx] = np.concatenate([train_set[i], test_set[i]])

        max_len = np.max([len(x) for k, x in ds.items()])
        idx = pd.date_range(
            end=pd.Timestamp("2023-11-01"),
            periods=max_len,
            freq=cls.frequency_pd[group],
        )

        ds = {
            k: pd.Series(series, index=idx[-len(series) :]) for k, series in ds.items()
        }
        df = pd.concat(ds, axis=1)
        df = df.reset_index().melt("index").dropna().reset_index(drop=True)
        df.columns = ["ds", "unique_id", "y"]

        return df
