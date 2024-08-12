import pandas as pd
from datasetsforecast.m4 import M4

from codebase.load_data.base import LoadDataset


class M4Dataset(LoadDataset):
    DATASET_NAME = "M4"

    @classmethod
    def load_data(cls, group):
        ds, *_ = M4.load(cls.DATASET_PATH, group=group)
        ds["ds"] = ds["ds"].astype(int)

        return ds
