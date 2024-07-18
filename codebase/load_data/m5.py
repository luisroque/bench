import pandas as pd
from datasetsforecast.m5 import M5

from codebase.load_data.base import LoadDataset


class M5Dataset(LoadDataset):
    DATASET_NAME = "M5"

    @classmethod
    def load_data(cls, group=None):
        ds, *_ = M5.load(cls.DATASET_PATH)
        return ds
