import pandas as pd
from datasetsforecast.m4 import M4

from codebase.load_data.base import LoadDataset


class M4Dataset(LoadDataset):
    DATASET_NAME = "M4"

    @classmethod
    def load_data(cls, group):
        ds, *_ = M4.load(cls.DATASET_PATH, group=group)
        ds["ds"] = ds["ds"].astype(int)

        # unq_periods = ds['ds'].sort_values().unique()
        #
        # dates = pd.date_range(start='2000-01-01',
        #                       periods=len(unq_periods),
        #                       freq=cls.frequency_pd[group])
        #
        # new_ds = {k: v for k, v in zip(unq_periods, dates)}
        #
        # ds['ds'] = ds['ds'].map(new_ds)

        return ds
