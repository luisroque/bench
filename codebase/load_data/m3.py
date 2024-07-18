from datasetsforecast.m3 import M3

from codebase.load_data.base import LoadDataset


class M3Dataset(LoadDataset):
    DATASET_NAME = "M3"

    @classmethod
    def load_data(cls, group):
        ds, *_ = M3.load(cls.DATASET_PATH, group=group)
        return ds
