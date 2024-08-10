from datasetsforecast.hierarchical import HierarchicalData

from codebase.load_data.base import LoadDataset


class LabourDataset(LoadDataset):
    DATASET_NAME = "Labour"

    @classmethod
    def load_data(cls, group):
        self = cls()
        ds, *_ = HierarchicalData.load(cls.DATASET_PATH, group=self.DATASET_NAME)
        return ds
