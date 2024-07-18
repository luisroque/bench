class LoadDataset:
    DATASET_PATH = "./datasets"
    DATASET_NAME = ""

    horizons_map = {
        "Yearly": 6,
        "Quarterly": 8,
        "Monthly": 18,
        "Daily": 30,
    }

    frequency_map = {"Yearly": 1, "Quarterly": 4, "Monthly": 12, "Daily": 365}

    context_length = {"Yearly": 8, "Quarterly": 10, "Monthly": 24, "Daily": 30}

    frequency_pd = {"Yearly": "Y", "Quarterly": "Q", "Monthly": "M", "Daily": "D"}

    data_group = [*horizons_map]
    frequency = [*frequency_map.values()]
    horizons = [*horizons_map.values()]

    @classmethod
    def load_data(cls, group):
        pass
