class LoadDataset:
    DATASET_PATH = "./datasets"
    DATASET_NAME = ""

    horizons_map = {
        "Yearly": 6,
        "Quarterly": 8,
        "Monthly": 18,
        "Daily": 30,
        "Hourly": 48,  # Assuming a horizon of 48 hours
        "15T": 96,     # Assuming a horizon of 96 periods (24 hours)
        "10M": 144     # Assuming a horizon of 24 hours
    }

    frequency_map = {
        "Yearly": 1,
        "Quarterly": 4,
        "Monthly": 12,
        "Daily": 365,
        "Hourly": 24 * 365,
        "15T": 4 * 24 * 365,
        "10M": 6 * 24 * 365
    }

    context_length = {
        "Yearly": 8,
        "Quarterly": 10,
        "Monthly": 24,
        "Daily": 30,
        "Hourly": 48,
        "15T": 96,
        "10M": 144
    }

    frequency_pd = {
        "Yearly": "Y",
        "Quarterly": "Q",
        "Monthly": "M",
        "Daily": "D",
        "Hourly": "H",
        "15T": "15T",
        "10M": "10T"  # 10-minute intervals are denoted as '10T' in pandas
    }

    data_group = [*horizons_map]
    frequency = [*frequency_map.values()]
    horizons = [*horizons_map.values()]

    @classmethod
    def load_data(cls, group):
        pass