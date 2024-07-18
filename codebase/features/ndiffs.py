from typing import List

import pandas as pd
from pmdarima.arima import ndiffs

from codebase.load_data.config import DATASETS


class StationarityWorkflow:
    TEST = 'adf'

    @classmethod
    def uid_ndiffs(cls, datasets: List[str]):

        ndiffs_by_uid = {}
        for data_name in datasets:
            data_cls_ = DATASETS[data_name]

            for group in data_cls_.data_group:
                print(data_name, group)

                ds = data_cls_.load_data(group)
                ds_grouped = ds.groupby('unique_id')

                for tsname, df_ in ds_grouped:
                    s = df_['y'].values
                    ndiffs_by_uid[f'{data_name}_{tsname}'] = ndiffs(s, test=cls.TEST)

        ndiffs_by_ids = pd.Series(ndiffs_by_uid).reset_index()
        ndiffs_by_ids.columns = ['index', 'No. diffs']

        ndiffs_by_ids['Stationary'] = ndiffs_by_ids['No. diffs'] < 1
        ndiffs_by_ids['Stationary'] = ndiffs_by_ids['Stationary'].map({True: 'Stationary',
                                                                       False: 'Non-stationary'})

        return ndiffs_by_ids
