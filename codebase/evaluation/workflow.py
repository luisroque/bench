import typing

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import mape, mae, smape, rmae

from codebase.load_data.config import DATASETS


class EvaluationWorkflow:
    # todo get metadata from index
    RESULTS_DIR = './assets/results/by_group'

    ALL_METADATA = ['unique_id', 'ds', 'cutoff', 'horizon',
                    'hi', 'lo', 'freq', 'y', 'is_anomaly', 'dataset', 'group']
    ORIGINAL_FEATURES = ['is_anomaly', 'horizon', 'unique_id', 'freq']

    def __init__(self,
                 cv: pd.DataFrame,
                 baseline: str,
                 reference: str):
        self.func = smape

        self.baseline = baseline
        self.reference = reference
        self.cv = cv
        self.hard_thr = -1
        self.hard_series = []
        self.hard_scores = pd.DataFrame()
        self.error_on_hard = pd.DataFrame()

        self.map_forecasting_horizon_col()
        self.models = self.get_model_names()

    def eval_by_horizon_full(self, cv_: typing.Optional[pd.DataFrame] = None):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_g = cv.groupby('freq')
        results_by_g = {}
        for g, df in cv_g:
            fh = df['horizon'].sort_values().unique()
            eval_fh = {}
            for h in fh:
                cv_fh = df.query(f'horizon<={h}')

                eval_fh[h] = self.run(cv_fh)

            results = pd.DataFrame(eval_fh).T
            results_by_g[g] = results

        results_df = pd.concat(results_by_g).reset_index()
        results_df = results_df.rename(columns={'level_0': 'Frequency', 'level_1': 'Horizon'})
        results_df = results_df.melt(['Frequency', 'Horizon'])
        results_df = results_df.rename(columns={'variable': 'Model', 'value': 'Error'})

        return results_df

    def eval_by_horizon_first_and_last(self):
        cv_grouped = self.cv.groupby('unique_id')

        first_horizon, last_horizon = [], []
        for g, df in cv_grouped:
            first_horizon.append(df.iloc[0, :])
            last_horizon.append(df.iloc[-1, :])

        first_h_df = pd.concat(first_horizon, axis=1).T
        last_h_df = pd.concat(last_horizon, axis=1).T

        errf_df = self.run(first_h_df, return_df=True)
        errl_df = self.run(last_h_df, return_df=True)
        err_df = errf_df.merge(errl_df, on='Model')
        err_df.columns = ['Model', 'First horizon', 'Last horizon']

        err_melted_df = err_df.melt('Model')
        err_melted_df.columns = ['Model', 'Horizon', 'Error']

        return err_melted_df

    def eval_by_series(self, cv_: typing.Optional[pd.DataFrame] = None):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_group = cv.groupby('unique_id')

        results_by_series = {}
        for g, df in cv_group:
            results_by_series[g] = self.run(df)

        results_df = pd.concat(results_by_series, axis=1).T

        return results_df

    def eval_by_anomalies(self, cv_: typing.Optional[pd.DataFrame] = None):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_group = cv.groupby('unique_id')

        results_by_series, cv_df = {}, []
        for g, df in cv_group:
            # print(g)
            df_ = df.loc[df['is_anomaly_95'] > 0, :]
            if df_.shape[0] > 0:
                cv_df.append(df_)
                results_by_series[g] = self.run(df_)

        cv_df = pd.concat(cv_df).reset_index(drop=True)
        result_all = self.run(cv_df)

        results_df = pd.concat(results_by_series, axis=1).T

        return results_df, result_all

    def eval_by_anomalous_series(self, cv_: typing.Optional[pd.DataFrame] = None):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_group = cv.groupby('unique_id')

        results_by_series, cv_df = {}, []
        for g, df in cv_group:
            # print(g)
            if df['is_anomaly_95'].sum() > 0:
                cv_df.append(df)
                results_by_series[g] = self.run(df)

        cv_df = pd.concat(cv_df).reset_index(drop=True)
        result_all = self.run(cv_df)
        results_df = pd.concat(results_by_series, axis=1).T

        return results_df, result_all

    @staticmethod
    def get_expected_shortfall(df, thr=0.9):
        sf = df.apply(lambda x: x[x > x.quantile(thr)].mean())
        sf = sf.reset_index()
        sf.columns = ['Model', 'Error']

        return sf

    def eval_by_frequency(self,
                          cv_: typing.Optional[pd.DataFrame] = None,
                          long_format: bool = False):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        cv_group = cv.groupby('freq')

        results_by_freq = {}
        for g, df in cv_group:
            results_by_freq[g] = self.run(df)

        results_df = pd.concat(results_by_freq, axis=1)

        if long_format:
            results_df = results_df.reset_index().melt('index')
            results_df.columns = ['Model', 'Frequency', 'Error']

        return results_df

    def run(self, cv_: typing.Optional[pd.DataFrame] = None, return_df: bool = False):
        if cv_ is None:
            cv = self.cv.copy()
        else:
            cv = cv_.copy()

        evaluation = {}
        for model in self.models:
            # evaluation[model] = self.func(y=cv['y'], y_hat1=cv[model], y_hat2=cv['SNaive'])
            evaluation[model] = self.func(y=cv['y'], y_hat=cv[model])

        evaluation = pd.Series(evaluation)

        if return_df:
            evaluation = evaluation.reset_index()
            evaluation.columns = ['Model', 'Error']

        return evaluation

    def get_hard_series(self, error_by_unique_id: pd.DataFrame):

        assert self.baseline in self.cv.columns

        self.hard_thr = error_by_unique_id[self.baseline].quantile(0.95)
        self.hard_series = error_by_unique_id.loc[error_by_unique_id[self.baseline] > self.hard_thr,
                           :].index.tolist()
        error_on_hard = error_by_unique_id.loc[self.hard_series, :]

        return error_on_hard

    def get_model_names(self):
        metadata = self.cv.columns.str.contains('|'.join(self.ALL_METADATA))
        models = self.cv.loc[:, ~metadata].columns.tolist()

        return models

    def map_forecasting_horizon_col(self):
        cv_g = self.cv.groupby('unique_id')

        horizon = []
        for g, df in cv_g:
            h = np.asarray(range(1, df.shape[0] + 1))
            hs = {
                'horizon': h,
                'ds': df['ds'].values,
                'unique_id': df['unique_id'].values,
            }
            hs = pd.DataFrame(hs)
            horizon.append(hs)

        horizon = pd.concat(horizon)
        horizon.head()

        self.cv = self.cv.merge(horizon, on=['unique_id', 'ds'])

    @classmethod
    def read_all_results(cls, dataset_list=None):

        if dataset_list is None:
            dataset_list = DATASETS

        results = []
        for ds in dataset_list:
            print(ds)
            for group in DATASETS[ds].data_group:
                print(group)

                try:
                    group_df = pd.read_csv(f'{cls.RESULTS_DIR}/{ds}_{group}_all.csv')
                except FileNotFoundError:
                    continue

                if 'Unnamed: 0' in group_df.columns:
                    group_df = group_df.drop('Unnamed: 0', axis=1)

                group_df['freq'] = DATASETS[ds].frequency_pd[group]
                group_df['dataset'] = ds

                results.append(group_df)

        results_df = pd.concat(results, axis=0)
        results_df['unique_id'] = results_df.apply(lambda x: f'{x["dataset"]}_{x["unique_id"]}', axis=1)
        results_df['freq'] = results_df['freq'].map({
            'QS': 'Quarterly',
            'MS': 'Monthly',
            'M': 'Monthly',
            'Q': 'Quarterly',
            'Y': 'Yearly',
        })

        # results_df = results_df.drop(['NHITS'], axis=1)
        # results_df = results_df.drop(['Ensemble', 'WindowAverage'], axis=1)
        results_df = results_df.rename(columns={'AutoARIMA': 'ARIMA',
                                                'SeasonalNaive': 'SNaive',
                                                'AutoETS': 'ETS',
                                                'SESOpt': 'SES',
                                                'AutoTheta': 'Theta', })

        return results_df

    @staticmethod
    def melt_data_by_series(df: pd.DataFrame):
        df_melted = df.melt()
        df_melted.columns = ['Model', 'Error']

        return df_melted
