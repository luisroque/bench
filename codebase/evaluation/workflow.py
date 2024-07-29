import typing
import os
import random

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import mape, mae, smape, rmae

from codebase.load_data.config import DATASETS


class EvaluationWorkflow:
    # todo get metadata from index
    RESULTS_DIR = "./assets/results/by_group"

    ALL_METADATA = [
        "unique_id",
        "ds",
        "cutoff",
        "horizon",
        "hi",
        "lo",
        "freq",
        "y",
        "is_anomaly",
        "dataset",
        "group",
    ]
    ORIGINAL_FEATURES = ["is_anomaly", "horizon", "unique_id", "freq"]

    def __init__(self, baseline: str, datasets: typing.List[str]):
        self.func = smape
        self.datasets = datasets

        self.baseline = baseline
        self.hard_thr = -1
        self.hard_series = []
        self.hard_scores = pd.DataFrame()
        self.error_on_hard = pd.DataFrame()

        self.cv = None
        self.read_all_results()
        self.models = self.get_model_names()

    def eval_by_horizon_full(self):
        cv_g = self.cv.groupby("freq")
        results_by_g = {}
        for g, df in cv_g:
            fh = df["horizon"].sort_values().unique()
            eval_fh = {}
            for h in fh:
                cv_fh = df.query(f"horizon<={h}")

                eval_fh[h] = self.run(cv_fh)

            results = pd.DataFrame(eval_fh).T
            results_by_g[g] = results

        results_df = pd.concat(results_by_g).reset_index()
        results_df = results_df.rename(
            columns={"level_0": "Frequency", "level_1": "Horizon"}
        )
        results_df = results_df.melt(["Frequency", "Horizon"])
        results_df = results_df.rename(columns={"variable": "Model", "value": "Error"})

        return results_df

    def eval_by_horizon_first_and_last(self):
        cv_grouped = self.cv.groupby("unique_id")

        first_horizon, last_horizon = [], []
        for g, df in cv_grouped:
            first_horizon.append(df.iloc[0, :])
            last_horizon.append(df.iloc[-1, :])

        first_h_df = pd.concat(first_horizon, axis=1).T
        last_h_df = pd.concat(last_horizon, axis=1).T

        errf_df = self.run(first_h_df)
        errl_df = self.run(last_h_df)
        err_df = errf_df.merge(errl_df, on="Model")
        err_df.columns = ["Model", "First horizon", "Last horizon"]

        err_melted_df = err_df.melt("Model")
        err_melted_df.columns = ["Model", "Horizon", "Error"]

        return err_melted_df

    def eval_by_series(self):
        cv_group = self.cv.groupby("unique_id")

        output_dir = "./assets/metrics/by_series/"
        os.makedirs(output_dir, exist_ok=True)

        dataset_names = "_".join(self.datasets)
        output_path = f"./assets/metrics/by_series/{dataset_names}_error_by_series.csv"

        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Loading existing data.")
            return pd.read_csv(output_path)

        results_by_series = {}
        total_groups = len(cv_group)
        for i, (g, df) in enumerate(cv_group, start=1):
            results_by_series[g] = self.run(df)
            if i % 100 == 0 or i == total_groups:
                print(f"Processed {i}/{total_groups} series")

        results_df = pd.concat(
            {k: df for k, df in results_by_series.items()}, names=["Series"]
        ).reset_index(level=0)

        results_df.to_csv(output_path, index=False)
        print(f"Storing eval by series on {output_path}")
        return results_df

    def eval_by_anomalies(self):
        cv_group = self.cv.groupby("unique_id")

        results_by_series, cv_df = {}, []
        for g, df in cv_group:
            # print(g)
            df_ = df.loc[df["is_anomaly_95"] > 0, :]
            if df_.shape[0] > 0:
                cv_df.append(df_)
                results_by_series[g] = self.run(df_)

        cv_df = pd.concat(cv_df).reset_index(drop=True)
        result_all = self.run(cv_df)

        results_df = pd.concat(results_by_series, axis=1).T

        return results_df, result_all

    def eval_by_anomalous_series(self):
        cv_group = self.cv.groupby("unique_id")

        results_by_series, cv_df = {}, []
        for g, df in cv_group:
            # print(g)
            if df["is_anomaly_95"].sum() > 0:
                cv_df.append(df)
                results_by_series[g] = self.run(df)

        cv_df = pd.concat(cv_df).reset_index(drop=True)
        result_all = self.run(cv_df)
        results_df = pd.concat(results_by_series, axis=1).T

        return results_df, result_all

    @staticmethod
    def get_expected_shortfall(df, thr=0.9):
        output_dir = "./assets/metrics/by_group/"
        os.makedirs(output_dir, exist_ok=True)

        # get the dataset
        df["Dataset"] = df["Series"].str.extract(r"([a-zA-Z0-9]+)")
        df["Dataset_Frequency"] = df["Series"].str.extract(
            r"([a-zA-Z0-9]+_[a-zA-Z0-9])"
        )

        dataset_names = "_".join(df["Dataset"].unique())
        output_path = os.path.join(
            output_dir, f"{dataset_names}_expected_shortfall.csv"
        )

        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Loading existing data.")
            return pd.read_csv(output_path)

        # calculate the 95th percentile for each Dataset and Model combination
        percentile_95 = (
            df.groupby(["Dataset", "Model"])["Error"].quantile(thr).reset_index()
        )

        df = df.merge(percentile_95, on=["Dataset", "Model"])
        df = df.rename(columns={"Error_x": "Error", "Error_y": "95th Percentile"})

        worst_5_percent = df[df["Error"] >= df["95th Percentile"]]
        worst_5_percent = worst_5_percent.drop(columns=["95th Percentile"])

        shortfall = worst_5_percent.groupby(["Model"])["Error"].mean().reset_index()
        shortfall.to_csv(output_path, index=False)
        print(f"Storing shortfall on {output_path}")

        return shortfall

    def eval_by_frequency(self):
        cv_group = self.cv.groupby("freq")

        results_by_freq = {}
        for g, df in cv_group:
            results_by_freq[g] = self.run(df)

        results_df = pd.concat(
            {k: df for k, df in results_by_freq.items()}, names=["Frequency"]
        ).reset_index(level=0)
        return results_df

    def run(self, cv: typing.Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if cv is None or cv.empty:
            cv = self.cv
        evaluation = {}
        for model in self.models:
            evaluation[model] = self.func(y=cv["y"], y_hat=cv[model])

        evaluation = pd.Series(evaluation)
        evaluation = evaluation.reset_index()
        evaluation.columns = ["Model", "Error"]

        return evaluation

    def get_hard_series(self, error_by_unique_id: pd.DataFrame):

        output_dir = "./assets/metrics/by_group/"
        os.makedirs(output_dir, exist_ok=True)

        error_by_unique_id["Dataset"] = error_by_unique_id["Series"].str.extract(
            r"([a-zA-Z0-9]+)"
        )
        error_by_unique_id["Dataset_Frequency"] = error_by_unique_id[
            "Series"
        ].str.extract(r"([a-zA-Z0-9]+_[a-zA-Z0-9])")

        dataset_names = "_".join(error_by_unique_id["Dataset"].unique())

        output_path = os.path.join(output_dir, f"{dataset_names}_hard_series.csv")
        output_path_thr = os.path.join(
            output_dir, f"{dataset_names}_hard_series_thr.csv"
        )

        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Loading existing data.")
            return pd.read_csv(output_path), pd.read_csv(output_path_thr)

        snaive_error = error_by_unique_id[error_by_unique_id.Model == self.baseline]

        percentile_95_baseline = (
            snaive_error.groupby(["Dataset_Frequency"])["Error"]
            .quantile(0.95)
            .reset_index()
        )

        error_by_unique_id = error_by_unique_id.merge(
            percentile_95_baseline, on=["Dataset_Frequency"]
        )
        error_by_unique_id = error_by_unique_id.rename(
            columns={"Error_x": "Error", "Error_y": f"95th Percentile {self.baseline}"}
        )

        hard_series = error_by_unique_id[
            error_by_unique_id["Error"]
            >= error_by_unique_id[f"95th Percentile {self.baseline}"]
        ]
        hard_series = hard_series.drop(columns=[f"95th Percentile {self.baseline}"])

        error_on_hard = hard_series.groupby(["Model"])["Error"].mean().reset_index()
        error_on_hard.to_csv(output_path, index=False)
        percentile_95_baseline.to_csv(output_path_thr, index=False)

        print(f"Storing error on hard series on {output_path}")

        return error_on_hard, percentile_95_baseline

    def get_model_names(self):
        metadata = self.cv.columns.str.contains("|".join(self.ALL_METADATA))
        models = self.cv.loc[:, ~metadata].columns.tolist()

        return models

    def map_forecasting_horizon_col(self):
        cv_g = self.cv.groupby("unique_id")

        horizon = []
        for g, df in cv_g:
            h = np.asarray(range(1, df.shape[0] + 1))
            hs = {
                "horizon": h,
                "ds": df["ds"].values,
                "unique_id": df["unique_id"].values,
            }
            hs = pd.DataFrame(hs)
            horizon.append(hs)

        horizon = pd.concat(horizon)
        horizon.head()

        self.cv = self.cv.merge(horizon, on=["unique_id", "ds"])

    def read_all_results(self):
        dataset_list = self.datasets

        output_dir = "./assets/metrics/all/"
        os.makedirs(output_dir, exist_ok=True)

        dataset_names = "_".join(dataset_list)

        output_path = os.path.join(output_dir, f"{dataset_names}_all_results.csv")

        if os.path.exists(output_path):
            print(f"Loading preprocessed data from {output_path}")
            self.cv = pd.read_csv(output_path)
            return

        results = []
        for ds in dataset_list:
            print(ds)
            for group in DATASETS[ds].data_group:
                print(group)

                try:
                    group_df = pd.read_csv(f"{self.RESULTS_DIR}/{ds}_{group}_all.csv")
                except FileNotFoundError:
                    continue

                if "Unnamed: 0" in group_df.columns:
                    group_df = group_df.drop("Unnamed: 0", axis=1)

                group_df["freq"] = DATASETS[ds].frequency_pd[group]
                group_df["dataset"] = ds

                results.append(group_df)

        results_df = pd.concat(results, axis=0)
        results_df["unique_id"] = results_df.apply(
            lambda x: f'{x["dataset"]}_{x["unique_id"]}', axis=1
        )
        results_df["freq"] = results_df["freq"].map(
            {
                "QS": "Quarterly",
                "MS": "Monthly",
                "M": "Monthly",
                "Q": "Quarterly",
                "Y": "Yearly",
            }
        )

        self.cv = results_df.rename(
            columns={
                "AutoARIMA": "ARIMA",
                "SeasonalNaive": "SNaive",
                "AutoETS": "ETS",
                "SESOpt": "SES",
                "AutoTheta": "Theta",
                "CrostonOptimized": "Croston",
            }
        )
        self.map_forecasting_horizon_col()
        self.cv.to_csv(output_path, index=False)
        print(f"Storing preprocessed data on {output_path}")

    @staticmethod
    def error_by_model(df: pd.DataFrame):
        df_output = df.groupby("Model")["Error"].mean().reset_index()

        return df_output

    @staticmethod
    def rank_by_model(df: pd.DataFrame):
        df["Dataset"] = df["Series"].str.extract(r"([a-zA-Z0-9]+)")
        df["Dataset_Frequency"] = df["Series"].str.extract(
            r"([a-zA-Z0-9]+_[a-zA-Z0-9])"
        )
        df["Rank"] = df.groupby("Series")["Error"].rank(method="average")

        return df

    def avg_rank_n_datasets_random(
        self, df: pd.DataFrame, N: typing.List[int], sample_count: int
    ):
        output_dir = "./assets/metrics/by_n/"
        os.makedirs(output_dir, exist_ok=True)

        # get the dataset
        df["Dataset"] = df["Series"].str.extract(r"([a-zA-Z0-9]+)")
        df["Dataset_Frequency"] = df["Series"].str.extract(
            r"([a-zA-Z0-9]+_[a-zA-Z0-9])"
        )

        dataset_names = "_".join(df["Dataset"].unique())
        output_path_raw = os.path.join(
            output_dir, f"{dataset_names}_raw_avg_rank_n_datasets.csv"
        )
        output_path_by_n = os.path.join(
            output_dir, f"{dataset_names}_avg_rank_by_n.csv"
        )
        output_path_by_n_by_series = os.path.join(
            output_dir, f"{dataset_names}_avg_rank_by_n_by_series.csv"
        )

        if os.path.exists(output_path_by_n) and os.path.exists(
            output_path_by_n_by_series
        ):
            print(
                f"File {output_path_by_n} and {output_path_by_n_by_series} already exists. Loading existing data."
            )
            return pd.read_csv(output_path_by_n), pd.read_csv(
                output_path_by_n_by_series
            )

        all_results = []
        for count in range(1, sample_count + 1):
            for n in N:
                datasets = df["Dataset_Frequency"].unique().tolist()
                selected_datasets = random.sample(datasets, n)

                # Filter the DataFrame to include only rows with selected datasets
                filtered_df = df[df["Dataset_Frequency"].isin(selected_datasets)].copy()

                # Add new columns for selected datasets, n, and count
                filtered_df["Selected_Datasets"] = [selected_datasets] * len(
                    filtered_df
                )
                filtered_df["n"] = n
                filtered_df["Sample_Count"] = count

                all_results.append(filtered_df)

        final_results_df = pd.concat(all_results, ignore_index=True)
        final_results_df["Selected_Datasets"] = final_results_df[
            "Selected_Datasets"
        ].apply(tuple)
        final_results_df.to_csv(output_path_raw, index=False)
        print(f"Storing RAW avg ranks for n datasets on {output_path_raw}")

        # Computing variance of the mean rank by samples for each N datasets that we random sample
        avg_by_n_datasets = self._compute_avg_rank_n_datasets(final_results_df)
        avg_by_n_datasets.to_csv(output_path_by_n, index=False)
        print(f"Storing avg ranks for n datasets on {output_path_by_n}")

        # Computing variance of the mean rank by samples for each N datasets that we random sample
        avg_by_n_datasets_by_series = self._compute_avg_rank_n_datasets_by_series(
            final_results_df
        )
        avg_by_n_datasets_by_series.to_csv(output_path_by_n_by_series, index=False)
        print(
            f"Storing avg ranks for n datasets BY series on {output_path_by_n_by_series}"
        )

        return avg_by_n_datasets, avg_by_n_datasets_by_series

    def avg_rank_n_datasets(self, df: pd.DataFrame, reference_model: str):
        output_dir = "./assets/metrics/by_n/"
        os.makedirs(output_dir, exist_ok=True)

        # get the dataset
        df["Dataset"] = df["Series"].str.extract(r"([a-zA-Z0-9]+)")
        df["Dataset_Frequency"] = df["Series"].str.extract(
            r"([a-zA-Z0-9]+_[a-zA-Z0-9])"
        )

        dataset_names = "_".join(df["Dataset"].unique())
        input_path_raw = os.path.join(
            output_dir, f"{dataset_names}_raw_avg_rank_n_datasets.csv"
        )

        if os.path.exists(input_path_raw):
            print(f"File {input_path_raw} already exists. Loading existing data.")
            raw_data = pd.read_csv(input_path_raw)
        grouped_by_experiment = (
            raw_data.groupby(["Model", "n", "Sample_Count", "Selected_Datasets"])[
                "Error"
            ]
            .mean()
            .reset_index(name="Error")
        )
        grouped_by_experiment["Rank"] = grouped_by_experiment.groupby(
            ["n", "Sample_Count", "Selected_Datasets"]
        )["Error"].rank(method="min", ascending=True)
        min_ranks = (
            grouped_by_experiment.groupby(["Model", "n"])["Rank"].min().reset_index()
        )
        min_ranks = pd.merge(
            min_ranks, grouped_by_experiment, on=["Model", "n", "Rank"]
        )
        min_ranks.rename(columns={"Rank": "Min_Rank"}, inplace=True)
        criteria = min_ranks[min_ranks.Model == reference_model][
            ["n", "Sample_Count", "Selected_Datasets"]
        ]
        reference_model_results = grouped_by_experiment.merge(
            criteria, on=["n", "Sample_Count", "Selected_Datasets"], how="inner"
        )

        return reference_model_results

    def _compute_avg_rank_n_datasets(self, df):
        grouped_by_experiment = (
            df.groupby(["Model", "n", "Sample_Count", "Selected_Datasets"])["Error"]
            .mean()
            .reset_index(name="Error")
        )
        grouped_by_experiment["Rank"] = grouped_by_experiment.groupby(
            ["n", "Sample_Count", "Selected_Datasets"]
        )["Error"].rank(method="min", ascending=True)

        return grouped_by_experiment

    def _compute_avg_rank_n_datasets_by_series(self, df):
        df["Selected_Datasets"] = df["Selected_Datasets"].apply(tuple)
        grouped_by_experiment = (
            df.groupby(["Model", "n", "Sample_Count", "Selected_Datasets"])["Rank"]
            .mean()
            .reset_index(name="Rank")
        )

        return grouped_by_experiment
