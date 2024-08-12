import typing
import os
import random

import numpy as np
import pandas as pd
from neuralforecast.losses.numpy import mape, mae, smape, rmae

from codebase.load_data.config import DATASETS, N, SAMPLE_COUNT, REFERENCE_MODELS


class EvaluationWorkflow:
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

        self.error_by_series = None
        self.eval_by_series()

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
        mean_errors = (
            self.error_by_series.groupby(["Dataset", "Frequency", "Model"])[
                ["First Horizon Error", "Last Horizon Error"]
            ]
            .mean()
            .reset_index()
        )
        mean_errors["Rank First Horizon"] = mean_errors.groupby(
            ["Dataset", "Frequency"]
        )["First Horizon Error"].rank(method="average")
        mean_errors["Rank Last Horizon"] = mean_errors.groupby(
            ["Dataset", "Frequency"]
        )["Last Horizon Error"].rank(method="average")

        melted_mean_errors = pd.melt(
            mean_errors,
            id_vars=["Model", "Dataset", "Frequency"],
            value_vars=["Rank First Horizon", "Rank Last Horizon"],
            var_name="Horizon",
            value_name="Rank",
        )

        melted_mean_errors["Horizon"] = melted_mean_errors["Horizon"].replace(
            {"Rank First Horizon": "First", "Rank Last Horizon": "Last"}
        )

        mean_rank = (
            melted_mean_errors.groupby(["Model", "Horizon"])["Rank"]
            .mean()
            .reset_index()
        )

        return mean_rank

    def eval_by_horizon_first_and_last_dist(self):
        mean_errors = (
            self.error_by_series.groupby(["Dataset", "Frequency", "Model"])[
                ["First Horizon Error", "Last Horizon Error"]
            ]
            .mean()
            .reset_index()
        )
        mean_errors["Rank First Horizon"] = mean_errors.groupby(
            ["Dataset", "Frequency"]
        )["First Horizon Error"].rank(method="average")
        mean_errors["Rank Last Horizon"] = mean_errors.groupby(
            ["Dataset", "Frequency"]
        )["Last Horizon Error"].rank(method="average")

        melted_mean_errors = pd.melt(
            mean_errors,
            id_vars=["Model", "Dataset", "Frequency"],
            value_vars=["Rank First Horizon", "Rank Last Horizon"],
            var_name="Horizon",
            value_name="Rank",
        )

        melted_mean_errors["Horizon"] = melted_mean_errors["Horizon"].replace(
            {"Rank First Horizon": "First", "Rank Last Horizon": "Last"}
        )

        return melted_mean_errors

    def eval_by_series(self):
        output_dir = "./assets/metrics/by_series/"
        os.makedirs(output_dir, exist_ok=True)

        dataset_names = "_".join(self.datasets)
        output_path = os.path.join(output_dir, f"{dataset_names}_error_by_series.csv")

        if os.path.exists(output_path):
            print(f"File {output_path} already exists. Loading existing data.")
            self.error_by_series = pd.read_csv(output_path)
            return

        self.read_all_results()
        cv_group = self.cv.groupby("unique_id")

        results_by_series = self._process_groups(cv_group)

        results_df = pd.concat(
            results_by_series.values(), keys=results_by_series.keys(), names=["Series"]
        ).reset_index(level=0)

        results_df.to_csv(output_path, index=False)
        print(f"Stored evaluation by series in {output_path}")
        self.error_by_series = results_df

    def _process_groups(self, cv_group):
        results_by_series = {}
        total_groups = len(cv_group)

        for i, (group_id, df) in enumerate(cv_group, start=1):
            results_by_series[group_id] = self._evaluate_group(df)
            if i % 100 == 0 or i == total_groups:
                print(f"Processed {i}/{total_groups} series")

        return results_by_series

    def _evaluate_group(self, df):
        result = self.run(df)
        result["Dataset"] = df["dataset"].unique()[0]
        result["Frequency"] = df["freq"].unique()[0]

        first_horizon_error = self._calculate_horizon_error(
            df, 1, "First Horizon Error"
        )
        last_horizon_error = self._calculate_horizon_error(
            df, df["horizon"].max(), "Last Horizon Error"
        )

        result = result.merge(first_horizon_error, on="Model", how="left")
        result = result.merge(last_horizon_error, on="Model", how="left")

        return result

    def _calculate_horizon_error(self, df, horizon_value, error_column_name):
        horizon_df = df[df["horizon"] == horizon_value]
        error_df = self.run(horizon_df)
        error_df.columns = ["Model", error_column_name]
        return error_df

    def eval_rank_total(self):
        mean_errors = (
            self.error_by_series.groupby(["Dataset", "Frequency", "Model"])["Error"]
            .mean()
            .reset_index()
        )
        mean_errors["Rank"] = mean_errors.groupby(["Dataset", "Frequency"])[
            "Error"
        ].rank(method="average")
        mean_rank = mean_errors.groupby("Model")["Rank"].mean().reset_index()

        return mean_rank

    def eval_rank_total_dist(self):
        mean_errors = (
            self.error_by_series.groupby(["Dataset", "Frequency", "Model"])["Error"]
            .mean()
            .reset_index()
        )
        mean_errors["Rank"] = mean_errors.groupby(["Dataset", "Frequency"])[
            "Error"
        ].rank(method="average")

        return mean_errors

    def eval_rank_by_freq(self):
        mean_errors = (
            self.error_by_series.groupby(["Dataset", "Frequency", "Model"])["Error"]
            .mean()
            .reset_index()
        )
        mean_errors["Rank"] = mean_errors.groupby(["Dataset", "Frequency"])[
            "Error"
        ].rank(method="average")
        mean_rank_by_freq = (
            mean_errors.groupby(["Frequency", "Model"])["Rank"].mean().reset_index()
        )

        return mean_rank_by_freq

    def eval_rank_by_dataset(self):
        mean_errors = (
            self.error_by_series.groupby(["Dataset", "Frequency", "Model"])["Error"]
            .mean()
            .reset_index()
        )
        mean_errors["Rank"] = mean_errors.groupby(["Dataset", "Frequency"])[
            "Error"
        ].rank(method="average")
        mean_rank_by_dataset = (
            mean_errors.groupby(["Dataset", "Model"])["Rank"].mean().reset_index()
        )

        return mean_rank_by_dataset

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
                print(f"    {group}")

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
                "Y": "Yearly",
                "Q": "Quarterly",
                "M": "Monthly",
                "D": "Daily",
                "H": "Hourly",
                "15T": "15T",
                "10T": "10M",
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

    def error_by_model(self):
        df = self.error_by_series
        df_output = df.groupby("Model")["Error"].mean().reset_index()

        return df_output

    def rank_by_model(self):
        df = self.error_by_series
        df["Dataset_Frequency"] = df["Dataset"] + df["Frequency"]
        df["Rank"] = df.groupby("Series")["Error"].rank(method="average")

        return df

    def avg_rank_n_datasets_random(self):
        output_dir = "./assets/metrics/by_n/"
        os.makedirs(output_dir, exist_ok=True)

        df = self.eval_rank_total_dist()
        df["Dataset_Frequency"] = df["Dataset"] + df["Frequency"]

        dataset_names = "_".join(df["Dataset"].unique())
        output_path_by_n = os.path.join(
            output_dir, f"{dataset_names}_avg_rank_by_n.csv"
        )

        if os.path.exists(output_path_by_n):
            print(f"File {output_path_by_n} already exists. Loading existing data.")
            return pd.read_csv(output_path_by_n)

        all_results = []
        for count in range(1, SAMPLE_COUNT + 1):
            for n in N:
                datasets = df["Dataset_Frequency"].unique().tolist()
                selected_datasets = random.sample(datasets, n)

                # Filter the DataFrame to include only rows with selected datasets
                filtered_df = df[df["Dataset_Frequency"].isin(selected_datasets)].copy()

                # Add new columns for selected datasets, n, and count
                filtered_df["Selected_Datasets"] = [selected_datasets] * len(
                    filtered_df
                )
                filtered_df["Selected_Datasets"] = filtered_df[
                    "Selected_Datasets"
                ].apply(tuple)
                filtered_df["n"] = n
                filtered_df["Sample_Count"] = count

                # Compute average rank for N
                filtered_df_avg = (
                    filtered_df.groupby(
                        ["Model", "Selected_Datasets", "n", "Sample_Count"]
                    )["Rank"]
                    .mean()
                    .reset_index()
                )

                # Rank again
                filtered_df_avg["Min_Rank"] = filtered_df_avg.groupby(
                    ["Selected_Datasets", "n", "Sample_Count"]
                )["Rank"].rank(method="min")

                all_results.append(filtered_df_avg)

        final_results_df = pd.concat(all_results, ignore_index=True)

        # Computing variance of the mean rank by samples for each N datasets that we random sample
        final_results_df.to_csv(output_path_by_n, index=False)
        print(f"Storing avg ranks for n datasets on {output_path_by_n}")

        return final_results_df

    def avg_rank_n_datasets(self, reference_model: str):
        df = self.error_by_series
        output_dir = "./assets/metrics/by_n/"
        os.makedirs(output_dir, exist_ok=True)

        df["Dataset_Frequency"] = df["Dataset"] + df["Frequency"]

        dataset_names = "_".join(df["Dataset"].unique())
        input_path_raw = os.path.join(output_dir, f"{dataset_names}_avg_rank_by_n.csv")

        if os.path.exists(input_path_raw):
            print(f"File {input_path_raw} already exists. Loading existing data.")
            raw_data = pd.read_csv(input_path_raw)
        else:
            self.avg_rank_n_datasets_random()
            raw_data = pd.read_csv(input_path_raw)

        ref_model_min_idx = (
            raw_data[raw_data["Model"] == reference_model]
            .groupby(["n"])["Min_Rank"]
            .idxmin()
        )
        best_model_data = raw_data.loc[ref_model_min_idx]

        filtered_data = raw_data[
            raw_data.set_index(["n", "Selected_Datasets", "Sample_Count"]).index.isin(
                best_model_data.set_index(
                    ["n", "Selected_Datasets", "Sample_Count"]
                ).index
            )
        ]
        filtered_data = filtered_data.drop("Rank", axis=1)
        filtered_data = filtered_data.sort_values(
            by=["n", "Selected_Datasets", "Sample_Count", "Min_Rank"]
        )
        filtered_data.rename(columns={"Min_Rank": "Rank"}, inplace=True)

        return filtered_data

    def _compute_avg_rank_n_datasets_by_series(self, df):
        df["Selected_Datasets"] = df["Selected_Datasets"].apply(tuple)
        grouped_by_experiment = (
            df.groupby(["Model", "n", "Sample_Count", "Selected_Datasets"])["Rank"]
            .mean()
            .reset_index(name="Rank")
        )

        return grouped_by_experiment

    def compute_agg_rank(self):
        all_models = []
        for model in REFERENCE_MODELS:
            avg_rank_model = self.avg_rank_n_datasets(model)
            ranks_reference_model = avg_rank_model.loc[avg_rank_model.Model == model]
            rank_n_model = avg_rank_model.loc[
                ranks_reference_model.groupby("n")["Rank"].idxmin()
            ]
            all_models.append(rank_n_model)
        all_models_concat = pd.concat(all_models, ignore_index=True)

        top = {}
        for i in range(1, 4):
            top_i = all_models_concat.loc[
                (all_models_concat["n"] == 1) & (all_models_concat["Rank"] <= i)
            ]
            top_i_perc = top_i.shape[0] / len(REFERENCE_MODELS)
            top[i] = top_i_perc

        top_df = pd.DataFrame(
            list(top.items()), columns=["Models in Top", "Percentage"]
        )
        return top_df
