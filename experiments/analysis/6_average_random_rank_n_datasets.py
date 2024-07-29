import plotnine as p9
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots

N = [1, 2, 3, 4, 5, 6]

sample_count = 20

eval_wf = EvaluationWorkflow(datasets=["M3", "Tourism", "M4"], baseline="SNaive")

df = eval_wf.eval_by_series()

df_m = eval_wf.error_by_model(df)
df_ranks_m = eval_wf.rank_by_model(df)

df_avg_rank_n_datasets, df_avg_rank_n_datasets_by_series = (
    eval_wf.avg_rank_n_datasets_random(df, N, sample_count)
)

ranks_dist_experiments_datasets = Plots.rank_dist_by_model(
    df_avg_rank_n_datasets
) + p9.labs(y="Rank distribution", x="")
ranks_dist_experiments_datasets_by_n = Plots.rank_dist_by_model_dataset(
    df_avg_rank_n_datasets, colname="n"
) + p9.labs(y="Rank distribution by number of datasets", x="")
ranks_dist_experiments_datasets_by_n_by_series = Plots.rank_dist_by_model_dataset(
    df_avg_rank_n_datasets_by_series, colname="n"
) + p9.labs(
    y="Rank distribution by number of datasets computed for individual series", x=""
)


ranks_dist_experiments_datasets.save(
    "assets/plots/ranks_dist_experiments_datasets.pdf", width=5, height=5
)
ranks_dist_experiments_datasets_by_n.save(
    "assets/plots/ranks_dist_experiments_datasets_by_n.pdf", width=5, height=5
)
ranks_dist_experiments_datasets_by_n_by_series.save(
    "assets/plots/ranks_dist_experiments_datasets_by_n_by_series.pdf", width=5, height=5
)
