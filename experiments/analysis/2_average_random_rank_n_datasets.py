import plotnine as p9
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots
from codebase.load_data.config import DATASETS_FREQ


datasets = list(DATASETS_FREQ.keys())
eval_wf = EvaluationWorkflow(datasets=datasets, baseline="SNaive")

df_ranks_m = eval_wf.rank_by_model()

df_avg_rank_n_datasets = eval_wf.avg_rank_n_datasets_random()

ranks_dist_experiments_datasets = Plots.rank_dist_by_model(
    df_avg_rank_n_datasets
) + p9.labs(y="Rank distribution", x="")
ranks_dist_experiments_datasets_by_n = Plots.rank_dist_by_model_dataset(
    df_avg_rank_n_datasets, colname="n"
) + p9.labs(y="Rank distribution by number of datasets", x="")


ranks_dist_experiments_datasets.save(
    "assets/plots/2.1_ranks_dist_experiments_datasets.pdf", width=5, height=5
)
ranks_dist_experiments_datasets_by_n.save(
    "assets/plots/2.2_ranks_dist_experiments_datasets_by_n.pdf", width=5, height=5
)
