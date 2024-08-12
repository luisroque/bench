import os
import plotnine as p9

from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots
from codebase.load_data.config import DATASETS_FREQ, DATASETS

datasets = list(DATASETS_FREQ.keys())

eval_wf = EvaluationWorkflow(datasets=datasets, baseline="SNaive")

rank_all = eval_wf.eval_rank_total()
rank_all_dist = eval_wf.eval_rank_total_dist()
rank_freq = eval_wf.eval_rank_by_freq()
rank_dataset = eval_wf.eval_rank_by_dataset()
rank_by_horizon = eval_wf.eval_by_horizon_first_and_last()
rank_by_horizon_dist = eval_wf.eval_by_horizon_first_and_last_dist()

overall_performance = (
    Plots.average_rank_barplot(rank_all)
    + p9.theme(
        axis_title_y=p9.element_text(size=7), axis_text_x=p9.element_text(size=11)
    )
    + p9.labs(y="Average rank across all datasets")
)
overall_performance_boxplot = (
    Plots.average_rank_boxplot(rank_all_dist)
    + p9.theme(
        axis_title_y=p9.element_text(size=7), axis_text_x=p9.element_text(size=11)
    )
    + p9.labs(y="Rank distribution across all datasets")
)
rank_freq = Plots.average_rank_by_freq(rank_freq) + p9.labs(y="Ranks")
rank_freq_dist = Plots.average_rank_by_freq_boxplot(rank_all_dist) + p9.labs(y="Ranks")

rank_dataset = Plots.average_rank_by_dataset(rank_dataset) + p9.labs(y="Rank")
rank_dataset_dist = Plots.average_rank_by_dataset_boxplot(rank_all_dist) + p9.labs(
    y="Rank"
)

rank_h = Plots().average_rank_by_horizons(df=rank_by_horizon)
rank_h_dist = Plots().average_rank_by_horizons_boxplot(df=rank_by_horizon_dist)

PATH_PLOTS = "assets/plots/"
os.makedirs(os.path.dirname(PATH_PLOTS), exist_ok=True)

overall_performance.save(PATH_PLOTS + "1.1_overall_rank.pdf", width=5, height=5)
overall_performance_boxplot.save(
    PATH_PLOTS + "1.2_overall_rank_boxplot.pdf", width=5, height=5
)
rank_freq.save(PATH_PLOTS + "1.3_rank_freq.pdf", width=9, height=5)
rank_freq_dist.save(PATH_PLOTS + "1.4_rank_freq_dist.pdf", width=9, height=5)
rank_dataset.save(PATH_PLOTS + "1.5_rank_dataset.pdf", width=9, height=5)
rank_dataset_dist.save(PATH_PLOTS + "1.6_rank_dataset_dist.pdf", width=9, height=5)
rank_h.save(PATH_PLOTS + "1.7_rank_h.pdf", width=8, height=5)
rank_h_dist.save(PATH_PLOTS + "1.8_rank_h_dist.pdf", width=8, height=5)
