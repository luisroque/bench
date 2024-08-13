import plotnine as p9
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots

from codebase.load_data.config import DATASETS_FREQ, REFERENCE_MODELS

datasets = list(DATASETS_FREQ.keys())
eval_wf = EvaluationWorkflow(datasets=datasets, baseline="SNaive")

for i, reference_model in enumerate(REFERENCE_MODELS):
    df_avg_rank_n_datasets = eval_wf.avg_rank_n_datasets(reference_model)

    cherry_picking = Plots.average_rank_n_barplot(
        df_avg_rank_n_datasets, "n", reference_model
    ) + p9.labs(y=f"{reference_model}")

    cherry_picking.save(
        f"assets/plots/3.{i}_cherry_picking_{reference_model}.pdf", width=5, height=5
    )
