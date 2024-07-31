import plotnine as p9
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots

N = [2, 3, 4, 5, 6]
REFERENCE_MODELS = ["TiDE", "DeepAR", "Informer", "NBEATS", "RNN"]

eval_wf = EvaluationWorkflow(datasets=["M3", "Tourism", "M4"], baseline="SNaive")

df = eval_wf.eval_by_series()

df_m = eval_wf.error_by_model(df)
df_ranks_m = eval_wf.rank_by_model(df)

for reference_model in REFERENCE_MODELS:
    df_avg_rank_n_datasets = eval_wf.avg_rank_n_datasets(df, reference_model)

    cherry_picking = Plots.average_rank_barplot(
        df_avg_rank_n_datasets, "n", reference_model
    ) + p9.labs(y="Cherry-Picking Datasets")

    cherry_picking.save(
        f"assets/plots/cherry_picking_{reference_model}.pdf", width=5, height=5
    )
