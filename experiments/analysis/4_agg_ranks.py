import plotnine as p9
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots

from codebase.load_data.config import DATASETS_FREQ

datasets = list(DATASETS_FREQ.keys())
eval_wf = EvaluationWorkflow(datasets=datasets, baseline="SNaive")

eval_agg_rank = eval_wf.compute_agg_rank()

top = Plots.top_barplot(eval_agg_rank)

top.save(f"assets/plots/4.1_top.pdf", width=5, height=5)
