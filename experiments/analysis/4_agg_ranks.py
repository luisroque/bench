import pandas as pd
import plotnine as p9
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots

from codebase.load_data.config import DATASETS_FREQ, N

datasets = list(DATASETS_FREQ.keys())
eval_wf = EvaluationWorkflow(datasets=datasets, baseline="SNaive")

agg_rank_all_n = []
for n in N:
    eval_agg_rank = eval_wf.compute_agg_rank(n=n)
    eval_agg_rank["n"] = n
    agg_rank_all_n.append(eval_agg_rank)

agg_rank_all_n_df = pd.concat(agg_rank_all_n)
agg_rank_all_n_df_4 = agg_rank_all_n_df.loc[agg_rank_all_n_df.n == 4].copy()

top = Plots.top_barplot(agg_rank_all_n_df_4)
top_n = Plots.top_n_lineplot(agg_rank_all_n_df)

top.save(f"assets/plots/4.1_top_4.pdf", width=5, height=5)
top_n.save(f"assets/plots/4.1_top_n.pdf", width=5, height=5)
