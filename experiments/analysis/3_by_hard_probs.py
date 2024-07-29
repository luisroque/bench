import pandas as pd
import plotnine as p9

from codebase.evaluation.rope import RopeAnalysis
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.utils import LogTransformation
from codebase.evaluation.plotting import Plots

ROPE = 5
LOG = False
REFERENCE = "DeepAR"

eval_wf = EvaluationWorkflow(datasets=["M3", "Tourism", "M4"], baseline="SNaive")

df_all = eval_wf.eval_by_series()
df_hard, df_hard_thr = eval_wf.get_hard_series(df_all)

shortfall = eval_wf.get_expected_shortfall(df_all, 0.95)

if LOG:
    df = LogTransformation.transform(df_hard)

df_m = eval_wf.error_by_model(df_all)
df_ranks_m = eval_wf.rank_by_model(df_all)

wr_rope = RopeAnalysis.get_probs(df_all, rope=ROPE, reference=REFERENCE)
wr_rope0 = RopeAnalysis.get_probs(df_all, rope=0, reference=REFERENCE)

error_dist_baseline = Plots.error_distribution_baseline(
    df=df_all,
    baseline=eval_wf.baseline,
    thr=df_all[df_all.Model == eval_wf.baseline]["Error"].quantile(0.95),
)
error_hard_series = Plots.average_error_barplot(df_hard) + p9.labs(
    y="SMAPE on difficult series"
)

shortfall_hard_series = Plots.average_error_barplot(shortfall) + p9.labs(
    y="Expected shortfall on difficult series"
)
rank_dist_hard_series = Plots.rank_dist_by_model(df_ranks_m) + p9.labs(
    y="Rank distribution on difficult series", x=""
)
rank_dist_hard_series_dataset_freq = Plots.rank_dist_by_model_dataset(
    df_ranks_m, "Dataset_Frequency"
) + p9.labs(y="Rank distribution on difficult series", x="")
rope_hard_series = Plots.result_with_rope_bars(wr_rope)
rope0_hard_series = Plots.result_with_rope_bars(wr_rope0)

error_dist_baseline.save("assets/plots/error_dist_baseline.pdf", width=9, height=5)
error_hard_series.save("assets/plots/error_hard_series.pdf", width=5, height=5)
shortfall_hard_series.save("assets/plots/shortfall_hard_series.pdf", width=5, height=5)
rank_dist_hard_series.save("assets/plots/rank_dist_hard_series.pdf", width=5, height=5)
rank_dist_hard_series_dataset_freq.save(
    "assets/plots/rank_dist_hard_series_dataset_freq.pdf", width=5, height=5
)
rope_hard_series.save("assets/plots/rope_hard_series.pdf", width=5, height=5)
rope0_hard_series.save("assets/plots/rope0_hard_series.pdf", width=5, height=5)
