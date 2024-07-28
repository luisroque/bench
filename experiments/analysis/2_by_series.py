import plotnine as p9

from codebase.evaluation.rope import RopeAnalysis
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.utils import LogTransformation
from codebase.evaluation.plotting import Plots

ROPE = 5
LOG = False
REFERENCE = "DeepAR"

eval_wf = EvaluationWorkflow(datasets=["M3", "Tourism", "M4"], baseline="SNaive")

df = eval_wf.eval_by_series()

shortfall = eval_wf.get_expected_shortfall(df, 0.95)

if LOG:
    df = LogTransformation.transform(df)

df_m = eval_wf.error_by_model(df)
df_ranks_m = eval_wf.rank_by_model(df)

wr_rope = RopeAnalysis.get_probs(df, rope=ROPE, reference=REFERENCE)
wr_rope0 = RopeAnalysis.get_probs(df, rope=0, reference=REFERENCE)

shortfall = Plots.average_error_barplot(shortfall) + p9.labs(
    y="Expected shortfall across all series"
)
# plot5_0 = Plots.error_dist_by_model(df_m)
ranks_dist = Plots.rank_dist_by_model(df_ranks_m) + p9.labs(y="Rank distribution", x="")
ranks_dist_dataset = Plots.rank_dist_by_model_dataset(
    df_ranks_m,
) + p9.labs(y="Rank distribution", x="")
ranks_dist_dataset_freq = Plots.rank_dist_by_model_dataset(
    df_ranks_m, "Dataset_Frequency"
) + p9.labs(y="Rank distribution", x="")
rope_analysis = Plots.result_with_rope_bars(wr_rope)
rope_analysis_rope0 = Plots.result_with_rope_bars(wr_rope0)

shortfall.save("assets/plots/overall_shortfall.pdf", width=5, height=5)
ranks_dist.save("assets/plots/ranks_dist.pdf", width=5, height=5)
ranks_dist_dataset.save("assets/plots/ranks_dist_dataset.pdf", width=5, height=5)
ranks_dist_dataset_freq.save(
    "assets/plots/ranks_dist_dataset_freq.pdf", width=5, height=5
)
rope_analysis.save("assets/plots/rope_analysis.pdf", width=5, height=5)
rope_analysis_rope0.save("assets/plots/rope_analysis_rope0.pdf", width=5, height=5)
