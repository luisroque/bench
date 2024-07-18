import plotnine as p9

from codebase.evaluation.rope import RopeAnalysis
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.utils import LogTransformation
from codebase.evaluation.plotting import Plots

ROPE = 5
LOG = False

# results_df = EvaluationWorkflow.read_all_results(['M3', 'Tourism', 'M4'])
results_df = EvaluationWorkflow.read_all_results(['M3'])

eval_wf = EvaluationWorkflow(results_df, baseline='SNaive', reference='NHITS')

df = eval_wf.eval_by_series()

shortfall = eval_wf.get_expected_shortfall(df, 0.95)

if LOG:
    df = LogTransformation.transform(df)

df_m = eval_wf.melt_data_by_series(df)
df_ranks_m = eval_wf.melt_data_by_series(df.rank(axis=1))

wr_rope = RopeAnalysis.get_probs(df, rope=ROPE, reference=eval_wf.reference)
wr_rope0 = RopeAnalysis.get_probs(df, rope=0, reference=eval_wf.reference)

plot4 = Plots.average_error_barplot(shortfall) + \
        p9.labs(y='Expected shortfall across all series')
# plot5_0 = Plots.error_dist_by_model(df_m)
plot5_1 = Plots.error_dist_by_model(df_ranks_m) + \
          p9.labs(y='Rank distribution', x='')
plot6_1 = Plots.result_with_rope_bars(wr_rope)
plot6_2 = Plots.result_with_rope_bars(wr_rope0)

plot4.save('assets/plots/plot4.pdf', width=5, height=5)
plot5_1.save('assets/plots/plot5_1.pdf', width=5, height=5)
plot6_1.save('assets/plots/plot6_1.pdf', width=5, height=5)
plot6_2.save('assets/plots/plot6_2.pdf', width=5, height=5)
