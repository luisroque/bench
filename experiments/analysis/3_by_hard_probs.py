import pandas as pd
import plotnine as p9

from codebase.evaluation.rope import RopeAnalysis
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.utils import LogTransformation
from codebase.evaluation.plotting import Plots

ROPE = 5
LOG = False

results_df = EvaluationWorkflow.read_all_results(['M3', 'Tourism', 'M4'])

eval_wf = EvaluationWorkflow(results_df,
                             baseline='SNaive',
                             reference='NHITS')

df_all = eval_wf.eval_by_series()
df = eval_wf.get_hard_series(df_all)

shortfall = eval_wf.get_expected_shortfall(df, 0.95)

if LOG:
    df = LogTransformation.transform(df)

df_m = eval_wf.melt_data_by_series(df)
df_ranks_m = eval_wf.melt_data_by_series(df.rank(axis=1))

# df_all.rank(axis=1).mean().sort_values()
# df_all.rank(axis=1).std()
# (df_all.rank(axis=1) < 2).mean()
#
# added_error = []
# for i, r in df_all.iterrows():
#     # print(i)
#     ranks = r.rank()
#     try:
#         best_mod = ranks[ranks < 2].index[0]
#
#         extra_err = r - r[best_mod]
#         # extra_err = 100 * ((r - r[best_mod]) / r[best_mod])
#
#         added_error.append(extra_err)
#     except IndexError:
#         continue
#
# added_error_df = pd.concat(added_error, axis=1).T
# added_error_df.mean()
#


wr_rope = RopeAnalysis.get_probs(df, rope=ROPE, reference=eval_wf.reference)
wr_rope0 = RopeAnalysis.get_probs(df, rope=0, reference=eval_wf.reference)

df_avg = df.mean().reset_index()
df_avg.columns = ['Model', 'Error']

plot7_0 = Plots.error_distribution_baseline(df=df_all,
                                            baseline='SNaive',
                                            thr=eval_wf.hard_thr)
plot7 = Plots.average_error_barplot(df_avg) + \
        p9.labs(y='SMAPE on difficult series')

plot8 = Plots.average_error_barplot(shortfall) + \
        p9.labs(y='Expected shortfall on difficult series')
# plot2 = Plots.error_dist_by_model(df_m)
plot9 = Plots.error_dist_by_model(df_ranks_m) + \
        p9.labs(y='Rank distribution on difficult series', x='')
plot10a = Plots.result_with_rope_bars(wr_rope)
plot10b = Plots.result_with_rope_bars(wr_rope0)

plot7_0.save('assets/plots/plot7_0.pdf', width=9, height=5)
plot7.save('assets/plots/plot7.pdf', width=5, height=5)
plot8.save('assets/plots/plot8.pdf', width=5, height=5)
plot10a.save('assets/plots/plot10a.pdf', width=5, height=5)
plot10b.save('assets/plots/plot10b.pdf', width=5, height=5)
