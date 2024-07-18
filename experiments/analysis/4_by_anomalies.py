import plotnine as p9

from codebase.evaluation.rope import RopeAnalysis
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.utils import LogTransformation
from codebase.evaluation.plotting import Plots

ROPE = 5
LOG = False
ANOMALY_TYPE = 'instances'
# ANOMALY_TYPE = 'series'

results_df = EvaluationWorkflow.read_all_results(['M3', 'Tourism', 'M4'])

eval_wf = EvaluationWorkflow(results_df, baseline='SNaive', reference='NHITS')

df_anomalies, summ_anomalies = eval_wf.eval_by_anomalies()
df_an_s, summ_anomalous = eval_wf.eval_by_anomalous_series()

if ANOMALY_TYPE == 'instances':
    expr = 'anomalies'
    df = df_anomalies.copy()
    summary = summ_anomalies.copy()
else:
    expr = 'series with anomalies'
    df = df_an_s.copy()  # 15725 time series
    summary = summ_anomalous.copy()

summary = summary.reset_index()
summary.columns = ['Model', 'Error']

shortfall = eval_wf.get_expected_shortfall(df, 0.95)

if LOG:
    df = LogTransformation.transform(df)

df_m = eval_wf.melt_data_by_series(df)
df_ranks_m = eval_wf.melt_data_by_series(df.rank(axis=1))

wr_rope = RopeAnalysis.get_probs(df,
                                 rope=ROPE,
                                 reference=eval_wf.reference)

plot11 = Plots.average_error_barplot(summary) + \
         p9.labs(y=f'SMAPE in {expr}')
plot12 = Plots.average_error_barplot(shortfall) + \
         p9.labs(y=f'Expected shortfall in {expr}')
# plot2 = Plots.error_dist_by_model(df_m)
plot13 = Plots.error_dist_by_model(df_ranks_m) + \
         p9.labs(y=f'Rank distribution on {expr}', x='')
plot14 = Plots.result_with_rope_bars(wr_rope)

plot11.save(f'assets/plots/plot11_{ANOMALY_TYPE}.pdf', width=5, height=5)
plot12.save(f'assets/plots/plot12_{ANOMALY_TYPE}.pdf', width=5, height=5)
