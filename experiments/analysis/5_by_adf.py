# from codebase.evaluation.workflow import EvaluationWorkflow
# from codebase.features.ndiffs import StationarityWorkflow
# from codebase.evaluation.plotting import Plots
#
# # DATA = ['M3']
# DATA = ['M3', 'Tourism', 'M4']
#
# stationary_df = StationarityWorkflow.uid_ndiffs(DATA)
# stationary_df.to_csv('./assets/results/stationarity.csv', index=False)
# results_df = EvaluationWorkflow.read_all_results(DATA)
#
# eval_wf = EvaluationWorkflow(results_df, baseline='SNaive', reference='NHITS')
#
# df = eval_wf.eval_by_series()
# df = df.reset_index()
#
# df = df.merge(stationary_df, on='index', how='left').set_index('index')
#
# df_st = df.drop('No. diffs', axis=1)
# df_st = df_st.groupby('Stationary').mean().reset_index()
# df_melted = df_st.melt('Stationary')
#
# plot13 = Plots.average_error_by_stationarity(df_melted, 'Stationary')
# plot13.save(f'assets/plots/plot13a.pdf', width=10, height=5)
#
# df_st = df.drop('Stationary', axis=1)
# df_st = df_st.groupby('No. diffs').mean().reset_index()
# df_melted = df_st.melt('No. diffs')
#
# plot13 = Plots.average_error_by_stationarity(df_melted, 'No. diffs')
# plot13.save(f'assets/plots/plot13b.pdf', width=10, height=5)
