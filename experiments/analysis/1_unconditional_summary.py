import plotnine as p9

from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots

results_df = EvaluationWorkflow.read_all_results(['M3', 'Tourism', 'M4'])
# results_df = EvaluationWorkflow.read_all_results(['M3'])

eval_wf = EvaluationWorkflow(results_df,
                             baseline='SNaive',
                             reference='NHITS')

error_all = eval_wf.run(return_df=True)
error_by_freq = eval_wf.eval_by_frequency(long_format=True)
# error_by_fullhorizon = eval_wf.eval_by_horizon_full()
error_by_flhorizon = eval_wf.eval_by_horizon_first_and_last()

# Overall performance
plot1 = Plots.average_error_barplot(error_all) + \
        p9.theme(axis_title_y=p9.element_text(size=7),
                 axis_text_x=p9.element_text(size=11)) + \
        p9.labs(y='Average SMAPE across all series')
# Performance by frequency
plot2 = Plots.average_error_by_freq(error_by_freq) + \
        p9.labs(y='SMAPE')
# Performance by forecasting horizon
# plot3 = Plots.average_error_by_horizon_freq(error_by_fullhorizon)
plot3 = Plots.average_error_by_horizons(error_by_flhorizon) + \
        p9.labs(y='SMAPE')

plot1.save('assets/plots/plot1.pdf', width=5, height=5)
plot2.save('assets/plots/plot2.pdf', width=9, height=5)
plot3.save('assets/plots/plot3.pdf', width=8, height=5)
