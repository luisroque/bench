import os
import plotnine as p9

from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots

eval_wf = EvaluationWorkflow(datasets=["M3", "Tourism", "M4", "M5"], baseline="SNaive")

error_all = eval_wf.run()
error_by_freq = eval_wf.eval_by_frequency()
error_by_fullhorizon = eval_wf.eval_by_horizon_first_and_last()

# Overall performance
overall_performance = (
    Plots.average_error_barplot(error_all)
    + p9.theme(
        axis_title_y=p9.element_text(size=7), axis_text_x=p9.element_text(size=11)
    )
    + p9.labs(y="Average SMAPE across all series")
)
# Performance by frequency
error_freq = Plots.average_error_by_freq(error_by_freq) + p9.labs(y="SMAPE")
# Performance by forecasting horizon
# plot3 = Plots.average_error_by_horizon_freq(error_by_fullhorizon)
error_h = Plots().average_error_by_horizons(df=error_by_fullhorizon)

PATH_PLOTS = "assets/plots/"
os.makedirs(os.path.dirname(PATH_PLOTS), exist_ok=True)

overall_performance.save(PATH_PLOTS + "overall_performance.pdf", width=5, height=5)
error_freq.save(PATH_PLOTS + "error_freq.pdf", width=9, height=5)
error_h.save(PATH_PLOTS + "error_h.pdf", width=8, height=5)
