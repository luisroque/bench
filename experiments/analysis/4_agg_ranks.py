import pandas as pd
from codebase.evaluation.workflow import EvaluationWorkflow
from codebase.evaluation.plotting import Plots
from plotnine import ggplot

from codebase.load_data.config import DATASETS_FREQ, N

datasets = list(DATASETS_FREQ.keys())
eval_wf = EvaluationWorkflow(datasets=datasets, baseline="SNaive")


def save_plot_with_font_embedding(
    plot: ggplot,
    filename: str,
    font: str = "Arial",
    width: float = 5,
    height: float = 5,
):
    from matplotlib import rcParams

    # set font for embedding
    rcParams["pdf.fonttype"] = 42  # embed TrueType fonts
    rcParams["ps.fonttype"] = 42  # for PostScript
    rcParams["font.family"] = font

    plot.save(filename, format="pdf", verbose=False, width=width, height=height)
    print(
        f"Plot saved to {filename} with font '{font}' embedded and dimensions {width}x{height} inches."
    )


agg_rank_all_n = []
for n in N:
    eval_agg_rank = eval_wf.compute_agg_rank(n=n)
    eval_agg_rank["n"] = n
    agg_rank_all_n.append(eval_agg_rank)

agg_rank_all_n_df = pd.concat(agg_rank_all_n)
agg_rank_all_n_df_4 = agg_rank_all_n_df.loc[agg_rank_all_n_df.n == 4].copy()

top = Plots.top_barplot(agg_rank_all_n_df_4)
top_n = Plots.top_n_lineplot(agg_rank_all_n_df)


save_plot_with_font_embedding(top, f"assets/plots/4.1_top_4.pdf", width=5, height=5)
save_plot_with_font_embedding(top_n, f"assets/plots/4.1_top_n.pdf", width=5, height=5)
