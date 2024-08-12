import pandas as pd
import plotnine as p9


class Plots:
    ORDER = [
        "SNaive",
        "RWD",
        "SES",
        "ARIMA",
        "ETS",
        "Theta",
        "Croston",
        "RNN",
        "TCN",
        "DeepAR",
        "NBEATS",
        "TiDE",
        "Informer",
    ]

    COLOR_MAP = {
        "RWD": "#69a765",
        "SNaive": "#69a765",
        "ETS": "#69a765",
        "ARIMA": "#69a765",
        "Theta": "#69a765",
        "SES": "#69a765",
        "Croston": "#69a765",
        "RNN": "#ed9121",
        "TCN": "#ed9121",
        "DeepAR": "#ed9121",
        "NHITS": "#ed9121",
        "TiDE": "#ed9121",
        "Informer": "#ed9121",
    }
    REFERENCE_COLOR = "red"

    @staticmethod
    def get_theme():
        theme_ = p9.theme_538(base_family="Palatino", base_size=12) + p9.theme(
            plot_margin=0.025,
            panel_background=p9.element_rect(fill="white"),
            plot_background=p9.element_rect(fill="white"),
            legend_box_background=p9.element_rect(fill="white"),
            strip_background=p9.element_rect(fill="white"),
            legend_background=p9.element_rect(fill="white"),
            axis_text_x=p9.element_text(size=9, angle=0),
            axis_text_y=p9.element_text(size=9),
            legend_title=p9.element_blank(),
        )

        return theme_

    @staticmethod
    def error_distribution_baseline(df: pd.DataFrame, baseline: str, thr: float):
        df_baseline = df[df["Model"] == baseline]

        plot = (
            p9.ggplot(df_baseline)
            + p9.aes(x="Error")
            + p9.geom_histogram(alpha=0.95, bins=30, color="black", fill="#69a765")
            + Plots.get_theme()
            + p9.geom_vline(xintercept=thr, colour="red", size=1)
            + p9.labs(x=f"Error distribution of {baseline}", y="Count")
        )

        return plot

    @classmethod
    def average_rank_barplot(cls, df: pd.DataFrame):
        df = df.sort_values("Rank", ascending=False).reset_index(drop=True)
        df["Model"] = pd.Categorical(
            df["Model"].values.tolist(), categories=df["Model"].values.tolist()
        )

        plot = (
            p9.ggplot(data=df, mapping=p9.aes(x="Model", y="Rank", fill="Model"))
            + p9.geom_bar(
                position="dodge",
                stat="identity",
                width=0.9,
                # fill='darkgreen'
            )
            + Plots.get_theme()
            + p9.theme(
                axis_title_y=p9.element_text(size=7),
                axis_text_x=p9.element_text(size=9),
            )
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.labs(x="", y="Rank across all datasets")
            + p9.coord_flip()
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def average_rank_n_barplot(
        cls, df: pd.DataFrame, facet_attribute=None, reference_model=None
    ):
        n1_sorted = df[df["n"] == 1].sort_values(by="Rank", ascending=False)
        model_order = n1_sorted["Model"].unique()
        df["Model"] = pd.Categorical(df["Model"], categories=model_order, ordered=True)

        color_map = cls.COLOR_MAP.copy()
        if reference_model:
            color_map[reference_model] = cls.REFERENCE_COLOR

        plot = (
            p9.ggplot(data=df, mapping=p9.aes(x="Model", y="Rank", fill="Model"))
            + p9.geom_bar(
                position="dodge",
                stat="identity",
                width=0.9,
                # fill='darkgreen'
            )
            + Plots.get_theme()
            + p9.theme(
                axis_title_y=p9.element_text(size=7),
                axis_text_x=p9.element_text(size=9),
            )
            + p9.scale_fill_manual(values=color_map)
            + p9.labs(x="", y="Error across all series")
            + p9.coord_flip()
            + p9.guides(fill=None)
        )

        if facet_attribute and facet_attribute in df.columns:
            plot += p9.facet_wrap(f"~ {facet_attribute}", scales="free")

        return plot

    @classmethod
    def average_rank_boxplot(cls, df: pd.DataFrame):
        model_medians = (
            df.groupby("Model")["Rank"]
            .median()
            .sort_values(ascending=False)
            .reset_index()
        )
        df_sorted = df.copy()
        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=model_medians["Model"].unique(), ordered=True
        )
        df_sorted = df_sorted.sort_values(["Model", "Rank"], ascending=[True, False])

        plot = (
            p9.ggplot(data=df_sorted, mapping=p9.aes(x="Model", y="Rank", fill="Model"))
            + p9.geom_boxplot()
            + Plots.get_theme()
            + p9.theme(
                axis_title_y=p9.element_text(size=7),
                axis_text_x=p9.element_text(size=9),
            )
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.labs(x="", y="Rank across all datasets")
            + p9.coord_flip()
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def average_rank_by_freq(cls, df: pd.DataFrame):
        df_sorted = df.sort_values(by=["Frequency", "Rank"]).reset_index(drop=True)

        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=df_sorted["Model"].unique(), ordered=True
        )

        plot = (
            p9.ggplot(
                data=df_sorted,
                mapping=p9.aes(x="Model", y="Rank", group="Frequency", fill="Model"),
            )
            + p9.facet_grid("~Frequency")
            + p9.geom_bar(position="dodge", stat="identity", width=0.9)
            + Plots.get_theme()
            + p9.theme(
                axis_text_x=p9.element_text(angle=60, size=7),
                strip_text=p9.element_text(size=10),
            )
            + p9.labs(x="", y="Rank")
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def average_rank_by_freq_boxplot(cls, df: pd.DataFrame):
        model_medians = (
            df.groupby("Model")["Rank"]
            .median()
            .sort_values(ascending=True)
            .reset_index()
        )
        df_sorted = df.copy()
        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=model_medians["Model"].unique(), ordered=True
        )
        df_sorted = df_sorted.sort_values(["Model", "Rank"], ascending=[True, False])

        plot = (
            p9.ggplot(
                data=df_sorted,
                mapping=p9.aes(x="Model", y="Rank", fill="Model"),
            )
            + p9.facet_grid("~Frequency")
            + p9.geom_boxplot()
            + Plots.get_theme()
            + p9.theme(
                axis_text_x=p9.element_text(angle=60, size=7),
                strip_text=p9.element_text(size=10),
            )
            + p9.labs(x="", y="Rank")
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def average_rank_by_dataset(cls, df: pd.DataFrame):
        df_sorted = df.sort_values(by=["Dataset", "Rank"]).reset_index(drop=True)

        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=df_sorted["Model"].unique(), ordered=True
        )

        plot = (
            p9.ggplot(
                data=df_sorted,
                mapping=p9.aes(x="Model", y="Rank", group="Dataset", fill="Model"),
            )
            + p9.facet_grid("~Dataset")
            + p9.geom_bar(position="dodge", stat="identity", width=0.9)
            + Plots.get_theme()
            + p9.theme(
                axis_text_x=p9.element_text(angle=60, size=7),
                strip_text=p9.element_text(size=10),
            )
            + p9.labs(x="", y="Rank")
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def average_rank_by_dataset_boxplot(cls, df: pd.DataFrame):
        model_medians = (
            df.groupby("Model")["Rank"]
            .median()
            .sort_values(ascending=True)
            .reset_index()
        )
        df_sorted = df.copy()
        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=model_medians["Model"].unique(), ordered=True
        )
        df_sorted = df_sorted.sort_values(["Model", "Rank"], ascending=[True, False])

        plot = (
            p9.ggplot(
                data=df_sorted,
                mapping=p9.aes(x="Model", y="Rank", fill="Model"),
            )
            + p9.facet_grid("~Dataset")
            + p9.geom_boxplot()
            + Plots.get_theme()
            + p9.theme(
                axis_text_x=p9.element_text(angle=60, size=7),
                strip_text=p9.element_text(size=10),
            )
            + p9.labs(x="", y="Rank")
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def average_win_rate_bar(cls, df: pd.DataFrame):
        df_sorted = df.sort_values(by=["Frequency", "Error"]).reset_index(drop=True)

        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=df_sorted["Model"].unique(), ordered=True
        )

        plot = (
            p9.ggplot(data=df, mapping=p9.aes(x="Group", y="Error", fill="Model"))
            + p9.geom_bar(position="dodge", stat="identity", width=0.9)
            + Plots.get_theme()
            + p9.labs(x="", y="Error")
            + p9.theme(axis_text_x=p9.element_text(size=12))
            + p9.geom_hline(yintercept=0.5, linetype="dashed", color="red", size=1.1)
        )

        return plot

    @classmethod
    def average_rank_by_horizons(cls, df: pd.DataFrame):
        df_sorted = df.sort_values(by=["Horizon", "Rank"]).reset_index(drop=True)

        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=df_sorted["Model"].unique(), ordered=True
        )

        plot = (
            p9.ggplot(
                data=df_sorted,
                mapping=p9.aes(x="Model", y="Rank", group="Horizon", fill="Model"),
            )
            + p9.facet_grid("~Horizon")
            + p9.geom_bar(position="dodge", stat="identity", width=0.9)
            + Plots.get_theme()
            + p9.theme(
                axis_text_x=p9.element_text(angle=60, size=7),
                strip_text=p9.element_text(size=10),
            )
            + p9.labs(x="", y="Rank")
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def average_rank_by_horizons_boxplot(cls, df: pd.DataFrame):
        model_medians = (
            df.groupby("Model")["Rank"]
            .median()
            .sort_values(ascending=True)
            .reset_index()
        )
        df_sorted = df.copy()
        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=model_medians["Model"].unique(), ordered=True
        )
        df_sorted = df_sorted.sort_values(["Model", "Rank"], ascending=[True, False])

        plot = (
            p9.ggplot(
                data=df_sorted,
                mapping=p9.aes(x="Model", y="Rank", fill="Model"),
            )
            + p9.facet_grid("~Horizon")
            + p9.geom_boxplot()
            + Plots.get_theme()
            + p9.theme(
                axis_text_x=p9.element_text(angle=60, size=7),
                strip_text=p9.element_text(size=10),
            )
            + p9.labs(x="", y="Rank")
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def average_error_by_stationarity(cls, df: pd.DataFrame, colname: str):
        df = df.rename(columns={"variable": "Model"})

        df["Model"] = pd.Categorical(df["Model"], categories=cls.ORDER[::-1])

        plot = (
            p9.ggplot(
                data=df,
                mapping=p9.aes(x="Model", y="value", group=colname, fill="Model"),
            )
            + p9.facet_grid(f"~{colname}")
            + p9.geom_bar(position="dodge", stat="identity", width=0.9)
            + Plots.get_theme()
            + p9.theme(
                axis_text_x=p9.element_text(angle=60, size=7),
                strip_text=p9.element_text(size=10),
            )
            + p9.labs(x="", y="SMAPE")
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.guides(fill=None)
        )

        return plot

    @staticmethod
    def average_error_by_horizon_freq(df: pd.DataFrame):
        plot = (
            p9.ggplot(df)
            + p9.aes(x="Horizon", y="Error", group="Model", color="Model")
            + p9.geom_line(size=1)
            + Plots.get_theme()
            + p9.facet_wrap("~Frequency", scales="free", ncol=1)
        )
        return plot

    @classmethod
    def rank_dist_by_model(cls, df: pd.DataFrame):
        model_medians = (
            df.groupby("Model")["Rank"].median().sort_values(ascending=False)
        )
        sorted_models = model_medians.index.tolist()

        df_sorted = df.copy()
        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=sorted_models, ordered=True
        )

        plot = (
            p9.ggplot(df_sorted, p9.aes(x="Model", y="Rank", fill="Model"))
            + Plots.get_theme()
            + p9.geom_boxplot(width=0.7, show_legend=False)
            + p9.coord_flip()
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.labs(x="Error distribution")
            + p9.guides(fill=None)
        )

        return plot

    @classmethod
    def rank_dist_by_model_dataset(cls, df: pd.DataFrame, colname: str = "Dataset"):
        model_medians = (
            df.groupby("Model")["Rank"].median().sort_values(ascending=False)
        )
        sorted_models = model_medians.index.tolist()

        df_sorted = df.copy()
        df_sorted["Model"] = pd.Categorical(
            df_sorted["Model"], categories=sorted_models, ordered=True
        )

        plot = (
            p9.ggplot(df_sorted, p9.aes(x="Model", y="Rank", fill="Model"))
            + Plots.get_theme()
            + p9.geom_boxplot(width=0.7, show_legend=False)
            + p9.coord_flip()
            + p9.scale_fill_manual(values=cls.COLOR_MAP)
            + p9.labs(x="Error distribution")
            + p9.guides(fill=None)
            + p9.facet_wrap(colname)
        )

        return plot

    @staticmethod
    def result_with_rope_bars(df: pd.DataFrame):
        plot = (
            p9.ggplot(df, p9.aes(fill="Result", y="Probability", x="Model"))
            + p9.geom_bar(position="stack", stat="identity")
            + p9.theme_classic(base_family="Palatino", base_size=12)
            + p9.theme(
                plot_margin=0.025,
                axis_text=p9.element_text(size=12),
                strip_text=p9.element_text(size=12),
                axis_text_x=p9.element_text(size=10, angle=0),
                legend_title=p9.element_blank(),
                legend_position="top",
            )
            + p9.labs(x="", y="Proportion of probability")
            + p9.scale_fill_hue()
            + p9.coord_flip()
        )

        return plot
