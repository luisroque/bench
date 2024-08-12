import pandas as pd

from codebase.load_data.config import DATASETS_FREQ, DATASETS

data_list = []

for data_name, groups in DATASETS_FREQ.items():
    for group in groups:
        data_cls = DATASETS[data_name]
        ds = data_cls.load_data(group)
        ds["group"] = group
        ds["unique_id"] = ds["unique_id"].apply(lambda x: f"{data_name}_{x}")
        ds["dataset"] = data_name
        ds["horizon"] = data_cls.horizons_map[group]
        ds["freq"] = data_cls.frequency_map[group]

        data_list.append(ds)

df = pd.concat(data_list)

df_groups = df.groupby(["dataset", "group", "horizon", "freq"])

info = {}
for g, df_g in df_groups:
    info[g] = {
        "n_ts": len(df_g["unique_id"].unique()),
        "n_obs": df_g.shape[0],
        "avg_len": df_g.groupby("unique_id").apply(lambda x: len(x)).median(),
        "h": g[2],
        "freq": g[3],
    }

df_info = pd.DataFrame(info).T.astype(int)
df_info.loc["Total", :] = df_info.sum().values


print(df_info.astype(str).to_latex(caption="asdasda", label="tab:data"))
