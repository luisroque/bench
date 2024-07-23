import pandas as pd

from codebase.load_data.config import DATASETS

data_list = []

datasets = {
    "Tourism": ["Monthly", "Quarterly"],
    "M3": ["Monthly", "Quarterly", "Yearly"],
    "M4": ["Monthly", "Quarterly", "Yearly"],
    "M5": ["Daily"],
}

for data_name, groups in datasets.items():
    for group in groups:
        data_cls = DATASETS[data_name]
        ds = data_cls.load_data(group)
        ds["group"] = group
        ds["unique_id"] = ds["unique_id"].apply(lambda x: f"{data_name}_{x}")
        ds["dataset"] = data_name

        data_list.append(ds)

df = pd.concat(data_list)

df_groups = df.groupby(["dataset", "group"])

# TODO: update the below - add each dataset correct metrics to the main class
info = {}
for g, df_g in df_groups:
    print(g)

    info[g] = {
        "n_ts": len(df_g["unique_id"].unique()),
        "n_obs": df_g.shape[0],
        "avg_len": df_g.groupby("unique_id").apply(lambda x: len(x)).median(),
        "h": 6,
        "input_size": 8,
        "freq": 1,
    }

df_info = pd.DataFrame(info).T.astype(int)
df_info.loc["Total", :] = df_info.sum().values


print(df_info.astype(str).to_latex(caption="asdasda", label="tab:data"))
