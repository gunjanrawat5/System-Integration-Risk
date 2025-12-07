import pandas as pd

df = pd.read_csv("all_systems_node_features.csv")

print(df.groupby(["system_id", "risk_label"]).size())
