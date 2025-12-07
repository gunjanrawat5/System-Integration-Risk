# make_pyg_graphs.py
import pandas as pd
import torch
from torch_geometric.data import Data

# Node feature columns to use
NODE_FEATURES = [
    "in_degree",
    "out_degree",
    "degree",
    "betweenness",
    "closeness",
    "eigenvector",
    "pagerank",
    "downstream_count",
    "upstream_count",
    "is_articulation",   # will be cast to 0/1
    "importance_score",
]

LABEL_MAP = {"low": 0, "medium": 1, "high": 2}


def load_system_graphs(nodes_csv="all_systems_node_features.csv",
                       edges_csv="all_systems_edges.csv"):
    """
    Returns a dict: system_id -> torch_geometric.data.Data
    Each Data is a separate system graph (no mixing).
    """
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)

    graphs = {}

    for system_id, df_nodes_sys in df_nodes.groupby("system_id"):
        df_nodes_sys = df_nodes_sys.reset_index(drop=True)

        node_names = df_nodes_sys["node"].tolist()
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        df_edges_sys = df_edges[df_edges["system_id"] == system_id]

        edge_list = []
        for _, row in df_edges_sys.iterrows():
            src = row["source"]
            dst = row["target"]
            if src in name_to_idx and dst in name_to_idx:
                edge_list.append([name_to_idx[src], name_to_idx[dst]])

        if not edge_list:
            continue

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

        # bool -> int
        df_nodes_sys = df_nodes_sys.copy()
        df_nodes_sys["is_articulation"] = df_nodes_sys["is_articulation"].astype(int)

        x = torch.tensor(df_nodes_sys[NODE_FEATURES].values, dtype=torch.float)
        y = torch.tensor(
            df_nodes_sys["risk_label"].map(LABEL_MAP).values,
            dtype=torch.long,
        )

        data = Data(x=x, edge_index=edge_index, y=y)
        data.node_names = node_names
        data.system_id = system_id

        graphs[system_id] = data

    return graphs


if __name__ == "__main__":
    gs = load_system_graphs()
    for sid, g in gs.items():
        print(sid, "| nodes:", g.num_nodes, "edges:", g.num_edges)
