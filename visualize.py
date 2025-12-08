import pandas as pd

SYSTEM_ID = "main_system_01"   # pick your system

# Read the files
nodes_df = pd.read_csv("all_systems_node_features.csv")
edges_df = pd.read_csv("all_systems_edges.csv")

# 1) Filter to one system

nodes_sys = nodes_df[nodes_df["system_id"] == SYSTEM_ID].copy()
edges_sys = edges_df[edges_df["system_id"] == SYSTEM_ID].copy()


# 2) Gephi NODES file
gephi_nodes = nodes_sys.copy()

# Insert Id and Label at the front
gephi_nodes.insert(0, "Id", gephi_nodes["node"])
gephi_nodes.insert(1, "Label", gephi_nodes["node"])


# 3) Gephi EDGES file
# Assumes edges_sys has columns: source, target, relation, system_id (plus anything else)
gephi_edges = edges_sys.copy()

# Rename to what Gephi expects
gephi_edges = gephi_edges.rename(columns={
    "source": "Source",
    "target": "Target"
})

# Edge type for Gephi
relation_to_type = {
    "unidirectional": "Directed",
    "bidirectional": "Undirected"
}
gephi_edges["Type"] = gephi_edges["relation"].map(relation_to_type).fillna("Directed")

# column order
edge_cols = ["Source", "Target", "Type"]
if "relation" in gephi_edges.columns:
    edge_cols.append("relation")
if "system_id" in gephi_edges.columns:
    edge_cols.append("system_id")

gephi_edges = gephi_edges[edge_cols]


# 4) Save CSVs for Gephi
gephi_nodes.to_csv("gephi_nodes_Main.csv", index=False)
gephi_edges.to_csv("gephi_edges_Main.csv", index=False)

print("Saved gephi_nodes_system.csv and gephi_edges_system.csv")
