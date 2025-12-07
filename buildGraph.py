import pandas as pd
import networkx as nx
import numpy as np

# ------------- CONFIG: your systems & file names -----------------

SYSTEMS = [
    {
        "system_id": "payment_gateway_01",
        "components_csv": "components_paymentgateway.csv",
        "edges_csv": "edgesPaymentGateway.csv",
    },
    {
        "system_id": "food_ordering_01",
        "components_csv": "components_foodordering.csv",
        "edges_csv": "edgesFoodOrdering.csv",
    },
    {
        "system_id": "rideshare_backend_01",
        "components_csv": "components_ridesharebackend.csv",
        "edges_csv": "edgesRideshareBackend.csv",
    },
    {
        "system_id": "notification_service_01",
        "components_csv": "components_notificationservice.csv",
        "edges_csv": "edgesNotificationService.csv",
    },
    {
        "system_id": "analytics_pipeline_01",
        "components_csv": "components_analytics.csv",
        "edges_csv": "edgesAnalyticsPipeline.csv",
    },

    # --- New 7 Systems ---
    {
        "system_id": "billing_platform_01",
        "components_csv": "components_billingplatform.csv",
        "edges_csv": "edgesBillingPlatform.csv",
    },
    {
        "system_id": "support_ticketing_01",
        "components_csv": "components_supportticketing.csv",
        "edges_csv": "edgesSupportTicketing.csv",
    },
    {
        "system_id": "recommendation_service_01",
        "components_csv": "components_recommendationservice.csv",
        "edges_csv": "edgesRecommendationService.csv",
    },
    {
        "system_id": "identity_platform_01",
        "components_csv": "components_identityplatform.csv",
        "edges_csv": "edgesIdentityPlatform.csv",
    },
    {
        "system_id": "media_transcoding_01",
        "components_csv": "components_mediapipeline.csv",
        "edges_csv": "edgesMediaPipeline.csv",
    },
    {
        "system_id": "ad_serving_platform_01",
        "components_csv": "components_adplatform.csv",
        "edges_csv": "edgesAdPlatform.csv",
    },
    {
        "system_id": "partner_integration_hub_01",
        "components_csv": "components_partnerhub.csv",
        "edges_csv": "edgesPartnerHub.csv",
    },
]



# ------------- GRAPH & FEATURE HELPERS -----------------


def build_graph(components_path: str, edges_path: str) -> nx.DiGraph:
    """Create a directed graph from components + edges CSV."""
    df_nodes = pd.read_csv(components_path)
    df_edges = pd.read_csv(edges_path)

    G = nx.DiGraph()

    # Use component name as node ID
    for name in df_nodes["name"]:
        G.add_node(name)

    for _, row in df_edges.iterrows():
        src = row["source"]
        dst = row["target"]
        relation = row.get("relation", "unidirectional")
        if src not in G:
            G.add_node(src)
        if dst not in G:
            G.add_node(dst)
        G.add_edge(src, dst, relation=relation)

    return G


def compute_node_features(system_id: str, G: nx.DiGraph) -> pd.DataFrame:
    """Compute features and risk labels for all nodes of a system."""
    nodes = list(G.nodes())

    # Degree-based
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    deg = dict(G.degree())

    # Centralities
    bet = nx.betweenness_centrality(G, normalized=True)
    clo = nx.closeness_centrality(G)
    try:
        eig = nx.eigenvector_centrality_numpy(G)
    except Exception:
        # Fallback if eigenvector fails (rare on tiny graphs)
        eig = {n: 0.0 for n in nodes}
    pr = nx.pagerank(G)

    # Reachability
    downstream = {n: len(nx.descendants(G, n)) for n in nodes}
    upstream = {n: len(nx.ancestors(G, n)) for n in nodes}

    # Articulation (in undirected view)
    undirected = G.to_undirected()
    articulation_set = set(nx.articulation_points(undirected))

    df = pd.DataFrame({"node": nodes})
    df["system_id"] = system_id
    df["in_degree"] = df["node"].map(in_deg)
    df["out_degree"] = df["node"].map(out_deg)
    df["degree"] = df["node"].map(deg)
    df["betweenness"] = df["node"].map(bet)
    df["closeness"] = df["node"].map(clo)
    df["eigenvector"] = df["node"].map(eig)
    df["pagerank"] = df["node"].map(pr)
    df["downstream_count"] = df["node"].map(downstream)
    df["upstream_count"] = df["node"].map(upstream)
    df["is_articulation"] = df["node"].apply(lambda n: n in articulation_set)

    # Add heuristic risk labels
    df["risk_label"] = assign_risk_labels(df)

    return df

def _minmax(series: pd.Series) -> pd.Series:
    smin, smax = series.min(), series.max()
    if smax == smin:
        return pd.Series(0.0, index=series.index)  # all same → 0
    return (series - smin) / (smax - smin)


def assign_risk_labels(df: pd.DataFrame) -> pd.Series:
    """
    Assign low / medium / high risk per system using a combined importance score.

    Score is based on:
        - degree
        - betweenness
        - downstream_count
        - articulation bonus

    Then:
        bottom 0.33  -> low
        middle        -> medium
        top 0.33      -> high
    """

    # Normalized sub-scores
    deg_norm  = _minmax(df["degree"])
    bet_norm  = _minmax(df["betweenness"])
    down_norm = _minmax(df["downstream_count"])

    # Articulation bonus (0 or 1, we’ll scale it down)
    art_bonus = df["is_articulation"].astype(float)

    # Combined importance score
    df["importance_score"] = (
        0.4 * deg_norm +
        0.4 * bet_norm +
        0.4 * down_norm +
        0.2 * art_bonus
    )

    # Quantile thresholds on this *single* score
    q_low  = df["importance_score"].quantile(0.33)
    q_high = df["importance_score"].quantile(0.66)

    def rule(score, is_art):
        if score >= q_high:
            return "high"
        if score <= q_low:
            return "low"
        return "medium"

    return df.apply(lambda row: rule(row["importance_score"], row["is_articulation"]), axis=1)




# ------------- MAIN PIPELINE -----------------


def main():
    all_dfs = []

    for cfg in SYSTEMS:
        print(f"Processing system: {cfg['system_id']}")
        G = build_graph(cfg["components_csv"], cfg["edges_csv"])
        df_feats = compute_node_features(cfg["system_id"], G)
        all_dfs.append(df_feats)

    all_nodes = pd.concat(all_dfs, ignore_index=True)
    all_nodes.to_csv("all_systems_node_features.csv", index=False)
    print("Saved node features to all_systems_node_features.csv")

    # Optional: also save a combined edges file for all systems
    all_edges = []
    for cfg in SYSTEMS:
        df_e = pd.read_csv(cfg["edges_csv"]).copy()
        df_e["system_id"] = cfg["system_id"]
        all_edges.append(df_e)
    all_edges_df = pd.concat(all_edges, ignore_index=True)
    all_edges_df.to_csv("all_systems_edges.csv", index=False)
    print("Saved all edges to all_systems_edges.csv")


if __name__ == "__main__":
    main()
