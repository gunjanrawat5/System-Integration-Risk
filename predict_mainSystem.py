import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from pyGraphs import load_system_graphs   # same module you used in trainGNN.py


# Same GCN architecture as in trainGNN.py
class RiskGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_classes=3):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x


LABEL_INV = {0: "low", 1: "medium", 2: "high"}


def main():
    # 1) Load graphs (now includes main_system_01)
    graphs = load_system_graphs()
    print("Available systems:", list(graphs.keys()))

    system_id = "main_system_01"  # <-- whatever you used
    data = graphs[system_id]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Build model and load trained weights
    in_dim = data.num_node_features
    model = RiskGCN(in_dim=in_dim, hidden_dim=32, num_classes=3).to(device)
    model.load_state_dict(torch.load("risk_gcn_system_split.pth", map_location=device))
    model.eval()

    # 3) Run inference
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1).cpu().numpy()

    # 4) Save predictions to CSV
    df_out = pd.DataFrame({
        "node": data.node_names,
        "system_id": data.system_id,
        "pred_label_idx": preds,
        "pred_label": [LABEL_INV[i] for i in preds],
        "p_low": probs[:, 0].cpu().numpy(),
        "p_medium": probs[:, 1].cpu().numpy(),
        "p_high": probs[:, 2].cpu().numpy(),
    })

    out_file = "main_system_risk_predictions.csv"
    df_out.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")


if __name__ == "__main__":
    main()
