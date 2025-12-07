import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from pyGraphs import load_system_graphs


class RiskGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_classes=3):
        super().__init__()
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


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0

    loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)  # node logits
        loss = loss_fn(out, batch.y)            # all nodes in all graphs in batch
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_nodes
        preds = out.argmax(dim=-1)
        total_correct += (preds == batch.y).sum().item()
        total_nodes += batch.num_nodes

    avg_loss = total_loss / total_nodes
    acc = total_correct / total_nodes
    return avg_loss, acc


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_nodes = 0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)

            total_loss += loss.item() * batch.num_nodes
            preds = out.argmax(dim=-1)
            total_correct += (preds == batch.y).sum().item()
            total_nodes += batch.num_nodes

    avg_loss = total_loss / total_nodes
    acc = total_correct / total_nodes
    return avg_loss, acc


def main():
    graphs = load_system_graphs()
    system_ids = sorted(graphs.keys())
    print("Available systems:", system_ids)

    # Choose 1 system as validation, rest as train
    # You can change this to use a different val system if you want
    val_system_id = "analytics_pipeline_01"
    if val_system_id not in graphs:
        val_system_id = system_ids[-1]  # fallback

    train_graphs = [g for sid, g in graphs.items() if sid != val_system_id]
    val_graphs = [graphs[val_system_id]]

    print("Train systems:", [g.system_id for g in train_graphs])
    print("Val system:", val_system_id)

    train_loader = DataLoader(train_graphs, batch_size=len(train_graphs), shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = train_graphs[0].num_node_features
    model = RiskGCN(in_dim=in_dim, hidden_dim=32, num_classes=3).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    for epoch in range(1, 201):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss {train_loss:.3f} | train_acc {train_acc:.3f} | "
                f"val_loss {val_loss:.3f} | val_acc {val_acc:.3f}"
            )

    torch.save(model.state_dict(), "risk_gcn_system_split.pth")
    print("Saved model to risk_gcn_system_split.pth")


if __name__ == "__main__":
    main()
