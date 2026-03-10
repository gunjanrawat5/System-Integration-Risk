# Knowledge Graphs for System Integration Risk

Understanding integration risk in modern distributed systems is difficult due to the complex dependencies between microservices, APIs, databases, messaging queues, and infrastructure components.

This project explores a hybrid approach combining **Large Language Models (LLMs), graph analytics, and Graph Neural Networks (GNNs)** to automatically analyze and predict integration risk across system architectures.

The pipeline converts system descriptions into dependency graphs, extracts structural features, generates heuristic risk labels, and trains a **Graph Convolutional Network (GCN)** to classify components as **low, medium, or high risk**.

---

# Problem

Modern distributed architectures contain dozens of interconnected components. A failure in one component can cascade through dependencies and disrupt the entire system.

### Key Question

> **Which components pose the highest integration risk in a system architecture?**

Traditional methods rely on manual analysis or simple heuristics that do not scale to large systems. This project investigates whether **graph learning models can better identify architectural bottlenecks and failure hotspots.**

---

# Project Pipeline

The project follows a multi-stage pipeline that transforms natural-language system descriptions into machine learning predictions.

```
LLM System Generator
        ↓
Architecture Description
        ↓
Parser
        ↓
Components CSV + Edges CSV
        ↓
Graph Construction (NetworkX)
        ↓
Feature Extraction
        ↓
Heuristic Risk Labeling
        ↓
Graph Neural Network Training
        ↓
Risk Prediction
```

---

# System Generation using LLMs

The pipeline begins by generating **synthetic system architectures** using prompts.

### Example Systems

- Payment gateway
- Notification service
- Food ordering platform
- Media processing pipeline

Each system typically contains:

- **10–20 components**
- **Directed dependencies between components**

The generated architecture is then converted into structured data.

### Outputs

```
components.csv
edges.csv
```

---

# Graph Construction

Each system is converted into a **directed dependency graph** using **NetworkX**.

- **Nodes** represent system components  
- **Edges** represent directed dependencies  

### Example

```
API Gateway → Lambda
Lambda → DynamoDB
Lambda → SQS
CloudFront → API Gateway
```

This representation enables structural analysis of system dependencies.

---

# Feature Extraction

Structural features are computed for every node in the graph.

## Degree Metrics

- `in_degree`
- `out_degree`
- `total_degree`

## Centrality Metrics

- betweenness centrality
- closeness centrality
- eigenvector centrality
- PageRank

## Reachability Metrics

- `upstream_count`
- `downstream_count`

## Structural Importance

- articulation points (nodes whose removal disconnects the graph)

These features capture both **local connectivity** and **global influence** within the architecture.

---

# Heuristic Risk Scoring

Initial risk labels are generated using a rule-based scoring function.

```
importance_score =
0.4 * degree_norm +
0.4 * betweenness_norm +
0.4 * downstream_norm +
0.2 * articulation_bonus
```

Nodes are classified into three categories:

| Risk Level | Criteria |
|------------|---------|
| Low | importance ≤ 33rd percentile |
| Medium | between 33rd and 66th percentile |
| High | ≥ 66th percentile |

These heuristic labels serve as **training targets for the GNN model**.

---

# Graph Neural Network

The model used is a **Graph Convolutional Network (GCN)** implemented using **PyTorch Geometric**.

### Architecture

```
GCNConv(input_dim, hidden_dim)
ReLU
GCNConv(hidden_dim, hidden_dim)
ReLU
Linear(hidden_dim, num_classes)
```

### Parameters

| Parameter | Value |
|----------|------|
| Node features | 11 |
| Hidden dimension | 32 |
| Classes | 3 |
| Training epochs | 200 |

---

# Dataset

The dataset consists of **12 synthetic system architectures**.

| Split | Systems |
|------|--------|
| Training | 8 |
| Validation | 4 |

Each graph contains:

- nodes = system components
- edges = dependencies
- features = structural metrics
- labels = risk categories

---

# Results

### Training Performance

- Training accuracy improved from **~62% → ~83%**
- Training loss decreased from **~0.85 → ~0.43**

### Validation Performance

- Validation accuracy: **~52%**
- Random baseline: **~33%**

Although the dataset is small, the model demonstrates **meaningful generalization across unseen architectures.**

---

# Key Observations

## Heuristic Model

The heuristic model tends to:

- Overestimate risk
- Label many nodes as high-risk
- Be overly sensitive to connectivity

## GNN Predictions

The GNN produces more realistic risk distributions.

### High-risk nodes

- Request coordination services
- Lambda processing layers
- API orchestration services

### Lower-risk nodes

- Infrastructure services
- Storage layers
- Monitoring systems

The model identifies **true coordination hubs rather than simply highly connected nodes.**

---

# Visualizations

Example outputs include:

### Heuristic Risk Graph

Shows many components labeled high risk due to high connectivity.

### GNN Predicted Risk Graph

More balanced predictions highlighting actual system bottlenecks.

### Architecture Dependency Graph

Visualization of component relationships and dependencies.

---

# Technologies Used

- Python
- NetworkX
- Pandas
- PyTorch
- PyTorch Geometric
- Matplotlib
- Large Language Models (for system generation)

---

# Future Improvements

### Larger Dataset

Training on more architectures would improve generalization.

### Edge Features

Incorporating edge types such as:

- data flow
- control flow
- external API calls

### Semantic Node Features

Adding component types:

- databases
- compute services
- messaging layers
- caching systems

### Advanced GNN Models

Possible upgrades:

- Graph Attention Networks (GAT)
- GraphSAGE
- heterogeneous graph models

---

# Conclusion

This project demonstrates a scalable approach for **system-level integration risk analysis** by combining:

- LLM-based architecture generation
- graph analytics
- Graph Neural Networks

The results show that **graph learning models can identify critical components and structural bottlenecks more effectively than simple heuristics.**

With richer datasets and features, this framework could evolve into a practical tool for **automated reliability analysis in distributed systems.**

---

# Author

**Gunjan Rawat**  
MS Computer Science  
Stevens Institute of Technology
