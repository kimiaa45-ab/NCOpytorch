import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from nets.encoder.gnn_encoder import GNNEncoder
from problem.problem_NCO import NCODataset
import torch

# تعریف انکودر
embedding_dim = 128
encoder = GNNEncoder(
    n_layers=3,
    hidden_dim=embedding_dim,
    aggregation="sum",
    norm="layer",
    learn_norm=True,
    track_norm=False,
    gated=True,
)

# گرفتن یک نمونه از دیتاست
dataset = NCODataset(
    json_dir="data/processed",
    num_samples=2,
    batch_size=2,
    max_components=90,
    max_edges=30,
    max_nodes=14,
    max_node_edges=196,
)
services_data, nodes_data = dataset[0]

# کدگذاری سرویس‌ها
service_embeddings = encoder(services_data.x, services_data.edge_index)
print("Service embeddings shape:", service_embeddings.shape)
print("Sample service embeddings:", service_embeddings[:5])
print("Shape of edge_index:", services_data.edge_index.shape)
print("Unique values in edge_index:", torch.unique(services_data.edge_index))
print("Unique values in edge_attr:", torch.unique(services_data.edge_attr))
# کدگذاری گره‌ها
node_embeddings = encoder(nodes_data.x, nodes_data.edge_index)
print("Node embeddings shape:", node_embeddings.shape)
print("Sample node embeddings:", node_embeddings[:5])

# بررسی وجود NaN یا مقادیر غیرعادی
assert not torch.isnan(service_embeddings).any(), "Service embeddings contain NaN"
assert not torch.isnan(node_embeddings).any(), "Node embeddings contain NaN"
