import json
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import yaml
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from problem.state_nco import StateNCO
from utils.beam_search import beam_search

# لود تنظیمات از فایل YAML
config_path = "configs/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# استخراج پارامترهای تنظیمات
num_epochs = config["model"]["num_epochs"]
num_samples = config["model"]["num_samples"]
num_layers = config["model"]["num_layers"]
gpu_hidden_dim = config["model"]["gpu_hidden_dim"]
cpu_hidden_dim = config["model"]["cpu_hidden_dim"]
device = config["model"]["device"]
charnum_s = config["model"]["charnum_s"]
charnum_n = config["model"]["charnum_n"]
charnum_se = config["model"]["charnum_se"]
charnum_ne = config["model"]["charnum_ne"]
charnum_node = config["model"]["charnum_node"]
charnum_component = config["model"]["charnum_component"]
charnum_service = config["model"]["charnum_service"]
charnum_user = config["model"]["charnum_user"]  # تعداد user nodes
charnum_helper = config["model"]["charnum_helper"]  # تعداد helper nodes

# تعیین دستگاه و dim
if device == "auto":
    if torch.cuda.is_available():
        device = "cuda"
        hidden_dim = gpu_hidden_dim
        batch_size = gpu_hidden_dim
    else:
        device = "cpu"
        hidden_dim = cpu_hidden_dim
        batch_size = cpu_hidden_dim
else:
    device = device.lower()
    hidden_dim = gpu_hidden_dim if device == "cuda" else cpu_hidden_dim


class NCOProblem(object):
    NAME = "nco"

    def __init__(self):
        self.device = torch.device(device)

    def get_costs(self, services_data, nodes_data, assignments):
        """
        Compute the execution time and mask for the given assignments.

        Args:
            services_data (Data): Contains x [num_services, 5], edge_index [2, num_edges], edge_attr
            nodes_data (Data): Contains x [num_nodes, 5], edge_index [2, num_node_edges], edge_attr
            assignments (Tensor): [batch_size, num_services], node indices assigned to each component

        Returns:
            cost (Tensor): [batch_size], total execution time
            mask (Tensor): [batch_size, num_services], mask of valid assignments
        """
        batch_size = assignments.size(0)
        num_services = services_data.x.size(0) if services_data.x.dim() == 2 else services_data.x.size(1)
        num_nodes = nodes_data.x.size(0) if nodes_data.x.dim() == 2 else nodes_data.x.size(1)

        # Initialize cost and mask
        cost = torch.zeros(batch_size, device=self.device)
        mask = torch.ones(batch_size, num_services, dtype=torch.bool, device=self.device)

        # Check resource constraints
        for b in range(batch_size):
            assigned_nodes = assignments[b]  # [num_services]
            for i in range(num_services):
                node_idx = assigned_nodes[i]
                if node_idx >= num_nodes:
                    mask[b, i] = False
                    continue
                comp_cpu = services_data.x[i, 0] if services_data.x.dim() == 2 else services_data.x[b, i, 0]
                comp_memory = services_data.x[i, 1] if services_data.x.dim() == 2 else services_data.x[b, i, 1]
                comp_disk = services_data.x[i, 2] if services_data.x.dim() == 2 else services_data.x[b, i, 2]
                node_cpu = nodes_data.x[node_idx, 0] if nodes_data.x.dim() == 2 else nodes_data.x[b, node_idx, 0]
                node_memory = nodes_data.x[node_idx, 1] if nodes_data.x.dim() == 2 else nodes_data.x[b, node_idx, 1]
                node_disk = nodes_data.x[node_idx, 2] if nodes_data.x.dim() == 2 else nodes_data.x[b, node_idx, 2]

                if not (node_cpu >= comp_cpu and node_memory >= comp_memory and node_disk >= comp_disk):
                    mask[b, i] = False

        # Check topological constraints
        edge_index = services_data.edge_index
        for b in range(batch_size):
            for i in range(num_services):
                incoming_edges = edge_index[1] == i
                prereqs = edge_index[0, incoming_edges]
                if prereqs.numel() > 0:
                    prereq_assigned = assignments[b, prereqs]
                    if (prereq_assigned >= num_nodes).any() or (prereq_assigned == 0).any():
                        mask[b, i] = False

        # Compute execution time
        for b in range(batch_size):
            for i in range(num_services):
                node_idx = assignments[b, i]
                if mask[b, i]:
                    # Add component execution time (simplified as CPU requirement)
                    comp_cpu = services_data.x[i, 0] if services_data.x.dim() == 2 else services_data.x[b, i, 0]
                    cost[b] += comp_cpu
                    # Add communication cost based on node edges
                    incoming_edges = services_data.edge_index[1] == i
                    prereqs = services_data.edge_index[0, incoming_edges]
                    for prereq in prereqs:
                        prereq_node = assignments[b, prereq]
                        if prereq_node < num_nodes and node_idx < num_nodes:
                            # Find edge weight between assigned nodes
                            node_edge_mask = (nodes_data.edge_index[0] == prereq_node) & (nodes_data.edge_index[1] == node_idx)
                            if node_edge_mask.any():
                                edge_idx = node_edge_mask.nonzero(as_tuple=True)[0]
                                cost[b] += nodes_data.edge_attr[edge_idx].sum()

        return cost, mask

    def make_state(self, services_data, nodes_data):
        """Create initial state for NCO problem."""
        return StateNCO.initialize(services_data, nodes_data, device=self.device)

    def get_mask(self, state):
        """Get mask of valid assignments for the current state."""
        return state.get_mask()

    def get_graph_mask(self, state):
        """Get mask based on dependency graph."""
        return state.get_graph_mask()

    def beam_search(self, services_data, nodes_data, model, beam_size=1280):
        """Perform beam search for NCO assignments."""
        return beam_search(services_data, nodes_data, model, beam_size=beam_size)


class NCODataset(Dataset):
    """PyTorch Geometric Dataset for NCO SPP problem from JSON files."""

    def __init__(
        self,
        json_dir,
        num_samples=num_samples,
        batch_size=batch_size,
        max_edges=charnum_component * (charnum_component - 1),
        max_components=charnum_component * charnum_service,
        max_nodes=charnum_node + charnum_user + charnum_helper,
        max_node_edges=(charnum_node + charnum_user + charnum_helper)
        * (charnum_node + charnum_user + charnum_helper),
    ):
        """
        Initialize dataset from JSON files for PyTorch Geometric.

        Args:
            json_dir (str): Directory containing JSON files (e.g., 'data/processed')
            num_samples (int): Number of JSON file pairs to process
            batch_size (int): Batch size for DataLoader
            max_components (int): Max number of components for padding
            max_edges (int): Max number of edges for padding
            max_nodes (int): Max number of nodes for padding
            max_node_edges (int): Max number of node edges for padding
        """
        name = "NCO"
        super(NCODataset, self).__init__()

        self.json_dir = json_dir
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.max_components = max_components
        self.max_edges = max_edges
        self.max_nodes = max_nodes
        self.max_node_edges = max_node_edges

        # Platform mapping
        self.platform_map = {"OS1": 0, "OS2": 1, "OS3": 2, "OS4": 4}
        self.data_list = []

        print(f"\nLoading {num_samples} JSON file pairs from {json_dir}...")
        for x in tqdm(range(1, min(num_samples + 1, 1001))):  # Limit to available files
            services_file = os.path.join(json_dir, f"services_{x}.json")
            edges_file = os.path.join(json_dir, f"services_edge_{x}.json")
            nodes_file = os.path.join(json_dir, f"nodes_{x}.json")
            nodes_edge_file = os.path.join(json_dir, f"nodes_edge_{x}.json")
            users_file = os.path.join(json_dir, f"users_{x}.json")
            helpers_file = os.path.join(json_dir, f"helpers_{x}.json")

            if not all(os.path.exists(f) for f in [services_file, edges_file, nodes_file, nodes_edge_file, users_file, helpers_file]):
                print(f"Warning: One or more files for sample {x} not found, skipping.")
                continue

            try:
                with open(services_file, "r") as f:
                    services_data = json.load(f)
                with open(edges_file, "r") as f:
                    edges_data = json.load(f)
                with open(nodes_file, "r") as f:
                    nodes_data = json.load(f)
                with open(nodes_edge_file, "r") as f:
                    nodes_edge_data = json.load(f)
                with open(users_file, "r") as f:
                    users_data = json.load(f)
                with open(helpers_file, "r") as f:
                    helpers_data = json.load(f)

                if not services_data or not isinstance(services_data, list):
                    print(f"Warning: Invalid services in {services_file}, skipping.")
                    continue

                # Combine all nodes
                all_nodes_data = nodes_data + users_data + helpers_data

                component_features = []
                service_ids = []
                user_ids = []
                helper_ids = []

                for service in services_data:
                    service_id = service.get("serviceID", 0)
                    user_id = service.get("userID", 0)
                    helper_id = service.get("helperID", 0)
                    components = service.get("components", [])

                    for component in components:
                        chars = component.get("characteristics", {})
                        features = [
                            chars.get("cpu", 0.0),
                            chars.get("memory", 0.0),
                            chars.get("disk", 0.0),
                            chars.get("reliabilityScore", 0.0),
                            chars.get("platform", 0.0),  # Will be mapped
                        ]
                        component_features.append(features)
                        service_ids.append(service_id)
                        user_ids.append(user_id)
                        helper_ids.append(helper_id)

                component_features = np.array(component_features, dtype=np.float32)

                if not edges_data or not isinstance(edges_data, list):
                    print(f"Warning: Invalid edges in {edges_file}, skipping.")
                    continue

                edges = []
                edge_weights = []
                component_map = {f"c{i+1}": i for i in range(len(component_features))}

                for edge in edges_data:
                    if len(edge) != 3:
                        continue
                    source, target, weight = edge
                    source_idx = component_map.get(source, -1)
                    target_idx = component_map.get(target, -1)
                    if source_idx == -1 or target_idx == -1:
                        continue
                    edges.append([source_idx, target_idx])
                    edge_weights.append(float(weight))

                edges = np.array(edges, dtype=np.int64).T
                edge_weights = np.array(edge_weights, dtype=np.float32)

                # Pad component data
                padded_components = np.zeros((self.max_components, 5), dtype=np.float32)
                padded_components[:len(component_features), :] = component_features

                padded_service_ids = np.zeros(self.max_components, dtype=np.int64)
                padded_service_ids[:len(service_ids)] = service_ids

                padded_user_ids = np.zeros(self.max_components, dtype=np.int64)
                padded_user_ids[:len(user_ids)] = user_ids

                padded_helper_ids = np.zeros(self.max_components, dtype=np.int64)
                padded_helper_ids[:len(helper_ids)] = helper_ids

                padded_edges = np.zeros((2, self.max_edges), dtype=np.int64)
                padded_edges[:, :len(edges[0])] = edges

                padded_edge_weights = np.zeros(self.max_edges, dtype=np.float32)
                padded_edge_weights[:len(edge_weights)] = edge_weights

                # Process nodes
                node_features = []
                node_ids = []
                node_tiers = []

                for node in all_nodes_data:
                    node_id = node.get("nodeID", 0)
                    node_tier = node.get("nodeTier", 0)
                    chars = node.get("characteristics", {})
                    platform = self.platform_map.get(chars.get("platform", "OS1"), 0)
                    features = [
                        chars.get("cpu", 0.0),
                        chars.get("memory", 0.0),
                        chars.get("disk", 0.0),
                        chars.get("reliabilityScore", 0.0),
                        platform,
                    ]
                    node_features.append(features)
                    node_ids.append(node_id)
                    node_tiers.append(node_tier)

                node_features = np.array(node_features, dtype=np.float32)

                if not nodes_edge_data or not isinstance(nodes_edge_data, list):
                    print(f"Warning: Invalid node edges in {nodes_edge_file}, skipping.")
                    continue

                node_edges = []
                node_edge_weights = []
                node_map = {node["nodeID"]: i for i, node in enumerate(all_nodes_data)}

                for source_idx, edges_list in enumerate(nodes_edge_data):
                    if source_idx >= len(all_nodes_data):
                        continue
                    source_id = all_nodes_data[source_idx]["nodeID"]
                    for edge in edges_list[1:]:
                        target_id, weight = edge
                        if target_id in node_map:
                            target_idx = node_map[target_id]
                            node_edges.append([source_idx, target_idx])
                            node_edge_weights.append(float(weight))

                node_edges = np.array(node_edges, dtype=np.int64).T if node_edges else np.zeros((2, 0), dtype=np.int64)
                node_edge_weights = np.array(node_edge_weights, dtype=np.float32)

                # Pad node data
                padded_nodes = np.zeros((self.max_nodes, 5), dtype=np.float32)
                padded_nodes[:len(node_features), :] = node_features

                padded_node_ids = np.zeros(self.max_nodes, dtype=np.int64)
                padded_node_ids[:len(node_ids)] = node_ids

                padded_node_tiers = np.zeros(self.max_nodes, dtype=np.int64)
                padded_node_tiers[:len(node_tiers)] = node_tiers

                padded_node_edges = np.zeros((2, self.max_node_edges), dtype=np.int64)
                padded_node_edges[:, :len(node_edges[0])] = node_edges

                padded_node_edge_weights = np.zeros(self.max_node_edges, dtype=np.float32)
                padded_node_edge_weights[:len(node_edge_weights)] = node_edge_weights

                # Create PyTorch Geometric Data objects
                services_data = Data(
                    x=torch.tensor(padded_components, dtype=torch.float),
                    edge_index=torch.tensor(padded_edges, dtype=torch.long),
                    edge_attr=torch.tensor(padded_edge_weights, dtype=torch.float),
                    service_ids=torch.tensor(padded_service_ids, dtype=torch.long),
                    user_ids=torch.tensor(padded_user_ids, dtype=torch.long),
                    helper_ids=torch.tensor(padded_helper_ids, dtype=torch.long),
                )

                nodes_data = Data(
                    x=torch.tensor(padded_nodes, dtype=torch.float),
                    edge_index=torch.tensor(padded_node_edges, dtype=torch.long),
                    edge_attr=torch.tensor(padded_node_edge_weights, dtype=torch.float),
                    node_ids=torch.tensor(padded_node_ids, dtype=torch.long),
                    node_tiers=torch.tensor(padded_node_tiers, dtype=torch.long),
                )

                self.data_list.append((services_data, nodes_data))

            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in sample {x}, skipping.")
                continue
            except Exception as e:
                print(f"Error processing sample {x}: {e}, skipping.")
                continue

        self.size = len(self.data_list)
        if self.size == 0:
            raise ValueError("No valid JSON file pairs were loaded.")

        if self.size % batch_size != 0:
            print(f"Warning: Dataset size ({self.size}) is not divisible by batch_size ({batch_size}).")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    dataset = NCODataset(
        json_dir="data/processed",
        num_samples=128,
        max_components=charnum_component * charnum_service,
        max_edges=charnum_component * (charnum_component - 1),
        max_nodes=charnum_node + charnum_user + charnum_helper,
        max_node_edges=(charnum_node + charnum_user + charnum_helper)
        * (charnum_node + charnum_user + charnum_helper),
    )
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    services_data, nodes_data = sample
    print("Services data keys:", services_data.keys())
    for key in services_data.keys():
        print(f"Services {key}: shape {services_data[key].shape}, dtype {services_data[key].dtype}")
    print("Nodes data keys:", nodes_data.keys())
    for key in nodes_data.keys():
        print(f"Nodes {key}: shape {nodes_data[key].shape}, dtype {nodes_data[key].dtype}")

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        collate_fn=lambda batch: (
            [b[0] for b in batch],  # services_data
            [b[1] for b in batch],  # nodes_data
        ),
    )
    for services_batch, nodes_batch in dataloader:
        print("Services batch keys:", services_batch[0].keys())
        for key in services_batch[0].keys():
            print(f"Services batch {key} shape: {services_batch[0][key].shape}")
        print("Nodes batch keys:", nodes_batch[0].keys())
        for key in nodes_batch[0].keys():
            print(f"Nodes batch {key} shape: {nodes_batch[0][key].shape}")
        break
