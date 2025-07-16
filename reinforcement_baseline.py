import torch
import yaml
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from train import rollout, get_inner_model
from problem.problem_NCO import NCOProblem, NCODataset

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
charnum_user = config["model"]["charnum_user"]
charnum_helper = config["model"]["charnum_helper"]

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


class Baseline(object):
    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, graph, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class GeneticBaseline(Baseline):
    def __init__(
        self,
        problem,
        opts,
        population_size=50,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.8,
    ):
        super(Baseline, self).__init__()
        self.problem = problem
        self.opts = opts
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def _generate_initial_population(
        self, batch_size, num_components, num_nodes, services_data, nodes_data
    ):
        """Generate initial population of assignments respecting topological constraints."""
        population = []
        for _ in range(self.population_size):
            assignments = torch.zeros(
                batch_size, num_components, dtype=torch.long, device=nodes_data.x.device
            )
            for b in range(batch_size):
                for i in range(num_components):
                    valid_nodes = self._get_valid_nodes(
                        services_data, nodes_data, assignments, b, i
                    )
                    if valid_nodes.sum() == 0:
                        assignments[b, i] = torch.randint(
                            0, num_nodes, (1,), device=nodes_data.x.device
                        )[0]
                    else:
                        valid_indices = torch.where(valid_nodes)[0]
                        assignments[b, i] = valid_indices[
                            torch.randint(0, valid_indices.size(0), (1,), device=nodes_data.x.device)[0]
                        ]
            population.append(assignments)
        return population

    def _get_valid_nodes(self, services_data, nodes_data, assignments, batch_idx, comp_idx):
        """Get valid nodes for a component considering topological constraints and resources."""
        #print("services_data.x shape:", services_data.x.shape)
        #print("nodes_data.x shape:", nodes_data.x.shape)

        edge_index = services_data.edge_index
        comp_cpu = services_data.x[comp_idx, 0]
        comp_memory = services_data.x[comp_idx, 1]
        comp_disk = services_data.x[comp_idx, 3]
        node_cpu = nodes_data.x[:, 0]
        node_memory = nodes_data.x[:, 1]
        node_disk = nodes_data.x[:, 2]

        valid_mask = torch.ones(
            nodes_data.x.size(0), dtype=torch.bool, device=services_data.x.device
        )
        incoming_edges = edge_index[1] == comp_idx
        prereqs = edge_index[0, incoming_edges]
        if prereqs.numel() > 0:
            assigned = assignments[batch_idx, prereqs]
            if (assigned == 0).any():
                valid_mask[:] = False
                return valid_mask

        valid_mask = (
            (node_cpu >= comp_cpu)
            & (node_memory >= comp_memory)
            & (node_disk >= comp_disk)
        )
        return valid_mask

    def _evaluate_population(self, population, services_data, nodes_data):
        """Evaluate fitness (negative execution_time) for each assignment in population."""
        fitness = []
        for assignments in population:
            cost, _ = self.problem.get_costs(services_data, nodes_data, assignments)
            fitness.append(-cost)
        return torch.stack(fitness)

    def _selection(self, fitness):
        """Select parents using tournament selection."""
        tournament_size = 5
        indices = torch.arange(self.population_size, device=fitness.device)
        selected = []
        for _ in range(self.population_size):
            competitors = indices[
                torch.randperm(self.population_size)[:tournament_size]
            ]
            winner = competitors[fitness[competitors].argmax()]
            selected.append(winner)
        return selected

    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        if torch.rand(1) > self.crossover_rate:
            return parent1.clone(), parent2.clone()
        batch_size, num_components = parent1.shape
        child1, child2 = parent1.clone(), parent2.clone()
        crossover_point = torch.randint(1, num_components, (1,)).item()
        child1[:, crossover_point:] = parent2[:, crossover_point:]
        child2[:, :crossover_point] = parent1[:, :crossover_point]
        return child1, child2

    def _mutate(self, assignments, num_nodes):
        """Apply mutation to assignments."""
        batch_size, num_components = assignments.shape
        mask = (
            torch.rand(batch_size, num_components, device=assignments.device)
            < self.mutation_rate
        )
        assignments[mask] = torch.randint(
            0, num_nodes, (int(mask.sum().item()),), device=assignments.device
        )
        return assignments

    def eval(self, services_data, nodes_data, c):
        """Run genetic algorithm to compute baseline value."""
        batch_size = 1
        num_components = services_data.x.size(0)
        num_nodes = nodes_data.x.size(0)

        population = self._generate_initial_population(
            batch_size, num_components, num_nodes, services_data, nodes_data
        )

        for _ in range(self.generations):
            fitness = self._evaluate_population(population, services_data, nodes_data)
            selected_indices = self._selection(fitness)
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = population[selected_indices[i]]
                parent2 = (
                    population[selected_indices[i + 1]]
                    if i + 1 < self.population_size
                    else parent1
                )
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1, num_nodes)
                child2 = self._mutate(child2, num_nodes)
                new_population.extend([child1, child2])
            population = new_population[: self.population_size]

        fitness = self._evaluate_population(population, services_data, nodes_data)
        best_fitness = fitness.max(dim=0)[0]
        bl_val = -best_fitness
        return bl_val.detach(), 0

    def wrap_dataset(self, dataset):
        """Wrap dataset with baseline values."""
        print("\nEvaluating Genetic baseline on dataset...")
        bl_vals = []
        for services_data, nodes_data in dataset:
            services_data = services_data.to(self.opts.device)
            nodes_data = nodes_data.to(self.opts.device)
            bl_val, _ = self.eval(services_data, nodes_data, None)
            bl_vals.append(bl_val)
        return BaselineDataset(dataset, torch.stack(bl_vals))

    def unwrap_batch(self, batch):
        return batch["data"], batch["baseline"].view(-1)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class BaselineDataset(Dataset):
    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert self.dataset is not None and self.baseline is not None
        assert len(self.dataset) == len(self.baseline)

    def __getitem__(self, item):
        assert self.dataset is not None and self.baseline is not None
        return {"data": self.dataset[item], "baseline": self.baseline[item]}

    def __len__(self):
        assert self.dataset is not None
        return len(self.dataset)