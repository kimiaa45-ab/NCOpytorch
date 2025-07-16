import torch
from typing import NamedTuple
from typing import NamedTuple

from utils.boolmask import (
    mask_long2bool,
    mask_long_scatter,
)  # فرض بر وجود توابع boolmask


class StateNCO(NamedTuple):
    """Class to manage the state of Network Service Composition (NCO) problem during beam search."""

    # Fixed input
    services_data: (
        torch.Tensor
    )  # PyTorch Geometric Data for services (components, edges, etc.)
    nodes_data: (
        torch.Tensor
    )  # PyTorch Geometric Data for nodes (computing, user, helper)

    # State
    assignments: (
        torch.Tensor
    )  # Tensor of shape [batch_size, beam_size, num_components] for assigned nodes
    visited: (
        torch.Tensor
    )  # Tensor of shape [batch_size, beam_size, num_components] for visited components
    costs: (
        torch.Tensor
    )  # Tensor of shape [batch_size, beam_size] for accumulated execution time
    ids: torch.Tensor  # Keeps track of original batch indices
    i: torch.Tensor  # Keeps track of step (number of assigned components)

    @property
    def num_components(self):
        return self.services_data.x.size(-2)

    @property
    def num_nodes(self):
        return self.nodes_data.x.size(-2)

    def __getitem__(self, key):
        """Index the state tensors by key (tensor or slice)."""
        if torch.is_tensor(key) or isinstance(key, slice):
            return self._replace(
                services_data=self.services_data[key],
                nodes_data=self.nodes_data[key],
                assignments=self.assignments[key],
                visited=self.visited[key],
                costs=self.costs[key],
                ids=self.ids[key],
            )
        return super(StateNCO, self).__getitem__(key)

    @staticmethod
    def initialize(services_data, nodes_data, visited_dtype=torch.uint8):
        """
        Initialize the state for a new beam search.

        Args:
            services_data (Data): PyTorch Geometric Data for services
            nodes_data (Data): PyTorch Geometric Data for nodes
            visited_dtype (torch.dtype): Data type for visited mask (uint8 or int64)

        Returns:
            StateNCO: Initialized state
        """
        batch_size = services_data.x.size(0)
        num_components = services_data.x.size(1)
        return StateNCO(
            services_data=services_data,
            nodes_data=nodes_data,
            assignments=torch.zeros(
                (batch_size, 1, num_components),
                dtype=torch.long,
                device=services_data.x.device,
            ),
            visited=(
                torch.zeros(
                    (batch_size, 1, num_components),
                    dtype=torch.uint8,
                    device=services_data.x.device,
                )
                if visited_dtype == torch.uint8
                else torch.zeros(
                    (batch_size, 1, (num_components + 63) // 64),
                    dtype=torch.int64,
                    device=services_data.x.device,
                )
            ),
            costs=torch.zeros(
                (batch_size, 1), dtype=torch.float, device=services_data.x.device
            ),
            ids=torch.arange(
                batch_size, dtype=torch.int64, device=services_data.x.device
            )[:, None],
            i=torch.zeros(1, dtype=torch.int64, device=services_data.x.device),
        )

    def update(self, selected_nodes, costs):
        """
        Update state with new assignments and costs.

        Args:
            selected_nodes (torch.Tensor): Selected node indices [batch_size, beam_size]
            costs (torch.Tensor): Execution time for the selected assignments [batch_size, beam_size]

        Returns:
            StateNCO: Updated state
        """
        batch_size, beam_size, num_components = self.assignments.size()
        selected = selected_nodes.unsqueeze(-1)  # [batch_size, beam_size, 1]
        assignments = self.assignments.clone()
        visited = self.visited.clone()
        costs = self.costs.clone() + costs

        # Update assignments and visited components
        step = (self.visited.sum(-1) == 0).sum(-1, keepdim=True)  # Current step
        assignments.scatter_(2, step, selected)
        if self.visited.dtype == torch.uint8:
            visited.scatter_(2, step, 1)
        else:
            visited = mask_long_scatter(self.visited, step)

        return self._replace(
            assignments=assignments, visited=visited, costs=costs, i=self.i + 1
        )

    def all_finished(self):
        """Check if all components have been assigned."""
        return self.i.item() >= self.num_components

    def get_current_component(self):
        """Return the index of the current component to be assigned."""
        return (self.visited.sum(-1) == 0).sum(-1, keepdim=True)

    def get_mask(self):
        """Return mask of visited components."""
        if self.visited.dtype == torch.uint8:
            return self.visited
        else:
            return mask_long2bool(self.visited, n=self.num_components)

    def get_valid_nodes(self):
        """
        Return mask of valid nodes considering topological constraints.

        Returns:
            valid_mask (torch.Tensor): Mask of shape [batch_size, beam_size, num_nodes]
        """
        edge_index = self.services_data.edge_index
        visited = self.get_mask()
        batch_size, beam_size, num_components = visited.size()
        valid_mask = torch.ones(
            (batch_size, beam_size, num_components),
            dtype=torch.uint8,
            device=visited.device,
        )

        for b in range(batch_size):
            for beam in range(beam_size):
                for i in range(num_components):
                    if visited[b, beam, i]:
                        valid_mask[b, beam, i] = 0
                        continue
                    # Check prerequisites (components that must be visited before i)
                    incoming_edges = edge_index[1] == i
                    prereqs = edge_index[0, incoming_edges]
                    if prereqs.numel() > 0 and not visited[b, beam, prereqs].all():
                        valid_mask[b, beam, i] = 0

        # Mask for valid nodes (all nodes are valid unless resource constraints apply)
        node_mask = torch.ones(
            (batch_size, beam_size, self.num_nodes),
            dtype=torch.uint8,
            device=visited.device,
        )
        return node_mask

    def get_compatible_nodes(self, k=None):
        """
        Return k nodes with sufficient resources for the current component.

        Args:
            k (int, optional): Number of compatible nodes to return

        Returns:
            compatible_nodes (torch.Tensor): Indices of compatible nodes or mask
        """
        batch_size, beam_size, num_components = self.visited.size()
        comp_cpu = self.services_data.x[:, :, 0]  # [batch_size, num_components]
        node_cpu = self.nodes_data.x[:, 0]  # [num_nodes]
        comp_memory = self.services_data.x[:, :, 1]
        node_memory = self.nodes_data.x[:, 1]
        comp_disk = self.services_data.x[:, :, 3]
        node_disk = self.nodes_data.x[:, 2]

        valid_nodes = torch.ones(
            (batch_size, beam_size, self.num_nodes),
            dtype=torch.float,
            device=self.services_data.x.device,
        )
        step = (self.visited.sum(-1) == 0).sum(-1, keepdim=True)  # Current component

        for b in range(batch_size):
            for beam in range(beam_size):
                comp_idx = step[b, beam].item()
                # Check resource compatibility
                valid_nodes[b, beam] = (
                    (node_cpu >= comp_cpu[b, comp_idx])
                    & (node_memory >= comp_memory[b, comp_idx])
                    & (node_disk >= comp_disk[b, comp_idx])
                ).float()

        if k is not None:
            # Penalize invalid nodes
            valid_nodes += self.get_mask().sum(-1, keepdim=True).float() * 1e6
            return valid_nodes.topk(k, dim=-1, largest=False)[1]
        return valid_nodes

    def construct_solutions(self, actions):
        """Construct final assignments from actions."""
        return actions
