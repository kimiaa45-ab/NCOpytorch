import math
import numpy as np
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn import DataParallel

from utils.tensor_functions import compute_in_batches
from utils.beam_search import CachedLookup
from utils.functions import sample_many


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):
    def __init__(
        self,
        problem,
        embedding_dim,
        encoder_class,
        n_encode_layers,
        aggregation="max",
        aggregation_graph="mean",
        normalization="layer",
        learn_norm=True,
        track_norm=False,
        gated=True,
        n_heads=8,
        tanh_clipping=10.0,
        mask_inner=True,
        mask_logits=True,
        mask_graph=True,
        checkpoint_encoder=False,
        shrink_size=None,
        extra_logging=False,
        *args,
        **kwargs
    ):
        """
        Models with a GNN encoder and autoregressive decoder using attention mechanism for NCO.

        Args:
            problem: NCOProblem, to define the problem-specific state and costs
            embedding_dim: Hidden dimension for encoder/decoder
            encoder_class: GNN encoder class
            n_encode_layers: Number of layers for encoder
            aggregation: Aggregation function for GNN encoder ('sum'/'mean'/'max')
            aggregation_graph: Graph aggregation function ('sum'/'mean'/'max')
            normalization: Normalization scheme ('batch'/'layer'/'none')
            learn_norm: Enable learnable affine transformation during normalization
            track_norm: Track training dataset stats for normalization
            gated: Enable anisotropic GNN aggregation
            n_heads: Number of attention heads
            tanh_clipping: Clip decoder logits with tanh
            mask_inner: Use visited mask during inner function
            mask_logits: Use visited mask during log computation
            mask_graph: Use dependency graph mask during decoding
            checkpoint_encoder: Use checkpoints for encoder embeddings
            shrink_size: N/A
            extra_logging: Enable extra logging for embeddings and probabilities
        """
        super(AttentionModel, self).__init__()

        self.problem = problem
        self.embedding_dim = embedding_dim
        self.encoder_class = encoder_class
        self.n_encode_layers = n_encode_layers
        self.aggregation = aggregation
        self.aggregation_graph = aggregation_graph
        self.normalization = normalization
        self.learn_norm = learn_norm
        self.track_norm = track_norm
        self.gated = gated
        self.n_heads = n_heads
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.mask_graph = mask_graph
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.extra_logging = extra_logging

        self.decode_type = None
        self.temp = 1.0

        # NCO-specific context parameters
        assert problem.NAME == "nco", f"Unsupported problem: {problem.NAME}"
        node_dim = 3  # cpu, memory, disk for nodes and services
        step_context_dim = embedding_dim  # Embedding of current component

        # Embedding layers for services and nodes
        self.init_embed_services = nn.Linear(node_dim, embedding_dim, bias=True)
        self.init_embed_nodes = nn.Linear(node_dim, embedding_dim, bias=True)

        # Encoder model
        self.embedder = self.encoder_class(
            n_layers=n_encode_layers,
            n_heads=n_heads,
            hidden_dim=embedding_dim,
            aggregation=aggregation,
            norm=normalization,
            learn_norm=learn_norm,
            track_norm=track_norm,
            gated=gated
        )

        # Projection layers
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, services_data, nodes_data, return_pi=False):
        """
        Args:
            services_data: Dict with 'features' [B, S, 3] (cpu, memory, disk) and 'edge_index' [B, 2, E]
            nodes_data: Dict with 'resources' [B, N, 3] (cpu, memory, disk)
            return_pi: Whether to return the output sequences
        """
        # Embed services and nodes
        if self.checkpoint_encoder:
            embeddings = checkpoint(self.embedder, self._init_embed(services_data, nodes_data), services_data['edge_index'])
        else:
            embeddings = self.embedder(self._init_embed(services_data, nodes_data), services_data['edge_index'])

        if self.extra_logging:
            self.embeddings_batch = embeddings

        # Run inner function
        _log_p, pi = self._inner(services_data, nodes_data, embeddings)

        if self.extra_logging:
            self.log_p_batch = _log_p
            self.log_p_sel_batch = _log_p.gather(2, pi.unsqueeze(-1)).squeeze(-1)

        # Get predicted costs
        cost, mask = self.problem.get_costs(services_data, nodes_data, pi)

        # Calculate log likelihood
        ll = self._calc_log_likelihood(_log_p, pi, mask)

        if return_pi:
            return cost, ll, pi
        return cost, ll

    def beam_search(self, services_data, nodes_data, beam_size=1280):
        """Helper method to call beam search for NCO"""
        return self.problem.beam_search(services_data, nodes_data, model=self, beam_size=beam_size)

    def precompute_fixed(self, services_data, nodes_data):
        embeddings = self.embedder(self._init_embed(services_data, nodes_data), services_data['edge_index'])
        return CachedLookup(self._precompute(embeddings))

    def propose_expansions(self, beam, fixed, expand_size=None, normalize=False, max_calc_batch_size=4096):
        log_p_topk, ind_topk = compute_in_batches(
            lambda b: self._get_log_p_topk(fixed[b.ids], b.state, k=expand_size, normalize=normalize),
            max_calc_batch_size, beam, n=beam.size()
        )

        score_expand = beam.score[:, None] + log_p_topk[:, 0, :]
        flat_action = ind_topk.view(-1)
        flat_score = score_expand.view(-1)
        flat_feas = flat_score > -1e10
        flat_parent = torch.arange(flat_action.size(-1), out=flat_action.new()) / ind_topk.size(-1)
        feas_ind_2d = torch.nonzero(flat_feas)

        if len(feas_ind_2d) == 0:
            return None, None, None

        feas_ind = feas_ind_2d[:, 0]
        return flat_parent[feas_ind], flat_action[feas_ind], flat_score[feas_ind]

    def _calc_log_likelihood(self, _log_p, a, mask):
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        if mask is not None:
            log_p[mask] = 0
        assert (log_p > -1000).data.all(), "Logprobs should not be -inf"
        return log_p.sum(1)

    def _init_embed(self, services_data, nodes_data):
        # Combine service and node embeddings
        service_emb = self.init_embed_services(services_data['features'])
        node_emb = self.init_embed_nodes(nodes_data['resources'])
        return torch.cat((service_emb, node_emb), dim=1)  # [B, S+N, embedding_dim]

    def _inner(self, services_data, nodes_data, embeddings):
        outputs = []
        sequences = []

        # Create problem state for tracking assignments
        state = self.problem.make_state(services_data, nodes_data)

        # Compute keys, values for the glimpse and keys for the logits
        fixed = self._precompute(embeddings)

        batch_size, num_services_plus_nodes, _ = embeddings.shape
        num_services = services_data['features'].shape[1]

        # Perform decoding steps
        i = 0
        while not state.all_finished():
            if self.shrink_size is not None:
                unfinished = torch.nonzero(state.get_finished() == 0)
                if len(unfinished) == 0:
                    break
                unfinished = unfinished[:, 0]
                if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                    state = state[unfinished]
                    fixed = fixed[unfinished]

            # Get log probabilities of next action
            log_p, mask = self._get_log_p(fixed, state)

            # Select the indices of the next nodes
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])

            # Update problem state
            state = state.update(selected)

            # Handle shrinking
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                log_p_, selected_ = log_p, selected
                log_p = log_p_.new_zeros(batch_size, *log_p_.size()[1:])
                selected = selected_.new_zeros(batch_size)
                log_p[state.ids[:, 0]] = log_p_
                selected[state.ids[:, 0]] = selected_

            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

            i += 1

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def sample_many(self, services_data, nodes_data, batch_rep=1, iter_rep=1):
        return sample_many(
            lambda input: self._inner(*input),
            lambda input, pi: self.problem.get_costs(input[0], input[1], pi),
            (services_data, nodes_data, self.embedder(self._init_embed(services_data, nodes_data), services_data['edge_index'])),
            batch_rep, iter_rep
        )

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain NaNs"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), "Infeasible action selected"
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):
        if self.aggregation_graph == "sum":
            graph_embed = embeddings.sum(1)
        elif self.aggregation_graph == "max":
            graph_embed = embeddings.max(1)[0]
        elif self.aggregation_graph == "mean":
            graph_embed = embeddings.mean(1)
        else:
            graph_embed = embeddings.sum(1) * 0.0

        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p_topk(self, fixed, state, k=None, normalize=True):
        log_p, _ = self._get_log_p(fixed, state, normalize=normalize)
        if k is not None and k < log_p.size(-1):
            return log_p.topk(k, -1)
        return (
            log_p,
            torch.arange(log_p.size(-1), device=log_p.device, dtype=torch.int64).repeat(log_p.size(0), 1)[:, None, :]
        )

    def _get_log_p(self, fixed, state, normalize=True):
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))
        glimpse_K, glimpse_V, logit_K = self._get_attention_node_data(fixed, state)
        mask = state.get_mask()
        graph_mask = state.get_graph_mask() if self.mask_graph else None
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, graph_mask)
        if normalize:
            log_p = F.log_softmax(log_p / self.temp, dim=-1)
        assert not torch.isnan(log_p).any()
        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        current_service = state.get_current_service()
        batch_size, num_steps = current_service.size()
        return embeddings.gather(
            1,
            current_service.contiguous().view(batch_size, num_steps, 1).expand(batch_size, num_steps, embeddings.size(-1))
        ).view(batch_size, num_steps, -1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, graph_mask=None):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -1e10
            if self.mask_graph and graph_mask is not None:
                compatibility[graph_mask[None, :, :, None, :].expand_as(compatibility)] = -1e10
        heads = torch.matmul(F.softmax(compatibility, dim=-1), glimpse_V)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))
        logits = torch.matmul(glimpse, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))
        if self.mask_logits and graph_mask is not None:
            logits[graph_mask] = -1e10
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -1e10
        return logits, glimpse.squeeze(-2)

    def _get_attention_node_data(self, fixed, state):
        return fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)
        )