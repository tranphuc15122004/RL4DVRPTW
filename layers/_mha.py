import torch
import torch.nn as nn
import torch.nn.functional as F


from functools import reduce
import operator


def scaled_dot_prod_attention(queries, keys, values, mask = None):
    r"""
    :param queries: :math:`... \times l_q  \times d_k`
    :param keys:    :math:`... \times l_kv \times d_k`
    :param values:  :math:`... \times l_kv \times d_v`
    :param mask:    :math:`... \times l_q  \times l_kv`
    :type queries:  torch.Tensor(dtype=torch.float32)
    :type keys:     torch.Tensor(dtype=torch.float32)
    :type values:   torch.Tensor(dtype=torch.float32)
    :type mask:     torch.Tensor(dtype=torch.uint8)
    :return:        :math:`... \times l_q  \times d_v`
    :rtype:         torch.Tensor(dtype=torch.float32)
    """
    weights = queries.matmul( keys.transpose(-1,-2) ) # ... x l_q x l_kv
    weights *= keys.size(-1) ** -0.5
    if mask is not None:
        if mask.dim() == weights.dim() - 1:
            mask = mask.unsqueeze(-2).expand_as(weights)
        weights[mask] = -float('inf')
    weights = F.softmax(weights, dim = -1 )
    return weights.matmul( values )


class _MHA_V1(nn.Module):
    def __init__(self, head_count, query_size, key_size = None, value_size = None,
            key_size_per_head = None, value_size_per_head = None):
        super().__init__()
        self.head_count = head_count

        self.query_size = query_size
        self.key_size = query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size

        self.key_size_per_head = self.key_size // self.head_count if key_size_per_head is None \
                else key_size_per_head
        self.value_size_per_head = self.value_size // self.head_count if value_size_per_head is None \
                else value_size_per_head

        self.query_project = nn.Linear(self.query_size, self.head_count * self.key_size_per_head, bias = False)
        self.key_project = nn.Linear(self.key_size, self.head_count * self.key_size_per_head, bias = False)
        self.value_project = nn.Linear(self.value_size, self.head_count * self.value_size_per_head, bias = False)
        self.recombine = nn.Linear(self.head_count * self.value_size_per_head, self.value_size, bias = False)


    def forward(self, queries, keys, values, mask = None):
        r"""
        :param queries: :math:`... \times l_q  \times d_q`
        :param keys:    :math:`... \times l_kv \times d_k`
        :param values:  :math:`... \times l_kv \times d_v`
        :param mask:    :math:`... \times l_q  \times l_kv`
        :type queries:  torch.Tensor(dtype=torch.float32)
        :type keys:     torch.Tensor(dtype=torch.float32)
        :type values:   torch.Tensor(dtype=torch.float32)
        :type mask:     torch.Tensor(dtype=torch.uint8)
        :return:        :math:`... \times l_q  \times d_v`
        :rtype:         torch.Tensor(dtype=torch.float32)
        """
        q_proj = self.query_project(queries).chunk(self.head_count, dim = -1)
        k_proj = self.key_project(keys).chunk(self.head_count, dim = -1)
        v_proj = self.value_project(values).chunk(self.head_count, dim = -1)

        att_applied = tuple(map(scaled_dot_prod_attention, \
            q_proj, k_proj, v_proj, (mask for _ in range(self.head_count))))

        return self.recombine(torch.cat(att_applied, dim = -1))



class _MHA_V2(nn.Module):
    def __init__(self, head_count, query_size, key_size = None, value_size = None,
            key_size_per_head = None, value_size_per_head = None):
        super().__init__()
        self.head_count = head_count

        self.query_size = query_size
        self.key_size = query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size

        self.key_size_per_head = self.key_size // self.head_count if key_size_per_head is None \
                else key_size_per_head
        self.value_size_per_head = self.value_size // self.head_count if value_size_per_head is None \
                else value_size_per_head

        self._inv_sqrt_d = self.key_size_per_head ** -0.5

        self.query_project = nn.Linear(self.query_size, self.head_count * self.key_size_per_head, bias = False)
        self.key_project = nn.Linear(self.key_size, self.head_count * self.key_size_per_head, bias = False)
        self.value_project = nn.Linear(self.value_size, self.head_count * self.value_size_per_head, bias = False)
        self.recombine = nn.Linear(self.head_count * self.value_size_per_head, self.value_size, bias = False)

        self._k_proj = None
        self._v_proj = None

        self.init_parameters()


    def init_parameters(self):
        nn.init.uniform_(self.query_project.weight, -self._inv_sqrt_d, self._inv_sqrt_d)
        nn.init.uniform_(self.key_project.weight, -self._inv_sqrt_d, self._inv_sqrt_d)
        inv_sq_dv = self.value_size_per_head**-0.5
        nn.init.uniform_(self.value_project.weight, -inv_sq_dv, inv_sq_dv)


    def precompute(self, keys, values = None):
        values = keys if values is None else values
        l_kv = keys.size(-2)
        self._k_proj = self.key_project(keys).view(
                -1, l_kv, self.head_count, self.key_size_per_head).permute(0,2,3,1)
        self._v_proj = self.value_project(values).view(
                -1, l_kv, self.head_count, self.value_size_per_head).permute(0,2,1,3)


    def forward(self, queries, keys = None, values = None, mask = None):
        *size, l_q, _ = queries.size()

        q_proj = self.query_project(queries).view(
                -1, l_q, self.head_count, self.key_size_per_head).permute(0,2,1,3)

        if keys is None:
            if self._k_proj is None: # self-attention
                l_kv = l_q
                k_proj = self.key_project(queries).view(
                        -1, l_kv, self.head_count, self.key_size_per_head).permute(0,2,3,1)
            else: # pre-computed
                l_kv = self._k_proj.size(-1)
                k_proj = self._k_proj
        else:
            l_kv = keys.size(-2)
            k_proj = self.key_project(keys).view(
                    -1, l_kv, self.head_count, self.key_size_per_head).permute(0,2,3,1)

        if values is None:
            if self._v_proj is None: # self-attention
                v_proj = self.value_project(queries).view(
                        -1, l_kv, self.head_count, self.value_size_per_head).permute(0,2,1,3)
            else: # pre-computed
                v_proj = self._v_proj
        else:
            v_proj = self.value_project(values).view(
                    -1, l_kv, self.head_count, self.value_size_per_head).permute(0,2,1,3)

        weights = q_proj.matmul( k_proj )
        weights *= self._inv_sqrt_d
        if mask is not None:
            if mask.numel() * self.head_count == weights.numel(): # one mask per query
                m = mask.view(-1,1,l_q,l_kv).expand_as(weights)
            else: # common mask for all queries
                m = mask.view(-1,1,1,l_kv).expand_as(weights)
            weights[m] = -float('inf')
        weights = F.softmax(weights, dim = -1)

        att_applied = weights.matmul(v_proj).permute(0,2,1,3).contiguous().view(
                *size, l_q, self.head_count * self.value_size_per_head)
        return self.recombine(att_applied)


class MixedScore_MultiHeadAttention(nn.Module):
    def __init__(self, head_count, query_size, key_size = None, value_size = None,
            key_size_per_head = None, value_size_per_head = None, ms_hidden_dim = 16):
        r"""
        Mixed-score Multi-Head Attention that combines dot-product attention with cost matrix.
        
        :param head_count: Number of attention heads
        :param query_size: Dimension of query vectors
        :param key_size: Dimension of key vectors (defaults to query_size)
        :param value_size: Dimension of value vectors (defaults to key_size)
        :param key_size_per_head: Dimension of keys per head (defaults to key_size // head_count)
        :param value_size_per_head: Dimension of values per head (defaults to value_size // head_count)
        :param ms_hidden_dim: Hidden dimension for mixed-score MLP (default: 16)
        """
        super().__init__()
        self.head_count = head_count

        self.query_size = query_size
        self.key_size = query_size if key_size is None else key_size
        self.value_size = self.key_size if value_size is None else value_size

        self.key_size_per_head = self.key_size // self.head_count if key_size_per_head is None \
                else key_size_per_head
        self.value_size_per_head = self.value_size // self.head_count if value_size_per_head is None \
                else value_size_per_head

        self._inv_sqrt_d = self.key_size_per_head ** -0.5
        self.ms_hidden_dim = ms_hidden_dim

        # Projection layers (same as _MHA_V2)
        self.query_project = nn.Linear(self.query_size, self.head_count * self.key_size_per_head, bias = False)
        self.key_project = nn.Linear(self.key_size, self.head_count * self.key_size_per_head, bias = False)
        self.value_project = nn.Linear(self.value_size, self.head_count * self.value_size_per_head, bias = False)
        self.recombine = nn.Linear(self.head_count * self.value_size_per_head, self.value_size, bias = False)

        # Cached projections for precompute
        self._k_proj = None
        self._v_proj = None

        # Mixed-score MLP layers
        mix1_init = (2 * ms_hidden_dim) ** -0.5
        mix2_init = ms_hidden_dim ** -0.5
        
        mix1_weight = torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_count, 2, ms_hidden_dim))
        mix1_bias = torch.distributions.Uniform(low=-mix1_init, high=mix1_init).sample((head_count, ms_hidden_dim))
        self.mix1_weight = nn.Parameter(mix1_weight)
        # shape: (head_count, 2, ms_hidden_dim)
        self.mix1_bias = nn.Parameter(mix1_bias)
        # shape: (head_count, ms_hidden_dim)

        mix2_weight = torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_count, ms_hidden_dim, 1))
        mix2_bias = torch.distributions.Uniform(low=-mix2_init, high=mix2_init).sample((head_count, 1))
        self.mix2_weight = nn.Parameter(mix2_weight)
        # shape: (head_count, ms_hidden_dim, 1)
        self.mix2_bias = nn.Parameter(mix2_bias)
        # shape: (head_count, 1)

        self.init_parameters()


    def init_parameters(self):
        nn.init.uniform_(self.query_project.weight, -self._inv_sqrt_d, self._inv_sqrt_d)
        nn.init.uniform_(self.key_project.weight, -self._inv_sqrt_d, self._inv_sqrt_d)
        inv_sq_dv = self.value_size_per_head**-0.5
        nn.init.uniform_(self.value_project.weight, -inv_sq_dv, inv_sq_dv)


    def precompute(self, keys, values = None):
        r"""
        Pre-compute and cache key/value projections for efficiency.
        
        :param keys: :math:`N \times L_{kv} \times D_k`
        :param values: :math:`N \times L_{kv} \times D_v` (defaults to keys)
        """
        values = keys if values is None else values
        l_kv = keys.size(-2)
        self._k_proj = self.key_project(keys).view(
                -1, l_kv, self.head_count, self.key_size_per_head).permute(0,2,3,1)
        self._v_proj = self.value_project(values).view(
                -1, l_kv, self.head_count, self.value_size_per_head).permute(0,2,1,3)


    def forward(self, queries, keys = None, values = None, mask = None, cost_mat = None):
        r"""
        :param queries: :math:`N \times L_q \times D_q`
        :param keys: :math:`N \times L_{kv} \times D_k` (None for self-attention or pre-computed)
        :param values: :math:`N \times L_{kv} \times D_v` (None for self-attention or pre-computed)
        :param mask: :math:`N \times L_q \times L_{kv}` or :math:`N \times L_{kv}` (optional)
        :param cost_mat: :math:`N \times L_q \times L_{kv}` cost matrix for mixed scoring (required)
        :return: :math:`N \times L_q \times D_v`
        """
        if cost_mat is None:
            raise ValueError("cost_mat is required for MixedScore_MultiHeadAttention")

        *size, l_q, _ = queries.size()
        batch_size = queries.size(0) if len(size) == 1 else size[0]

        # Project queries
        q_proj = self.query_project(queries).view(
                -1, l_q, self.head_count, self.key_size_per_head).permute(0,2,1,3)
        # shape: (batch, head_count, l_q, key_size_per_head)

        # Get or compute key projections
        if keys is None:
            if self._k_proj is None:  # self-attention
                l_kv = l_q
                k_proj = self.key_project(queries).view(
                        -1, l_kv, self.head_count, self.key_size_per_head).permute(0,2,3,1)
            else:  # pre-computed
                l_kv = self._k_proj.size(-1)
                k_proj = self._k_proj
        else:
            l_kv = keys.size(-2)
            k_proj = self.key_project(keys).view(
                    -1, l_kv, self.head_count, self.key_size_per_head).permute(0,2,3,1)
        # shape: (batch, head_count, key_size_per_head, l_kv)

        # Get or compute value projections
        if values is None:
            if self._v_proj is None:  # self-attention
                v_proj = self.value_project(queries).view(
                        -1, l_kv, self.head_count, self.value_size_per_head).permute(0,2,1,3)
            else:  # pre-computed
                v_proj = self._v_proj
        else:
            v_proj = self.value_project(values).view(
                    -1, l_kv, self.head_count, self.value_size_per_head).permute(0,2,1,3)
        # shape: (batch, head_count, l_kv, value_size_per_head)

        # Compute dot-product score
        dot_product = q_proj.matmul(k_proj)
        # shape: (batch, head_count, l_q, l_kv)
        dot_product_score = dot_product * self._inv_sqrt_d
        # shape: (batch, head_count, l_q, l_kv)

        # Expand cost matrix to match head dimension
        cost_mat_score = cost_mat[:, None, :, :].expand(batch_size, self.head_count, l_q, l_kv)
        # shape: (batch, head_count, l_q, l_kv)

        # Stack two scores and prepare for MLP
        two_scores = torch.stack((dot_product_score, cost_mat_score), dim=4)
        # shape: (batch, head_count, l_q, l_kv, 2)
        two_scores_transposed = two_scores.transpose(1, 2)
        # shape: (batch, l_q, head_count, l_kv, 2)

        # Mixed-score MLP layer 1
        ms1 = torch.matmul(two_scores_transposed, self.mix1_weight)
        # shape: (batch, l_q, head_count, l_kv, ms_hidden_dim)
        ms1 = ms1 + self.mix1_bias[None, None, :, None, :]
        # shape: (batch, l_q, head_count, l_kv, ms_hidden_dim)
        ms1_activated = F.relu(ms1)

        # Mixed-score MLP layer 2
        ms2 = torch.matmul(ms1_activated, self.mix2_weight)
        # shape: (batch, l_q, head_count, l_kv, 1)
        ms2 = ms2 + self.mix2_bias[None, None, :, None, :]
        # shape: (batch, l_q, head_count, l_kv, 1)
        
        mixed_scores = ms2.transpose(1, 2).squeeze(4)
        # shape: (batch, head_count, l_q, l_kv)

        # Apply mask if provided
        if mask is not None:
            if mask.numel() * self.head_count == mixed_scores.numel():  # one mask per query
                m = mask.view(-1, 1, l_q, l_kv).expand_as(mixed_scores)
            else:  # common mask for all queries
                m = mask.view(-1, 1, 1, l_kv).expand_as(mixed_scores)
            mixed_scores[m] = -float('inf')

        # Compute attention weights
        weights = F.softmax(mixed_scores, dim=-1)
        # shape: (batch, head_count, l_q, l_kv)

        # Apply attention to values
        att_applied = weights.matmul(v_proj).permute(0,2,1,3).contiguous().view(
                *size, l_q, self.head_count * self.value_size_per_head)
        # shape: (N, l_q, head_count * value_size_per_head)

        return self.recombine(att_applied)

