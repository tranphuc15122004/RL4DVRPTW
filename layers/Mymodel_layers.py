import torch
import torch.nn as nn
import torch.nn.functional as F

from ._mha import _MHA_V2


class GraphEncoderLayer(nn.Module):
	def __init__(self, head_count, model_size, ff_size, rbf_bins = 16, dropout = 0.1):
		super().__init__()
		self.head_count = head_count
		self.model_size = model_size
		self.rbf_bins = rbf_bins
		self.dropout = nn.Dropout(dropout)
		self.ff_dropout = nn.Dropout(dropout)
		if model_size % head_count != 0:
			raise ValueError("model_size must be divisible by head_count")

		head_dim = model_size // head_count
		self.q_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.k_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.v_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.out_proj = nn.Linear(head_count * head_dim, model_size, bias = False)

		self.edge_mlp = nn.Sequential(
			nn.Linear(rbf_bins, rbf_bins),
			nn.ReLU(),
			nn.Linear(rbf_bins, head_count)
		)

		self.norm1 = nn.LayerNorm(model_size)
		self.ff1 = nn.Linear(model_size, ff_size)
		self.ff2 = nn.Linear(ff_size, model_size)
		self.norm2 = nn.LayerNorm(model_size)

	def _rbf(self, dist):
		centers = torch.linspace(0, 1, self.rbf_bins, device = dist.device, dtype = dist.dtype)
		centers = centers.view(1, 1, 1, -1)
		width = (centers[..., 1] - centers[..., 0]).clamp(min = 1e-6)
		d = dist.unsqueeze(-1)
		return torch.exp(-((d - centers) ** 2) / (2 * width ** 2))

	def forward(self, h_in, cost_mat, mask = None):
		batch, length, _ = h_in.size()
		head_dim = self.model_size // self.head_count

		q = self.q_proj(h_in).view(batch, length, self.head_count, head_dim).transpose(1, 2)
		k = self.k_proj(h_in).view(batch, length, self.head_count, head_dim).transpose(1, 2)
		v = self.v_proj(h_in).view(batch, length, self.head_count, head_dim).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)

		rbf = self._rbf(cost_mat)
		edge_bias = self.edge_mlp(rbf).permute(0, 3, 1, 2)
		scores = scores + edge_bias

		if mask is not None:
			if mask.dim() == 2:
				key_mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.head_count, length, -1)
			elif mask.dim() == 3:
				key_mask = mask.unsqueeze(1).expand(-1, self.head_count, -1, -1)
			else:
				raise ValueError("mask must be 2D or 3D")
			
			# Check if any row in any head is fully masked
			all_masked = key_mask.all(dim = -1, keepdim = True)
			if all_masked.any():
				# If all entries are masked, unmask the first one to avoid nan in softmax
				key_mask = key_mask.clone()
				key_mask.masked_fill_(all_masked, False)
				
			scores = scores.masked_fill(key_mask, float('-inf'))

		attn = F.softmax(scores, dim = -1)
		attn = self.dropout(attn)
		context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, length, -1)
		att = self.out_proj(context)

		h = self.norm1(h_in + att)
		ff = self.ff2(F.relu(self.ff1(h)))
		h_out = self.norm2(h + ff)
		if mask is not None:
			if mask.dim() == 2:
				h_out[mask] = 0
			elif mask.dim() == 3:
				node_mask = mask.all(dim = -1)
				if node_mask.any():
					h_out = h_out.masked_fill(node_mask.unsqueeze(-1), 0.0)
		return h_out


class GraphEncoder(nn.Module):
	def __init__(self, layer_count, head_count, model_size, ff_size, k = None,
			adaptive_depth = False, min_layers = 1, easy_ratio = 0.6):
		super().__init__()
		self.k = k
		self.adaptive_depth = adaptive_depth
		self.min_layers = max(1, min_layers)
		self.easy_ratio = easy_ratio
		self.layers = nn.ModuleList([
			GraphEncoderLayer(head_count, model_size, ff_size) for _ in range(layer_count)
		])

	def _resolve_layer_count(self, mask, total_layers):
		if (not self.adaptive_depth) or total_layers <= 1:
			return total_layers
		min_layers = min(self.min_layers, total_layers)
		if mask is None:
			return total_layers
		visible_ratio = 1.0 - mask.float().mean().item()
		if visible_ratio >= self.easy_ratio:
			return min_layers
		span = total_layers - min_layers
		hardness = (self.easy_ratio - visible_ratio) / max(self.easy_ratio, 1e-6)
		hardness = max(0.0, min(1.0, hardness))
		return min_layers + int(round(span * hardness))

	def forward(self, inputs, mask = None, cost_mat = None, coords = None):
		"""
		:param inputs: N x L_c x D
		:param mask:   N x L_c (optional)
		:param cost_mat: N x L_c x L_c (optional)
		:param coords: N x L_c x 2 (optional, raw positions)
		:return:       N x L_c x D
		"""
		if cost_mat is None:
			pos = coords if coords is not None else inputs[:, :, :2]
			cost_mat = torch.cdist(pos, pos)
		max_val = cost_mat.amax(dim = (-1, -2), keepdim = True).clamp(min = 1e-6)
		cost_mat = cost_mat / max_val
		attn_mask = None
		if self.k is not None and self.k > 0:
			k = min(self.k, cost_mat.size(-1))
			_, knn_idx = torch.topk(cost_mat, k, largest = False)
			attn_mask = cost_mat.new_ones(cost_mat.size(), dtype = torch.bool)
			attn_mask.scatter_(2, knn_idx, False)
		if mask is not None:
			key_mask = mask[:, None, :].expand(-1, cost_mat.size(1), -1)
			attn_mask = key_mask if attn_mask is None else (attn_mask | key_mask)
			query_mask = mask[:, :, None].expand(-1, -1, cost_mat.size(2))
			attn_mask = query_mask if attn_mask is None else (attn_mask | query_mask)
		h = inputs
		use_layers = self._resolve_layer_count(mask, len(self.layers))
		for layer in self.layers[:use_layers]:
			h = layer(h, cost_mat, attn_mask if attn_mask is not None else mask)
		return h


class FleetEncoderLayer(nn.Module):
	def __init__(self, head_count, model_size, ff_size):
		super().__init__()
		self.mha = _MHA_V2(head_count, model_size)
		self.norm1 = nn.LayerNorm(model_size)
		self.ff1 = nn.Linear(model_size, ff_size)
		self.ff2 = nn.Linear(ff_size, model_size)
		self.norm2 = nn.LayerNorm(model_size)

	def forward(self, veh_repr, cust_repr, mask = None):
		att = self.mha(veh_repr, keys = cust_repr, values = cust_repr, mask = mask)
		h = self.norm1(veh_repr + att)
		ff = self.ff2(F.relu(self.ff1(h)))
		h_out = self.norm2(h + ff)
		if mask is not None:
			if mask.dim() == 2 and mask.size(1) == h_out.size(1):
				h_out = h_out.masked_fill(mask.unsqueeze(-1), 0.0)
			elif mask.dim() == 3:
				node_mask = mask.all(dim = -1)
				if node_mask.any():
					h_out = h_out.masked_fill(node_mask.unsqueeze(-1), 0.0)
		return h_out


class FleetEncoder(nn.Module):
	def __init__(self, layer_count, head_count, model_size, ff_size,
			adaptive_depth = False, min_layers = 1, easy_ratio = 0.6):
		super().__init__()
		self.input_proj = nn.LazyLinear(model_size)
		self.adaptive_depth = adaptive_depth
		self.min_layers = max(1, min_layers)
		self.easy_ratio = easy_ratio
		self.layers = nn.ModuleList([
			FleetEncoderLayer(head_count, model_size, ff_size) for _ in range(layer_count)
		])

	def _resolve_layer_count(self, mask, total_layers):
		if (not self.adaptive_depth) or total_layers <= 1:
			return total_layers
		min_layers = min(self.min_layers, total_layers)
		if mask is None:
			return total_layers
		if mask.dim() == 3:
			feasible_ratio = (~mask).float().mean().item()
		elif mask.dim() == 2:
			feasible_ratio = (~mask).float().mean().item()
		else:
			return total_layers
		if feasible_ratio >= self.easy_ratio:
			return min_layers
		span = total_layers - min_layers
		hardness = (self.easy_ratio - feasible_ratio) / max(self.easy_ratio, 1e-6)
		hardness = max(0.0, min(1.0, hardness))
		return min_layers + int(round(span * hardness))

	def forward(self, vehicles, cust_repr, mask = None):
		"""
		:param vehicles:    N x L_v x D_v
		:param cust_repr:   N x L_c x D
		:param mask:        N x L_v x L_c or N x L_c (optional)
		:return:            N x L_v x D
		"""
		h = self.input_proj(vehicles)
		use_layers = self._resolve_layer_count(mask, len(self.layers))
		for layer in self.layers[:use_layers]:
			h = layer(h, cust_repr, mask)
		return h


class EdgeFeatureEncoder(nn.Module):
	def __init__(self, edge_feat_size, model_size):
		super().__init__()
		self.edge_feat_size = edge_feat_size
		self.model_size = model_size
		self.net = nn.Sequential(
			nn.Linear(edge_feat_size, model_size),
			nn.ReLU(),
			nn.Linear(model_size, model_size),
			nn.LayerNorm(model_size)
		)

	def forward(self, edge_feat):
		"""
		:param edge_feat: N x 1 x L_c x E
		:return:          N x 1 x L_c x D
		"""
		return self.net(edge_feat)


class CrossEdgeFusion(nn.Module):
	def __init__(self, head_count, model_size):
		super().__init__()
		self.head_count = head_count
		self.model_size = model_size
		if model_size % head_count != 0:
			raise ValueError("model_size must be divisible by head_count")
		head_dim = model_size // head_count
		self.q_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.k_proj = nn.Linear(model_size, head_count * head_dim, bias = False)
		self.edge_bias = nn.Linear(model_size, head_count, bias = False)

	def forward(self, veh_repr, cust_repr, edge_emb):
		"""
		:param veh_repr: N x 1 x D
		:param cust_repr: N x L_c x D
		:param edge_emb: N x 1 x L_c x D
		:return:         N x 1 x L_c
		"""
		batch, _, dim = veh_repr.size()
		_, length, _ = cust_repr.size()
		head_dim = dim // self.head_count

		q = self.q_proj(veh_repr).view(batch, 1, self.head_count, head_dim).transpose(1, 2)
		k = self.k_proj(cust_repr).view(batch, length, self.head_count, head_dim).transpose(1, 2)
		scores = torch.matmul(q, k.transpose(-1, -2)) * (head_dim ** -0.5)

		bias = self.edge_bias(edge_emb).permute(0, 3, 1, 2)
		scores = scores + bias

		return scores.mean(dim = 1)


class CoordinationMemory(nn.Module):
	def __init__(self, veh_state_size, hidden_size):
		super().__init__()
		self.veh_state_size = veh_state_size
		self.hidden_size = hidden_size
		self.input_proj = nn.LazyLinear(hidden_size)
		self.hidden_proj = nn.Linear(hidden_size, hidden_size)

	def update(self, memory, veh_idx, veh_repr, cust_repr, edge_emb):
		"""
		:param memory:   N x L_v x H
		:param veh_idx:  N x 1
		:param veh_repr: N x 1 x D
		:param cust_repr: N x 1 x D
		:param edge_emb: N x 1 x 1 x D
		:return:         N x L_v x H (updated)
		"""
		cur_h = memory.gather(1, veh_idx[:, :, None].expand(-1, -1, memory.size(-1)))
		x = torch.cat((
			veh_repr.squeeze(1),
			cust_repr.squeeze(1),
			edge_emb.squeeze(2).squeeze(1)
		), dim = -1)
		next_h = torch.tanh(self.input_proj(x) + self.hidden_proj(cur_h.squeeze(1)))
		next_h = next_h.to(memory.dtype)
		updated = memory.scatter(1, veh_idx[:, :, None].expand(-1, -1, memory.size(-1)), next_h.unsqueeze(1))
		return updated


class OwnershipHead(nn.Module):
	def __init__(self, model_size):
		super().__init__()
		self.model_size = model_size
		self.veh_proj = nn.LazyLinear(model_size)
		self.cust_proj = nn.Linear(model_size, model_size, bias = False)

	def forward(self, veh_memory, cust_repr):
		"""
		:param veh_memory: N x L_v x H
		:param cust_repr:  N x L_c x D
		:return:           N x L_v x L_c
		"""
		v = self.veh_proj(veh_memory)
		c = self.cust_proj(cust_repr)
		logits = torch.matmul(v, c.transpose(1, 2))
		logits *= (self.model_size ** -0.5)
		return logits


class LookaheadHead(nn.Module):
	def __init__(self, model_size, hidden_size = 128, dropout = 0.1):
		super().__init__()
		self.model_size = model_size
		self.hidden_size = hidden_size
		self.net = nn.Sequential(
			nn.LazyLinear(hidden_size),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_size, 1)
		)

	def forward(self, veh_repr, cust_repr, edge_emb):
		"""
		:param veh_repr: N x 1 x D
		:param cust_repr: N x L_c x D
		:param edge_emb: N x 1 x L_c x D
		:return:         N x 1 x L_c
		"""
		batch, _, dim = veh_repr.size()
		length = cust_repr.size(1)
		veh_expand = veh_repr.expand(-1, length, -1)
		edge_flat = edge_emb.squeeze(1)
		feat = torch.cat((veh_expand, cust_repr, edge_flat), dim = -1)
		out = self.net(feat).transpose(1, 2)
		return out