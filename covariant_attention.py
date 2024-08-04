import math
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import softmax

class CovariantAttention():
	def __init__(self, hidden_dim: int,
				 heads: int = 1, gate: bool = True,
				 dropout: float = 0.,
				 bias: bool = True, d_space=3, update_p=False, uniform_attention=False, **kwargs):
		super().__init__(node_dim=0, **kwargs)

		self.hidden_dim = hidden_dim
		self.heads = heads
		self.gate = gate
		self.dropout = dropout
		self.d_space = d_space
		self.update_p = update_p
		self._alpha = None
		self._phi_message_norm = None
		self.pi = 3.1415926
		self.uniform_attention = uniform_attention

		self.lin_query = Linear(hidden_dim, hidden_dim, bias=False)
		self.lin_key = Linear(hidden_dim, hidden_dim, bias=False)
		self.lin_value = Linear(hidden_dim, hidden_dim, bias=False)
		
		if self.gate:
			self.gate_nn = Sequential(Dropout(dropout), Linear(2 * hidden_dim, hidden_dim), Sigmoid())

		self.reset_parameters()

	def reset_parameters(self):
		self.lin_key.reset_parameters()
		self.lin_query.reset_parameters()
		self.lin_value.reset_parameters()
		if self.gate:
			self.gate_nn.apply(gate_linear_init_weights) # w = 0, b = 1

	def forward(self, x: Union[Tensor, PairTensor],
				edge_attr: OptTensor = None):
	
		if isinstance(x, Tensor):
			in_feat: PairTensor = (x, x)
		else:
			in_feat = x

		# propagate_type: (x: PairTensor, edge_attr: OptTensor)
		m = self.propagate(edge_index, x=in_feat, edge_attr=edge_attr, size=None) # aggreagated messgaes 
		m = m.contiguous().view(-1, self.hidden_dim) # (|E|, D)
        x = in_feat[1] # initial target feat
        p = None
        m_x = m
        m_p = None

		if self.gate:
			m_x = self.gate_nn(torch.cat([x, m_x], dim=-1)) * m_x

		return m_x, p

	def normalize_phi_vec(self, p, eps=1e-8, store_norm=False):
		phi_vec = p[:, 1:]
		phi_message_norm = phi_vec.norm(dim=-1, keepdim=True)
		if store_norm:
			self._phi_message_norm = phi_message_norm.view(-1)
		phi_vec = phi_vec / (phi_message_norm + eps)
		return torch.cat([p[:, 0].view(-1, 1), phi_vec], dim=-1)


	def to_angle(self, phi):
		# return phi in [-pi, pi]
		return (phi + self.pi).remainder(2*self.pi) - self.pi 

	def rotation_from_vec(self, v):
		""" 
		v = [c, s] # (B, 2)
		R = [[c, -s]
			 [s, c]]
		"""
		c = v[..., 0] # (B, )
		s = v[..., 1] # (B, )
		r = torch.stack([c, -s, s, c], dim=-1) # (B, 4)
		r = r.reshape(-1, 2, 2) # (B, 2, 2)
		return r

	def rotate(self, phi2, phi1, inverse=False):
		# phi : (B, 2)
		r = self.rotation_from_vec(phi1) # (B, 2, 2)
		if inverse:
			r = torch.transpose(r, -1, -2) # transpose to get the inverse (B, 2, 2)
		phi21 = torch.einsum('...ij, ...j', r, phi2) # rotate phi2 by -phi1
		return phi21

	def boost(self, p2, p1):
		# boost p2 into p1's frame
		# p = [eta, cos(phi), sin(phi)], (cos(phi), sin(phi)) is a unit vector representing the phi-angle
		eta21 = p2[..., 0] - p1[..., 0]
		phi1, phi2 = p1[..., 1:], p2[..., 1:] 
		phi21 = self.rotate(phi2, phi1, inverse=True)
		p21 = torch.cat([eta21.view(-1, 1), phi21], dim=-1)
		return p21

	def message(self, x_i: Tensor, x_j: Tensor, 
				index: Tensor, ptr: OptTensor,
				size_i: Optional[int]) -> Tensor:

		query = self.lin_query(x_i).view(-1, self.heads, self.hidden_dim // self.heads)
        key = self.lin_key(x_j).view(-1, self.heads, self.hidden_dim // self.heads)
        value = self.lin_value(x_j).view(-1, self.heads, self.hidden_dim // self.heads)

		if not self.uniform_attention:
			alpha = (query * key).sum(dim=-1) / math.sqrt(self.hidden_dim // self.heads) + edge_bias
		else:
			alpha = torch.zeros(query.shape[0], query.shape[1]).to(query.device)
		alpha = softmax(alpha, index, ptr, size_i)
		self._alpha = alpha

		m = value * alpha.view(-1, self.heads, 1) # invariant messages (|E|, H, D // H)
		return m

	def __repr__(self):
		return '{}({}, {}, heads={})'.format(self.__class__.__name__,
											 self.hidden_dim,
											 self.hidden_dim, self.heads)