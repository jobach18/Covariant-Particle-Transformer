'''
columnar rewrite of the transformer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU, Dropout, LayerNorm, Identity, Sigmoid
from typing import Optional
import torch
from torch import Tensor
from torch_scatter import scatter_mean
from torch_geometric.nn.inits import zeros, ones
import utils


def zero_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0)

def gate_linear_init_weights(m):
    if isinstance(m, nn.Linear)
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(1)



class TransformerEncoderBlock(torch.nn.Module):
	def __init__(self, hidden_dim, heads,  dropout=0., update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.self_attn = ResidualSelfAttention(hidden_dim, heads, edge_dim, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention)
		self.feedforward = ResidualFeedForward(hidden_dim, dropout=dropout)
		self.d_space = d_space
		if self.edge_dim != None:
			self.edge_nn = Sequential(Dropout(dropout), LayerNorm(edge_dim + 2 * hidden_dim), Linear(edge_dim + 2 * hidden_dim, edge_dim), LeakyReLU(), Dropout(dropout), Linear(edge_dim, edge_dim))
		
	def forward(self, x, p, edge_index, edge_attr):
		x, p = self.self_attn(x, p, edge_index, edge_attr)
		x = self.feedforward(x)
		if self.edge_dim != None:
			x_i, x_j = x[edge_index[0]], x[edge_index[1]]
			edge_attr = edge_attr + self.edge_nn(torch.cat([edge_attr, x_i, x_j], dim=-1))
		return x, p, edge_attr


class TransformerEncoder(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, num_convs=6, heads=8, dropout=0., recycle=0, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.num_convs = num_convs
		self.heads = heads
		self.recycle = recycle
		self.uniform_attention = uniform_attention
		self.input_embedding = MLP(in_dim, hidden_dim, hidden_dim, dropout, normalize_input=False)
		self.d_space = d_space
		self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(hidden_dim=hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention) for i in range(self.num_convs)])

	def forward(self, x, edge_index, edge_attr, p=None):
		x = self.input_embedding(x)
		for _ in range(self.recycle + 1):
			for encoder_block in self.encoder_blocks:
				x, p, edge_attr = encoder_block(x, p, edge_index, edge_attr)
		if self.edge_dim:
			return x, p, edge_attr
		return x, p



class MLP(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, dropout, normalize_input):
		super().__init__()
		if normalize_input:
			self.norm_input = LayerNorm(in_dim)
		else:
			self.norm_input = torch.nn.Identity(in_dim)
		self.linear1 = Linear(in_dim, hidden_dim)
		self.act1 = LeakyReLU()
		self.norm2 = LayerNorm(hidden_dim)
		self.dropout2 = Dropout(dropout)
		self.linear2 = Linear(hidden_dim, out_dim)
	def forward(self, x):
		x = self.linear1(x)
		x = self.act1(x)
		x = self.norm2(x)
		x = self.dropout2(x)
		x = self.linear2(x)
		return x



class ResidualSelfAttention(torch.nn.Module):
	def __init__(self, hidden_dim, heads, edge_dim, dropout=0., geometric=False, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.norm_conv = LayerNorm(hidden_dim)
		self.d_space = d_space
		self.uniform_attention = uniform_attention
		self.conv = CovariantGraphAttention(hidden_dim, heads=heads, edge_dim=edge_dim, dropout=dropout, geometric=geometric, update_p=update_p, d_space=d_space, uniform_attention=self.uniform_attention)

	def forward(self, x, p, edge_index, edge_attr):
		x_normed = self.norm_conv(x)
		if self.geometric:
			in_feat = torch.cat([p, x_normed], dim=-1)
		else:
			in_feat = x_normed
		if self.edge_dim != None:
			m_x, p = self.conv(in_feat, edge_index, self.norm_edge(edge_attr))
			x = x + m_x # p = p + m_p, already
		else:
			m_x, p = self.conv(in_feat, edge_index)
			x = x + m_x # p = p + m_p, already
		# if self.geometric:
		# 	print('self_attn:\n', p) # __debug__
		return x, p