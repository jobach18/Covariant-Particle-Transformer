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
#from torch_scatter import scatter_mean
#from torch_geometric.nn.inits import zeros, ones
import utils
from covariant_attention import CovariantAttention


def zero_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0)

def gate_linear_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(1)



class TransformerEncoderBlock(torch.nn.Module):
	def __init__(self, hidden_dim, heads,  dropout=0., update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.self_attn = ResidualSelfAttention(hidden_dim, heads, dropout=dropout, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention)
		self.feedforward = ResidualFeedForward(hidden_dim, dropout=dropout)
		
	def forward(self, x, p):
		x, p = self.self_attn(x, p)
		x = self.feedforward(x)
		return x, p


class TransformerEncoder(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, num_convs=6, heads=8, dropout=0., recycle=0, update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.num_convs = num_convs
		self.heads = heads
		self.recycle = recycle
		self.uniform_attention = uniform_attention
		self.input_embedding = MLP(in_dim, hidden_dim, hidden_dim, dropout, normalize_input=False)
		self.d_space = d_space
		self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout,  update_p=update_p, d_space=d_space, uniform_attention=uniform_attention) for i in range(self.num_convs)])

	def forward(self, x, p=None):
		x = self.input_embedding(x)
		for _ in range(self.recycle + 1):
			for encoder_block in self.encoder_blocks:
				x, p = encoder_block(x, p)
		return x, p


class TransformerDecoderBlock(torch.nn.Module):
	def __init__(self, hidden_dim,  heads, dropout=0., update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.self_attn = ResidualSelfAttention(hidden_dim, heads, dropout=dropout, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention)
		self.cross_attn = ResidualCrossAttentionBlock(hidden_dim, heads, dropout=dropout, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention)
		self.feedforward = ResidualFeedForward(hidden_dim, dropout=dropout)
		self.get_alpha = self.cross_attn.get_alpha
		self.get_phi_message_norm = self.cross_attn.get_phi_message_norm

	def forward(self, x_source, p_source, x_out, p_out):
		x_out, p_out = self.self_attn(x_out, p_out)
		x_out, p_out = self.cross_attn(x_source, p_source, x_out, p_out)
		x_out = self.feedforward(x_out)
		return x_out, p_out



class TransformerDecoder(torch.nn.Module):
	def __init__(self, in_dim, hidden_dim, num_convs=6, heads=8, dropout=0., update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.num_convs = num_convs
		self.output_embedding = MLP(in_dim, hidden_dim, hidden_dim, dropout, normalize_input=True)
		self.update_p = update_p
		self.d_space = d_space
		self.uniform_attention = uniform_attention
		self.decoder_blocks = nn.ModuleList([TransformerDecoderBlock(hidden_dim=hidden_dim, heads=heads, dropout=dropout, update_p=update_p, d_space=d_space, uniform_attention=uniform_attention) for i in range(self.num_convs)])
	
	def get_alpha(self):
		alphas = [block.get_alpha() for block in self.decoder_blocks]
		alpha = torch.stack(alphas, dim=1)
		return alpha # (|E|, L, H)

	def get_phi_message_norm(self):
		phi_message_norms = [block.get_phi_message_norm() for block in self.decoder_blocks]
		return torch.cat(phi_message_norms)
	
	def forward(self, x_source, x_out,  p_source=None, p_out=None):
		self.y_intermediates = []
		x_out = self.output_embedding(x_out)
		for decoder_block in self.decoder_blocks:
			x_out, p_out = decoder_block(x_source, p_source, x_out, p_out)
			# record intermediate predictions
		return x_out, p_out



###################################Modules###################################################

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
	def __init__(self, hidden_dim, heads, dropout=0., update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.norm_conv = LayerNorm(hidden_dim)
		self.d_space = d_space
		self.uniform_attention = uniform_attention
		self.conv = CovariantAttention(hidden_dim, heads=heads, dropout=dropout, update_p=update_p, d_space=d_space, uniform_attention=self.uniform_attention)

	def forward(self, x, p):
		x_normed = self.norm_conv(x)
		in_feat = x_normed
		m_x, p = self.conv(in_feat)
		x = x + m_x # p = p + m_p, already
		return x, p


class ResidualFeedForward(torch.nn.Module):
	def __init__(self, hidden_dim, dropout=0.):
		super().__init__()
		self.norm_fc = LayerNorm(hidden_dim)
		self.fc = Sequential(Dropout(dropout), Linear(hidden_dim, hidden_dim), LeakyReLU(), Dropout(dropout), Linear(hidden_dim, hidden_dim))

	def forward(self, x):
		return x + self.fc(self.norm_fc(x))


class ResidualCrossAttentionBlock(torch.nn.Module):
	def __init__(self, hidden_dim, heads, dropout=0., update_p=False, d_space=3, uniform_attention=False):
		super().__init__()
		self.norm_conv_source = LayerNorm(hidden_dim)
		self.norm_conv_out = LayerNorm(hidden_dim)
		self.uniform_attention = uniform_attention
		self.conv = CovariantAttention(hidden_dim, heads=heads, dropout=dropout, update_p=update_p, d_space=d_space, uniform_attention=self.uniform_attention)
	
	def get_alpha(self):
		return self.conv._alpha

	def get_phi_message_norm(self):
		return self.conv._phi_message_norm
	
	def forward(self, x_source, p_source, x_out, p_out):
		x_normed_source = self.norm_conv_source(x_source)
		x_normed_out = self.norm_conv_out(x_out)
		in_feat = torch.cat([x_normed_source, x_normed_out], axis=-1)
		m_x_out, p_out = self.conv(in_feat) # not using edge attr
		x_out = x_out + m_x_out
		return x_out, p_out


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Define the attention mechanism (a simple linear layer)
        self.attention_nn1 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Assuming x is of shape [batch_size, 2 * hidden_dim]
        
        # Compute attention scores
        attention_score1 = torch.sigmoid(self.attention_nn1(x))  # shape: [batch_size, 1]
        
        # Apply attention scores to the respective tensors
        attended_x1 = attention_score1 * x  # shape: [batch_size, hidden_dim]
        
        # Pool the information by summing 
        pooled_output = torch.sum(attended_x1, axis=1)  # shape: [batch_size, hidden_dim])
        
        return pooled_output