'''
CPT Implementation for columnar data
'''
import os
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, LeakyReLU, Dropout, LayerNorm
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from transformer_columnar import TransformerEncoder, TransformerDecoder, GraphMLP, GlobalAttention
from base_model import BaseModel
import matplotlib.pyplot as plt
import utils
from tqdm import tqdm
from functools import partial
import numpy as np
tqdm = partial(tqdm, position=0, leave=True)


class ColumnarCovariantTopFormer(BaseModel):
	def __init__(self, in_dim, hidden_dim, out_dim, max_num_output, output_dir, use_gpu=True, lr=1e-4, schedule_lr=False, num_convs=(3, 3), heads=8, dist_scale=1, beta=0.5, dropout=0., match_scale_factor=1, p_norm=2, mass=172.76, d_space=3, uniform_attention=False):
		self.hidden_dim = hidden_dim
		self.num_convs = num_convs
		self.dropout = dropout
		self.max_num_output = max_num_output
		self.heads = heads
		self.d_space = d_space
		self.uniform_attention = uniform_attention
		super().__init__(in_dim, out_dim, output_dir, use_gpu, lr, schedule_lr)
		self.cross_entropy_loss = nn.CrossEntropyLoss()
		self.beta = beta # weight for the loss on predicting the number of tops
		self.match_scale_factor = match_scale_factor.to(self.device)
		self.p_norm = p_norm
		self.loss_scale_factor = 1/torch.FloatTensor([100, 100, 1, 5]).to(self.device) # px, py, eta, m
		self.ghost_value = utils.get_ghost_value()
		self.mass = mass
		self.detector = True
		self.dim_labels = utils.get_plot_configs(detector=True)[0]
		self.pi = 3.1415926
		
    def define_modules(self):
        self.invariant_encoder = TransformerEncoder(in_dim=self.in_dim - 2, hidden_dim=self.hidden_dim, num_convs=self.num_convs[0], heads=self.heads, dropout=self.dropout, geometric=self.geometric, update_p=False, d_space=self.d_space, uniform_attention=self.uniform_attention)
        self.pre_decoder = TransformerDecoder(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, num_convs=1, heads=self.heads, dropout=self.dropout, geometric=False)
        self.covariant_decoder = TransformerDecoder(in_dim=self.hidden_dim, hidden_dim=self.hidden_dim, num_convs=self.num_convs[1], heads=self.heads, dropout=self.dropout, geometric=self.geometric, update_p=True, d_space=self.d_space, uniform_attention=self.uniform_attention)
        self.out_linear = Linear(self.hidden_dim, 2) # pT and mass
        self.pool_attn_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, 1))
        self.pool_value_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim))
        self.pool = GlobalAttention(self.pool_attn_nn, self.pool_value_nn)
        self.readout_key_nn = GraphMLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout, normalize_input=True)
        self.readout_value_nn = GraphMLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.dropout, normalize_input=True)
        self.readout = Set2Set(in_channels=self.hidden_dim, num_outputs=self.max_num_output, num_layers=1)
        self.count_logits_nn = Sequential(LayerNorm(self.hidden_dim), Dropout(self.dropout), Linear(self.hidden_dim, self.hidden_dim), LeakyReLU(), Dropout(self.dropout), Linear(self.hidden_dim, self.max_num_output))



