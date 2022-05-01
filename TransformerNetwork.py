#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 09:44:23 2022

@author: wangcatherine
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from MultiheadAttn import *
from t2v import *


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        a = self.fc1(F.relu(self.fc2(x)))
        x = self.norm2(x + a)

        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)

        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)

        a = self.fc1(F.relu(self.fc2(x)))

        x = self.norm3(x + a)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, batch_size, enc_seq_len, dec_seq_len, out_seq_len,
                 n_decoder_layers=1, n_encoder_layers=1, n_heads=1, time_embd=2):
        super(Transformer, self).__init__()
        self.dim_val = dim_val
        self.input_size = input_size + time_embd
        self.dec_seq_len = dec_seq_len
        self.batch_size = batch_size
        self.enc_seq_len = enc_seq_len
        self.time_embd = time_embd

        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for _ in range(n_encoder_layers):
            self.encs.append(EncoderLayer(self.dim_val, dim_attn, n_heads))

        # self.decs = nn.ModuleList()
        # for i in range(n_decoder_layers):
        #     self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads))

        self.time = SineActivation(input_size, time_embd)
        # self.pos = PositionalEncoding(dim_val)

        # /////////////////////////////--------------------------------------------------
        # self.fc1 = nn.linear(self.dim_val,)

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(self.input_size, dim_val)
        # self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(self.dim_val * enc_seq_len, 1)

    def forward(self, x):
        # encoder
        # x = self.enc_input_fc(x)
        x = torch.reshape(x, (self.batch_size * self.enc_seq_len, self.input_size - self.time_embd))
        t2v = self.time(x)
        x = torch.cat((x, t2v), axis=-1)  # concatenate t2v to original data
        x = torch.reshape(x, (self.batch_size, self.enc_seq_len, self.input_size))
        x = self.enc_input_fc(x)
        # x = self.pos(x)

        e = self.encs[0](x)
        for enc in self.encs[1:]:
            e = enc(e)

        # decoder
        # d = self.decs[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), e)
        # for dec in self.decs[1:]:
        # d = dec(d, e)

        # output
        e = F.relu(e)
        x = self.out_fc(e.flatten(start_dim=1))

        return x
