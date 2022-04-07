#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 15:30:28 2022

@author: wangcatherine
"""

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from TransformerNetwork import * 
from t2v import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



# hyperparameter
window_size = 128
enc_seq_len = window_size
dec_seq_len = 2
output_sequence_length = 1

dim_val = 5 # input data # features
dim_attn = 128
lr = 0.000001
epochs = 50

n_heads = 4 

n_decoder_layers = 2
n_encoder_layers = 2
batch_size = 32


# time to vec
time_embed_size = 2
#=======================================================================

df = pd.read_csv("cleaned_data")
X = df.iloc[:,1:]
y = df["Close"]
x,y = X.to_numpy(),y.to_numpy()

def sliding_window(x,y, window_size):
    xx, yy = [], []
    for i in range(window_size,len(x)):
        xx.append(x[i-window_size:i])
        yy.append(y[i])
    xx, yy = np.array(xx), np.array(yy)
    return xx, yy

x,y = sliding_window(x,y,window_size)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

x_train, x_test = train_test_split(x, test_size=0.2, random_state=1, shuffle=False)

x_train, x_val= train_test_split(x_train, test_size=0.25, random_state=1, shuffle=False) # 0.25 x 0.8 = 0.2

model = Transformer(dim_val, dim_attn, batch_size, enc_seq_len, dec_seq_len, output_sequence_length, 
                    n_decoder_layers, n_encoder_layers, n_heads, time_embed_size)


loader = DataLoader(x_train.astype(np.float32), batch_size=batch_size, 
                    shuffle=False, drop_last=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fc = torch.nn.MSELoss()
#print(x_train[:900,:,3].shape)

losses = []
losses1 = []

for i in range(epochs):
    print("epoch \n",i)
    for data in loader:
        x_train = data
        #print("oooo",x_train.shape)
        #x_train = torch.from_numpy(x_train)
        y_train = data[:,:,3] #  close price
      
        y_pred = model(x_train)
        #print(net_out.shape,Y.shape)
        loss = loss_fc(y_pred,y_train)
        #loss = torch.mean((y_pred - y_train) ** 2)
        loss1 = torch.mean(torch.div(torch.abs((y_pred - y_train)),y_train))

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        losses1.append(loss1.item())
    

print(losses1)
plt.plot(losses)


    
    
    
    


