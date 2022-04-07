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
from cleanData import clean_data

# hyperparameter
window_size = 128
enc_seq_len = window_size
dec_seq_len = 2
output_sequence_length = 1

dim_val = 5 # input data # features
dim_attn = 128
lr = 0.000001
epochs = 20

n_heads = 4 

n_decoder_layers = 2
n_encoder_layers = 3
batch_size = 64


# time to vec
time_embed_size = 2
#=======================================================================

filename = 'AMZN'
clean_data(filename)
df = pd.read_csv(filename+'_cleaned.csv')
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




optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fc = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
#print(x_train[:900,:,3].shape)


def epoch_train(dataloader):
    model.train()
    batch_size = len(next(iter(dataloader)))
    losses = 0 # mse
    losses1 = 0 # percentage loss
    
    for data in iter(dataloader):
        x_train = data
        y_train = data[:,:,3] #  close price
  
        y_pred = model(x_train)
        loss = loss_fc(y_pred,y_train)
        #loss1 = torch.mean(torch.div(torch.abs(y_pred - y_train),y_train))
        loss1 = mae(y_pred,y_train)
        
        loss.backward()
        optimizer.step()
        
        losses += loss.item()/batch_size
        losses1 += loss1.item()/batch_size #mae
                
    return losses,losses1

def epoch_test(dataloader):
    model.eval()
    batch_size = len(next(iter(dataloader)))
    losses = 0 # mse
    losses1 = 0 # percentage loss
    for data in dataloader:
        x_test = data
        y_test = data[:,:,3]
        y_pred = model(x_test)
        loss = loss_fc(y_pred,y_test)
        loss1 = mae(y_pred,y_test)

        #loss1 = torch.mean(torch.div(torch.abs(y_pred - y_test),y_test))
        
        losses += loss.item()/batch_size
        losses1 += loss1.item()/batch_size #mae
        
        
    return losses,losses1
    

def run_epoch(epochs):
    train = []
    val = []
    train_loader = DataLoader(x_train.astype(np.float32), batch_size=batch_size, 
                    shuffle=False, drop_last=True)
    val_loader = DataLoader(x_val.astype(np.float32), batch_size=batch_size, 
                    shuffle=False, drop_last=True)
    for epoch in range(epochs):
        train_mse, train_mae = epoch_train(train_loader)
        train.append(train_mse)
        val_mse, val_mae = epoch_test(val_loader)
        val.append(val_mse)
        print('Epoch[{}/{}] | train(mse):{:.6f}, mae:{:.6f} | test(mse):{:.6f}, mae:{:.6f}'
              .format(epoch+1, epochs, train_mse, train_mae, val_mse, val_mae))
    plt.plot(train)
    plt.plot(val)
    plt.show()

run_epoch(epochs)    
    
    
    


